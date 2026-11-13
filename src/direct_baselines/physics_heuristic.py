"""Physics-informed *direct* scorers (policy-side).

These are not A6's learned / residual world model. They are cheap, explicit
mechanical heuristics that map (observation, optional probe feedback, candidate)
to a predicted settled object position, then score by -distance-to-goal.

Toy mechanism (same statements as A4 USAGE, used as inductive bias):
- During continuous contact the object is assumed to follow the pusher (mu-independent).
- After release, the object slides and stops with roughly slide ≈ v^2 / (2 a), a = mu * g.
- The history version estimates a single scalar a from the *same* probe feedback the
  world-model route would see (obs before/after the executed probe, plus the probe actions).
  Informative case: tap_fast (push then trailing zeros). Uninformative probes leave a at
  a fixed nominal prior (mu_nominal=0.25, A4's mid value, a constant — not env.hidden).

Never reads env.hidden, never branches the real env.
"""
from __future__ import annotations

import numpy as np
from shared.interface import DT, SUBSTEP, TABLE, V_MAX, clip_action

G = 9.81
R_P, R_O = 0.03, 0.05
# Fixed prior used when history does not identify deceleration. Not a hidden input.
MU_NOMINAL = 0.35  # midpoint of official main pair 0.1 vs 0.6; not fitted on test
A_NOMINAL = MU_NOMINAL * G
A_MIN, A_MAX = 0.05 * G, 1.0 * G
SETTLE_MAX = 20  # A10 frozen_sets.json offline.settle_steps
REST_SPEED = 1e-3
N_SUB = int(round(DT / SUBSTEP))
EPS = 1e-9


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def estimate_decel(observation: dict, history) -> tuple[float, bool, dict]:
    """Estimate Coulomb deceleration a [m/s^2] from executed history.

    Returns (a_est, informed, debug). `informed=True` only when the last segment
    has a push-then-release (trailing zero commands) and the object actually moved
    or has residual velocity — the tap_fast pattern. push_slow / retreat / empty
    history are not treated as identifying and keep A_NOMINAL.
    """
    debug = {"a_est": A_NOMINAL, "informed": False, "reason": "no_history"}
    if not history:
        return A_NOMINAL, False, debug
    last = history[-1]
    acts = np.asarray(last.actions, dtype=np.float64).reshape(-1, 2)
    if acts.shape[0] == 0:
        return A_NOMINAL, False, debug
    speeds = np.linalg.norm(acts, axis=1)
    # trailing zeros = release window
    rel = 0
    for s in speeds[::-1]:
        if s < 1e-9:
            rel += 1
        else:
            break
    t_rel = rel * DT
    push_speeds = speeds[speeds >= 1e-9]
    if t_rel < DT - 1e-12 or push_speeds.size == 0:
        debug["reason"] = "no_release_or_no_push"
        return A_NOMINAL, False, debug
    v_cmd = float(np.max(push_speeds))  # commanded contact speed
    v_end = float(np.linalg.norm(_v(last.next_obs["obj_vel"])))
    # If the object never got going, this probe did not identify a.
    obj_disp = float(np.linalg.norm(_v(last.next_obs["obj"]) - _v(last.obs["obj"])))
    if obj_disp < 1e-4 and v_end < REST_SPEED:
        debug["reason"] = "no_object_motion"
        return A_NOMINAL, False, debug
    # v_end ≈ max(0, v_release - a * t_rel); v_release ≈ v_cmd under kinematic contact.
    if v_end < REST_SPEED:
        # stopped before the end of the release window => a >= v_cmd / t_rel
        a_est = v_cmd / max(t_rel, EPS)
        debug["reason"] = "stopped_during_release"
    else:
        a_est = (v_cmd - v_end) / max(t_rel, EPS)
        debug["reason"] = "residual_vel"
    a_est = float(np.clip(a_est, A_MIN, A_MAX))
    debug.update({"a_est": a_est, "informed": True, "t_rel": t_rel, "v_cmd": v_cmd, "v_end": v_end})
    return a_est, True, debug


def rollout_settled(observation: dict, actions: np.ndarray, a_fric: float,
                    settle: bool = True, settle_max: int = SETTLE_MAX) -> np.ndarray:
    """Cheap kinematic pusher + contact + Coulomb slide. Returns settled object xy."""
    p = _v(observation["pusher"]).copy()
    o = _v(observation["obj"]).copy()
    ov = _v(observation["obj_vel"]).copy()
    a_fric = float(np.clip(a_fric, A_MIN, A_MAX))
    acts = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    lim = TABLE - R_P

    def _sub(vcmd):
        nonlocal p, o, ov
        vcmd = np.clip(vcmd, -V_MAX, V_MAX)
        newp = np.clip(p + vcmd * SUBSTEP, -lim, lim)
        pvel = (newp - p) / SUBSTEP
        p = newp
        d = o - p
        dist = float(np.linalg.norm(d))
        if dist < R_P + R_O and dist > 1e-12:
            n = d / dist
            o = p + n * (R_P + R_O)
            vn = float(pvel @ n)
            ovn = float(ov @ n)
            if vn > ovn:
                ov = ov + (vn - ovn) * n
        sp = float(np.linalg.norm(ov))
        if sp > 0:
            ov = ov * (max(0.0, sp - a_fric * SUBSTEP) / sp)
        o = o + ov * SUBSTEP

    for v in acts:
        for _ in range(N_SUB):
            _sub(v)
    if settle:
        k = 0
        while k < settle_max and float(np.linalg.norm(ov)) >= REST_SPEED:
            for _ in range(N_SUB):
                _sub(np.zeros(2))
            k += 1
    return o


def predicted_goal_error(observation: dict, actions: np.ndarray, a_fric: float) -> float:
    goal = _v(observation["goal"])
    end = rollout_settled(observation, actions, a_fric)
    return float(np.linalg.norm(end - goal))


def score_candidates_physics(observation, history, candidates, use_history: bool) -> np.ndarray:
    if use_history:
        a, _, _ = estimate_decel(observation, history)
    else:
        a = A_NOMINAL
    cands = np.asarray(candidates, dtype=np.float64)
    return np.array([-predicted_goal_error(observation, c, a) for c in cands], dtype=np.float64)


def physics_features(observation: dict, actions: np.ndarray, history, use_history: bool) -> np.ndarray:
    """Extra features (5,): a_used, informed, pred_err_used, pred_err_nominal, delta_err."""
    a_nom = A_NOMINAL
    if use_history:
        a_use, informed, _ = estimate_decel(observation, history)
    else:
        a_use, informed = a_nom, False
    err_use = predicted_goal_error(observation, actions, a_use)
    if informed and abs(a_use - a_nom) > 1e-12:
        err_nom = predicted_goal_error(observation, actions, a_nom)
    else:
        err_nom = err_use
    return np.array([
        a_use,
        1.0 if informed else 0.0,
        err_use,
        err_nom,
        err_use - err_nom,
    ], dtype=np.float64)
