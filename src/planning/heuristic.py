"""Low-cost nominal-physics predictor and a goal-distance scorer.

Predictor internally simulates the published A4 *mechanism* (actuator lag, Coulomb
slide, inelastic disc contact) with a FIXED nominal mu (default 0.3). It never reads
the live env or its private physical parameters. Mid-episode state is taken from the visible observation
(including obj_vel). Actuator lag is initialised at 0 when only an observation is
available (mid-episode lag state is not in the observation) — a documented approximation.

This is a stand-in so Best-of-N can run before A6's predictor exists. Label all
numbers that use it as [toy-run] / nominal-mu predictor, not a learned world model.
"""
from __future__ import annotations

import math

import numpy as np

from shared.interface import DT, SUBSTEP, TABLE, Prediction, clip_action

G = 9.81
R_P, R_O = 0.03, 0.05
TAU_ACT = 0.02
REST_SPEED = 1e-3
NOMINAL_MU = 0.3
N_SUB = int(round(DT / SUBSTEP))


def _substep(p, o, ov, pv, v_cmd, mu: float, tau_act: float):
    """A4-like order: friction on object, then actuator+pusher, then contact, then integrate o."""
    sp = float(np.linalg.norm(ov))
    if sp > 0:
        ov = ov * (max(0.0, sp - mu * G * SUBSTEP) / sp)
    if tau_act > 0:
        pv = pv + (v_cmd - pv) * (1.0 - math.exp(-SUBSTEP / tau_act))
    else:
        pv = v_cmd.copy()
    lim = TABLE - R_P
    newp = np.clip(p + pv * SUBSTEP, -lim, lim)
    pvel = (newp - p) / SUBSTEP
    p = newp
    d = o - p
    dist = float(np.linalg.norm(d))
    if dist < R_P + R_O - 1e-12 and dist > 1e-12:
        n = d / dist
        o = p + n * (R_P + R_O)
        vn = float(pvel @ n)
        ovn = float(ov @ n)
        if vn > ovn:
            ov = ov + (vn - ovn) * n
    o = o + ov * SUBSTEP
    return p, o, ov, pv


def rollout_nominal(observation: dict, actions: np.ndarray, mu: float = NOMINAL_MU,
                    max_time: float = 6.0, tau_act: float = TAU_ACT,
                    pv0: np.ndarray | None = None):
    """Deterministic nominal rollout from a visible observation.

    Returns dict with endpoint (2,), traj (H,2), success, done, termination, goal_error.
    """
    p = np.asarray(observation["pusher"], dtype=np.float64).copy()
    o = np.asarray(observation["obj"], dtype=np.float64).copy()
    ov = np.asarray(observation["obj_vel"], dtype=np.float64).copy()
    pv = np.zeros(2, dtype=np.float64) if pv0 is None else np.asarray(pv0, dtype=np.float64).copy()
    goal = np.asarray(observation["goal"], dtype=np.float64)
    goal_radius = float(observation.get("goal_radius", 0.05))
    t = float(observation.get("t", 0.0))
    actions = np.asarray(actions, dtype=np.float64)
    H = int(actions.shape[0])
    traj = np.zeros((H, 2), dtype=np.float64)
    success = False
    done = False
    termination = None
    last_o = o.copy()
    for i, a in enumerate(actions):
        v = clip_action(np.asarray(a, dtype=np.float64).reshape(2))
        for _ in range(N_SUB):
            p, o, ov, pv = _substep(p, o, ov, pv, v, mu, tau_act)
        t = round(t + DT, 10)
        traj[i] = o
        last_o = o
        if np.any(np.abs(o) > TABLE):
            done, termination = True, "out_of_bounds"
            if i + 1 < H:
                traj[i + 1 :] = o
            break
        if np.linalg.norm(o - goal) < goal_radius and np.linalg.norm(ov) < REST_SPEED:
            done, success, termination = True, True, "success"
            if i + 1 < H:
                traj[i + 1 :] = o
            break
        if t >= max_time - 1e-9:
            done, termination = True, "timeout"
            if i + 1 < H:
                traj[i + 1 :] = o
            break
    goal_error = float(np.linalg.norm(last_o - goal))
    return {
        "endpoint": last_o.copy(),
        "traj": traj,
        "success": success,
        "done": done,
        "termination": termination,
        "goal_error": goal_error,
        "t": t,
        "pusher": p.copy(),
        "obj_vel": ov.copy(),
    }


class NominalPredictor:
    """Predictor protocol: predict(observation, actions=(K,H,2), context) -> Prediction."""

    def __init__(self, mu: float = NOMINAL_MU, max_time: float = 6.0, tau_act: float = TAU_ACT):
        self.mu = float(mu)
        self.max_time = float(max_time)
        self.tau_act = float(tau_act)

    def predict(self, observation, actions, context) -> Prediction:
        del context
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 3:
            raise ValueError(f"actions must be (K,H,2), got {actions.shape}")
        K, H, _ = actions.shape
        ends = np.zeros((K, 2), dtype=np.float64)
        traj = np.zeros((K, H, 2), dtype=np.float64)
        extras = {"success": [], "goal_error": [], "termination": [], "nominal_mu": self.mu}
        for k in range(K):
            r = rollout_nominal(
                observation, actions[k], mu=self.mu, max_time=self.max_time, tau_act=self.tau_act
            )
            ends[k] = r["endpoint"]
            traj[k] = r["traj"]
            extras["success"].append(bool(r["success"]))
            extras["goal_error"].append(float(r["goal_error"]))
            extras["termination"].append(r["termination"])
        return Prediction(endpoints=ends, samples=None, traj=traj, extra=extras)


class GoalDistanceScorer:
    """score = -||endpoint - goal||. Higher is better. Ties are left as equal scores
    (Best-of-N breaks ties by the smallest index). Does not use live-env parameters.
    """

    def __init__(self, success_bonus: float = 0.0):
        self.success_bonus = float(success_bonus)

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        g = np.asarray(goal["goal"], dtype=np.float64)
        ends = np.asarray(predictions.endpoints, dtype=np.float64)
        d = np.linalg.norm(ends - g.reshape(1, 2), axis=-1)
        s = -d
        if self.success_bonus and predictions.extra.get("success"):
            succ = np.asarray(predictions.extra["success"], dtype=np.float64)
            s = s + self.success_bonus * succ
        return np.asarray(s, dtype=np.float64)
