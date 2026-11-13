"""Visible-only features for direct candidate scoring.

Policy-side module: never reads env.hidden, evaluator internals, or counterfactual
rollouts. All features are functions of the current observation, the candidate
action segment, and (optionally) the executed History.

Observations are idealised structured poses, not vision results.
"""
from __future__ import annotations

import numpy as np
from shared.interface import DT

EPS = 1e-9

N_OBS = 16
N_CAND = 20
N_HIST = 20
N_HX = 8

N_PHYS = 5  # physics-heuristic features (appended)
# Set False for large-K official training (physics scorers still run independently).
INCLUDE_PHYS_FEATURES = True
OBS_ONLY_DIM = N_OBS + N_CAND + N_PHYS
HISTORY_DIM = N_OBS + N_CAND + N_HIST + N_HX + N_PHYS

R_P, R_O = 0.03, 0.05  # disc radii from the shared interface / A4 env docs


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _unit(v: np.ndarray) -> np.ndarray:
    v = _v(v)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def obs_features(observation: dict) -> np.ndarray:
    """Shape (N_OBS,)."""
    obj = _v(observation["obj"])
    pusher = _v(observation["pusher"])
    goal = _v(observation["goal"])
    vel = _v(observation["obj_vel"])
    gr = float(observation.get("goal_radius", 0.03))
    t = float(observation.get("t", 0.0))
    og = obj - goal
    po = pusher - obj
    d_goal = float(np.linalg.norm(og))
    d_push = float(np.linalg.norm(po))
    u_goal = _unit(goal - obj)
    u_to_obj = _unit(obj - pusher)
    speed = float(np.linalg.norm(vel))
    close = float(vel @ u_goal)
    return np.array([
        og[0], og[1], po[0], po[1], vel[0], vel[1], gr,
        d_goal, d_push, u_goal[0], u_goal[1], u_to_obj[0], u_to_obj[1],
        t, speed, close,
    ], dtype=np.float64)


def candidate_features(observation: dict, actions: np.ndarray) -> np.ndarray:
    """Shape (N_CAND,). Segment descriptors plus geometry interactions (no rollout)."""
    a = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    H = int(a.shape[0])
    if H == 0:
        return np.zeros(N_CAND, dtype=np.float64)
    duration = H * DT
    mean_v = a.mean(axis=0)
    first_v = a[0]
    last_v = a[-1]
    disp = a.sum(axis=0) * DT
    mean_speed = float(np.linalg.norm(mean_v))
    disp_n = float(np.linalg.norm(disp))
    step_speeds = np.linalg.norm(a, axis=1)
    speed_std = float(step_speeds.std()) if H > 1 else 0.0
    max_speed = float(step_speeds.max())

    obj = _v(observation["obj"])
    pusher = _v(observation["pusher"])
    goal = _v(observation["goal"])
    u_goal = _unit(goal - obj)
    u_to_obj = _unit(obj - pusher)
    d_goal = float(np.linalg.norm(goal - obj))
    gap = float(np.linalg.norm(obj - pusher))
    clearance = max(0.0, gap - (R_P + R_O))
    toward = float(mean_v @ u_to_obj)
    contact_likely = 1.0 if (toward > 0.0 and disp_n >= 0.5 * clearance) else 0.0
    disp_along = float(disp @ u_goal)
    return np.array([
        mean_v[0], mean_v[1], first_v[0], first_v[1], last_v[0], last_v[1],
        disp[0], disp[1], mean_speed, disp_n, duration, float(H),
        speed_std, max_speed,
        float(mean_v @ u_goal), float(mean_v @ u_to_obj), float(first_v @ u_to_obj),
        disp_along, d_goal - disp_along, contact_likely,
    ], dtype=np.float64)


def history_features(history) -> np.ndarray:
    """Shape (N_HIST,). Empty history -> zeros with has_history=0."""
    out = np.zeros(N_HIST, dtype=np.float64)
    if not history:
        return out
    n_probe = sum(1 for tr in history if getattr(tr, "kind", "") == "probe")
    last = history[-1]
    acts = np.asarray(last.actions, dtype=np.float64).reshape(-1, 2)
    obj0, obj1 = _v(last.obs["obj"]), _v(last.next_obs["obj"])
    p0, p1 = _v(last.obs["pusher"]), _v(last.next_obs["pusher"])
    v0, v1 = _v(last.obs["obj_vel"]), _v(last.next_obs["obj_vel"])
    od, pd = obj1 - obj0, p1 - p0
    odn, pdn = float(np.linalg.norm(od)), float(np.linalg.norm(pd))
    mobility = odn / (pdn + EPS)
    dur = acts.shape[0] * DT
    kind_probe = 1.0 if getattr(last, "kind", "") == "probe" else 0.0
    speed1 = float(np.linalg.norm(v1))
    cos = float(_unit(od) @ _unit(pd)) if (odn > EPS and pdn > EPS) else 0.0
    cum = np.zeros(2)
    cum_n = 0.0
    mobs = []
    for tr in history:
        d = _v(tr.next_obs["obj"]) - _v(tr.obs["obj"])
        pdd = _v(tr.next_obs["pusher"]) - _v(tr.obs["pusher"])
        cum = cum + d
        cum_n += float(np.linalg.norm(d))
        mobs.append(float(np.linalg.norm(d)) / (float(np.linalg.norm(pdd)) + EPS))
    out[:] = [
        1.0, float(len(history)), float(n_probe),
        od[0], od[1], pd[0], pd[1], odn, pdn, _safe(mobility),
        dur, kind_probe, v1[0], v1[1], speed1, cos,
        cum[0], cum[1], cum_n,
        _safe(float(np.mean(mobs)) if mobs else 0.0),
    ]
    return out


def hist_cand_interactions(observation: dict, actions: np.ndarray, hist_f: np.ndarray) -> np.ndarray:
    """Shape (N_HX,). Linear model can specialise candidate value by observed mobility."""
    cand = candidate_features(observation, actions)
    obj, goal = _v(observation["obj"]), _v(observation["goal"])
    u_goal = _unit(goal - obj)
    d_goal = float(np.linalg.norm(goal - obj))
    disp = np.array([cand[6], cand[7]])
    disp_along = float(disp @ u_goal)
    mobility, has = float(hist_f[9]), float(hist_f[0])
    speed_after, duration = float(hist_f[14]), float(cand[10])
    mean_speed, align = float(cand[8]), float(cand[14])
    pred_err = d_goal - mobility * disp_along
    last_od = np.array([hist_f[3], hist_f[4]])
    mean_v = np.array([cand[0], cand[1]])
    return np.array([
        has * mobility * mean_speed,
        has * mobility * align,
        has * mobility * float(cand[9]),
        has * speed_after * duration,
        has * pred_err,
        has * float(last_od @ mean_v),
        has * mobility * (d_goal - disp_along),
        has * (1.0 if mobility < 0.5 else 0.0) * mean_speed,
    ], dtype=np.float64)


def build_features(observation: dict, actions: np.ndarray, history=None, use_history: bool = False) -> np.ndarray:
    of = obs_features(observation)
    cf = candidate_features(observation, actions)
    # Physics features: obs-only uses a fixed nominal decelerations; history may estimate a from probe.
    if not use_history:
        parts = [of, cf]
    else:
        hf = history_features(history)
        hx = hist_cand_interactions(observation, actions, hf)
        parts = [of, cf, hf, hx]
    if INCLUDE_PHYS_FEATURES:
        from .physics_heuristic import physics_features
        parts.append(physics_features(observation, actions, history, use_history=use_history))
    return np.concatenate(parts)


def build_feature_matrix(observation: dict, candidates: np.ndarray, history=None, use_history: bool = False) -> np.ndarray:
    cands = np.asarray(candidates, dtype=np.float64)
    return np.stack([build_features(observation, c, history, use_history=use_history) for c in cands], axis=0)


FEATURE_NAMES_OBS = [
    "og_x", "og_y", "po_x", "po_y", "vel_x", "vel_y", "goal_radius",
    "d_goal", "d_push", "u_goal_x", "u_goal_y", "u_to_obj_x", "u_to_obj_y",
    "t", "speed", "vel_close",
]
FEATURE_NAMES_CAND = [
    "mean_vx", "mean_vy", "first_vx", "first_vy", "last_vx", "last_vy",
    "disp_x", "disp_y", "mean_speed", "disp_n", "duration", "H",
    "speed_std", "max_speed", "align_goal", "align_approach", "first_approach",
    "disp_along_goal", "overshoot_proxy", "contact_likely",
]
FEATURE_NAMES_HIST = [
    "has_history", "n_trans", "n_probe", "last_od_x", "last_od_y",
    "last_pd_x", "last_pd_y", "last_od_n", "last_pd_n", "mobility",
    "last_dur", "last_is_probe", "vel_after_x", "vel_after_y", "speed_after",
    "cos_od_pd", "cum_od_x", "cum_od_y", "cum_od_n", "mean_mobility",
]
FEATURE_NAMES_HX = [
    "h_mob_speed", "h_mob_align", "h_mob_disp", "h_speed_dur",
    "h_mob_pred_err", "h_od_dot_v", "h_mob_overshoot", "h_lowmob_speed",
]
FEATURE_NAMES_PHYS = ["a_used", "decel_informed", "phys_err_used", "phys_err_nominal", "phys_delta_err"]


def feature_names(use_history: bool) -> list[str]:
    names = FEATURE_NAMES_OBS + FEATURE_NAMES_CAND
    if use_history:
        names = names + FEATURE_NAMES_HIST + FEATURE_NAMES_HX
    if INCLUDE_PHYS_FEATURES:
        names = names + FEATURE_NAMES_PHYS
    return names
