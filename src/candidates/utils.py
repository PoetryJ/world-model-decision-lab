"""Shared candidate helpers. Actions are clipped to legal V_MAX; units m/s, world frame."""
from __future__ import annotations

import numpy as np

from shared.interface import DT, V_MAX, clip_action

DEFAULT_K = 4
DEFAULT_H = 10
# Comparable commanded speed used by the fixed library and the random proposer.
NOMINAL_SPEED = 0.4  # m/s  (V_MAX = 0.5)
R_PUSHER = 0.03
R_OBJ = 0.05
BEHIND_DIST = R_PUSHER + R_OBJ + 0.02  # 0.10 m


def as_actions(x, k: int | None = None, h: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f"candidates must have shape (K,H,2), got {a.shape}")
    if k is not None and a.shape[0] != k:
        raise ValueError(f"expected K={k}, got {a.shape[0]}")
    if h is not None and a.shape[1] != h:
        raise ValueError(f"expected H={h}, got {a.shape[1]}")
    return clip_action(a)


def segment_rms(actions: np.ndarray) -> np.ndarray:
    """RMS speed of each candidate, shape (K,)."""
    a = np.asarray(actions, dtype=np.float64)
    return np.sqrt(np.mean(np.sum(a * a, axis=-1), axis=-1))


def mean_speed(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    return np.mean(np.linalg.norm(a, axis=-1), axis=-1)


def pairwise_action_distance(actions: np.ndarray) -> np.ndarray:
    """Mean per-step L2 distance between commanded velocities, shape (K,K), m/s."""
    a = np.asarray(actions, dtype=np.float64)
    k = a.shape[0]
    d = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            val = float(np.mean(np.linalg.norm(a[i] - a[j], axis=-1)))
            d[i, j] = d[j, i] = val
    return d


def unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(2) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return v / n


def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float64)


def geometry(observation: dict) -> dict:
    p = np.asarray(observation["pusher"], dtype=np.float64)
    o = np.asarray(observation["obj"], dtype=np.float64)
    g = np.asarray(observation["goal"], dtype=np.float64)
    goal_dir = unit(g - o, fallback=np.array([1.0, 0.0]))
    to_obj = unit(o - p, fallback=goal_dir)
    return {
        "pusher": p,
        "obj": o,
        "goal": g,
        "goal_dir": goal_dir,
        "tangent": rot90(goal_dir),
        "to_obj": to_obj,
        "behind": o - goal_dir * BEHIND_DIST,
        "goal_radius": float(observation.get("goal_radius", 0.05)),
    }


def go_to_then_push(p0: np.ndarray, waypoint: np.ndarray, push_dir: np.ndarray,
                    H: int, speed: float, arrive_tol: float = 0.02) -> np.ndarray:
    """Open-loop pusher plan: drive to waypoint without overshoot, then hold push_dir.

    A previous version commanded `speed` even when the remaining distance was less than
    speed*DT, which made the pusher oscillate about the waypoint and often miss the object.
    """
    acts = np.zeros((H, 2), dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64).copy()
    push_dir = unit(push_dir, fallback=np.array([1.0, 0.0]))
    speed = float(np.clip(speed, 0.0, V_MAX))
    arrived = False
    for t in range(H):
        if not arrived:
            delta = waypoint - p
            dist = float(np.linalg.norm(delta))
            if dist > speed * DT + 1e-12:
                v = speed * delta / dist
            elif dist > 1e-12:
                # arrive exactly this step (commanded speed <= nominal; no reversal)
                v = delta / DT
                arrived = True
            else:
                arrived = True
                v = speed * push_dir
        else:
            v = speed * push_dir
        v = clip_action(v)
        acts[t] = v
        p = p + v * DT
    return acts
