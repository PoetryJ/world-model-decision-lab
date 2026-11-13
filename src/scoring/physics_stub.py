"""Deterministic 2D push rollout matching shared/fixture_env.py physics.

Used as a LOCAL stub world model. The caller supplies mu; this module never
reads env.hidden. Structured pose input is idealised perception, not vision.
[toy] kinematic disc pusher + Coulomb sliding friction.
"""
from __future__ import annotations
import numpy as np
from shared.interface import DT, SUBSTEP, TABLE, clip_action

G = 9.81
R_P, R_O = 0.03, 0.05
# Must match FixturePushEnv success / bound checks.
VEL_SUCC = 1e-3


def rollout_segment(
    pusher: np.ndarray,
    obj: np.ndarray,
    obj_vel: np.ndarray,
    actions: np.ndarray,
    mu: float,
    t0: float = 0.0,
    goal: np.ndarray | None = None,
    goal_radius: float = 0.05,
    max_time: float = 6.0,
    stop_on_done: bool = True,
) -> dict:
    """Roll one action segment (H, 2). Returns object traj, endpoint, flags.

    stop_on_done=True mirrors FixturePushEnv.step_segment (pads remaining
    frames with the terminal object position). Predictors should typically
    use the same termination so scored endpoints match executed ones.
    """
    p = np.asarray(pusher, dtype=np.float64).copy()
    o = np.asarray(obj, dtype=np.float64).copy()
    ov = np.asarray(obj_vel, dtype=np.float64).copy()
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != 2:
        raise ValueError(f"actions must be (H,2), got {acts.shape}")
    H = acts.shape[0]
    traj = np.zeros((H, 2), dtype=np.float64)
    vel_traj = np.zeros((H, 2), dtype=np.float64)
    t = float(t0)
    done = False
    success = False
    reason = None
    h_exec = 0
    for i, a in enumerate(acts):
        if done:
            traj[i] = o
            vel_traj[i] = ov
            continue
        v = clip_action(a)
        for _ in range(int(round(DT / SUBSTEP))):
            p, o, ov = _substep(p, o, ov, v, mu)
        t = round(t + DT, 10)
        h_exec = i + 1
        if np.any(np.abs(o) > TABLE):
            done, reason = True, "out_of_bounds"
        elif (
            goal is not None
            and np.linalg.norm(o - goal) < goal_radius
            and np.linalg.norm(ov) < VEL_SUCC
        ):
            done, success, reason = True, True, "success"
        elif t >= max_time - 1e-9:
            done, reason = True, "timeout"
        traj[i] = o
        vel_traj[i] = ov
    return {
        "traj": traj,
        "vel_traj": vel_traj,
        "endpoint": o.copy(),
        "end_vel": ov.copy(),
        "t": t,
        "done": done,
        "success": success,
        "termination": reason,
        "h_exec": h_exec,
        "pusher_end": p.copy(),
    }


def _substep(p, o, ov, v, mu):
    p = p + v * SUBSTEP
    d = o - p
    dist = np.linalg.norm(d)
    if dist < R_P + R_O and dist > 1e-9:
        n = d / dist
        o = p + n * (R_P + R_O)
        vn = float(v @ n)
        if vn > float(ov @ n):
            ov = ov + (vn - float(ov @ n)) * n
    sp = np.linalg.norm(ov)
    if sp > 0:
        dec = mu * G * SUBSTEP
        ov = ov * max(0.0, sp - dec) / sp
    o = o + ov * SUBSTEP
    return p, o, ov
