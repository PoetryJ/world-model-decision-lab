"""OFFLINE / ORACLE-ONLY helpers (A4).  Uses snapshot branching from the TRUE env state => only for
evaluation/, diagnostics and E0/E1 oracle tables.  NEVER call from policy-side code (it would leak
hidden dynamics through counterfactual execution)."""
from __future__ import annotations
import numpy as np
from .push_env import REST_SPEED

SETTLE_MAX = 30   # zero-action control steps after the segment to let the object come to rest (3 s)


def rollout_from(env, snap, actions, settle_max: int = SETTLE_MAX):
    """Restore snap, execute actions (H,2), then zero actions until object at rest / done / settle_max.
    Returns dict(final_obj, dist, success, termination, steps, seg_end_obj). Leaves env in post-rollout state."""
    env.restore(snap)
    res = env.step_segment(actions)
    seg_end = env.observe()["obj"].copy()
    k = 0
    while not env.done and k < settle_max and np.linalg.norm(env.ov) >= REST_SPEED:
        env.step(np.zeros(2)); k += 1
    o = env.observe()
    dist = float(np.linalg.norm(env.o - env.goal))
    return {"final_obj": env.o.copy(), "dist": dist,
            "success": bool(env.success or (dist < env.goal_radius and np.linalg.norm(env.ov) < REST_SPEED)),
            "termination": env.termination, "steps": len(res) + k, "seg_end_obj": seg_end}


def evaluate_candidates(env, snap, cands, settle_max: int = SETTLE_MAX):
    """Branch every candidate (K,H,2) from the same snapshot. Returns arrays dist (K,), success (K,), final (K,2).
    Env is restored to snap at the end."""
    out = [rollout_from(env, snap, c, settle_max) for c in cands]
    env.restore(snap)
    return (np.array([r["dist"] for r in out]), np.array([r["success"] for r in out]),
            np.stack([r["final_obj"] for r in out]))
