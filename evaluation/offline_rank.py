"""A10 offline ground truth: branch every candidate from the SAME full snapshot (OFFLINE ONLY).
This is a diagnostic oracle (counterfactual execution). It must never be called from policy-side code or
from an online episode loop; online methods only execute their chosen action.

Outcome of candidate k under rep seed r:
  restore(snapshot) -> reseed env RNG with r (common random numbers: same r for every candidate)
  -> step_segment(candidate) -> settle: zero-velocity steps until |obj_vel| < 1e-3 or settle_steps or done
  -> settled goal error |obj-goal| [m], success (done&success, or in goal & at rest), out_of_bounds.
Oracle selection uses select_rep_seeds; the oracle's reported value uses DISJOINT eval_rep_seeds.
"""
from __future__ import annotations
import copy
import numpy as np

REST = 1e-3

def reseed_snapshot(snap: dict, seed: int) -> dict:
    s = copy.deepcopy(snap)
    if "rng" in s:
        s["rng"] = np.random.default_rng(seed).bit_generator.state
    return s

def _true_state(env):
    """Evaluator-side access to true (noise-free) object state; falls back to observation."""
    if hasattr(env, "o") and hasattr(env, "ov") and hasattr(env, "goal"):
        return np.array(env.o, float), np.array(env.ov, float), np.array(env.goal, float), float(env.goal_radius)
    ob = env.observe()
    return np.asarray(ob["obj"]), np.asarray(ob["obj_vel"]), np.asarray(ob["goal"]), float(ob["goal_radius"])

def rollout_outcome(env, snap, actions, seed, settle_steps=20, continuation=None):
    env.restore(reseed_snapshot(snap, seed))
    res = env.step_segment(np.asarray(actions, float))
    term = res[-1].info.get("termination") if res else None
    n = len(res)
    k = 0
    while not (res and res[-1].done) and k < settle_steps:
        o, ov, _, _ = _true_state(env)
        if continuation is None and np.linalg.norm(ov) < REST:
            break
        a = np.zeros(2) if continuation is None else continuation(env.observe())
        res.append(env.step(a)); k += 1
        term = res[-1].info.get("termination")
    o, ov, g, gr = _true_state(env)
    err = float(np.linalg.norm(o - g))
    success = bool((res and res[-1].success) or (err < gr and np.linalg.norm(ov) < REST))
    return {"goal_error": err, "success": success, "out_of_bounds": term == "out_of_bounds",
            "steps": n + k, "termination": term}

def segment_endpoint(env, snap, actions, seed):
    """True object position right at the end of the segment (for endpoint-prediction error)."""
    env.restore(reseed_snapshot(snap, seed))
    env.step_segment(np.asarray(actions, float))
    return _true_state(env)[0]

def rank_candidates(env, snap, candidates, select_rep_seeds=(11, 12, 13), eval_rep_seeds=(21, 22, 23),
                    settle_steps=20, continuation=None):
    """candidates (K,H,2). Returns dict with per-candidate errors for select and eval reps, true endpoints,
    oracle index (chosen on select reps), oracle value on eval reps, and a determinism flag."""
    C = np.asarray(candidates, float)
    K = C.shape[0]
    sel = np.array([[rollout_outcome(env, snap, C[k], r, settle_steps, continuation)["goal_error"]
                     for r in select_rep_seeds] for k in range(K)])
    ev_out = [[rollout_outcome(env, snap, C[k], r, settle_steps, continuation) for r in eval_rep_seeds]
              for k in range(K)]
    ev = np.array([[o["goal_error"] for o in row] for row in ev_out])
    succ = np.array([[o["success"] for o in row] for row in ev_out], float)
    endpoints = np.array([segment_endpoint(env, snap, C[k], eval_rep_seeds[0]) for k in range(K)])
    sel_mean, ev_mean = sel.mean(1), ev.mean(1)
    oracle = int(np.argmin(sel_mean))
    env.restore(snap)  # leave env exactly as given
    return {"K": K, "select_err": sel.tolist(), "eval_err": ev.tolist(), "eval_success": succ.tolist(),
            "true_err": ev_mean.tolist(), "true_success_rate": succ.mean(1).tolist(),
            "true_endpoints": endpoints.tolist(), "oracle_index": oracle,
            "oracle_eval_err": float(ev_mean[oracle]), "oracle_select_err": float(sel_mean[oracle]),
            "best_eval_err_insample": float(ev_mean.min()),
            "deterministic": bool(np.allclose(sel, sel[:, :1]) and np.allclose(ev, ev[:, :1]) and np.allclose(sel[:, :1], ev[:, :1])),
            "spread_err": float(ev_mean.max() - ev_mean.min()),
            "all_fail": bool(succ.mean(1).max() == 0.0),
            "select_rep_seeds": list(select_rep_seeds), "eval_rep_seeds": list(eval_rep_seeds)}
