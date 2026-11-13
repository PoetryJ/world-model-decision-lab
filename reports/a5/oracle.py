"""Ground-truth oracle on a FIXED candidate set, via snapshot branching.

THIS IS A DIAGNOSTIC UPPER BOUND, NOT A DEPLOYABLE ALGORITHM.
It uses free snapshot/restore (counterfactual execution from the same state) and
the real environment as a simulator. A deployed policy cannot do this.

Scoring uses the same GoalDistanceScorer convention: higher = better = closer
endpoint (after optional coast-to-termination). Tie-break: smallest index.
"""
from __future__ import annotations

import numpy as np

from shared.interface import clip_action


def _goal_error(obs: dict) -> float:
    return float(np.linalg.norm(np.asarray(obs["obj"]) - np.asarray(obs["goal"])))


REST = 1e-3
SETTLE_STEPS = 20  # A10 offline protocol: zero actions until |obj_vel|<REST or this many steps


def _true_state(env):
    """Eval-side true (noise-free) object state; falls back to observation."""
    if hasattr(env, "o") and hasattr(env, "ov") and hasattr(env, "goal"):
        return (np.array(env.o, float), np.array(env.ov, float),
                np.array(env.goal, float), float(env.goal_radius))
    ob = env.observe()
    return (np.asarray(ob["obj"]), np.asarray(ob["obj_vel"]),
            np.asarray(ob["goal"]), float(ob["goal_radius"]))


def _settle(env, settle_steps: int = SETTLE_STEPS):
    """Zero-action settle until at rest, done, or settle_steps (A10 offline protocol)."""
    results = []
    for _ in range(int(settle_steps)):
        if env.done:
            break
        _o, ov, _g, _gr = _true_state(env)
        if np.linalg.norm(ov) < REST:
            break
        results.append(env.step(np.zeros(2, dtype=np.float64)))
        if results[-1].done:
            break
    return results


def rollout_candidate(env, actions: np.ndarray, coast: bool = True,
                      settle_steps: int = SETTLE_STEPS) -> dict:
    """Execute one segment (and optional A10-style settle) on the CURRENT env state. Mutates env."""
    actions = clip_action(np.asarray(actions, dtype=np.float64))
    steps = env.step_segment(actions)
    if coast and not env.done:
        steps = list(steps) + _settle(env, settle_steps=settle_steps)
    last = steps[-1] if steps else env.step(np.zeros(2))
    o, ov, g, gr = _true_state(env)
    err = float(np.linalg.norm(o - g))
    success = bool(last.success or (err < gr and np.linalg.norm(ov) < REST))
    obs = last.obs
    return {
        "endpoint": np.asarray(o, dtype=np.float64).copy(),
        "success": success,
        "done": bool(last.done),
        "termination": last.info.get("termination") if last.info else None,
        "goal_error": err,
        "n_steps": len(steps),
        "t": float(obs["t"]),
    }


def evaluate_candidates(env, candidates: np.ndarray, coast: bool = True,
                        n_repeats: int = 1, settle_steps: int = SETTLE_STEPS) -> dict:
    """Offline branching: restore the same snapshot for each candidate.

    If n_repeats>1 (stochastic continuation), each candidate is executed n_repeats
    times from the same snapshot (env RNG is part of the snapshot, so repeats are
    identical unless the caller advances RNG between repeats). For a noisy env we
    re-restore and step; with action_noise=0 repeats are exact duplicates — the
    caller should keep n_repeats=1 in that case.

    Leaves the env restored to the original snapshot.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[-1] != 2:
        raise ValueError(f"candidates must be (K,H,2), got {candidates.shape}")
    K = candidates.shape[0]
    snap = env.snapshot()
    records = []
    for k in range(K):
        repeat_rows = []
        for r in range(n_repeats):
            env.restore(snap)
            repeat_rows.append(rollout_candidate(env, candidates[k], coast=coast, settle_steps=settle_steps))
        # Aggregate repeats: mean endpoint / error; success rate.
        ends = np.stack([row["endpoint"] for row in repeat_rows], axis=0)
        errors = np.array([row["goal_error"] for row in repeat_rows], dtype=np.float64)
        succs = np.array([row["success"] for row in repeat_rows], dtype=np.float64)
        records.append(
            {
                "index": k,
                "endpoint": ends.mean(axis=0),
                "endpoint_std": ends.std(axis=0),
                "goal_error": float(errors.mean()),
                "goal_error_std": float(errors.std()),
                "success_rate": float(succs.mean()),
                "success": bool(succs.mean() >= 0.5),
                "termination": repeat_rows[0]["termination"],
                "n_repeats": n_repeats,
                "repeats": repeat_rows if n_repeats > 1 else None,
            }
        )
    env.restore(snap)
    endpoints = np.stack([r["endpoint"] for r in records], axis=0)
    goal_error = np.array([r["goal_error"] for r in records], dtype=np.float64)
    success = np.array([r["success_rate"] for r in records], dtype=np.float64)
    # Oracle score: prefer success, then smaller goal error. Equivalent to
    # score = success_rate * big_bonus - goal_error, but we also expose raw fields.
    scores = success * 10.0 - goal_error
    # Tie-break: smallest index among argmax.
    oracle_index = int(np.argmax(scores))  # numpy argmax: first occurrence
    random_expected_error = float(goal_error.mean())
    random_expected_success = float(success.mean())
    return {
        "K": K,
        "endpoints": endpoints,
        "goal_error": goal_error,
        "success_rate": success,
        "scores": scores,
        "oracle_index": oracle_index,
        "oracle_goal_error": float(goal_error[oracle_index]),
        "oracle_success": float(success[oracle_index]),
        "random_expected_goal_error": random_expected_error,
        "random_expected_success": random_expected_success,
        "regret_vs_random_error": float(random_expected_error - goal_error[oracle_index]),
        "all_fail": bool(np.all(success < 0.5)),
        "all_succeed": bool(np.all(success >= 0.5)),
        "records": records,
        "n_repeats": n_repeats,
        "coast": coast,
        "note": "diagnostic upper bound on this fixed set; not a deployable algorithm",
    }


class GroundTruthOracle:
    """Picks the best candidate by true branched rollout. Diagnostic only."""

    def __init__(self, env, coast: bool = True, n_repeats: int = 1, settle_steps: int = SETTLE_STEPS):
        self.env = env
        self.coast = coast
        self.n_repeats = n_repeats
        self.settle_steps = settle_steps

    def choose(self, candidates: np.ndarray) -> tuple[int, dict]:
        ev = evaluate_candidates(self.env, candidates, coast=self.coast, n_repeats=self.n_repeats, settle_steps=self.settle_steps)
        return ev["oracle_index"], ev
