"""Diversity + outcome-spread diagnostics on a fixed candidate set (offline branching)."""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]  # repo root (was hardcoded /workspace/wmprobe)
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
if str(_ROOT/'src') not in sys.path: sys.path.insert(0, str(_ROOT/'src'))
if str(_ROOT/'reports/a5') not in sys.path: sys.path.insert(0, str(_ROOT/'reports/a5'))
from candidates.utils import pairwise_action_distance
from oracle import evaluate_candidates

# Thresholds (documented in REPORT): "near-identical" if mean per-step action L2
# is below ACTION_EPS m/s AND endpoint L2 is below OUTCOME_EPS m.
ACTION_EPS = 0.05  # m/s
OUTCOME_EPS = 0.01  # m


def pairwise_outcome_distance(endpoints: np.ndarray) -> np.ndarray:
    e = np.asarray(endpoints, dtype=np.float64)
    k = e.shape[0]
    d = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            val = float(np.linalg.norm(e[i] - e[j]))
            d[i, j] = d[j, i] = val
    return d


def offdiag(m: np.ndarray) -> np.ndarray:
    k = m.shape[0]
    return np.array([m[i, j] for i in range(k) for j in range(i + 1, k)], dtype=np.float64)


def diagnose_set(env, candidates: np.ndarray, coast: bool = True, n_repeats: int = 1, settle_steps: int = 20) -> dict:
    ev = evaluate_candidates(env, candidates, coast=coast, n_repeats=n_repeats, settle_steps=settle_steps)
    a_dist = pairwise_action_distance(candidates)
    o_dist = pairwise_outcome_distance(ev["endpoints"])
    a_pairs = offdiag(a_dist)
    o_pairs = offdiag(o_dist)
    near = []
    k = candidates.shape[0]
    for i in range(k):
        for j in range(i + 1, k):
            if a_dist[i, j] < ACTION_EPS and o_dist[i, j] < OUTCOME_EPS:
                near.append((i, j, float(a_dist[i, j]), float(o_dist[i, j])))
    spread = ev["goal_error"].max() - ev["goal_error"].min()
    return {
        **{key: ev[key] for key in ev if key != "records"},
        "records": ev["records"],
        "action_pairwise": a_dist,
        "outcome_pairwise": o_dist,
        "action_pair_mean": float(a_pairs.mean()) if len(a_pairs) else 0.0,
        "action_pair_min": float(a_pairs.min()) if len(a_pairs) else 0.0,
        "action_pair_max": float(a_pairs.max()) if len(a_pairs) else 0.0,
        "outcome_pair_mean": float(o_pairs.mean()) if len(o_pairs) else 0.0,
        "outcome_pair_min": float(o_pairs.min()) if len(o_pairs) else 0.0,
        "outcome_pair_max": float(o_pairs.max()) if len(o_pairs) else 0.0,
        "goal_error_spread": float(spread),
        "near_identical_pairs": near,
        "has_near_identical": bool(near),
        "action_eps": ACTION_EPS,
        "outcome_eps": OUTCOME_EPS,
    }
