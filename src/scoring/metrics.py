"""Ranking / regret metrics. Tie handling is explicit.

Oracle ranking uses true goal distances (lower is better). Two candidates
are tied if |d_i - d_j| <= tie_tol (default 1e-4 m). A model's top-1 is
correct if its true distance is within tie_tol of the oracle best (it is
an oracle-tied winner). Pairwise accuracy ignores tied oracle pairs.
"""
from __future__ import annotations
import numpy as np

TIE_TOL_M = 1e-4  # 0.1 mm; fixture is deterministic to well below this


def pairwise_ranking_accuracy(pred_scores: np.ndarray, true_distances: np.ndarray,
                              tie_tol: float = TIE_TOL_M) -> dict:
    """pred_scores: (K,) higher better; true_distances: (K,) lower better."""
    s = np.asarray(pred_scores, dtype=np.float64)
    d = np.asarray(true_distances, dtype=np.float64)
    K = len(d)
    n_comp, n_correct, n_ties = 0, 0, 0
    for i in range(K):
        for j in range(i + 1, K):
            if abs(d[i] - d[j]) <= tie_tol:
                n_ties += 1
                continue
            n_comp += 1
            true_i_better = d[i] < d[j]
            pred_i_better = s[i] > s[j]
            if pred_i_better == true_i_better:
                n_correct += 1
    acc = n_correct / n_comp if n_comp else np.nan
    return {"pairwise_acc": acc, "n_compared": n_comp, "n_correct": n_correct,
            "n_oracle_ties": n_ties, "K": K}


def top1_regret(chosen: int, true_distances: np.ndarray, tie_tol: float = TIE_TOL_M) -> dict:
    d = np.asarray(true_distances, dtype=np.float64)
    best = float(d.min())
    d_ch = float(d[int(chosen)])
    regret = d_ch - best
    correct = regret <= tie_tol
    oracle_idx = int(np.argmin(d))
    return {
        "regret_m": regret,
        "top1_correct": bool(correct),
        "chosen": int(chosen),
        "oracle_idx": oracle_idx,
        "true_dist_chosen": d_ch,
        "true_dist_oracle": best,
        "n_oracle_tied": int(np.sum(np.abs(d - best) <= tie_tol)),
    }


def along_lateral_error(pred_endpoint: np.ndarray, true_endpoint: np.ndarray,
                        obj0: np.ndarray, goal: np.ndarray) -> dict:
    """Decompose endpoint error into along-goal and lateral components.

    u_along points from the INITIAL object position to the goal (task axis),
    not along the action. If the goal is essentially at obj0, u_along = [1,0].
    """
    pred = np.asarray(pred_endpoint, dtype=np.float64).reshape(-1, 2)
    tru = np.asarray(true_endpoint, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(goal, dtype=np.float64).reshape(2)
    o = np.asarray(obj0, dtype=np.float64).reshape(2)
    axis = g - o
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:
        u = np.array([1.0, 0.0])
    else:
        u = axis / nrm
    v = np.array([-u[1], u[0]])
    err = pred - tru
    e_along = err @ u
    e_lat = err @ v
    return {
        "e_along": e_along, "e_lat": e_lat,
        "e_norm": np.linalg.norm(err, axis=-1),
        "u_along": u, "u_lat": v,
    }
