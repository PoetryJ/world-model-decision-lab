"""A10 metrics (frozen definitions; see PROTOCOL.md). Pure numpy, no access to policy internals.

Conventions
- Outcome quality of a candidate = true settled goal error e (metres, lower better) from offline_rank.
  Utility u = -e. A predictor/scorer produces scores s (higher better).
- Ties: true tie if |e_i - e_j| <= tie_tol (default 1e-3 m): pair EXCLUDED from pairwise accuracy denominator
  (count reported). Predicted tie (|s_i - s_j| <= 1e-12) on a non-tied true pair counts 0.5.
- Top-1 regret: e(chosen) - min_k e_k. If the scorer max is tied, regret = mean over the tied argmax set
  (expected regret under uniform random tie-break).
- Uncertainty: cluster bootstrap over init states (cluster = init_state_id; all conditions/candidates/reps
  of one init state move together). Never treat candidates of one state as independent samples.
"""
from __future__ import annotations
import numpy as np

TIE_TOL = 1e-3
SCORE_TIE = 1e-12

def pairwise_accuracy(true_err, scores, tie_tol=TIE_TOL):
    """Returns (acc, n_pairs_used, n_true_ties). acc is nan if no usable pair."""
    e = np.asarray(true_err, float); s = np.asarray(scores, float)
    K = len(e); num = 0.0; n = 0; ties = 0
    for i in range(K):
        for j in range(i + 1, K):
            if abs(e[i] - e[j]) <= tie_tol:
                ties += 1; continue
            n += 1
            better_true = i if e[i] < e[j] else j
            ds = s[i] - s[j]
            if abs(ds) <= SCORE_TIE:
                num += 0.5
            elif (ds > 0) == (better_true == i):
                num += 1.0
    return (num / n if n else float("nan")), n, ties

def top1_regret(true_err, scores):
    e = np.asarray(true_err, float); s = np.asarray(scores, float)
    tied = np.flatnonzero(s >= s.max() - SCORE_TIE)
    return float(np.mean(e[tied]) - e.min())

def top1_correct(true_err, scores, tie_tol=TIE_TOL):
    """Fraction of tied-argmax set whose true error is within tie_tol of the best."""
    e = np.asarray(true_err, float); s = np.asarray(scores, float)
    tied = np.flatnonzero(s >= s.max() - SCORE_TIE)
    return float(np.mean(e[tied] <= e.min() + tie_tol))

def candidate_diversity(cands):
    """cands (K,H,2): mean pairwise L2 distance between flattened action segments (m/s * sqrt(steps))."""
    c = np.asarray(cands, float).reshape(len(cands), -1)
    K = len(c)
    if K < 2: return 0.0
    d = [np.linalg.norm(c[i] - c[j]) for i in range(K) for j in range(i + 1, K)]
    return float(np.mean(d))

# ---------------- episode-level (online) metrics ----------------
EPISODE_FIELDS = ["method", "init_state_id", "hidden_condition_id", "rep", "success", "goal_error",
                  "steps_total", "probe_steps", "task_steps", "physical_time_s", "termination",
                  "compute_s"]  # compute_s: {"propose","predict","update","score","probe"}

def summarize_episodes(eps: list[dict]) -> dict:
    """eps: list of per-episode dicts from records_to_episodes. Returns flat means (no CI)."""
    if not eps: return {}
    keys = ["propose", "predict", "update", "score", "probe"]
    out = {"n_episodes": len(eps),
           "success_rate": float(np.mean([e["success"] for e in eps])),
           "goal_error_mean": float(np.mean([e["goal_error"] for e in eps])),
           "goal_error_median": float(np.median([e["goal_error"] for e in eps])),
           "steps_total_mean": float(np.mean([e["steps_total"] for e in eps])),
           "probe_steps_mean": float(np.mean([e["probe_steps"] for e in eps])),
           "physical_time_s_mean": float(np.mean([e["physical_time_s"] for e in eps]))}
    for k in keys:
        out[f"compute_{k}_s_mean"] = float(np.mean([e["compute_s"].get(k, 0.0) for e in eps]))
    out["compute_total_s_mean"] = float(sum(out[f"compute_{k}_s_mean"] for k in keys))
    terms = {}
    for e in eps: terms[e["termination"]] = terms.get(e["termination"], 0) + 1
    out["terminations"] = terms
    return out

def records_to_episodes(records: list[dict], dt: float = 0.1) -> list[dict]:
    """Group DecisionRecord dicts by (method, episode_id) into episode summaries.
    goal_error = |obj - goal| from obs_after of the LAST record. Steps counted from len(executed)."""
    by = {}
    for r in records:
        by.setdefault((r["method"], r["episode_id"]), []).append(r)
    eps = []
    for (m, eid), rs in by.items():
        rs = sorted(rs, key=lambda r: r["step_index"])
        last = rs[-1]["obs_after"]
        err = float(np.linalg.norm(np.asarray(last["obj"]) - np.asarray(last["goal"])))
        probe = sum(len(r["executed"]) for r in rs if r["kind"] == "probe")
        task = sum(len(r["executed"]) for r in rs if r["kind"] == "task")
        comp = {}
        for r in rs:
            for k, v in (r.get("wall_time_s") or {}).items():
                comp[k] = comp.get(k, 0.0) + float(v)
        term = rs[-1].get("termination")
        eps.append({"method": m, "episode_id": eid, "init_state_id": rs[0]["init_state_id"],
                    "hidden_condition_id": rs[0]["hidden_condition_id"], "seed": rs[0]["seed"],
                    "success": term == "success", "goal_error": err, "steps_total": probe + task,
                    "probe_steps": probe, "task_steps": task, "physical_time_s": round((probe + task) * dt, 10),
                    "termination": term, "compute_s": comp, "n_decisions": len(rs)})
    return eps

# ---------------- cluster bootstrap ----------------
def cluster_bootstrap(values_by_cluster: dict, stat=np.mean, B=2000, seed=0, alpha=0.05):
    """values_by_cluster: {cluster_id: array of per-unit values (e.g. per condition)}.
    Statistic computed on the concatenation of resampled clusters. Returns (point, lo, hi)."""
    ids = sorted(values_by_cluster)
    arrs = [np.atleast_1d(np.asarray(values_by_cluster[i], float)) for i in ids]
    point = float(stat(np.concatenate(arrs)))
    rng = np.random.default_rng(seed)
    n = len(ids); bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        bs[b] = stat(np.concatenate([arrs[i] for i in idx]))
    return point, float(np.nanquantile(bs, alpha / 2)), float(np.nanquantile(bs, 1 - alpha / 2))

def paired_diff_by_cluster(eps_a, eps_b, field):
    """Pair episodes of methods a,b on (init_state_id, hidden_condition_id, rep); return
    {init_state_id: [a-b per condition/rep]} plus list of unpaired keys (must be empty for a fair comparison)."""
    key = lambda e: (e["init_state_id"], e["hidden_condition_id"], e.get("rep", 0))
    A = {key(e): e for e in eps_a}; Bm = {key(e): e for e in eps_b}
    out = {}
    for k in sorted(set(A) & set(Bm)):
        out.setdefault(k[0], []).append(float(A[k][field]) - float(Bm[k][field]))
    unpaired = sorted(set(A) ^ set(Bm))
    return out, unpaired

def summarize_with_ci(eps, field, B=2000, seed=0):
    vb = {}
    for e in eps: vb.setdefault(e["init_state_id"], []).append(float(e[field]))
    return cluster_bootstrap(vb, B=B, seed=seed)

def per_condition(eps):
    out = {}
    for e in eps: out.setdefault(e["hidden_condition_id"], []).append(e)
    return {c: summarize_episodes(v) for c, v in sorted(out.items())}
