"""A10 E2/E3 analysis of DecisionRecord logs (any agent / controller runner). [toy-run]
python -m evaluation.analyze <decisions.jsonl> [--pairs a:b,c:d] [--split test] [--regret]
 -> prints JSON summary; writes <decisions>.a10_summary.json
Checks first (audit.online_accounting with PROTOCOL A-3 budget); summary is flagged INVALID if any issue.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from evaluation import metrics as M
from evaluation.audit import online_accounting, load_frozen, episode_classes

BUDGET_A3 = {"max_steps_total": 60, "max_probe_steps": 5, "max_probes": 1, "max_task_decisions": 1, "max_task_segment_steps": 12}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--pairs", default="")
    ap.add_argument("--split", default="test"); ap.add_argument("--regret", action="store_true")
    ap.add_argument("--budget", default="A3", help="A3 (E2 single-segment) or legacy (60/10 only)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.path)]
    budget = BUDGET_A3 if a.budget == "A3" else {"max_steps_total": 60, "max_probe_steps": 10}
    issues = online_accounting(recs, budget)
    eps = M.records_to_episodes(recs)
    sets = load_frozen("__audit__", who=f"A10 analyze {a.path}")
    expected = {x["init_state_id"] for x in sets[a.split]} if a.split in sets else set()
    out = {"source": a.path, "label": "[toy-run]", "budget": budget, "n_issues": len(issues), "issues_head": issues[:30],
           "valid": not issues, "methods": {}, "episode_classes": episode_classes(recs)}
    for m in sorted({e["method"] for e in eps}):
        E = [e for e in eps if e["method"] == m]
        got = {e["init_state_id"] for e in E}
        d = M.summarize_episodes(E)
        d["coverage_missing_states"] = sorted(expected - got)[:10] if expected else None
        for f in ("goal_error", "success"):
            p, lo, hi = M.summarize_with_ci([{**e, "success": float(e["success"])} for e in E], f)
            d[f"{f}_ci95_cluster"] = [p, lo, hi]
        d["per_condition"] = M.per_condition(E)
        ec = out["episode_classes"].get(m, {})
        d["probe_rate"] = ec.get("probe_rate"); d["n_terminated_during_probe"] = ec.get("n_terminated_during_probe")
        d["tdp_by_termination"] = ec.get("tdp_by_termination")
        out["methods"][m] = d
    if a.regret:
        from evaluation.online_regret import analyse
        ids = {x["init_state_id"]: x for k in ("test", "test_neg", "val", "val_neg", "e0", "e0_neg", "e1") for x in sets[k]}
        rows = analyse(recs, ids, sets["conditions"]["test"])
        out["replay_mismatches"] = sum(r["replay_max_abs_err"] > 1e-9 for r in rows)
        for m in out["methods"]:
            R = [r for r in rows if r["method"] == m]
            vb = {}
            for r in R: vb.setdefault(r["init_state_id"], []).append(r["regret_chosen"])
            out["methods"][m]["online_regret_ci95_cluster"] = list(M.cluster_bootstrap(vb)) if vb else None
            pa = [r["pairwise_acc"] for r in R if r.get("pairwise_acc") is not None and not np.isnan(r["pairwise_acc"])]
            out["methods"][m]["pairwise_acc_mean"] = float(np.mean(pa)) if pa else None
            out["methods"][m]["all_fail_frac"] = float(np.mean([r["all_fail"] for r in R])) if R else None
    for pr in filter(None, a.pairs.split(",")):
        x, y = pr.split(":")
        EX = [e for e in eps if e["method"] == x]; EY = [e for e in eps if e["method"] == y]
        res = {}
        for f in ("goal_error", "success"):
            vb, unp = M.paired_diff_by_cluster([{**e, "success": float(e["success"])} for e in EX],
                                               [{**e, "success": float(e["success"])} for e in EY], f)
            res[f] = {"diff_ci95_cluster": list(M.cluster_bootstrap(vb)) if vb else None, "n_unpaired": len(unp),
                      "n_clusters": len(vb)}
        out.setdefault("paired", {})[f"{x} - {y}"] = res
    with open(a.path.replace(".jsonl", "") + ".a10_summary.json", "w") as f: json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=1, default=str)[:3000])
    for m, d in out["methods"].items():
        print(m, {k: d[k] for k in ("n_episodes", "success_rate", "goal_error_mean", "probe_rate", "n_terminated_during_probe", "probe_steps_mean", "compute_total_s_mean") if k in d},
              "succCI", np.round(d["success_ci95_cluster"], 3).tolist())

if __name__ == "__main__":
    main()
