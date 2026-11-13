"""Controller-side analysis wrapper around A10 evaluation functions (no evaluation/ code copied or modified). [toy-run]
Needed because evaluation.analyze --regret looks up init states only in frozen_sets.json (sealed ids are not there),
and its replay env is noise-free. This wrapper feeds the correct id table (val/val_neg or test_sealed/test_sealed_neg)
and, for the stochastic variant, a PushEnv with the same action_noise.
  python -m runner.analyze_ctrl runs/controller/<x>/decisions.jsonl --split {val|sealed} [--regret] [--pairs a:b,...] [--action_noise 0.02]
Writes <x>/ctrl_summary.json: per method overall + per (subset pos/neg) x condition, paired cluster-bootstrap diffs."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from evaluation import metrics as M
from evaluation.audit import online_accounting, episode_classes, load_frozen, load_sealed
from evaluation.analyze import BUDGET_A3

_RG = None
def _regret_part(rs):
    from evaluation.online_regret import analyse
    from src.env.push_env import PushEnv
    ids, conds, noise = _RG
    return analyse(rs, ids, conds, env_factory=lambda: PushEnv(action_noise=noise))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--split", required=True, choices=["val", "sealed"])
    ap.add_argument("--regret", action="store_true"); ap.add_argument("--pairs", default=""); ap.add_argument("--action_noise", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.path)]
    issues = online_accounting(recs, BUDGET_A3)
    if a.split == "val":
        s = load_frozen("__audit__", who=f"controller analyze_ctrl {a.path}"); rows = s["val"] + s["val_neg"]
    else:
        s = load_sealed(who="controller analyze_ctrl", split="__all__"); rows = s["test_sealed"] + s["test_sealed_neg"]
    ids = {r["init_state_id"]: r for r in rows}; conds = s["conditions"]["test"]
    eps = M.records_to_episodes(recs)
    sub = lambda e: "neg" if "_neg" in e["init_state_id"] else "pos"
    out = {"source": a.path, "label": "[toy-run]", "split": a.split, "n_issues": len(issues), "issues_head": issues[:30],
           "valid": not issues, "episode_classes": episode_classes(recs), "methods": {}}
    for m in sorted({e["method"] for e in eps}):
        E = [e for e in eps if e["method"] == m]
        d = M.summarize_episodes(E)
        for f in ("goal_error", "success"):
            d[f"{f}_ci95_cluster"] = list(M.summarize_with_ci([{**e, "success": float(e["success"])} for e in E], f))
        d["by_subset_condition"] = {sb: M.per_condition([e for e in E if sub(e) == sb]) for sb in ("pos", "neg")}
        ec = out["episode_classes"].get(m, {}); d["probe_rate"] = ec.get("probe_rate"); d["n_terminated_during_probe"] = ec.get("n_terminated_during_probe")
        d["probe_rate_by_subset"] = {sb: float(np.mean([e["probe_steps"] > 0 for e in E if sub(e) == sb])) for sb in ("pos", "neg") if any(sub(e) == sb for e in E)}
        out["methods"][m] = d
    if a.regret:
        from evaluation.online_regret import analyse
        from src.env.push_env import PushEnv
        import multiprocessing as mp, time as _t
        t0 = _t.time(); groups = {}
        for r in recs: groups.setdefault(r["method"], []).append(r)
        global _RG; _RG = (ids, conds, a.action_noise)
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_regret_part, list(groups.values()))
        R = [x for p in parts for x in p]; out["regret_wall_s"] = _t.time() - t0
        out["replay_mismatches"] = int(sum(r["replay_max_abs_err"] > 1e-9 for r in R))
        for m in out["methods"]:
            Rm = [r for r in R if r["method"] == m]; vb = {}
            for r in Rm: vb.setdefault(r["init_state_id"], []).append(r["regret_chosen"])
            d = out["methods"][m]
            d["online_regret_ci95_cluster"] = list(M.cluster_bootstrap(vb)) if vb else None
            pa = [r["pairwise_acc"] for r in Rm if r.get("pairwise_acc") is not None and not np.isnan(r["pairwise_acc"])]
            d["pairwise_acc_mean"] = float(np.mean(pa)) if pa else None
            d["top1_correct_mean"] = float(np.mean([r["top1_correct"] for r in Rm if "top1_correct" in r])) if Rm else None
            d["all_fail_frac"] = float(np.mean([r["all_fail"] for r in Rm])) if Rm else None
            d["regret_by_subset_condition"] = {f"{sb}/{c}": float(np.mean([r["regret_chosen"] for r in Rm if ("neg" if "_neg" in r["init_state_id"] else "pos") == sb and r["hidden_condition_id"] == c]))
                                               for sb in ("pos", "neg") for c in sorted(conds) if any(("neg" if "_neg" in r["init_state_id"] else "pos") == sb and r["hidden_condition_id"] == c for r in Rm)}
    for pr in filter(None, a.pairs.split(",")):
        x, y = pr.split(":"); res = {}
        for scope in ("all", "pos", "neg"):
            EX = [e for e in eps if e["method"] == x and (scope == "all" or sub(e) == scope)]
            EY = [e for e in eps if e["method"] == y and (scope == "all" or sub(e) == scope)]
            for f in ("goal_error", "success"):
                vb, unp = M.paired_diff_by_cluster([{**e, "success": float(e["success"])} for e in EX], [{**e, "success": float(e["success"])} for e in EY], f)
                res[f"{scope}/{f}"] = {"diff_ci95_cluster": list(M.cluster_bootstrap(vb)) if vb else None, "n_unpaired": len(unp), "n_clusters": len(vb)}
        out.setdefault("paired", {})[f"{x} - {y}"] = res
    p = os.path.join(os.path.dirname(a.path), "ctrl_summary.json"); json.dump(out, open(p, "w"), indent=1, default=str)
    print("valid", out["valid"], "issues", len(issues), "replay_mismatches", out.get("replay_mismatches"))
    print(f"{'method':34s} {'n':>4s} {'succ':>6s} {'err':>7s} {'pos':>5s} {'neg':>5s} {'prb+':>5s} {'prb-':>5s} {'tdp':>3s} {'regret':>7s} {'pacc':>5s} {'cmp_s':>6s}")
    for m, d in out["methods"].items():
        bs = d["by_subset_condition"]
        ps = np.mean([v["success_rate"] * v["n_episodes"] for v in bs["pos"].values()]) * len(bs["pos"]) / max(1, sum(v["n_episodes"] for v in bs["pos"].values())) if bs["pos"] else float("nan")
        ns = np.mean([v["success_rate"] * v["n_episodes"] for v in bs["neg"].values()]) * len(bs["neg"]) / max(1, sum(v["n_episodes"] for v in bs["neg"].values())) if bs["neg"] else float("nan")
        rg = d.get("online_regret_ci95_cluster"); pa = d.get("pairwise_acc_mean")
        print(f"{m:34s} {d['n_episodes']:4d} {d['success_rate']:6.3f} {d['goal_error_mean']:7.4f} {ps:5.2f} {ns:5.2f} "
              f"{d['probe_rate_by_subset'].get('pos', float('nan')):5.2f} {d['probe_rate_by_subset'].get('neg', float('nan')):5.2f} {d['n_terminated_during_probe'] or 0:3d} "
              f"{(rg[0] if rg else float('nan')):7.4f} {(pa if pa is not None else float('nan')):5.2f} {d['compute_total_s_mean']:6.3f}")
        print("    pos per-cond succ:", {c: round(v["success_rate"], 3) for c, v in bs["pos"].items()}, " neg:", {c: round(v["success_rate"], 3) for c, v in bs["neg"].items()})
    for k, v in out.get("paired", {}).items():
        print("PAIR", k, {s: np.round(v[s]["diff_ci95_cluster"], 3).tolist() if v[s]["diff_ci95_cluster"] else None for s in v if s.endswith("success")})

if __name__ == "__main__":
    main()
