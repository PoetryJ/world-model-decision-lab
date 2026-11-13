"""Summarise runs/a8/<tag>/episodes.jsonl per method x hidden condition. Cluster bootstrap over init states
(cluster = init_state_id) for success-rate and goal-error differences vs no_probe (paired)."""
from __future__ import annotations
import json, sys, collections
import numpy as np

def load(tag):
    return [json.loads(l) for l in open(f"runs/a8/{tag}/episodes.jsonl")]

def boot_diff(eps, m, base="no_probe", key="success", n=2000, seed=0):
    by = collections.defaultdict(dict)
    for e in eps:
        by[e["init_state_id"]].setdefault(e["method"], []).append(float(e[key]))
    ids = [i for i in by if m in by[i] and base in by[i]]
    d = np.array([np.mean(by[i][m]) - np.mean(by[i][base]) for i in ids])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def main(tag):
    eps = load(tag)
    methods = list(dict.fromkeys(e["method"] for e in eps)); conds = sorted(set(e["hidden_condition_id"] for e in eps))
    lines = [f"## {tag}  [toy-run]", "", "| method | cond | n | success | mean final err (m) | probe rate | mean probe steps | mean steps | compute s/ep |", "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for c in conds + ["ALL"]:
            E = [e for e in eps if e["method"] == m and (c == "ALL" or e["hidden_condition_id"] == c)]
            if not E: continue
            lines.append(f"| {m} | {c} | {len(E)} | {np.mean([e['success'] for e in E]):.3f} | {np.mean([e['final_goal_err'] for e in E]):.4f} | "
                         f"{np.mean([e['n_probes'] > 0 for e in E]):.2f} | {np.mean([e['probe_steps'] for e in E]):.2f} | "
                         f"{np.mean([e['steps_used'] for e in E]):.1f} | {np.mean([sum(e['compute_s'].values()) for e in E]):.3f} |")
    lines += ["", "Paired diff vs no_probe (cluster bootstrap over init states, 95% CI):", ""]
    for m in methods:
        if m == "no_probe": continue
        s = boot_diff(eps, m, key="success"); g = boot_diff(eps, m, key="final_goal_err")
        lines.append(f"- {m}: d_success={s[0]:+.3f} [{s[1]:+.3f},{s[2]:+.3f}]; d_err={g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] m")
    txt = "\n".join(lines); print(txt)
    open(f"runs/a8/{tag}/summary.md", "w").write(txt + "\n")

if __name__ == "__main__":
    main(sys.argv[1])
