"""A10: post-hoc (evaluator-side) true ranking at the ACTUAL online decision state. [toy-run]
For each E2 episode (deterministic env): reset with frozen init/hidden/seed, replay the executed actions recorded
before the task decision (must reproduce recorded obs_before exactly -> replay check), snapshot, then
offline_rank the method's own recorded candidates. Gives per-episode top-1 regret / pairwise acc of the
method's scores on the state it really was in (after its probe). Never feeds back into the method.
Usage: python -m evaluation.online_regret <decisions.jsonl> [--hidden-map test]  -> <decisions>.regret.jsonl
"""
from __future__ import annotations
import json, sys
import numpy as np
from evaluation.audit import load_frozen
from evaluation.offline_rank import rank_candidates
from evaluation import metrics as M

def _env(noise=0.0):
    from src.env.push_env import PushEnv
    return PushEnv(action_noise=noise)

def analyse(records, sets_by_id, conds, env_factory=_env):
    by = {}
    for r in records: by.setdefault((r["method"], r["episode_id"]), []).append(r)
    out = []
    for (m, eid), rs in by.items():
        rs = sorted(rs, key=lambda r: r["step_index"])
        dec = [r for r in rs if r["kind"] == "task" and r.get("candidates") is not None]
        if not dec: continue
        d = dec[0]; st = sets_by_id[d["init_state_id"]]
        env = env_factory(); env.reset(st["init"], conds[d["hidden_condition_id"]], d["seed"])
        for r in rs:
            if r["step_index"] >= d["step_index"]: break
            env.step_segment(np.asarray(r["executed"], float))
        ob = env.observe()
        rep_err = max(float(np.max(np.abs(np.asarray(ob[k], float) - np.asarray(d["obs_before"][k], float)))) for k in ("obj", "pusher", "obj_vel"))
        rk = rank_candidates(env, env.snapshot(), np.asarray(d["candidates"], float))
        e = np.asarray(rk["true_err"]); s = np.asarray(d["scores"], float) if d.get("scores") is not None else None
        row = {"method": m, "episode_id": eid, "init_state_id": d["init_state_id"], "hidden_condition_id": d["hidden_condition_id"],
               "replay_max_abs_err": rep_err, "K": len(e), "chosen": d["chosen_index"],
               "true_err_chosen": float(e[d["chosen_index"]]), "true_err_best": float(e.min()), "oracle_index": rk["oracle_index"],
               "regret_chosen": float(e[d["chosen_index"]] - e.min()), "all_fail": rk["all_fail"]}
        if s is not None:
            acc, n, ties = M.pairwise_accuracy(e, s)
            row.update({"pairwise_acc": acc, "n_pairs": n, "n_true_ties": ties, "top1_regret_scores": M.top1_regret(e, s),
                        "top1_correct": M.top1_correct(e, s)})
        if d.get("predictions") is not None:
            row["endpoint_err_mean"] = float(np.mean(np.linalg.norm(np.asarray(d["predictions"], float) - np.asarray(rk["true_endpoints"]), axis=1)))
        out.append(row)
    return out

if __name__ == "__main__":
    path = sys.argv[1]
    allsets = load_frozen("__audit__", who=f"A10 online_regret {path}")
    ids = {x["init_state_id"]: x for k in ("test", "test_neg", "val", "val_neg", "e0", "e0_neg", "e1") for x in allsets[k]}
    conds = allsets["conditions"]["test"]
    recs = [json.loads(l) for l in open(path)]
    rows = analyse(recs, ids, conds)
    with open(path.replace(".jsonl", "") + ".regret.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    bad = [r for r in rows if r["replay_max_abs_err"] > 1e-9]
    print(len(rows), "episodes; replay mismatches:", len(bad))
