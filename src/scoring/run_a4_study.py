#!/usr/bin/env python3
"""A7 follow-up on A4 PushEnv (TOY): conservative scoring vs greedy distance.

Context (A4, [toy-run]): in the negative-control (near-goal) set a candidate
that succeeds under all mu exists in 12/12 states, but the distance-argmin
under the WRONG mu still fails sometimes (det. lo-best under hi success 0.58).

Question: if the predictor's ensemble covers a hypothesis set over mu (not
the current episode's hidden value), do conservative aggregations
(worst-case distance, robust success, official LCB) pick a robust candidate
more often than greedy mean-endpoint distance?

No threshold is fitted here; e0 is the reproduction set, e1 is held-out
confirmation. Hidden pair = mu_lo 0.1 vs mu_hi 0.5 (controller).
Evaluation-only oracle: src.env.branch_eval.evaluate_candidates.

  PYTHONPATH=/workspace/wmprobe python -m src.scoring.run_a4_study
"""
from __future__ import annotations
import json, os, time, sys
import numpy as np

from src.env.push_env import PushEnv
from src.env import scenarios as S
from src.env.branch_eval import evaluate_candidates
from src.scoring.scorers import score_detailed, choose_index
from src.scoring.pushenv_stub import PushEnvStubPredictor
from src.scoring.calibration import dump_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs", "a7")
REPS = os.path.join(ROOT, "reports", "a7")
os.makedirs(RUNS, exist_ok=True)

HIDDENS = S.hidden_conditions(2)  # lo 0.1, hi 0.5
SCORE_NAMES = (
    "goal_distance",
    "average_value",
    "lcb",
    "worst_case_value",
    "worst_case_distance",
    "robust_success",
    "robust_success_then_distance",
    "success_prob",
)
# Deployable predictor settings (do not depend on the current episode's mu)
PREDICTORS = {
    "point_lo": dict(assumed_mu=0.10, n_samples=1, mu_perturb_mode="none"),
    "point_mid": dict(assumed_mu=0.25, n_samples=1, mu_perturb_mode="none"),
    "point_hi": dict(assumed_mu=0.50, n_samples=1, mu_perturb_mode="none"),
    "narrow_mid": dict(assumed_mu=0.25, n_samples=5, mu_perturb=0.05, mu_perturb_mode="fixed_grid"),
    "hyp_lohi": dict(mu_list=(0.10, 0.50), assumed_mu=0.25),          # experiment pair as prior
    "wide_grid": dict(mu_list=(0.10, 0.20, 0.30, 0.40, 0.50), assumed_mu=0.30),
}


def env_factory():
    return PushEnv(action_noise=0.0, obs_noise=0.0, max_time=6.0)


def oracle_table(init, seed=0):
    """dist (C,K), success (C,K), final (C,K,2), labels, obs0, cands. Eval only."""
    env = env_factory()
    # same visible obs for all hidden; build grid once
    obs0 = env.reset(init, HIDDENS[0][1], seed)
    cands, labels = S.strike_grid(obs0)
    C = len(HIDDENS)
    K = cands.shape[0]
    dist = np.zeros((C, K))
    succ = np.zeros((C, K), dtype=bool)
    final = np.zeros((C, K, 2))
    for c, (cid, hid) in enumerate(HIDDENS):
        env = env_factory()
        env.reset(init, hid, seed)
        snap = env.snapshot()
        d, su, fin = evaluate_candidates(env, snap, cands)
        dist[c], succ[c], final[c] = d, su.astype(bool), fin
    robust = succ.all(axis=0)  # succeeds under ALL hidden in the pair
    return {
        "obs0": obs0, "cands": cands, "labels": labels,
        "dist": dist, "succ": succ, "final": final,
        "robust": robust, "n_robust": int(robust.sum()),
        "goal": {"goal": np.asarray(init["goal"], float),
                 "goal_radius": float(init["goal_radius"])},
    }


def predict_all(pred, obs, cands, seed):
    return pred.predict(obs, cands, {"seed": seed, "history": [], "adapt_state": None})


def choose_metrics(chosen, dist_row, succ_row, robust, oracle_best_succ):
    d = dist_row[chosen]
    best_d = float(dist_row.min())
    return {
        "chosen": int(chosen),
        "success": bool(succ_row[chosen]),
        "is_robust": bool(robust[chosen]),
        "regret_m": float(d - best_d),
        "true_dist": float(d),
        "oracle_best_success_exists": bool(oracle_best_succ),
    }


def cross_wrong_mu_greedy(ora):
    """A4 diagnostic: argmin distance under condition a, success under b (eval only)."""
    out = {}
    C = len(HIDDENS)
    for a in range(C):
        for b in range(C):
            if a == b:
                continue
            chosen = int(np.argmin(ora["dist"][a]))  # same as argmax(-D)
            aid, bid = HIDDENS[a][0], HIDDENS[b][0]
            out[f"{aid}->{bid}"] = {
                "success": bool(ora["succ"][b, chosen]),
                "is_robust": bool(ora["robust"][chosen]),
                "chosen": chosen,
                "label": ora["labels"][chosen],
            }
    return out


def run_set(states, set_name, split_label):
    """states: list of (sid, init)."""
    rows = []
    t0 = time.perf_counter()
    # build predictors once
    predictors = {name: PushEnvStubPredictor(seed=0, **kw) for name, kw in PREDICTORS.items()}
    for i, (sid, init) in enumerate(states):
        ora = oracle_table(init, seed=0)
        cross = cross_wrong_mu_greedy(ora)
        rec_base = {
            "init_state_id": sid, "set": set_name, "split": split_label,
            "n_robust": ora["n_robust"],
            "robust_exists": bool(ora["n_robust"] > 0),
            "n_succ_by_cond": {HIDDENS[c][0]: int(ora["succ"][c].sum()) for c in range(len(HIDDENS))},
            "cross_wrong_mu_greedy": cross,
            "goal_radius": float(init["goal_radius"]),
            "goal_dist_meta": (init.get("meta") or {}).get("goal_dist"),
        }
        # predictions (shared cands)
        preds = {}
        for pname, pred in predictors.items():
            preds[pname] = predict_all(pred, ora["obs0"], ora["cands"], seed=1000 + i)
        for c, (cid, hid) in enumerate(HIDDENS):
            rec = dict(rec_base)
            rec["hidden_condition_id"] = cid
            rec["mu"] = hid["mu"]
            rec["methods"] = {"oracle_wrong_mu_greedy": {}}
            # diagnostic: wrong-mu oracle greedy
            other = 1 - c
            ch = int(np.argmin(ora["dist"][other]))
            rec["methods"]["oracle_wrong_mu_greedy"]["goal_distance"] = choose_metrics(
                ch, ora["dist"][c], ora["succ"][c], ora["robust"], True
            )
            rec["methods"]["oracle_wrong_mu_greedy"]["goal_distance"]["from_condition"] = HIDDENS[other][0]
            # also oracle known-mu greedy and any-robust
            ch_known = int(np.argmin(ora["dist"][c]))
            rec["methods"]["oracle_known_mu_greedy"] = {
                "goal_distance": choose_metrics(ch_known, ora["dist"][c], ora["succ"][c], ora["robust"], True)
            }
            if ora["n_robust"] > 0:
                # pick robust cand with smallest mean true dist across conditions (eval diagnostic)
                mean_d = ora["dist"].mean(0)
                masked = np.where(ora["robust"], mean_d, np.inf)
                ch_rob = int(np.argmin(masked))
            else:
                ch_rob = int(np.argmin(ora["dist"].mean(0)))
            rec["methods"]["oracle_robust"] = {
                "robust_pick": choose_metrics(ch_rob, ora["dist"][c], ora["succ"][c], ora["robust"], True)
            }
            for pname, P in preds.items():
                rec["methods"][pname] = {}
                for sname in SCORE_NAMES:
                    bun = score_detailed(P, ora["goal"], score_name=sname)
                    # For S=1, LCB safe path equals mean; official NaN avoided by safe=True default
                    ch = choose_index(bun.scores)
                    rec["methods"][pname][sname] = choose_metrics(
                        ch, ora["dist"][c], ora["succ"][c], ora["robust"], True
                    )
                    rec["methods"][pname][sname]["n_samples"] = bun.n_samples
            rows.append(rec)
        if (i + 1) % 4 == 0:
            print(f"  [{set_name}] {i+1}/{len(states)} states  {time.perf_counter()-t0:.1f}s", flush=True)
    return rows, time.perf_counter() - t0


def summarise(rows):
    """Success / robust-pick rates by set, mu, method, scorer."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        for mname, scorers in r["methods"].items():
            for sname, m in scorers.items():
                key = (r["set"], r["split"], r["hidden_condition_id"], mname, sname)
                buckets[key].append(m)
                key_all = (r["set"], r["split"], "ALL", mname, sname)
                buckets[key_all].append(m)
    out = {}
    for key, ms in buckets.items():
        setn, split, cond, mname, sname = key
        rec = out.setdefault(setn, {}).setdefault(split, {}).setdefault(cond, {}).setdefault(mname, {})
        rec[sname] = {
            "n": len(ms),
            "success_rate": float(np.mean([m["success"] for m in ms])),
            "robust_pick_rate": float(np.mean([m["is_robust"] for m in ms])),
            "mean_regret_m": float(np.mean([m["regret_m"] for m in ms])),
            "n_success": int(sum(m["success"] for m in ms)),
        }
    # reproduction of A4 cross success on each set/split
    cross = {}
    for r in rows:
        if r["hidden_condition_id"] != HIDDENS[0][0]:
            continue  # count once per state
        k = (r["set"], r["split"])
        cross.setdefault(k, [])
        cross[k].append(r["cross_wrong_mu_greedy"])
    cross_sum = {}
    for (setn, split), lst in cross.items():
        acc = {}
        for direction in lst[0]:
            acc[direction] = float(np.mean([d[direction]["success"] for d in lst]))
            acc[direction + "_robust_pick"] = float(np.mean([d[direction]["is_robust"] for d in lst]))
        acc["n"] = len(lst)
        cross_sum.setdefault(setn, {})[split] = acc
    # robust existence
    exist = {}
    seen = set()
    for r in rows:
        k = (r["set"], r["split"], r["init_state_id"])
        if k in seen:
            continue
        seen.add(k)
        exist.setdefault((r["set"], r["split"]), []).append(r["robust_exists"])
    exist_sum = {setn: {split: {"n": len(v), "frac_robust_exists": float(np.mean(v))}
                        for (sn, split), v in exist.items() if sn == setn}
                 for setn in {k[0] for k in exist}}
    return {"rates": out, "cross_wrong_mu_greedy": cross_sum, "robust_existence": exist_sum}


def slim_rows(rows):
    slim = []
    for r in rows:
        item = {k: r[k] for k in r if k != "methods"}
        item["methods"] = {}
        for m, sc in r["methods"].items():
            item["methods"][m] = {sn: {kk: vv for kk, vv in d.items()} for sn, d in sc.items()}
        slim.append(item)
    return slim


def main():
    t_all = time.perf_counter()
    print("[a7-a4] e0 pos/neg (reproduction of A4 C4 setting) ...", flush=True)
    e0pos, t1 = run_set(S.e0_set(False), "pos", "e0")
    e0neg, t2 = run_set(S.e0_set(True), "neg", "e0")
    print("[a7-a4] e1 pos/neg (held-out confirmation, seed 1100/2100) ...", flush=True)
    e1pos, t3 = run_set(S.e1_set(False), "pos", "e1")
    e1neg, t4 = run_set(S.e1_set(True), "neg", "e1")
    rows = e0pos + e0neg + e1pos + e1neg
    summary = summarise(rows)
    pack = {
        "label": "toy-run",
        "env": "PushEnv-v1",
        "hiddens": [(c, h) for c, h in HIDDENS],
        "goal_radius": S.GOAL_RADIUS,
        "task_H": S.TASK_H,
        "K": len(S.STRIKE_SPEEDS) * S.TASK_H,
        "score_names": SCORE_NAMES,
        "predictors": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in kw.items()}
                       for k, kw in PREDICTORS.items()},
        "wall_s": {"e0pos": t1, "e0neg": t2, "e1pos": t3, "e1neg": t4,
                   "total": time.perf_counter() - t_all},
        "n_rows": len(rows),
        "summary": summary,
        "note": "Oracle uses branch_eval (evaluation only). Predictors use a private PushEnv with assumed/hypothesis mus; they never read env.hidden. hyp_lohi uses the experiment pair {0.1,0.5} as a prior, not the current episode mu.",
    }
    dump_json(pack, os.path.join(RUNS, "a4_study_summary.json"))
    dump_json(slim_rows(rows), os.path.join(RUNS, "a4_study_raw.json"))
    dump_json(pack, os.path.join(REPS, "a4_study_summary.json"))
    # compact highlight for the report
    def pick(setn, split, cond, method, scorer):
        return summary["rates"][setn][split][cond][method][scorer]
    highlight = {
        "label": "toy-run",
        "cross_wrong_mu_greedy": summary["cross_wrong_mu_greedy"],
        "robust_existence": summary["robust_existence"],
        "neg_e0_ALL": summary["rates"]["neg"]["e0"]["ALL"],
        "neg_e1_ALL": summary["rates"]["neg"]["e1"]["ALL"],
        "pos_e0_ALL": summary["rates"]["pos"]["e0"]["ALL"],
        "pos_e1_ALL": summary["rates"]["pos"]["e1"]["ALL"],
    }
    dump_json(highlight, os.path.join(RUNS, "a4_study_compact.json"))
    dump_json(highlight, os.path.join(REPS, "a4_study_compact.json"))
    print(json.dumps({
        "wall_s": pack["wall_s"],
        "cross_e0_neg": summary["cross_wrong_mu_greedy"].get("neg", {}).get("e0"),
        "cross_e0_pos": summary["cross_wrong_mu_greedy"].get("pos", {}).get("e0"),
        "robust_exist": summary["robust_existence"],
    }, indent=2), flush=True)
    print("[a7-a4] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
