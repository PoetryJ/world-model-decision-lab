#!/usr/bin/env python3
"""A7 frozen-protocol study (A-8): fit on TRAIN, report VAL. Never load test.

A10 sets via evaluation.audit.load_frozen only for train/train_neg/val/val_neg.
PushEnv, conditions mu 0.1 / 0.3 / 0.6 (main pair lo vs hi=0.6; mid reported separately).
Primary predictor: A6 EnsemblePredictor v3 (ens_push_env_v3_s0), settled by zero-padding.
Secondary: PushEnvStub hypothesis-set {0.1, 0.6} (train-pool endpoints, mid not in the prior).

  PYTHONPATH=/workspace/wmprobe python -m src.scoring.run_frozen_study
"""
from __future__ import annotations
import json, os, time, sys, hashlib
import numpy as np

from evaluation.audit import load_frozen, frozen_sha
from src.env.push_env import PushEnv
from src.env import scenarios as S
from src.env.branch_eval import evaluate_candidates
from src.scoring.scorers import (
    score_detailed, choose_index, LCB_BETA_OFFICIAL, MAJORITY_THRESH_OFFICIAL,
    VALUE_SCALE, GAMMA_DEFAULT,
)
from src.scoring.pushenv_stub import PushEnvStubPredictor
from src.scoring.config import pad_for_settle, SETTLE_PAD, CONDITIONS, MAIN_PAIR, DEFAULT_CONFIG_PATH
from src.scoring.calibration import dump_json, reliability_table, logistic_prob, write_reliability_svg
from src.scoring.gate import FEATURE_NAMES, extract_feature, LowConfidenceGate, evaluate_gate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs", "a7")
REPS = os.path.join(ROOT, "reports", "a7")
FIGS = os.path.join(REPS, "figures")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

ALLOWED = {"train", "train_neg", "val", "val_neg"}
FORBIDDEN_PREFIX = "test"
TEST_CONDS = tuple((k, dict(v)) for k, v in CONDITIONS.items())  # lo, mid, hi in that order
SCORE_NAMES = (
    "goal_distance", "average_value", "lcb",
    "worst_case_value", "worst_case_distance",
    "robust_success", "robust_success_then_distance", "success_prob",
)
PROBE_COST = 0.4


def load_split(split: str):
    if split.startswith(FORBIDDEN_PREFIX) or split not in ALLOWED:
        raise RuntimeError(f"A7 A-8 forbids loading split={split!r}; allowed={sorted(ALLOWED)}")
    return load_frozen(split, who="a7")


def env_factory():
    return PushEnv(action_noise=0.0, obs_noise=0.0, max_time=6.0)


def load_a6():
    from src.dynamics.model import load_or_train
    pred = load_or_train(seed=0, retrain=False)
    meta = getattr(pred, "meta", {}) or {}
    path = os.path.join(ROOT, "runs", "a6", "models", "ens_push_env_v3_s0.pkl")
    return pred, {"path": path, "meta_cfg": meta.get("cfg"), "n_members": len(getattr(pred, "members", [])),
                  "n_params": (meta.get("n_params_total") if meta else None)}


class A6SettledPredictor:
    """Wrap A6: pad zero settle steps so endpoints are resting positions. Never reads hidden."""
    name = "a6_v3_settled"

    def __init__(self, base, settle=SETTLE_PAD):
        self.base = base
        self.settle = int(settle)

    def predict(self, observation, actions, context=None):
        padded = pad_for_settle(actions, self.settle)
        P = self.base.predict(observation, padded, context or {"history": [], "adapt_state": None})
        # traj includes settle frames; scoring uses endpoints/samples (settled)
        if P.extra is None:
            P.extra = {}
        P.extra["model"] = self.name
        P.extra["settle_pad"] = self.settle
        return P


def oracle_table(init, seed):
    env = env_factory()
    obs0 = env.reset(init, TEST_CONDS[0][1], seed)
    cands, labels = S.strike_grid(obs0)
    C = len(TEST_CONDS)
    K = cands.shape[0]
    dist = np.zeros((C, K))
    succ = np.zeros((C, K), dtype=bool)
    for c, (cid, hid) in enumerate(TEST_CONDS):
        env = env_factory()
        env.reset(init, hid, seed)
        d, su, _ = evaluate_candidates(env, env.snapshot(), cands, settle_max=SETTLE_PAD)
        dist[c], succ[c] = d, su.astype(bool)
    idx = {TEST_CONDS[c][0]: c for c in range(C)}
    robust3 = succ.all(0)
    robust_pair = succ[idx["mu_lo"]] & succ[idx["mu_hi"]]
    return dict(obs0=obs0, cands=cands, labels=labels, dist=dist, succ=succ,
                robust3=robust3, n_robust3=int(robust3.sum()),
                robust_pair=robust_pair, n_robust_pair=int(robust_pair.sum()),
                goal={"goal": np.asarray(init["goal"], float), "goal_radius": float(init["goal_radius"])},
                idx=idx)


def metrics(chosen, dist_row, succ_row, robust3, robust_pair):
    d = float(dist_row[chosen])
    return {"chosen": int(chosen), "success": bool(succ_row[chosen]),
            "is_robust3": bool(robust3[chosen]), "is_robust_pair": bool(robust_pair[chosen]),
            "regret_m": float(d - dist_row.min()), "true_dist": d}


def run_states(records, set_name, split, predictors):
    rows, t0 = [], time.perf_counter()
    C = len(TEST_CONDS)
    for i, rec in enumerate(records):
        init, sid, seed = rec["init"], rec["init_state_id"], int(rec["seed"])
        ora = oracle_table(init, seed)
        preds = {n: p.predict(ora["obs0"], ora["cands"], {"seed": 10_000 + seed, "history": [], "adapt_state": None})
                 for n, p in predictors.items()}
        cross = {}
        for a in range(C):
            for b in range(C):
                if a == b:
                    continue
                ch = int(np.argmin(ora["dist"][a]))
                cross[f"{TEST_CONDS[a][0]}->{TEST_CONDS[b][0]}"] = {
                    "success": bool(ora["succ"][b, ch]),
                    "is_robust3": bool(ora["robust3"][ch]),
                    "is_robust_pair": bool(ora["robust_pair"][ch]),
                }
        for c, (cid, hid) in enumerate(TEST_CONDS):
            if cid == "mu_lo":
                w = ora["idx"]["mu_hi"]
            elif cid == "mu_hi":
                w = ora["idx"]["mu_lo"]
            else:
                w = ora["idx"]["mu_hi"]
            row = {"init_state_id": sid, "set": set_name, "split": split,
                   "hidden_condition_id": cid, "mu": hid["mu"], "seed": seed,
                   "n_robust3": ora["n_robust3"], "n_robust_pair": ora["n_robust_pair"],
                   "cross_wrong_mu_greedy": cross, "methods": {}}
            row["methods"]["oracle_wrong_main_greedy"] = {"goal_distance": metrics(
                int(np.argmin(ora["dist"][w])), ora["dist"][c], ora["succ"][c], ora["robust3"], ora["robust_pair"])}
            row["methods"]["oracle_known_mu_greedy"] = {"goal_distance": metrics(
                int(np.argmin(ora["dist"][c])), ora["dist"][c], ora["succ"][c], ora["robust3"], ora["robust_pair"])}
            row["methods"]["a6"] = {}
            row["methods"]["hyp_lohi"] = {}
            for pname, P in preds.items():
                for sname in SCORE_NAMES:
                    bun = score_detailed(P, ora["goal"], score_name=sname)
                    ch = choose_index(bun.scores)
                    met = metrics(ch, ora["dist"][c], ora["succ"][c], ora["robust3"], ora["robust_pair"])
                    met.update({"n_samples": bun.n_samples,
                                "chosen_success_prob": float(bun.success_prob[ch]),
                                "value_std": float(bun.uncertainty_value_std[ch]),
                                "agreement": float(bun.sample_argmax_agreement)})
                    met["_feat_src"] = bun
                    row["methods"][pname][sname] = met
            rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{split}/{set_name}] {i+1}/{len(records)}  {time.perf_counter()-t0:.1f}s", flush=True)
    return rows, time.perf_counter() - t0


def summarise(rows):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        for mname, scorers in r["methods"].items():
            for sname, m in scorers.items():
                for cond in (r["hidden_condition_id"], "ALL",
                             "MAIN_PAIR" if r["hidden_condition_id"] in MAIN_PAIR else None):
                    if cond:
                        buckets[(r["set"], r["split"], cond, mname, sname)].append(m)
    rates = {}
    for (setn, split, cond, mname, sname), ms in buckets.items():
        rates.setdefault(setn, {}).setdefault(split, {}).setdefault(cond, {}).setdefault(mname, {})[sname] = {
            "n": len(ms),
            "success_rate": float(np.mean([m["success"] for m in ms])),
            "robust3_pick_rate": float(np.mean([m["is_robust3"] for m in ms])),
            "robust_pair_pick_rate": float(np.mean([m["is_robust_pair"] for m in ms])),
            "mean_regret_m": float(np.mean([m["regret_m"] for m in ms])),
            "n_success": int(sum(m["success"] for m in ms)),
        }
    cross_sum, acc, seen = {}, {}, set()
    for r in rows:
        k = (r["set"], r["split"], r["init_state_id"])
        if k in seen:
            continue
        seen.add(k)
        acc.setdefault((r["set"], r["split"]), []).append(r["cross_wrong_mu_greedy"])
    for (setn, split), lst in acc.items():
        dirs = {"n": len(lst)}
        for direction in lst[0]:
            dirs[direction] = float(np.mean([d[direction]["success"] for d in lst]))
            dirs[direction + "_robust3"] = float(np.mean([d[direction]["is_robust3"] for d in lst]))
        cross_sum.setdefault(setn, {})[split] = dirs
    exist, tmp, seen = {}, {}, set()
    for r in rows:
        k = (r["set"], r["split"], r["init_state_id"])
        if k in seen:
            continue
        seen.add(k)
        tmp.setdefault((r["set"], r["split"]), []).append(
            (r["n_robust3"] > 0, r["n_robust_pair"] > 0, r["n_robust3"], r["n_robust_pair"]))
    for (setn, split), vs in tmp.items():
        exist.setdefault(setn, {})[split] = {
            "n": len(vs),
            "frac_robust3_exists": float(np.mean([v[0] for v in vs])),
            "frac_robust_pair_exists": float(np.mean([v[1] for v in vs])),
            "mean_n_robust3": float(np.mean([v[2] for v in vs])),
            "mean_n_robust_pair": float(np.mean([v[3] for v in vs])),
        }
    return {"rates": rates, "cross_wrong_mu_greedy": cross_sum, "robust_existence": exist}


def pick_default_on_train(summary):
    """Select default (predictor, scorer) using TRAIN only (pooled pos+neg, ALL conditions)."""
    cand = []
    for setn in ("pos", "neg"):
        pass
    # average success over pos+neg train ALL
    methods = ("a6", "hyp_lohi")
    scorers = ("goal_distance", "lcb", "worst_case_distance", "robust_success_then_distance", "average_value")
    scored = []
    for m in methods:
        for s in scorers:
            succs, regrets, ns = [], [], 0
            for setn in ("pos", "neg"):
                blk = summary["rates"].get(setn, {}).get("train", {}).get("ALL", {}).get(m, {}).get(s)
                if not blk:
                    continue
                succs.append(blk["success_rate"] * blk["n"])
                regrets.append(blk["mean_regret_m"] * blk["n"])
                ns += blk["n"]
            if ns == 0:
                continue
            scored.append(((sum(succs) / ns), -(sum(regrets) / ns), m, s, ns))
    scored.sort(reverse=True)
    return scored


def fit_gates(rows_train, method, scorer):
    out = {}
    for feat in FEATURE_NAMES:
        f, w = [], []
        for r in rows_train:
            if method not in r["methods"] or scorer not in r["methods"][method]:
                continue
            m = r["methods"][method][scorer]
            bun = m.get("_feat_src")
            if bun is None:
                continue
            f.append(extract_feature(bun, feat, m["chosen"]))
            w.append(0.0 if m["success"] else 1.0)
        gate = LowConfidenceGate.fit(np.array(f), np.array(w), feat, probe_cost=PROBE_COST, fitted_on="train")
        ev = evaluate_gate(gate, np.array(f), np.array(w))
        out[feat] = {"feature": feat, "threshold": gate.threshold, "logistic_ab": gate.logistic_ab,
                     "fitted_on": "train", "probe_cost": PROBE_COST, "n_fit": gate.n_fit,
                     "train_eval": ev, "_gate": gate}
    return out


def apply_gates(gpack, rows, method, scorer):
    applied = {}
    for feat, info in gpack.items():
        gate = info["_gate"]
        groups = {"ALL": ([], []), "pos": ([], []), "neg": ([], []),
                  "mu_lo": ([], []), "mu_mid": ([], []), "mu_hi": ([], []), "MAIN_PAIR": ([], [])}
        for r in rows:
            if method not in r["methods"] or scorer not in r["methods"][method]:
                continue
            m = r["methods"][method][scorer]
            bun = m.get("_feat_src")
            if bun is None:
                continue
            f = extract_feature(bun, feat, m["chosen"])
            w = 0.0 if m["success"] else 1.0
            tags = ["ALL", r["set"], r["hidden_condition_id"]]
            if r["hidden_condition_id"] in MAIN_PAIR:
                tags.append("MAIN_PAIR")
            for g in tags:
                groups[g][0].append(f); groups[g][1].append(w)
        applied[feat] = {
            "threshold": gate.threshold, "fitted_on": "train", "probe_cost": PROBE_COST,
            "slices": {g: evaluate_gate(gate, np.array(fv), np.array(wv))
                       for g, (fv, wv) in groups.items() if fv},
        }
        if info.get("logistic_ab") and groups["ALL"][0]:
            a, b = info["logistic_ab"]
            applied[feat]["logistic_ece"] = reliability_table(
                logistic_prob(np.array(groups["ALL"][0]), a, b), np.array(groups["ALL"][1]),
                n_bins=6, strategy="uniform")
    return applied


def calib_block(rows, method, scorer="goal_distance"):
    out = {}
    for split in sorted({r["split"] for r in rows}):
        for cond in ("ALL", "MAIN_PAIR", "mu_lo", "mu_mid", "mu_hi"):
            sel = [r for r in rows if r["split"] == split
                   and method in r["methods"] and scorer in r["methods"][method]
                   and (cond == "ALL" or (cond == "MAIN_PAIR" and r["hidden_condition_id"] in MAIN_PAIR)
                        or r["hidden_condition_id"] == cond)]
            if not sel:
                continue
            p = np.array([r["methods"][method][scorer]["chosen_success_prob"] for r in sel])
            y = np.array([1.0 if r["methods"][method][scorer]["success"] else 0.0 for r in sel])
            out.setdefault(split, {})[cond] = reliability_table(p, y, n_bins=8, strategy="uniform")
    return out


def slim(rows):
    out = []
    for r in rows:
        item = {k: r[k] for k in r if k != "methods"}
        item["methods"] = {m: {sn: {k: v for k, v in d.items() if k != "_feat_src"}
                               for sn, d in sc.items()} for m, sc in r["methods"].items()}
        out.append(item)
    return out


def pub_gates(gpack):
    return {feat: {k: v for k, v in info.items() if not k.startswith("_")} for feat, info in gpack.items()}


def main():
    t_all = time.perf_counter()
    print("[a7-frozen A-8] load A6 v3 + train/val only ...", flush=True)
    a6_base, a6_info = load_a6()
    predictors = {
        "a6": A6SettledPredictor(a6_base, SETTLE_PAD),
        "hyp_lohi": PushEnvStubPredictor(mu_list=(0.10, 0.60), assumed_mu=0.20, seed=0, settle_max=SETTLE_PAD),
    }
    print(f"  A6 members={a6_info['n_members']} path exists={os.path.isfile(a6_info['path'])}", flush=True)

    train = load_split("train")
    train_neg = load_split("train_neg")
    val = load_split("val")
    val_neg = load_split("val_neg")
    sha = frozen_sha()
    print(f"  sha={sha}  train={len(train)} train_neg={len(train_neg)} val={len(val)} val_neg={len(val_neg)}", flush=True)

    tr_pos, t1 = run_states(train, "pos", "train", predictors)
    tr_neg, t2 = run_states(train_neg, "neg", "train", predictors)
    va_pos, t3 = run_states(val, "pos", "val", predictors)
    va_neg, t4 = run_states(val_neg, "neg", "val", predictors)
    train_rows, val_rows = tr_pos + tr_neg, va_pos + va_neg
    all_rows = train_rows + val_rows
    summary = summarise(all_rows)

    ranked = pick_default_on_train(summary)
    default = {"predictor": ranked[0][2], "score_name": ranked[0][3],
               "train_success": ranked[0][0], "train_n": ranked[0][4],
               "ranked": [{"predictor": m, "score_name": s, "train_success": suc, "n": n}
                          for suc, _r, m, s, n in ranked]}
    print(f"  default from TRAIN: {default['predictor']}::{default['score_name']} "
          f"succ={default['train_success']:.3f}", flush=True)

    gpack = fit_gates(train_rows, default["predictor"], default["score_name"])
    # pick gate feature by TRAIN expected cost (lowest)
    feat_rank = sorted(gpack.items(), key=lambda kv: kv[1]["train_eval"]["expected_cost"])
    best_feat = feat_rank[0][0]
    gate_train_eval = apply_gates(gpack, train_rows, default["predictor"], default["score_name"])
    gate_val_eval = apply_gates(gpack, val_rows, default["predictor"], default["score_name"])

    calib = {m: {"train": calib_block(train_rows, m), "val": calib_block(val_rows, m)}
             for m in ("a6", "hyp_lohi")}
    for split, block in (("train", calib["a6"]["train"]), ("val", calib["a6"]["val"])):
        inner = block.get(split, block)
        for cond, tab in inner.items():
            if cond in ("ALL", "mu_mid", "mu_lo", "mu_hi", "MAIN_PAIR"):
                write_reliability_svg(
                    tab, os.path.join(FIGS, f"frozen_reliability_a6_{split}_{cond}.svg"),
                    title=f"[frozen {split} {cond}] A6 success_prob vs success [toy-run; thresh from train]",
                )

    # Frozen config for the controller / sealed test (no val-tuned numbers used as thresholds)
    best_g = gpack[best_feat]
    config = {
        "version": "a7-scorer-v1",
        "frozen_sets_sha256": sha,
        "fit_split": "train+train_neg",
        "report_split": "val+val_neg",
        "conditions": CONDITIONS,
        "main_pair": list(MAIN_PAIR),
        "mu_mid_note": "held out of A6 train mus {0.2,0.4,0.5}; report separately as interpolation",
        "default_predictor": default["predictor"],
        "default_score_name": default["score_name"],
        "a6_model": a6_info,
        "settle_pad": SETTLE_PAD,
        "lcb_beta": LCB_BETA_OFFICIAL,
        "lcb_beta_provenance": "official Cosmos aggregate_value_predictions beta=1.0 @ 18a2acc",
        "lcb_safe_single": True,
        "lcb_safe_single_note": "official LCB with S=1 is NaN; safe fallback=mean. A6 has S=5 so official LCB is used in practice.",
        "value_scale_m": VALUE_SCALE,
        "value_scale_provenance": "A7 shaped-value map (table half-width 0.5 m), NOT official Cosmos value head",
        "gamma": GAMMA_DEFAULT,
        "majority_thresh_official": MAJORITY_THRESH_OFFICIAL,
        "majority_thresh_provenance": "official Cosmos 0.05; scale-sensitive, not the default ranker",
        "gate": {
            "feature": best_feat,
            "threshold": best_g["threshold"],
            "logistic_ab": best_g["logistic_ab"],
            "fitted_on": "train",
            "probe_cost": PROBE_COST,
            "n_fit": best_g["n_fit"],
            "train_eval": best_g["train_eval"],
            "note": "Threshold and logistic fitted on train only. Ensemble variance is a FEATURE, not a calibrated confidence.",
        },
        "gate_all_features_train": pub_gates(gpack),
        "default_selection_from_train": default,
        "import": {
            "make_scorer": "from src.scoring import make_scorer, make_gate, score_detailed, choose_index, pad_for_settle, load_scorer_config",
            "a6": "from src.dynamics.model import load_or_train; pred=load_or_train(seed=0, retrain=False)",
            "score": "P=pred.predict(obs, pad_for_settle(cands), ctx); scores=make_scorer().score(P, {'goal': obs['goal'], 'goal_radius': obs['goal_radius']})",
        },
        "label": "toy-run, frozen protocol, thresholds from train only",
    }
    # write config first without sha, then add sha
    dump_json(config, DEFAULT_CONFIG_PATH)
    dump_json(config, os.path.join(RUNS, "frozen_scorer_config.json"))
    with open(DEFAULT_CONFIG_PATH, "rb") as f:
        config["config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    dump_json(config, DEFAULT_CONFIG_PATH)
    dump_json(config, os.path.join(RUNS, "frozen_scorer_config.json"))

    pack = {
        "label": "toy-run, A10 frozen protocol, fit=train report=val, no test load",
        "frozen_sets_sha256": sha,
        "config_sha256": config["config_sha256"],
        "wall_s": {"train_pos": t1, "train_neg": t2, "val_pos": t3, "val_neg": t4,
                   "total": time.perf_counter() - t_all},
        "n_states": {"train": len(train), "train_neg": len(train_neg), "val": len(val), "val_neg": len(val_neg)},
        "summary": summary,
        "default_from_train": default,
        "gate_train": gate_train_eval,
        "gate_val": gate_val_eval,  # reported; thresholds not refit
        "calibration": calib,
        "a6": a6_info,
    }
    dump_json(pack, os.path.join(RUNS, "frozen_trainval_summary.json"))
    dump_json(slim(all_rows), os.path.join(RUNS, "frozen_trainval_raw.json"))
    compact = {
        "label": pack["label"],
        "config_sha256": config["config_sha256"],
        "default_from_train": default,
        "wall_s": pack["wall_s"],
        "robust_existence": summary["robust_existence"],
        "cross_wrong_mu_greedy": summary["cross_wrong_mu_greedy"],
        "rates": summary["rates"],
        "gate_val": gate_val_eval,
        "calib_a6": calib["a6"],
    }
    dump_json(compact, os.path.join(RUNS, "frozen_trainval_compact.json"))
    dump_json(compact, os.path.join(REPS, "frozen_trainval_compact.json"))
    print(json.dumps({
        "wall_s": pack["wall_s"],
        "default": {k: default[k] for k in default if k != "ranked"},
        "robust_existence": summary["robust_existence"],
        "cross_neg": summary["cross_wrong_mu_greedy"].get("neg"),
        "config_sha256": config["config_sha256"],
        "config_path": "reports/a7/frozen_scorer_config.json",
    }, indent=2), flush=True)
    print("[a7-frozen A-8] done (no test loaded)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
