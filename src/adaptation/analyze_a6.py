"""A6 post-hoc analysis on VAL rows (no new env runs). python -m src.adaptation.analyze_a6 > reports/a6/analysis_val.json"""
import json, glob, numpy as np
from scipy.stats import spearmanr
R = "/workspace/wmprobe/runs/a6/e2/"
def load(t): return [json.loads(l) for l in open(R + f"rows_{t}.jsonl")]
out = {}
for t in ["val_s0_p5.0_n0.005_f2", "val_s1_p5.0_n0.005_f2", "val_s2_p5.0_n0.005_f2", "val_s0_latent_n0.005"]:
    rows = load(t); o = {}
    for c in sorted({r["hidden_condition_id"] for r in rows}):
        for neg in (False, True):
            for arm in ["none", "tap_noupdate", "tap_update", "slow_noupdate", "slow_update"]:
                r = [x for x in rows if x["arm"] == arm and x["hidden_condition_id"] == c and x["neg"] == neg]
                o[f"{'neg' if neg else 'pos'}|{c}|{arm}"] = {k: float(np.nanmean([x[k] for x in r])) for k in
                    ["endpoint_err_seg", "goalerr_pred_err", "pairwise_acc", "top1_regret", "online_success", "online_goal_err", "oracle_success"]} | {"n": len(r)}
    # confidence vs error (pos set)
    conf = {}
    for arm in ["none", "tap_noupdate", "tap_update"]:
        r = [x for x in rows if x["arm"] == arm and not x["neg"]]
        e = [x["goalerr_pred_err_chosen"] for x in r]; s = [x["ens_std_chosen"] for x in r]; reg = [x["top1_regret"] for x in r]
        d = {"spearman_ensstd_vs_prederr": spearmanr(s, e)[0], "spearman_ensstd_vs_regret": spearmanr(s, reg)[0],
             "frac_regret_gt_1cm": float(np.mean(np.array(reg) > 0.01)),
             "frac_regret_gt_1cm_and_ensstd_below_median": float(np.mean([(g > 0.01) and (q < np.median(s)) for g, q in zip(reg, s)]))}
        if r and r[0].get("res_std_chosen") is not None:
            rs = [x["res_std_chosen"] for x in r]
            d |= {"spearman_rlsstd_vs_prederr": spearmanr(rs, e)[0], "spearman_rlsstd_vs_regret": spearmanr(rs, reg)[0]}
        conf[arm] = {k: float(v) for k, v in d.items()}
    # prediction error vs control: per-episode paired change (tap_update - tap_noupdate)
    A = {(x["init_state_id"], x["hidden_condition_id"]): x for x in rows if x["arm"] == "tap_update" and not x["neg"]}
    B = {(x["init_state_id"], x["hidden_condition_id"]): x for x in rows if x["arm"] == "tap_noupdate" and not x["neg"]}
    de = np.array([A[k]["endpoint_err_seg"] - B[k]["endpoint_err_seg"] for k in A]); dr = np.array([A[k]["top1_regret"] - B[k]["top1_regret"] for k in A])
    dg = np.array([A[k]["goalerr_pred_err"] - B[k]["goalerr_pred_err"] for k in A])
    pe = {"n": len(de), "frac_endpoint_err_down": float(np.mean(de < 0)), "frac_regret_down": float(np.mean(dr < -1e-4)),
          "frac_endpoint_down_but_regret_up": float(np.mean((de < 0) & (dr > 1e-4))),
          "frac_endpoint_up_but_regret_down": float(np.mean((de > 0) & (dr < -1e-4))),
          "spearman_d_endpoint_vs_d_regret": float(spearmanr(de, dr)[0]),
          "spearman_d_settledgoalerr_pred_vs_d_regret": float(spearmanr(dg, dr)[0])}
    out[t] = {"by_set_cond_arm": o, "confidence": conf, "prederr_vs_control": pe}
print(json.dumps(out, indent=1))
