#!/usr/bin/env python3
"""A7 study: value vs credibility, high-confidence errors, low-confidence gate.

All numbers are [toy-run] on FixturePushEnv (CPU 2D, idealised pose, not vision)
unless a comment says [static-review].

Repro:
  cd /workspace/wmprobe && . .venv/bin/activate
  PYTHONPATH=/workspace/wmprobe python -m src.scoring.run_study
  PYTHONPATH=/workspace/wmprobe python -m src.scoring.test_scoring

Seeds, splits, and paths are written to runs/a7/meta.json.
Calibration / gate thresholds are fit on VAL only; TEST is reported separately.
"""
from __future__ import annotations
import json, os, time, sys
from collections import defaultdict
import numpy as np

from shared.fixture_env import FixturePushEnv, FIXTURE_INIT, FIXTURE_HIDDEN
from shared.interface import Prediction, to_jsonable
from src.scoring.scorers import (
    VALUE_SCALE, LCB_BETA_OFFICIAL, MAJORITY_THRESH_OFFICIAL, GAMMA_DEFAULT,
    CompositeScorer, score_detailed, choose_index, official_aggregate,
    dist_to_value,
)
from src.scoring.stub_predictor import WrongMuStubPredictor
from src.scoring.metrics import pairwise_ranking_accuracy, top1_regret, along_lateral_error, TIE_TOL_M
from src.scoring.calibration import (
    reliability_table, fit_logistic, logistic_prob, write_reliability_svg, dump_json,
)
from src.scoring.gate import FEATURE_NAMES, extract_feature, LowConfidenceGate, evaluate_gate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs", "a7")
REPS = os.path.join(ROOT, "reports", "a7")
FIGS = os.path.join(REPS, "figures")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# ---------- frozen experiment config ----------
MASTER_SEED = 7
H = 20
PUSH_VX = 0.4
PUSH_NS = (5, 7, 9, 11)          # 4 fixed candidates (push then coast)
HIDDENS = tuple(FIXTURE_HIDDEN)  # mu 0.1, 0.3, 0.6
ASSUMED_MU = 0.3                 # stub's misspecified nominal mu (matches mid condition)
N_SAMPLES = 9
# Two uncertainty widths: narrow (overconfident under misspec) vs wide
SETTINGS = {
    "narrow": dict(assumed_mu=ASSUMED_MU, n_samples=N_SAMPLES, mu_perturb=0.05,
                   mu_perturb_mode="fixed_grid", endpoint_noise=0.0, seed=MASTER_SEED),
    "wide": dict(assumed_mu=ASSUMED_MU, n_samples=N_SAMPLES, mu_perturb=0.25,
                 mu_perturb_mode="fixed_grid", endpoint_noise=0.0, seed=MASTER_SEED),
}
SCORE_NAMES = (
    "goal_distance",          # simple baseline
    "success_prob",           # P(in goal)
    "discounted_value",       # discounted shaped return from traj
    "average_value",          # official average of shaped values
    "lcb",                    # official LCB baseline (safe S=1; here S=9)
    "majority_mean",          # official majority_mean, thresh=0.05
    "majority_mean_at_radius",# scale-aware variant (NOT official)
    "success_vote",           # official success_vote, thresh=0.05
    "success_vote_at_radius",
)
PROBE_COSTS = (0.2, 0.4, 0.8)   # relative cost of abstain vs a wrong commit
N_BINS = 8


def make_candidates(H=H, ns=PUSH_NS, vx=PUSH_VX):
    K = len(ns)
    c = np.zeros((K, H, 2), dtype=np.float64)
    for i, n in enumerate(ns):
        c[i, :n, 0] = vx
    return c


def build_init_bank(seed: int = MASTER_SEED):
    """30 inits, split 10/10/10 by init id (all mus appear in each split).

    Includes SAME-best (goal x≈0.15 with this candidate set) and DIFF-best
    cases. Split assignment is by init index after a seeded shuffle, not by
    outcome, and is frozen in runs/a7/inits.json.
    """
    rng = np.random.default_rng(seed)
    # A structured grid, then a few jittered extras, then shuffle.
    grid = []
    for gx in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35):
        for gy in (0.0, -0.08, 0.08):
            grid.append((gx, gy, 0.0))  # start y offset
    # 24 grid cells; add 6 jittered
    extras = []
    for _ in range(6):
        gx = float(rng.choice([0.09, 0.14, 0.18, 0.22, 0.28, 0.32]))
        gy = float(rng.uniform(-0.10, 0.10))
        oy = float(rng.uniform(-0.06, 0.06))
        extras.append((gx, gy, oy))
    specs = grid + extras  # 30
    assert len(specs) == 30
    order = rng.permutation(len(specs))
    # first 10 train, next 10 val, last 10 test in shuffled order
    split_of = {}
    for rank, idx in enumerate(order):
        split_of[int(idx)] = "train" if rank < 10 else ("val" if rank < 20 else "test")
    inits = []
    for i, (gx, gy, oy) in enumerate(specs):
        init = {
            "pusher": [-0.25, oy],
            "obj": [-0.15, oy],
            "goal": [float(gx), float(gy)],
            "goal_radius": 0.05,
        }
        inits.append({
            "init_state_id": f"init_{i:02d}",
            "split": split_of[i],
            "init": init,
            "gx": float(gx), "gy": float(gy), "oy": float(oy),
        })
    return inits


def oracle_rollouts(env, init, hidden, seed, candidates):
    """Offline branching: restore the same snapshot, execute each candidate."""
    obs0 = env.reset(init, hidden, seed=seed)
    # never hand hidden to policy; we only keep a label
    snap = env.snapshot()
    K = candidates.shape[0]
    true_ep = np.zeros((K, 2))
    true_dist = np.zeros(K)
    true_succ = np.zeros(K, dtype=bool)
    true_term = []
    for k in range(K):
        env.restore(snap)
        rs = env.step_segment(candidates[k])
        last = rs[-1]
        true_ep[k] = last.obs["obj"]
        true_dist[k] = float(np.linalg.norm(last.obs["obj"] - last.obs["goal"]))
        true_succ[k] = bool(last.success)
        true_term.append(last.info.get("termination"))
    env.restore(snap)
    return {
        "obs0": obs0,
        "true_ep": true_ep,
        "true_dist": true_dist,
        "true_succ": true_succ,
        "true_term": true_term,
        "goal": {"goal": np.asarray(init["goal"], float), "goal_radius": float(init["goal_radius"])},
    }


def best_action_pattern(true_by_mu: dict, tie_tol=TIE_TOL_M) -> str:
    """SAME if every hidden condition shares an oracle-tied winner set overlap;
    DIFF otherwise. Uses the intersection of argmin sets."""
    sets = []
    for mu, d in true_by_mu.items():
        best = d.min()
        sets.append(set(np.flatnonzero(np.abs(d - best) <= tie_tol).tolist()))
    inter = set.intersection(*sets) if sets else set()
    return "SAME_BEST" if inter else "DIFF_BEST"


def run_offline(setting_name: str, pred_kwargs: dict, inits, candidates, env_factory):
    predictor = WrongMuStubPredictor(**pred_kwargs)
    rows = []
    t0 = time.perf_counter()
    score_t = 0.0
    predict_t = 0.0
    for spec in inits:
        # first collect oracle across mus to label SAME/DIFF for this init
        oracle_mu = {}
        for hid in HIDDENS:
            env = env_factory()
            # episode seed depends only on init id (same visible start)
            ep_seed = _episode_seed(spec["init_state_id"], hid["mu"])
            ora = oracle_rollouts(env, spec["init"], hid, ep_seed, candidates)
            oracle_mu[hid["mu"]] = ora["true_dist"]
        pattern = best_action_pattern(oracle_mu)

        for hid in HIDDENS:
            env = env_factory()
            ep_seed = _episode_seed(spec["init_state_id"], hid["mu"])
            ora = oracle_rollouts(env, spec["init"], hid, ep_seed, candidates)
            # predictor sees only the observation, not hidden
            ctx = {"history": [], "adapt_state": None, "seed": ep_seed + 1000}
            t_p = time.perf_counter()
            pred = predictor.predict(ora["obs0"], candidates, ctx)
            predict_t += time.perf_counter() - t_p
            goal = ora["goal"]
            # error decomposition vs mean predicted endpoint
            decomp = along_lateral_error(pred.endpoints, ora["true_ep"],
                                         spec["init"]["obj"], spec["init"]["goal"])
            rec = {
                "init_state_id": spec["init_state_id"],
                "split": spec["split"],
                "hidden_condition_id": f"mu{hid['mu']}",
                "mu": hid["mu"],  # evaluation-only label (not given to scorer)
                "seed": ep_seed,
                "pattern": pattern,
                "true_dist": ora["true_dist"],
                "true_ep": ora["true_ep"],
                "true_succ": ora["true_succ"],
                "pred_ep": pred.endpoints,
                "e_along": decomp["e_along"],
                "e_lat": decomp["e_lat"],
                "e_norm": decomp["e_norm"],
                "bundles": {},
            }
            t_s = time.perf_counter()
            for name in SCORE_NAMES:
                bun = score_detailed(pred, goal, score_name=name)
                chosen = choose_index(bun.scores)
                pw = pairwise_ranking_accuracy(bun.scores, ora["true_dist"])
                rg = top1_regret(chosen, ora["true_dist"])
                rec["bundles"][name] = {
                    "scores": bun.scores,
                    "chosen": chosen,
                    "success_prob": bun.success_prob,
                    "mean_distance": bun.mean_distance,
                    "mean_value": bun.mean_value,
                    "lcb_value": bun.lcb_value,
                    "discounted_value": bun.discounted_value,
                    "value_std": bun.uncertainty_value_std,
                    "endpoint_rms": bun.uncertainty_endpoint_rms,
                    "credibility": bun.credibility,
                    "agreement": bun.sample_argmax_agreement,
                    "n_samples": bun.n_samples,
                    **pw, **rg,
                    "chosen_success_prob": float(bun.success_prob[chosen]),
                    "chosen_true_success": bool(ora["true_succ"][chosen]),
                    "bundle": bun,
                }
            score_t += time.perf_counter() - t_s
            rows.append(rec)
    wall = time.perf_counter() - t0
    meta = {
        "setting": setting_name, "pred_kwargs": pred_kwargs, "n_rows": len(rows),
        "wall_s": wall, "predict_s": predict_t, "score_s": score_t,
        "K": int(candidates.shape[0]), "H": int(candidates.shape[1]),
        "n_samples": pred_kwargs["n_samples"],
    }
    return rows, meta


def _episode_seed(init_id: str, mu: float) -> int:
    # stable, explicit, independent of global RNG
    n = int(init_id.split("_")[1])
    return MASTER_SEED * 10_000 + n * 10 + int(round(mu * 10))


def _summarise(rows, score_name, split=None, mu=None, pattern=None):
    sel = rows
    if split is not None:
        sel = [r for r in sel if r["split"] == split]
    if mu is not None:
        sel = [r for r in sel if abs(r["mu"] - mu) < 1e-12]
    if pattern is not None:
        sel = [r for r in sel if r["pattern"] == pattern]
    if not sel:
        return {"n": 0}
    accs, regrets, correct = [], [], []
    pair_num = pair_den = 0
    for r in sel:
        b = r["bundles"][score_name]
        accs.append(b["pairwise_acc"] if b["n_compared"] else np.nan)
        regrets.append(b["regret_m"])
        correct.append(1.0 if b["top1_correct"] else 0.0)
        pair_num += b["n_correct"]
        pair_den += b["n_compared"]
    regrets = np.asarray(regrets)
    return {
        "n": len(sel),
        "top1_acc": float(np.mean(correct)),
        "mean_regret_m": float(np.mean(regrets)),
        "median_regret_m": float(np.median(regrets)),
        "pairwise_acc_micro": (pair_num / pair_den) if pair_den else np.nan,
        "pairwise_acc_macro": float(np.nanmean(accs)),
        "n_pairwise": int(pair_den),
    }


def high_conf_wrong(rows, score_name, split, feature="value_std",
                    conf_quantile=0.3, quantile_from="val"):
    """High confidence = feature in the lowest `conf_quantile` of VAL features
    (low std / we will invert for features where low=confident).

    For value_std and endpoint_rms and inv_agreement: low = confident.
    For neg_margin*: more negative gap magnitude... actually neg_margin is
    -gap, so low (more negative) = large gap = confident. Same direction:
    LOW feature => confident for all current features.
    Threshold is the conf_quantile quantile of VAL features (fit on val only).
    """
    val_f = np.array([extract_feature(r["bundles"][score_name]["bundle"], feature,
                                      r["bundles"][score_name]["chosen"])
                      for r in rows if r["split"] == quantile_from])
    if len(val_f) == 0:
        return {"n": 0}
    thr = float(np.quantile(val_f, conf_quantile))
    sel = [r for r in rows if r["split"] == split]
    feats = []
    wrong = []
    for r in sel:
        b = r["bundles"][score_name]
        feats.append(extract_feature(b["bundle"], feature, b["chosen"]))
        wrong.append(not b["top1_correct"])
    feats = np.asarray(feats)
    wrong = np.asarray(wrong, dtype=bool)
    high = feats <= thr  # low uncertainty => high confidence
    n_high = int(high.sum())
    n_high_wrong = int((high & wrong).sum())
    by_mu = {}
    for mu in [h["mu"] for h in HIDDENS]:
        m = np.array([abs(r["mu"] - mu) < 1e-12 for r in sel], dtype=bool)
        by_mu[f"mu{mu}"] = {
            "n_high": int((high & m).sum()),
            "n_high_wrong": int((high & m & wrong).sum()),
            "n": int(m.sum()),
            "n_wrong": int((m & wrong).sum()),
        }
    return {
        "feature": feature, "conf_quantile": conf_quantile, "threshold_from": quantile_from,
        "threshold": thr, "split": split, "score_name": score_name,
        "n": len(sel), "n_high": n_high, "n_high_wrong": n_high_wrong,
        "high_conf_error_rate": (n_high_wrong / n_high) if n_high else np.nan,
        "overall_error_rate": float(wrong.mean()) if len(wrong) else np.nan,
        "by_mu": by_mu,
    }


def error_cause_table(rows, score_name, split):
    """Which prediction errors coincide with a wrong top-1 choice."""
    sel = [r for r in rows if r["split"] == split]
    recs = []
    for r in sel:
        b = r["bundles"][score_name]
        # mean |e_along|, |e_lat| over candidates; also of chosen vs oracle
        e_al = np.abs(r["e_along"])
        e_la = np.abs(r["e_lat"])
        recs.append({
            "wrong": 0 if b["top1_correct"] else 1,
            "pattern": r["pattern"],
            "mu": r["mu"],
            "mean_abs_e_along": float(e_al.mean()),
            "mean_abs_e_lat": float(e_la.mean()),
            "max_abs_e_along": float(e_al.max()),
            "chosen_abs_e_along": float(e_al[b["chosen"]]),
            "oracle_abs_e_along": float(e_al[b["oracle_idx"]]),
            "oracle_margin": float(np.sort(r["true_dist"])[1] - np.min(r["true_dist"])) if len(r["true_dist"]) > 1 else 0.0,
            "regret": b["regret_m"],
        })
    arr = recs
    def corr(key):
        x = np.array([z[key] for z in arr])
        y = np.array([z["wrong"] for z in arr], dtype=float)
        if x.std() < 1e-15 or y.std() < 1e-15:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # Counterfactual: re-rank with along-error removed from predicted endpoints
    # pred_corrected = pred_ep - e_along * u  (i.e. replace along component with truth)
    recovered = 0
    still_wrong = 0
    was_wrong = 0
    for r in sel:
        b = r["bundles"][score_name]
        if b["top1_correct"]:
            continue
        was_wrong += 1
        obj0 = np.array([r["true_ep"][0][0] * 0, 0])  # unused; use stored e and pred
        # reconstruct u from e and endpoints
        # pred - true = e_along u + e_lat v  => corrected = pred - e_along u = true + e_lat v
        # We only correct the along component:
        # For scoring by distance, build corrected endpoints
        # Need u. Recompute from init — we don't have init xy in the row except true/pred.
        # Store wasn't including u. Recompute via: we have e_along, e_lat, pred, true.
        # err = pred-true; u is unit along. We can get u if e_along is not ~0:
        # Actually we stored e_along as a vector of scalars. corrected_ep = pred_ep - e_along[:,None]*u
        # We can recover u: if any |e_along|>eps, err = e_along u + e_lat v, two eqs.
        # Simpler: corrected_ep[...,0] wait — just use true_along + pred_lat in goal frame.
        # We didn't store u. Recover from first candidate with large error, or from pred-true
        # least squares. Let's compute u as the direction that best fits.
        err = np.asarray(r["pred_ep"]) - np.asarray(r["true_ep"])
        ea = np.asarray(r["e_along"]).reshape(-1, 1)
        el = np.asarray(r["e_lat"]).reshape(-1, 1)
        # err ≈ ea * u + el * v; u,v in R^2. Solve with first two well-conditioned rows.
        # Direct: u = (goal-obj)/||.|| but we don't have init here...
        # Use least squares: err = [ea el] @ [u;v]
        A = np.hstack([ea, el])  # (K,2)
        # 2-column design, 2-dim target (K,2)
        # For each dim, or jointly: [u_x v_x; u_y v_y] ...
        # err_x = ea * u_x + el * v_x
        try:
            uv, *_ = np.linalg.lstsq(A, err, rcond=None)
            u = uv[0]
        except np.linalg.LinAlgError:
            u = np.array([1.0, 0.0])
        nrm = np.linalg.norm(u)
        if nrm < 1e-9:
            u = np.array([1.0, 0.0])
        else:
            u = u / nrm
        corrected = np.asarray(r["pred_ep"]) - np.asarray(r["e_along"])[:, None] * u[None, :]
        # distance of corrected mean endpoints
        # need goal position: true_ep is object end, not goal. We have true_dist and true_ep
        # goal is the unique g s.t. ||true_ep_k - g|| = true_dist_k. Use first two if possible
        # or: we stored nothing. From any k: circle. Intersection of two circles.
        # Easier: I should have stored goal in the row. Let me add it in run_offline...
        # I'll store goal in rec. For now skip if not present and compute from true_ep[0] and
        # we'll add goal to rec.
        g = r.get("goal_xy")
        if g is None:
            continue
        d_corr = np.linalg.norm(corrected - np.asarray(g)[None, :], axis=-1)
        new_choice = int(np.argmin(d_corr))
        best = np.min(r["true_dist"])
        if r["true_dist"][new_choice] - best <= TIE_TOL_M:
            recovered += 1
        else:
            still_wrong += 1
    return {
        "n": len(sel),
        "n_wrong": int(sum(z["wrong"] for z in arr)),
        "corr_wrong_vs_mean_abs_e_along": corr("mean_abs_e_along"),
        "corr_wrong_vs_mean_abs_e_lat": corr("mean_abs_e_lat"),
        "corr_wrong_vs_oracle_margin": corr("oracle_margin"),
        "mean_abs_e_along_correct": _mean([z["mean_abs_e_along"] for z in arr if z["wrong"] == 0]),
        "mean_abs_e_along_wrong": _mean([z["mean_abs_e_along"] for z in arr if z["wrong"] == 1]),
        "mean_abs_e_lat_correct": _mean([z["mean_abs_e_lat"] for z in arr if z["wrong"] == 0]),
        "mean_abs_e_lat_wrong": _mean([z["mean_abs_e_lat"] for z in arr if z["wrong"] == 1]),
        "mean_oracle_margin_correct": _mean([z["oracle_margin"] for z in arr if z["wrong"] == 0]),
        "mean_oracle_margin_wrong": _mean([z["oracle_margin"] for z in arr if z["wrong"] == 1]),
        "along_correction_recovered": recovered,
        "along_correction_still_wrong": still_wrong,
        "along_correction_was_wrong": was_wrong,
        "along_correction_recovery_rate": (recovered / was_wrong) if was_wrong else np.nan,
    }


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def attach_goal_xy(rows, inits):
    m = {s["init_state_id"]: s["init"]["goal"] for s in inits}
    for r in rows:
        r["goal_xy"] = np.asarray(m[r["init_state_id"]], float)


def calibration_block(rows, score_name, split):
    """Primary: chosen-candidate success_prob vs true success (1 per episode).
    Also: P(top-1 correct) mapped from value_std via val-fitted logistic — that
    mapping is applied later; here we report raw success_prob ECE.
    """
    sel = [r for r in rows if r["split"] == split]
    p = np.array([r["bundles"][score_name]["chosen_success_prob"] for r in sel])
    y = np.array([1.0 if r["bundles"][score_name]["chosen_true_success"] else 0.0 for r in sel])
    tab = reliability_table(p, y, n_bins=N_BINS, strategy="uniform")
    tab_q = reliability_table(p, y, n_bins=min(N_BINS, max(4, len(sel)//8)), strategy="quantile")
    by_mu = {}
    for mu in [h["mu"] for h in HIDDENS]:
        sub = [r for r in sel if abs(r["mu"] - mu) < 1e-12]
        if not sub:
            continue
        pi = np.array([r["bundles"][score_name]["chosen_success_prob"] for r in sub])
        yi = np.array([1.0 if r["bundles"][score_name]["chosen_true_success"] else 0.0 for r in sub])
        by_mu[f"mu{mu}"] = reliability_table(pi, yi, n_bins=max(4, N_BINS // 2), strategy="uniform")
    # secondary dependent: all candidates
    p_all, y_all = [], []
    for r in sel:
        b = r["bundles"][score_name]
        p_all.extend(b["success_prob"].tolist())
        y_all.extend([1.0 if s else 0.0 for s in r["true_succ"]])
    tab_all = reliability_table(np.array(p_all), np.array(y_all), n_bins=N_BINS, strategy="uniform")
    return {
        "split": split, "score_name": score_name,
        "chosen_uniform": tab, "chosen_quantile": tab_q,
        "chosen_by_mu": by_mu,
        "all_candidates_dependent_uniform": tab_all,
        "note": "all_candidates_dependent_uniform counts K candidates per episode; they are not independent.",
    }


def gate_block(rows, score_name, probe_cost=0.4):
    out = {"score_name": score_name, "probe_cost": probe_cost, "features": {}}
    for feat in FEATURE_NAMES:
        val_f, val_w, test_f, test_w = [], [], [], []
        # also by pattern on test
        test_meta = []
        for r in rows:
            b = r["bundles"][score_name]
            f = extract_feature(b["bundle"], feat, b["chosen"])
            w = 0.0 if b["top1_correct"] else 1.0
            if r["split"] == "val":
                val_f.append(f); val_w.append(w)
            elif r["split"] == "test":
                test_f.append(f); test_w.append(w)
                test_meta.append(r)
        gate = LowConfidenceGate.fit(np.array(val_f), np.array(val_w), feat,
                                     probe_cost=probe_cost, fitted_on="val")
        val_eval = evaluate_gate(gate, np.array(val_f), np.array(val_w))
        test_eval = evaluate_gate(gate, np.array(test_f), np.array(test_w))
        # test by mu and by pattern (threshold still from val)
        by = {}
        for label, maskfn in {
            **{f"mu{h['mu']}": (lambda r, mu=h["mu"]: abs(r["mu"] - mu) < 1e-12) for h in HIDDENS},
            "SAME_BEST": lambda r: r["pattern"] == "SAME_BEST",
            "DIFF_BEST": lambda r: r["pattern"] == "DIFF_BEST",
        }.items():
            idx = [i for i, r in enumerate(test_meta) if maskfn(r)]
            if not idx:
                continue
            by[label] = evaluate_gate(gate, np.array(test_f)[idx], np.array(test_w)[idx])
        # logistic P(wrong|feature) fitted on val, ECE on val vs test
        if gate.logistic_ab is not None:
            a, b = gate.logistic_ab
            p_val = logistic_prob(np.array(val_f), a, b)
            p_te = logistic_prob(np.array(test_f), a, b)
            cal_val = reliability_table(p_val, np.array(val_w), n_bins=6, strategy="uniform")
            cal_te = reliability_table(p_te, np.array(test_w), n_bins=6, strategy="uniform")
        else:
            cal_val = cal_te = None
        out["features"][feat] = {
            "gate": {"feature": gate.feature, "threshold": gate.threshold,
                     "logistic_ab": gate.logistic_ab, "n_fit": gate.n_fit,
                     "fit_stats": getattr(gate, "_fit_stats", {})},
            "val": val_eval, "test": test_eval, "test_slices": by,
            "logistic_ece_val": cal_val, "logistic_ece_test": cal_te,
        }
    return out


def majority_scale_probe():
    """[toy-run] Demonstrate official threshold 0.05 vs value scale.

    On v = clip(1 - dist/0.5, 0, 1), v>0.05 <=> dist < 0.475 m, which is
    almost every on-table endpoint. Official majority_mean then collapses
    to (approximately) the mean value, or 0 only if a majority of samples
    predict an almost-off-table miss.
    """
    dists = np.array([
        [0.01, 0.02, 0.03, 0.04],   # all clearly in a 0.05 radius, v≈0.96
        [0.10, 0.12, 0.08, 0.09],   # all miss the 0.05 radius, v≈0.8
        [0.40, 0.45, 0.30, 0.20],   # far, still v = 0.2..0.6 > 0.05
        [0.49, 0.48, 0.47, 0.50],   # near table edge, some v<=0.05
    ])
    # each row is a candidate's S=4 sample distances? Wait we need (K,S)
    vals = dist_to_value(dists, VALUE_SCALE)
    official = official_aggregate(vals, "majority_mean", success_threshold=0.05)
    at_r = official_aggregate(vals, "majority_mean", success_threshold=dist_to_value(np.array([0.05]))[0])
    average = official_aggregate(vals, "average")
    return {
        "dists": dists, "values": vals,
        "majority_mean_official_0.05": official,
        "majority_mean_at_radius_0.9": at_r,
        "average": average,
        "thresh_at_radius": float(dist_to_value(np.array([0.05]))[0]),
        "note": "official 0.05 is almost always 'success' on this value scale",
    }


def lcb_single_sample_probe():
    vals = np.array([[0.8], [0.2], [0.5]])  # K=3, S=1
    official = official_aggregate(vals, "lcb", safe_single_sample=False)
    safe = official_aggregate(vals, "lcb", safe_single_sample=True)
    return {"official_S1": official, "safe_S1": safe,
            "official_is_nan": bool(np.all(np.isnan(official)))}


def write_raw(rows, path):
    slim = []
    for r in rows:
        item = {
            "init_state_id": r["init_state_id"], "split": r["split"],
            "hidden_condition_id": r["hidden_condition_id"], "mu": r["mu"],
            "seed": r["seed"], "pattern": r["pattern"],
            "true_dist": r["true_dist"].tolist(),
            "true_ep": r["true_ep"].tolist(),
            "true_succ": r["true_succ"].astype(int).tolist(),
            "pred_ep": r["pred_ep"].tolist(),
            "e_along": np.asarray(r["e_along"]).tolist(),
            "e_lat": np.asarray(r["e_lat"]).tolist(),
            "e_norm": np.asarray(r["e_norm"]).tolist(),
            "scores": {},
        }
        for name, b in r["bundles"].items():
            item["scores"][name] = {
                "scores": np.asarray(b["scores"]).tolist(),
                "chosen": b["chosen"], "oracle_idx": b["oracle_idx"],
                "top1_correct": b["top1_correct"], "regret_m": b["regret_m"],
                "pairwise_acc": b["pairwise_acc"], "n_compared": b["n_compared"],
                "success_prob": np.asarray(b["success_prob"]).tolist(),
                "mean_distance": np.asarray(b["mean_distance"]).tolist(),
                "value_std": np.asarray(b["value_std"]).tolist(),
                "endpoint_rms": np.asarray(b["endpoint_rms"]).tolist(),
                "agreement": b["agreement"],
                "chosen_success_prob": b["chosen_success_prob"],
                "chosen_true_success": b["chosen_true_success"],
            }
        slim.append(item)
    dump_json(slim, path)


def try_a6_predictor(inits, candidates, env_factory):
    """If A6 has a cached model, score it too. Optional; must not fail the study."""
    path_candidates = [
        os.path.join(ROOT, "runs", "a6", "models", "ens_fixture_s0.pkl"),
    ]
    # also any pkl in that dir
    model_dir = os.path.join(ROOT, "runs", "a6", "models")
    found = None
    if os.path.isdir(model_dir):
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith(".pkl"):
                found = os.path.join(model_dir, fn)
                break
    if not found:
        return {"status": "skipped", "reason": "no A6 model cache yet"}
    try:
        from src.dynamics.model import load_or_train
        pred = load_or_train(seed=0, retrain=False)
    except Exception as e:
        return {"status": "skipped", "reason": f"load failed: {type(e).__name__}: {e}"}
    rows = []
    t0 = time.perf_counter()
    try:
        for spec in inits:
            oracle_mu = {}
            for hid in HIDDENS:
                env = env_factory()
                ep_seed = _episode_seed(spec["init_state_id"], hid["mu"])
                ora = oracle_rollouts(env, spec["init"], hid, ep_seed, candidates)
                oracle_mu[hid["mu"]] = ora["true_dist"]
            pattern = best_action_pattern(oracle_mu)
            for hid in HIDDENS:
                env = env_factory()
                ep_seed = _episode_seed(spec["init_state_id"], hid["mu"])
                ora = oracle_rollouts(env, spec["init"], hid, ep_seed, candidates)
                P = pred.predict(ora["obs0"], candidates, {"history": [], "adapt_state": None})
                goal = ora["goal"]
                decomp = along_lateral_error(P.endpoints, ora["true_ep"],
                                             spec["init"]["obj"], spec["init"]["goal"])
                rec = {
                    "init_state_id": spec["init_state_id"], "split": spec["split"],
                    "hidden_condition_id": f"mu{hid['mu']}", "mu": hid["mu"],
                    "seed": ep_seed, "pattern": pattern,
                    "true_dist": ora["true_dist"], "true_ep": ora["true_ep"],
                    "true_succ": ora["true_succ"], "pred_ep": P.endpoints,
                    "e_along": decomp["e_along"], "e_lat": decomp["e_lat"],
                    "e_norm": decomp["e_norm"], "bundles": {},
                }
                for name in ("goal_distance", "success_prob", "lcb", "average_value"):
                    bun = score_detailed(P, goal, score_name=name)
                    chosen = choose_index(bun.scores)
                    pw = pairwise_ranking_accuracy(bun.scores, ora["true_dist"])
                    rg = top1_regret(chosen, ora["true_dist"])
                    rec["bundles"][name] = {
                        "scores": bun.scores, "chosen": chosen,
                        "success_prob": bun.success_prob,
                        "mean_distance": bun.mean_distance,
                        "mean_value": bun.mean_value, "lcb_value": bun.lcb_value,
                        "discounted_value": bun.discounted_value,
                        "value_std": bun.uncertainty_value_std,
                        "endpoint_rms": bun.uncertainty_endpoint_rms,
                        "credibility": bun.credibility,
                        "agreement": bun.sample_argmax_agreement,
                        "n_samples": bun.n_samples,
                        **pw, **rg,
                        "chosen_success_prob": float(bun.success_prob[chosen]),
                        "chosen_true_success": bool(ora["true_succ"][chosen]),
                        "bundle": bun,
                    }
                rows.append(rec)
        attach_goal_xy(rows, inits)
        wall = time.perf_counter() - t0
        summary = {sp: {nm: _summarise(rows, nm, split=sp) for nm in ("goal_distance", "lcb", "success_prob", "average_value")}
                   for sp in ("train", "val", "test")}
        write_raw(rows, os.path.join(RUNS, "raw_a6.json"))
        return {"status": "ok", "model_path": found, "wall_s": wall, "n_rows": len(rows),
                "summary": summary, "n_members": len(getattr(pred, "members", []))}
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}


def main():
    t_all = time.perf_counter()
    env_factory = lambda: FixturePushEnv(action_noise=0.0, max_time=6.0)
    candidates = make_candidates()
    inits = build_init_bank(MASTER_SEED)
    dump_json({
        "master_seed": MASTER_SEED, "H": H, "PUSH_NS": PUSH_NS, "PUSH_VX": PUSH_VX,
        "hiddens": [h["mu"] for h in HIDDENS], "assumed_mu": ASSUMED_MU,
        "n_samples": N_SAMPLES, "score_names": SCORE_NAMES,
        "value_scale": VALUE_SCALE, "lcb_beta_official": LCB_BETA_OFFICIAL,
        "majority_thresh_official": MAJORITY_THRESH_OFFICIAL, "gamma": GAMMA_DEFAULT,
        "tie_tol_m": TIE_TOL_M,
        "inits": inits,
        "splits": {s: sum(1 for i in inits if i["split"] == s) for s in ("train", "val", "test")},
        "note": "hidden mu is an evaluation label only; predictors/scorers never receive it",
        "label": "toy-run",
    }, os.path.join(RUNS, "inits.json"))

    probes = {
        "majority_scale": majority_scale_probe(),
        "lcb_S1": lcb_single_sample_probe(),
        "label": "toy-run (synthetic values, not env rollouts) except they use the same mapping",
    }
    dump_json({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else
                   {kk: (np.asarray(vv).tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()} if isinstance(v, dict) else v)
               for k, v in probes.items()}, os.path.join(RUNS, "unit_probes.json"))

    all_settings = {}
    for sname, kwargs in SETTINGS.items():
        print(f"[a7] running setting={sname} ...", flush=True)
        rows, meta = run_offline(sname, kwargs, inits, candidates, env_factory)
        attach_goal_xy(rows, inits)
        write_raw(rows, os.path.join(RUNS, f"raw_{sname}.json"))
        # summaries
        summary = {}
        for split in ("train", "val", "test", None):
            sl = split or "all"
            summary[sl] = {}
            for nm in SCORE_NAMES:
                summary[sl][nm] = {
                    "overall": _summarise(rows, nm, split=split),
                    "by_mu": {f"mu{h['mu']}": _summarise(rows, nm, split=split, mu=h["mu"]) for h in HIDDENS},
                    "by_pattern": {p: _summarise(rows, nm, split=split, pattern=p) for p in ("SAME_BEST", "DIFF_BEST")},
                }
        # error causes on TEST (and VAL for inspection; threshold-free descriptive stats)
        causes = {sp: error_cause_table(rows, "goal_distance", sp) for sp in ("val", "test")}
        # high-confidence wrong (quantile from VAL)
        hcw = {}
        for feat in ("value_std", "endpoint_rms", "inv_agreement"):
            hcw[feat] = {sp: high_conf_wrong(rows, "goal_distance", sp, feature=feat,
                                             conf_quantile=0.3, quantile_from="val")
                         for sp in ("val", "test")}
        # calibration of success_prob (no fitted params except bin edges are fixed uniform)
        calib = {sp: calibration_block(rows, "goal_distance", sp) for sp in ("val", "test")}
        # also success_prob scorer's chosen p
        calib_sp = {sp: calibration_block(rows, "success_prob", sp) for sp in ("val", "test")}
        # write reliability SVGs (val and test, overall)
        for sp, block in calib.items():
            write_reliability_svg(
                block["chosen_uniform"],
                os.path.join(FIGS, f"reliability_chosen_success_{sname}_{sp}.svg"),
                title=f"[{sname} {sp}] chosen success_prob vs success [toy-run]",
            )
        # gates: fit on VAL, eval TEST; several probe costs
        gates = {str(c): gate_block(rows, "goal_distance", probe_cost=c) for c in PROBE_COSTS}
        # pick default cost 0.4, feature with best VAL expected cost for a short highlight
        default_gates = gates[str(0.4)]["features"]
        ranked = sorted(default_gates.items(),
                        key=lambda kv: kv[1]["val"]["expected_cost"])
        # logistic reliability for best feature
        best_feat = ranked[0][0]
        if default_gates[best_feat]["logistic_ece_val"]:
            write_reliability_svg(
                default_gates[best_feat]["logistic_ece_val"],
                os.path.join(FIGS, f"reliability_pwrong_{sname}_val_{best_feat}.svg"),
                title=f"[{sname} val] logistic P(wrong|{best_feat}) [toy-run, fit on val]",
            )
            write_reliability_svg(
                default_gates[best_feat]["logistic_ece_test"],
                os.path.join(FIGS, f"reliability_pwrong_{sname}_test_{best_feat}.svg"),
                title=f"[{sname} test] logistic P(wrong|{best_feat}) [toy-run, params from val]",
            )

        pack = {
            "meta": meta, "summary": summary, "causes": causes,
            "high_conf_wrong": hcw, "calibration_goal_distance": calib,
            "calibration_success_prob_scorer": calib_sp, "gates": gates,
            "best_gate_feature_by_val_cost_probe0.4": best_feat,
        }
        dump_json(pack, os.path.join(RUNS, f"summary_{sname}.json"))
        all_settings[sname] = pack
        print(f"  wall={meta['wall_s']:.2f}s  test top1 goal_distance="
              f"{summary['test']['goal_distance']['overall']['top1_acc']:.3f}  "
              f"lcb={summary['test']['lcb']['overall']['top1_acc']:.3f}", flush=True)

    print("[a7] optional A6 predictor ...", flush=True)
    a6 = try_a6_predictor(inits, candidates, env_factory)
    dump_json(a6, os.path.join(RUNS, "a6_optional.json"))

    # init-level pattern counts
    patterns = defaultdict(lambda: defaultdict(int))
    # use first setting rows via raw file? we have all_settings but not rows.
    # Recompute quickly from inits.json labels by loading raw
    raw = json.load(open(os.path.join(RUNS, "raw_narrow.json")))
    for r in raw:
        patterns[r["split"]][r["pattern"]] += 1
        patterns["all"][r["pattern"]] += 1

    overview = {
        "label": "toy-run",
        "wall_s_total": time.perf_counter() - t_all,
        "master_seed": MASTER_SEED,
        "n_inits": len(inits),
        "n_hidden": len(HIDDENS),
        "n_episodes_total": len(inits) * len(HIDDENS),
        "pattern_counts": patterns,
        "probes": {
            "lcb_S1_official_all_nan": probes["lcb_S1"]["official_is_nan"],
            "majority_scale_note": probes["majority_scale"]["note"],
        },
        "a6_optional_status": a6.get("status"),
        "paths": {
            "raw_narrow": "runs/a7/raw_narrow.json",
            "raw_wide": "runs/a7/raw_wide.json",
            "summary_narrow": "runs/a7/summary_narrow.json",
            "summary_wide": "runs/a7/summary_wide.json",
            "inits": "runs/a7/inits.json",
        },
    }
    dump_json(overview, os.path.join(RUNS, "overview.json"))
    # also a compact numbers file for the report
    compact = {"label": "toy-run", "overview": overview, "settings": {}}
    for sname, pack in all_settings.items():
        compact["settings"][sname] = {
            "meta_wall_s": pack["meta"]["wall_s"],
            "test": {nm: pack["summary"]["test"][nm]["overall"] for nm in SCORE_NAMES},
            "test_by_mu_goal_distance": pack["summary"]["test"]["goal_distance"]["by_mu"],
            "test_by_pattern_goal_distance": pack["summary"]["test"]["goal_distance"]["by_pattern"],
            "val_test_causes": pack["causes"],
            "high_conf_wrong_value_std": pack["high_conf_wrong"]["value_std"],
            "calib_chosen_ece": {
                sp: {
                    "ece_uniform": pack["calibration_goal_distance"][sp]["chosen_uniform"]["ece"],
                    "brier": pack["calibration_goal_distance"][sp]["chosen_uniform"]["brier"],
                    "n": pack["calibration_goal_distance"][sp]["chosen_uniform"]["n"],
                    "by_mu_ece": {k: v["ece"] for k, v in pack["calibration_goal_distance"][sp]["chosen_by_mu"].items()},
                } for sp in ("val", "test")
            },
            "gate_probe0.4": {
                feat: {"val": pack["gates"]["0.4"]["features"][feat]["val"],
                       "test": pack["gates"]["0.4"]["features"][feat]["test"],
                       "threshold": pack["gates"]["0.4"]["features"][feat]["gate"]["threshold"],
                       "logistic_ece_val": (pack["gates"]["0.4"]["features"][feat]["logistic_ece_val"] or {}).get("ece"),
                       "logistic_ece_test": (pack["gates"]["0.4"]["features"][feat]["logistic_ece_test"] or {}).get("ece"),
                       "test_slices_abstain": {k: v["abstain_rate"] for k, v in pack["gates"]["0.4"]["features"][feat]["test_slices"].items()},
                       "test_slices_commit_err": {k: v["commit_error_rate"] for k, v in pack["gates"]["0.4"]["features"][feat]["test_slices"].items()},
                       } for feat in FEATURE_NAMES
            },
            "best_gate_feature_by_val_cost_probe0.4": pack["best_gate_feature_by_val_cost_probe0.4"],
        }
    dump_json(compact, os.path.join(RUNS, "compact.json"))
    dump_json(compact, os.path.join(REPS, "results_compact.json"))
    print(f"[a7] done total_wall={overview['wall_s_total']:.2f}s  a6={a6.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
