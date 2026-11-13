#!/usr/bin/env python3
"""A5 experiments: freeze candidates, diversity/oracle E1, long-vs-short horizon.

Reproducible entry point (from repo root, venv on):
  . .venv/bin/activate
  python reports/a5/run_all.py

Seeds, config, and output paths are written to runs/a5/run_meta.json.
All numbers are [toy-run] unless A4's env is picked up, still toy CPU 2D.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/wmprobe")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reports/a5"))

from candidates import (
    FixedLibraryProposer,
    RandomProposer,
    Ref4Proposer,
    TimedReleaseProposer,
    freeze_set,
    load_candidates,
)
from candidates.utils import mean_speed, segment_rms
from planning import BestOfNPlanner, GoalDistanceScorer, NominalPredictor
from diagnose import diagnose_set
from closed_loop import run_episode
from env_loader import (
    A10_OFFLINE,
    hidden_id,
    load_hidden_conditions,
    load_init_states,
    make_env,
    env_is_noisy,
)

RUNS = ROOT / "runs/a5"
LOGS = ROOT / "reports/a5/logs"
RUNS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ---- frozen config ----
E1_SEED = 510
HORIZON_SEED = 511
RANDOM_PROP_SEED = 510
E1_K = 4
E1_H = 10
E1_N_INIT = 20
RANDOM_K = 8
SHORT_H = 3
LONG_H = 10
MAX_TIME = 6.0
NOMINAL_MU = 0.3
TIE_BREAK = "smallest_index"
COAST = True  # A10 settle: zero actions until |obj_vel|<1e-3 or 20 steps
SETTLE_STEPS = 30  # A4 branch_eval.SETTLE_MAX (3 s); A10 offline listed 20
TIE_TOL_M = float(A10_OFFLINE['tie_tol_m'])


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def _summarize(rows: list[dict], keys_mean=("goal_error", "success", "regret_vs_random_error",
                                            "oracle_success", "random_expected_success",
                                            "oracle_goal_error", "random_expected_goal_error",
                                            "median_goal_error", "oracle_vs_median_gap",
                                            "goal_error_spread", "action_pair_mean",
                                            "outcome_pair_mean", "all_fail", "all_succeed",
                                            "has_near_identical", "compute_s", "n_plans",
                                            "n_env_steps", "t_final", "episode_wall_s")) -> dict:
    if not rows:
        return {}
    out = {"n": len(rows)}
    for k in keys_mean:
        vals = []
        for r in rows:
            if k in r and r[k] is not None:
                try:
                    vals.append(float(r[k]))
                except (TypeError, ValueError):
                    pass
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return out


def reset_init(env, st: dict, hidden: dict, seed: int):
    init = {k: st[k] for k in ("pusher", "obj", "goal", "goal_radius")}
    # hidden dict passed to env.reset must be only {"mu": ...} for A4
    h = {"mu": float(hidden["mu"])}
    return env.reset(init, h, int(seed))


def _env_seed(st: dict, default: int) -> int:
    return int(st.get("seed", default))


def _library_diag(env, inits, hiddens, proposer, proposer_name, freeze_seed, n_repeats):
    manifest = freeze_set(proposer, inits, seed=freeze_seed, proposer_name=proposer_name, out_dir=RUNS)
    rows = []
    for st in inits:
        packed = load_candidates(RUNS / f"{proposer_name}__{st['id']}__seed{freeze_seed}.npz")
        cands = packed["actions"]
        rms = segment_rms(cands)
        mspd = mean_speed(cands)
        for hid in hiddens:
            hid_s = hidden_id(hid)
            ep_seed = _env_seed(st, freeze_seed)
            reset_init(env, st, hid, ep_seed)
            diag = diagnose_set(env, cands, coast=COAST, n_repeats=n_repeats, settle_steps=SETTLE_STEPS)
            err = np.asarray(diag["goal_error"], dtype=np.float64)
            median_err = float(np.median(err))
            oracle_vs_median = float(median_err - diag["oracle_goal_error"])
            row = {
                "init_state_id": st["id"],
                "hidden_condition_id": hid_s,
                "seed": ep_seed,
                "proposer_name": proposer_name,
                "K": int(cands.shape[0]),
                "H": int(cands.shape[1]),
                "n_repeats": n_repeats,
                "coast": COAST,
                "settle_steps": SETTLE_STEPS,
                "candidate_rms_speed": rms.tolist(),
                "candidate_mean_speed": mspd.tolist(),
                "oracle_index": diag["oracle_index"],
                "oracle_goal_error": diag["oracle_goal_error"],
                "oracle_success": diag["oracle_success"],
                "random_expected_goal_error": diag["random_expected_goal_error"],
                "random_expected_success": diag["random_expected_success"],
                "median_goal_error": median_err,
                "oracle_vs_median_gap": oracle_vs_median,
                "regret_vs_random_error": diag["regret_vs_random_error"],
                "goal_error": err.tolist(),
                "success_rate": diag["success_rate"].tolist(),
                "all_fail": diag["all_fail"],
                "all_succeed": diag["all_succeed"],
                "goal_error_spread": diag["goal_error_spread"],
                "action_pair_mean": diag["action_pair_mean"],
                "action_pair_min": diag["action_pair_min"],
                "action_pair_max": diag["action_pair_max"],
                "outcome_pair_mean": diag["outcome_pair_mean"],
                "outcome_pair_min": diag["outcome_pair_min"],
                "outcome_pair_max": diag["outcome_pair_max"],
                "has_near_identical": diag["has_near_identical"],
                "near_identical_pairs": diag["near_identical_pairs"],
                "endpoints": diag["endpoints"].tolist(),
                "note": diag["note"],
            }
            rows.append(row)
    return rows, manifest


def _e1_summary(rows, env_label, init_source, hidden_source, proposer_name, K, H, seed, n_repeats, wall_s):
    by_hidden = {}
    hids = sorted({r["hidden_condition_id"] for r in rows})
    for hid_s in hids:
        sub = [r for r in rows if r["hidden_condition_id"] == hid_s]
        by_hidden[hid_s] = _summarize(sub)
    mixed = [r for r in rows if (not r["all_fail"]) and (not r["all_succeed"])]
    improve_success = [r for r in rows if r["oracle_success"] > r["random_expected_success"] + 1e-12]
    improve_error = [r for r in rows if r["regret_vs_random_error"] > 1e-12]
    gaps = [r["oracle_vs_median_gap"] for r in rows]
    headroom = float(np.mean(gaps)) >= 0.01 if gaps else False
    return {
        "label": "[toy-run]",
        "env": env_label,
        "init_source": init_source,
        "hidden_source": hidden_source,
        "proposer_name": proposer_name,
        "n_init": len({r["init_state_id"] for r in rows}),
        "n_hidden": len(hids),
        "n_rows": len(rows),
        "K": K,
        "H": H,
        "seed": seed,
        "n_repeats": n_repeats,
        "coast": COAST,
        "settle_steps": SETTLE_STEPS,
        "tie_break": TIE_BREAK,
        "tie_tol_m": TIE_TOL_M,
        "oracle_is_diagnostic_upper_bound": True,
        "oracle_not_deployable": True,
        "wall_s": wall_s,
        "overall": _summarize(rows),
        "by_hidden": by_hidden,
        "n_all_fail": int(sum(1 for r in rows if r["all_fail"])),
        "n_all_succeed": int(sum(1 for r in rows if r["all_succeed"])),
        "n_mixed": len(mixed),
        "n_has_near_identical": int(sum(1 for r in rows if r["has_near_identical"])),
        "n_oracle_success_beats_random": len(improve_success),
        "n_oracle_error_beats_random": len(improve_error),
        "mean_oracle_vs_median_gap": float(np.mean(gaps)) if gaps else None,
        "headroom_gate_mean_gap_ge_0p01": headroom,
    }


def run_e1(env, inits, hiddens, env_label, init_source, hidden_source):
    n_repeats = 5 if env_is_noisy(env) else 1
    t0 = time.perf_counter()
    # Primary recommended A5 library for E1/E2: timed-release, H=6 (A4 task segment).
    rel = TimedReleaseProposer(K=E1_K, H=6, speed=0.4)
    rows_rel, man = _library_diag(env, inits, hiddens, rel, "release_e1", E1_SEED, n_repeats)
    # Geometry-aware approach library (bugfixed open-loop, H=10 and duration-matched H=6)
    a5_prop = FixedLibraryProposer(K=E1_K, H=E1_H)
    rows_a5, _ = _library_diag(env, inits, hiddens, a5_prop, "fixed_e1", E1_SEED, n_repeats)
    a5_h6 = FixedLibraryProposer(K=E1_K, H=6)
    rows_h6, _ = _library_diag(env, inits, hiddens, a5_h6, "fixed_e1_H6", E1_SEED, n_repeats)
    ref4 = Ref4Proposer()
    rows_ref, _ = _library_diag(env, inits, hiddens, ref4, "ref4", E1_SEED, n_repeats)
    elapsed = time.perf_counter() - t0

    # Random-K8 diversity spotcheck on first init, all hiddens
    rand_prop = RandomProposer(K=RANDOM_K, H=E1_H)
    freeze_set(rand_prop, inits, seed=RANDOM_PROP_SEED, proposer_name="random_k8", out_dir=RUNS)
    rand_rows = []
    st0 = inits[0]
    from diagnose import diagnose_set as _ds
    c_rand = load_candidates(RUNS / f"random_k8__{st0['id']}__seed{RANDOM_PROP_SEED}.npz")["actions"]
    for hid in hiddens:
        reset_init(env, st0, hid, _env_seed(st0, E1_SEED))
        d = _ds(env, c_rand, coast=COAST, n_repeats=n_repeats, settle_steps=SETTLE_STEPS)
        rand_rows.append(
            {
                "init_state_id": st0["id"],
                "hidden_condition_id": hidden_id(hid),
                "K": RANDOM_K,
                "action_pair_mean": d["action_pair_mean"],
                "action_pair_min": d["action_pair_min"],
                "outcome_pair_mean": d["outcome_pair_mean"],
                "outcome_pair_min": d["outcome_pair_min"],
                "has_near_identical": d["has_near_identical"],
                "all_fail": d["all_fail"],
                "oracle_success": d["oracle_success"],
                "random_expected_success": d["random_expected_success"],
                "goal_error_spread": d["goal_error_spread"],
            }
        )

    summary = _e1_summary(rows_rel, env_label, init_source, hidden_source, "release_e1", E1_K, 6, E1_SEED, n_repeats, elapsed)
    summary["primary_library"] = "release_e1"
    summary["random_k8_spotcheck"] = rand_rows
    summary["fixed_e1"] = _e1_summary(rows_a5, env_label, init_source, hidden_source, "fixed_e1", E1_K, E1_H, E1_SEED, n_repeats, elapsed)
    summary["fixed_e1_H6"] = _e1_summary(rows_h6, env_label, init_source, hidden_source, "fixed_e1_H6", E1_K, 6, E1_SEED, n_repeats, elapsed)
    summary["ref4"] = _e1_summary(rows_ref, env_label, init_source, hidden_source, "ref4", 4, 5, E1_SEED, n_repeats, elapsed)
    payload = {
        "summary": summary,
        "rows": rows_rel,
        "rows_fixed_e1": rows_a5,
        "rows_fixed_e1_H6": rows_h6,
        "rows_ref4": rows_ref,
    }
    # leftover stub so the old tail of this function is replaced:
    (LOGS / 'e1_results.json').write_text(json.dumps(_jsonable(payload), indent=2))
    (RUNS / 'e1_results.json').write_text(json.dumps(_jsonable(payload), indent=2))
    return payload, man


def run_horizon(env, inits, hiddens, env_label, init_source, hidden_source):
    """Long (H=10) vs short (H=3 with replanning) Best-of-N + nominal predictor.

    Same physical budget (max_time=6s), same K=4 fixed-library proposer (rebuilt
    at each replan from the current observation), same scorer. Wall time recorded
    for propose/predict/score vs env steps.
    """
    predictor = NominalPredictor(mu=NOMINAL_MU, max_time=MAX_TIME)
    scorer = GoalDistanceScorer(success_bonus=0.05)
    rows = []
    t0 = time.perf_counter()
    for H, tag in ((LONG_H, "long_H10"), (SHORT_H, "short_H3")):
        proposer = FixedLibraryProposer(K=E1_K, H=H)
        planner = BestOfNPlanner(proposer, predictor, scorer)
        for st in inits:
            init = {k: st[k] for k in ("pusher", "obj", "goal", "goal_radius")}
            for hid in hiddens:
                hid_s = hidden_id(hid)
                # Per-episode seed: shared across the two horizons for pairing.
                ep_seed = int(st.get("seed", HORIZON_SEED))
                rec = run_episode(
                    env,
                    planner,
                    init,
                    {"mu": float(hid["mu"])},
                    seed=_env_seed(st, ep_seed),
                    init_state_id=st["id"],
                    hidden_condition_id=hid_s,
                    method=f"bon_nominal_{tag}",
                    run_id="a5_horizon",
                )
                rec = {k: v for k, v in rec.items() if k != "records"}
                rec["H"] = H
                rec["horizon_tag"] = tag
                rec["predictor"] = f"nominal_mu_{NOMINAL_MU}"
                rec["K"] = E1_K
                rows.append(rec)
    elapsed = time.perf_counter() - t0

    def pack(tag):
        sub = [r for r in rows if r["horizon_tag"] == tag]
        by_h = {}
        for hid in hiddens:
            hid_s = hidden_id(hid)
            by_h[hid_s] = _summarize([r for r in sub if r["hidden_condition_id"] == hid_s])
        return {"overall": _summarize(sub), "by_hidden": by_h, "n": len(sub)}

    long_p = pack("long_H10")
    short_p = pack("short_H3")

    # Paired comparison on (init, hidden)
    paired = []
    idx = {(r["init_state_id"], r["hidden_condition_id"], r["horizon_tag"]): r for r in rows}
    for st in inits:
        for hid in hiddens:
            hid_s = hidden_id(hid)
            a = idx[(st["id"], hid_s, "long_H10")]
            b = idx[(st["id"], hid_s, "short_H3")]
            paired.append(
                {
                    "init_state_id": st["id"],
                    "hidden_condition_id": hid_s,
                    "long_success": a["success"],
                    "short_success": b["success"],
                    "long_error": a["goal_error"],
                    "short_error": b["goal_error"],
                    "delta_error_short_minus_long": b["goal_error"] - a["goal_error"],
                    "long_compute_s": a["compute_s"],
                    "short_compute_s": b["compute_s"],
                    "delta_compute_short_minus_long": b["compute_s"] - a["compute_s"],
                    "long_n_plans": a["n_plans"],
                    "short_n_plans": b["n_plans"],
                    "short_wins_error": bool(b["goal_error"] < a["goal_error"] - 1e-12),
                    "short_wins_success": bool(b["success"] and not a["success"]),
                    "long_wins_success": bool(a["success"] and not b["success"]),
                }
            )
    d_err = np.array([p["delta_error_short_minus_long"] for p in paired])
    d_cmp = np.array([p["delta_compute_short_minus_long"] for p in paired])
    summary = {
        "label": "[toy-run]",
        "env": env_label,
        "init_source": init_source,
        "hidden_source": hidden_source,
        "predictor": f"NominalPredictor(mu={NOMINAL_MU})  # NOT ground truth; NOT A6",
        "scorer": "GoalDistanceScorer(success_bonus=0.05)",
        "K": E1_K,
        "long_H": LONG_H,
        "short_H": SHORT_H,
        "max_time": MAX_TIME,
        "horizon_base_seed": HORIZON_SEED,
        "wall_s": elapsed,
        "long": long_p,
        "short": short_p,
        "n_paired": len(paired),
        "short_better_error_count": int(sum(1 for p in paired if p["short_wins_error"])),
        "long_better_error_count": int(sum(1 for p in paired if p["delta_error_short_minus_long"] > 1e-12)),
        "short_only_success": int(sum(1 for p in paired if p["short_wins_success"])),
        "long_only_success": int(sum(1 for p in paired if p["long_wins_success"])),
        "mean_delta_error_short_minus_long": float(d_err.mean()) if len(d_err) else None,
        "mean_delta_compute_short_minus_long": float(d_cmp.mean()) if len(d_cmp) else None,
        "mean_long_compute_s": float(np.mean([p["long_compute_s"] for p in paired])),
        "mean_short_compute_s": float(np.mean([p["short_compute_s"] for p in paired])),
        "e2_note": (
            "This replan comparison is A5's assigned diagnostic only. "
            "Controller E2 budget is optional probe (<=5) + ONE task segment + settle; "
            "no corrective re-pushes. Do not treat short-H receding-horizon as an E2 method."
        ),
    }
    payload = {"summary": summary, "rows": rows, "paired": paired}
    (LOGS / "horizon_results.json").write_text(json.dumps(_jsonable(payload), indent=2))
    (RUNS / "horizon_results.json").write_text(json.dumps(_jsonable(payload), indent=2))
    return payload


def main():
    t0 = time.perf_counter()
    env, env_label = make_env(action_noise=0.0, max_time=MAX_TIME)
    inits, init_source = load_init_states(n=E1_N_INIT)
    hiddens, hidden_source = load_hidden_conditions()
    meta = {
        "agent": "A5",
        "label": "[toy-run]",
        "env": env_label,
        "env_type": type(env).__name__,
        "init_source": init_source,
        "hidden_source": hidden_source,
        "n_init": len(inits),
        "init_ids": [s["id"] for s in inits],
        "hiddens_ids": [hidden_id(h) for h in hiddens],
        "e1_seed": E1_SEED,
        "horizon_seed": HORIZON_SEED,
        "random_prop_seed": RANDOM_PROP_SEED,
        "fallback_init_seed": 512,
        "E1_K": E1_K,
        "E1_H": E1_H,
        "SHORT_H": SHORT_H,
        "LONG_H": LONG_H,
        "MAX_TIME": MAX_TIME,
        "NOMINAL_MU": NOMINAL_MU,
        "COAST": COAST,
        "TIE_BREAK": TIE_BREAK,
        "note_oracle": (
            "Oracle on a fixed candidate set is a diagnostic upper bound, "
            "not a deployable algorithm (uses free snapshot branching + ground-truth env)."
        ),
    }
    (RUNS / "run_meta.json").write_text(json.dumps(_jsonable(meta), indent=2))
    print(json.dumps({k: meta[k] for k in ("env", "init_source", "hidden_source", "n_init", "hiddens_ids")}, indent=2))
    e1, _man = run_e1(env, inits, hiddens, env_label, init_source, hidden_source)
    print("E1 overall:", json.dumps(_jsonable(e1["summary"]["overall"]), indent=2))
    print("E1 by_hidden keys:", list(e1["summary"]["by_hidden"].keys()))
    print(
        "E1 flags:",
        {k: e1["summary"][k] for k in (
            "n_all_fail", "n_all_succeed", "n_mixed",
            "n_has_near_identical", "n_oracle_success_beats_random",
            "n_oracle_error_beats_random", "wall_s",
        )},
    )
    hz = run_horizon(env, inits, hiddens, env_label, init_source, hidden_source)
    print("HORIZON summary:", json.dumps(_jsonable(hz["summary"]), indent=2, default=str)[:4000])
    meta["total_wall_s"] = time.perf_counter() - t0
    (RUNS / "run_meta.json").write_text(json.dumps(_jsonable(meta), indent=2))
    print(f"done in {meta['total_wall_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
