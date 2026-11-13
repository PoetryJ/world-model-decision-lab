"""Train/tune A9 on A10 frozen train/val only. NEVER loads test/test_neg/test_sealed*.

Primary candidate set: A4 strike_grid (K=54, H=6).
Episode structure: PROTOCOL A-3 (optional one probe <=5 steps, one task segment H<=12, settle).
Controller runs the sealed test. This script freezes weights + config and reports VAL numbers only.

    PYTHONPATH=/workspace/wmprobe python -m src.direct_baselines.frozen_eval \
        --out /workspace/wmprobe/runs/a9/frozen_strike54

Load for the sealed runner: src.direct_baselines.api.load_frozen_a9()
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit import load_frozen, frozen_sha
import src.direct_baselines.features as feat
from src.direct_baselines.collect import collect_split, set_candidate_source
from src.direct_baselines import experiment as exp
from src.direct_baselines.experiment import fit_models, evaluate_rows, _write_json, _dump_jsonl
from src.direct_baselines.selectors import (
    ObsOnlySelector, HistoryDirectSelector,
    HeuristicObsSelector, HeuristicHistorySelector, RandomSelector,
    PhysicsObsSelector, PhysicsHistorySelector,
)

SETTLE_STEPS = 20
TIE_TOL = 0.001
WHO = "A9"
SEED_RANDOM = 7
CONTEXTS = ("no_probe", "tap_fast", "push_slow", "retreat")
METHODS_PRINT = [
    "obs_only", "history_direct", "physics_obs", "physics_history",
    "heuristic_obs", "heuristic_history", "random", "oracle",
]
FORBIDDEN_SPLITS = ("test", "test_neg", "test_sealed", "test_sealed_pos", "test_sealed_neg")


def _items_to_inits(items):
    out = []
    for it in items:
        init = dict(it["init"])
        init["meta"] = dict(it.get("meta") or {})
        out.append((it["init_state_id"], init, int(it["seed"])))
    return out


def _conds(d: dict):
    return [(k, v) for k, v in d.items()]


def _print_block(title, block):
    print(f"-- {title} n={block.get('n_states')} --")
    for m in METHODS_PRINT:
        if m not in block:
            continue
        s = block[m]
        pw = s.get("mean_pairwise_acc")
        pw_s = f"{pw:.3f}" if pw is not None else "na"
        print(f"  {m:18s} succ={s['success_rate']:.3f} dist={s['mean_dist']:.4f} "
              f"top1={s['top1_match']:.3f} regret={s['mean_regret_dist']:.4f} pair={pw_s}")


def _load(split: str):
    if split in FORBIDDEN_SPLITS or split.startswith("test"):
        raise RuntimeError(f"A-8: A9 must not load split {split!r}")
    return load_frozen(split, who=WHO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "a9" / "frozen_strike54")
    ap.add_argument("--candidate-source", default="strikegrid",
                    choices=("strikegrid", "a5", "strike4", "speedgrid"))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    exp.DIST_TIE = TIE_TOL
    # K=54: do not bake physics rollouts into ridge features (heuristics still run at score time).
    feat.INCLUDE_PHYS_FEATURES = False
    set_candidate_source(args.candidate_source)

    sha = frozen_sha()
    print(f"== A9 TRAIN/VAL ONLY source={args.candidate_source} sha256={sha} ==")
    print("A-8: will not load test/test_neg/test_sealed*")
    print("leakage", exp.leakage_selfcheck())

    conditions = _load("conditions")
    train_items = _load("train") + _load("train_neg")
    val_items = _load("val") + _load("val_neg")
    train_h = _conds(conditions["train_val"])  # 0.1,0.15,0.2,0.4,0.5,0.6 — no 0.3
    train_inits = _items_to_inits(train_items)
    val_inits = _items_to_inits(val_items)
    print(f"train inits={len(train_inits)} hidden={[h[0] for h in train_h]}")
    print(f"val   inits={len(val_inits)}")

    train_rows, train_meta = collect_split(
        train_inits, train_h, CONTEXTS, settle_max=SETTLE_STEPS)
    print(f"  train {train_meta['n_rows']} rows in {train_meta['collect_seconds']:.2f}s "
          f"src={train_meta['candidate_source']}")
    val_rows, val_meta = collect_split(
        val_inits, train_h, CONTEXTS, settle_max=SETTLE_STEPS)
    print(f"  val   {val_meta['n_rows']} rows in {val_meta['collect_seconds']:.2f}s")
    _dump_jsonl(out / "logs" / "train_rows.jsonl", train_rows)
    _dump_jsonl(out / "logs" / "val_rows.jsonl", val_rows)

    models, fit_info = fit_models(train_rows, val_rows, out)

    selectors = {
        "obs_only": ObsOnlySelector(models["obs_only"]),
        "history_direct": HistoryDirectSelector(models["history_direct"]),
        "physics_obs": PhysicsObsSelector(),
        "physics_history": PhysicsHistorySelector(),
        "heuristic_obs": HeuristicObsSelector(),
        "heuristic_history": HeuristicHistorySelector(),
        "random": RandomSelector(SEED_RANDOM),
    }

    print("evaluate VAL only (tuning set; not frozen test)")
    val_recs, val_sum = evaluate_rows(val_rows, selectors, infer_repeats=8)
    _dump_jsonl(out / "logs" / "val_records.jsonl", val_recs)

    cfg = {
        "status": "FROZEN_FOR_SEALED_TEST",
        "candidate_source": train_meta.get("candidate_source"),
        "proposer": "A4 src.env.scenarios.strike_grid (K=54, H=6) via DirectProposer(source='strikegrid')",
        "episode": "PROTOCOL A-3: optional ONE probe <=5 steps, exactly ONE task segment H<=12, settle zeros until |obj_vel|<1e-3, cap 60 steps, no corrective pushes",
        "settle_steps": SETTLE_STEPS,
        "tie_tol_m": TIE_TOL,
        "tie_rule": "argmax lowest index; label ties |d|<=0.001 m excluded from pairwise (score ties=0.5)",
        "train_hidden": [h[0] for h in train_h],
        "note_conditions": "train_val mus 0.1/0.15/0.2/0.4/0.5/0.6; mu_mid=0.3 held out. Test mus (for controller) 0.1/0.3/0.6; main pair lo vs hi=0.6.",
        "n_train_inits": len(train_inits),
        "n_val_inits": len(val_inits),
        "n_train_rows": train_meta["n_rows"],
        "n_val_rows": val_meta["n_rows"],
        "fit": {k: {kk: vv for kk, vv in v.items() if kk != "summary"} for k, v in fit_info.items()},
        "frozen_sha256": sha,
        "include_phys_features_in_ridge": bool(feat.INCLUDE_PHYS_FEATURES),
        "models": {
            "obs_only": str(out / "models" / "obs_only.npz"),
            "history_direct": str(out / "models" / "history_direct.npz"),
        },
        "api": "from src.direct_baselines.api import load_frozen_a9, decide",
        "protocol": "A10 FROZEN v2 + A-3 + A-7 + A-8 (A9 does not load test)",
    }
    cfg_bytes = json.dumps(cfg, sort_keys=True, default=str).encode()
    cfg["config_sha256"] = hashlib.sha256(cfg_bytes).hexdigest()
    _write_json(out / "logs" / "config_frozen.json", cfg)
    _write_json(ROOT / "reports" / "a9" / "config_frozen.json", cfg)

    report = {
        "label": "[toy-run] frozen-protocol TRAIN/VAL only (A-8: no test load)",
        "protocol": cfg["protocol"],
        "frozen_sha256": sha,
        "config_sha256": cfg["config_sha256"],
        "env": "src.env.push_env.PushEnv (A4, CPU 2D mechanism, idealised pose, NOT vision)",
        "e2_episode": cfg["episode"],
        "main_pair_for_controller_test": {"mu_lo": 0.1, "mu_hi": 0.6},
        "mu_mid_secondary_held_out": 0.3,
        "candidate_source": train_meta.get("candidate_source"),
        "collect": {"train": train_meta, "val": val_meta},
        "fit": fit_info,
        "val": val_sum,
        "leakage_selfcheck": exp.leakage_selfcheck(),
        "prior_a4_grid_runs": "runs/a9/logs and runs/a9/speedgrid are A4 e1_set at mu 0.1/0.25/0.5 — NOT frozen protocol (A-7).",
    }
    _write_json(out / "logs" / "summary.json", report)
    rep = ROOT / "reports" / "a9" / "logs"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "summary_frozen_strike54_val.json").write_text((out / "logs" / "summary.json").read_text())

    print("\n======= VAL only [toy-run] frozen-protocol (not test) =======")
    _print_block("val all", val_sum["aggregates"]["all"])
    for k, v in val_sum["aggregates"]["by_context"].items():
        _print_block(f"val ctx:{k}", v)
    for k, v in val_sum["aggregates"]["by_set"].items():
        _print_block(f"val set:{k}", v)
    print("-- train_val mus on VAL (0.3 not present) --")
    for hid in sorted(val_sum["aggregates"]["by_hidden"]):
        _print_block(hid, val_sum["aggregates"]["by_hidden"][hid])
    print("inference_s", {k: v["mean_s"] for k, v in val_sum["inference_time"].items()})
    print("FROZE config", cfg["config_sha256"])
    print("wrote", out / "logs" / "config_frozen.json")


if __name__ == "__main__":
    main()
