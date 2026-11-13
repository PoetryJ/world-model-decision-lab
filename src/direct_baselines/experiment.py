"""Train and evaluate A9 direct baselines.

Repro (from repo root, venv on):

    PYTHONPATH=/workspace/wmprobe python -m src.direct_baselines.experiment \\
        --out /workspace/wmprobe/runs/a9

All numbers written under --out (and copied to reports/a9/logs/) are [toy-run]
on PushEnv (CPU 2D mechanism, idealised pose, not vision, not a real robot).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.scenarios import train_set, e1_set, hidden_conditions
from src.direct_baselines.collect import collect_split, rows_to_xy, hist_from_json, _arrays_obs, set_candidate_source, fill_group_fields
from src.direct_baselines.models import train_ridge, RidgeScorerModel
from src.direct_baselines.selectors import (
    ObsOnlySelector, HistoryDirectSelector,
    HeuristicObsSelector, HeuristicHistorySelector, RandomSelector,
    PhysicsObsSelector, PhysicsHistorySelector,
)
from src.direct_baselines.candidates import load_shared_candidates

DIST_TIE = 1e-4  # label-tie threshold for pairwise / oracle ties
SEED_RANDOM = 7


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:

    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    # keep candidate order
    for k in g:
        g[k] = sorted(g[k], key=lambda r: r["cand_index"])
    return dict(g)


def oracle_index(rows: list[dict]) -> int:
    """Best candidate: min settled dist; ties -> success preferred, then lowest index."""
    best_i, best = 0, rows[0]
    for r in rows[1:]:
        if r["y_dist"] < best["y_dist"] - DIST_TIE:
            best_i, best = r["cand_index"], r
        elif abs(r["y_dist"] - best["y_dist"]) <= DIST_TIE:
            if r["y_success"] > best["y_success"]:
                best_i, best = r["cand_index"], r
            # else keep lower index
    return best_i


def pairwise_acc(scores: np.ndarray, dists: np.ndarray) -> float:
    """Among pairs with |d_i-d_j| > DIST_TIE, fraction correctly ordered (score ties = 0.5)."""
    k = len(scores)
    n, hits = 0, 0.0
    for i in range(k):
        for j in range(i + 1, k):
            if abs(dists[i] - dists[j]) <= DIST_TIE:
                continue
            n += 1
            if abs(scores[i] - scores[j]) <= 1e-15:
                hits += 0.5
            else:
                # higher score should have lower dist
                correct = (scores[i] > scores[j] and dists[i] < dists[j]) or (
                    scores[j] > scores[i] and dists[j] < dists[i]
                )
                hits += float(correct)
    return float(hits / n) if n else float("nan")


def evaluate_rows(rows: list[dict], selectors: dict, infer_repeats: int = 200) -> tuple[list[dict], dict]:
    grouped = group_rows(fill_group_fields(rows))
    recs = []
    times = {name: [] for name in selectors}
    for group, rs in grouped.items():
        obs = _arrays_obs(rs[0]["obs"])
        hist = hist_from_json(rs[0]["history"])
        cands = np.asarray(rs[0]["candidates"], dtype=np.float64)
        dists = np.array([r["y_dist"] for r in rs], dtype=np.float64)
        succs = np.array([r["y_success"] for r in rs], dtype=np.int32)
        orb = oracle_index(rs)
        rec = {
            "group": group,
            "init_id": rs[0]["init_id"],
            "hid_id": rs[0]["hid_id"],
            "context": rs[0]["context"],
            "set_name": rs[0]["set_name"],
            "n_cand": len(rs),
            "oracle_index": orb,
            "oracle_dist": float(dists[orb]),
            "oracle_success": int(succs[orb]),
            "cand_dists": dists.tolist(),
            "cand_success": succs.tolist(),
            "methods": {},
        }
        for name, sel in selectors.items():
            t0 = time.perf_counter()
            scores = np.asarray(sel.score_candidates(obs, hist, cands), dtype=np.float64)
            t1 = time.perf_counter()
            times[name].append(t1 - t0)
            # extra repeats only on first group for a stabler inference estimate later
            idx = int(np.argmax(scores))
            rec["methods"][name] = {
                "scores": scores.tolist(),
                "chosen_index": idx,
                "chosen_dist": float(dists[idx]),
                "chosen_success": int(succs[idx]),
                "top1_match": int(idx == orb),
                "regret_dist": float(dists[idx] - dists[orb]),
                "pairwise_acc": pairwise_acc(scores, dists),
                "score_time_s": t1 - t0,
            }
        recs.append(rec)

    # more precise inference timing on a handful of groups
    infer = {}
    sample = list(grouped.values())[: min(20, len(grouped))]
    for name, sel in selectors.items():
        ts = []
        for rs in sample:
            obs = _arrays_obs(rs[0]["obs"])
            hist = hist_from_json(rs[0]["history"])
            cands = np.asarray(rs[0]["candidates"], dtype=np.float64)
            sel.score_candidates(obs, hist, cands)  # warmup
            t0 = time.perf_counter()
            for _ in range(infer_repeats):
                sel.score_candidates(obs, hist, cands)
            ts.append((time.perf_counter() - t0) / infer_repeats)
        infer[name] = {
            "mean_s": float(np.mean(ts)) if ts else None,
            "median_s": float(np.median(ts)) if ts else None,
            "n_groups_timed": len(ts),
            "repeats": infer_repeats,
            "one_shot_mean_s": float(np.mean(times[name])) if times[name] else None,
        }

    summary = summarize(recs, infer, selectors)
    return recs, summary


def summarize(recs: list[dict], infer: dict, selectors: dict) -> dict:
    methods = list(selectors.keys()) + ["oracle"]
    def agg(subset: list[dict]) -> dict:
        out = {}
        n = len(subset)
        out["n_states"] = n  # one state = one (init, hidden, context); not K candidates
        if n == 0:
            return out
        for name in methods:
            if name == "oracle":
                succs = [r["oracle_success"] for r in subset]
                dists = [r["oracle_dist"] for r in subset]
                out[name] = {
                    "success_rate": float(np.mean(succs)),
                    "mean_dist": float(np.mean(dists)),
                    "median_dist": float(np.median(dists)),
                    "top1_match": 1.0,
                    "mean_regret_dist": 0.0,
                    "mean_pairwise_acc": None,
                }
                continue
            m = [r["methods"][name] for r in subset]
            pw = [x["pairwise_acc"] for x in m if x["pairwise_acc"] == x["pairwise_acc"]]
            out[name] = {
                "success_rate": float(np.mean([x["chosen_success"] for x in m])),
                "mean_dist": float(np.mean([x["chosen_dist"] for x in m])),
                "median_dist": float(np.median([x["chosen_dist"] for x in m])),
                "top1_match": float(np.mean([x["top1_match"] for x in m])),
                "mean_regret_dist": float(np.mean([x["regret_dist"] for x in m])),
                "mean_pairwise_acc": float(np.mean(pw)) if pw else None,
                "n_pairwise_defined": len(pw),
            }
        return out

    by = {
        "all": agg(recs),
        "by_context": {},
        "by_hidden": {},
        "by_set": {},
        "by_context_hidden": {},
        "by_set_hidden_context": {},
    }
    for ctx in sorted({r["context"] for r in recs}):
        by["by_context"][ctx] = agg([r for r in recs if r["context"] == ctx])
    for hid in sorted({r["hid_id"] for r in recs}):
        by["by_hidden"][hid] = agg([r for r in recs if r["hid_id"] == hid])
    for st in sorted({r["set_name"] for r in recs}):
        by["by_set"][st] = agg([r for r in recs if r["set_name"] == st])
    for r0 in recs:
        key = f"{r0['context']}|{r0['hid_id']}"
        if key not in by["by_context_hidden"]:
            by["by_context_hidden"][key] = None
    for key in list(by["by_context_hidden"]):
        ctx, hid = key.split("|", 1)
        by["by_context_hidden"][key] = agg([r for r in recs if r["context"] == ctx and r["hid_id"] == hid])
    return {"aggregates": by, "inference_time": infer, "n_records": len(recs), "tie_rule": "lowest index; label ties |d|<1e-4 skipped in pairwise, oracle ties prefer success then lowest index"}


def fit_models(train_rows, val_rows, out_dir: Path) -> dict:
    models = {}
    timings = {}
    for name, use_hist in (("obs_only", False), ("history_direct", True)):
        t0 = time.perf_counter()
        Xtr, ytr, _ = rows_to_xy(train_rows, use_history=use_hist)
        Xva, yva, gva = rows_to_xy(val_rows, use_history=use_hist)
        model = train_ridge(Xtr, ytr, use_history=use_hist, X_val=Xva, y_val=yva, groups_val=gva,
                            train_seconds=0.0)
        model.train_seconds = time.perf_counter() - t0
        path = out_dir / "models" / f"{name}.npz"
        model.save(path)
        models[name] = model
        timings[name] = {
            "train_seconds": model.train_seconds,
            "n_train_rows": model.n_train,
            "n_val_rows": int(Xva.shape[0]),
            "n_params": model.n_params,
            "n_features": model.n_features,
            "lam": model.lam,
            "path": str(path),
            "summary": model.summary(),
        }
        print(f"[fit] {name}: params={model.n_params} rows={model.n_train} lam={model.lam} "
              f"t={model.train_seconds:.3f}s -> {path}")
    return models, timings


def leakage_selfcheck() -> dict:
    """Static review: policy-side files must not mention env.hidden access."""
    pol = [
        ROOT / "src" / "direct_baselines" / "features.py",
        ROOT / "src" / "direct_baselines" / "models.py",
        ROOT / "src" / "direct_baselines" / "selectors.py",
        ROOT / "src" / "direct_baselines" / "candidates.py",
        ROOT / "src" / "direct_baselines" / "physics_heuristic.py",
    ]
    hits = []
    for p in pol:
        text = p.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if (".hidden[" in line or 'hidden["mu"]' in line) and not line.strip().startswith(("#", "\"")) and "never" not in line.lower():
                hits.append(f"{p.name}:{i}:{line.strip()}")
    return {"policy_files_checked": [str(p) for p in pol], "hits": hits, "ok": len(hits) == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "a9")
    ap.add_argument("--n-train", type=int, default=80, help="A4 train_set size (half pos, half neg)")
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--train-seed", type=int, default=9000)
    ap.add_argument("--val-seed", type=int, default=9100)
    ap.add_argument("--holdout-mu-mid", action="store_true", default=True,
                    help="Train on mu_lo & mu_hi only; evaluate also on held-out mu_mid")
    ap.add_argument("--no-holdout-mu-mid", action="store_false", dest="holdout_mu_mid")
    ap.add_argument("--candidate-source", default="a5",
                    choices=("a5", "auto", "speedgrid", "strikegrid"),
                    help="a5 = A5 FixedLibraryProposer; speedgrid = A9 amplitude set; "
                         "strikegrid = A4 strike_grid (K=54, H=6, official E2-style task segment)")
    ap.add_argument("--rows-dir", type=Path, default=None,
                    help="Reuse previously collected jsonl (train/val/test_e1_rows.jsonl) and only refit+eval")
    args = ap.parse_args()
    set_candidate_source(args.candidate_source)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    print(f"== A9 experiment starting source={args.candidate_source} ==")
    leak = leakage_selfcheck()
    print("leakage_selfcheck", leak)

    train_inits = train_set(n=args.n_train, seed=args.train_seed, negative_control=None)
    val_inits = train_set(n=args.n_val, seed=args.val_seed, negative_control=None)
    # official suggested E1 sets (A10 may freeze a different one later)
    e1_pos = e1_set(False)
    e1_neg = e1_set(True)
    test_inits = e1_pos + e1_neg

    all_h = hidden_conditions(3)          # mu_lo, mu_mid, mu_hi
    train_h = hidden_conditions(2) if args.holdout_mu_mid else all_h  # lo & hi

    contexts = ("no_probe", "tap_fast", "push_slow", "retreat")
    if args.rows_dir is not None:
        rd = Path(args.rows_dir) / "logs"
        print(f"reuse rows from {rd}")
        train_rows = load_jsonl(rd / "train_rows.jsonl")
        val_rows = load_jsonl(rd / "val_rows.jsonl")
        test_rows = load_jsonl(rd / "test_e1_rows.jsonl")
        train_meta = {"n_rows": len(train_rows), "reused_from": str(rd), "n_inits": len(train_inits)}
        val_meta = {"n_rows": len(val_rows), "reused_from": str(rd), "n_inits": len(val_inits)}
        test_meta = {"n_rows": len(test_rows), "reused_from": str(rd), "n_inits": len(test_inits),
                     "candidate_source": f"reused:{args.candidate_source}"}
        print(f"  train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}")
    else:
        print(f"collect train: {len(train_inits)} inits x {len(train_h)} hidden x {len(contexts)} ctx")
        train_rows, train_meta = collect_split(train_inits, train_h, contexts)
        print(f"  {train_meta['n_rows']} rows in {train_meta['collect_seconds']:.2f}s "
              f"source={train_meta['candidate_source']}")
        print(f"collect val: {len(val_inits)} inits")
        val_rows, val_meta = collect_split(val_inits, train_h, contexts)
        print(f"  {val_meta['n_rows']} rows in {val_meta['collect_seconds']:.2f}s")
        print(f"collect test E1 pos+neg: {len(test_inits)} inits x {len(all_h)} hidden")
        test_rows, test_meta = collect_split(test_inits, all_h, contexts)
        print(f"  {test_meta['n_rows']} rows in {test_meta['collect_seconds']:.2f}s")
        _dump_jsonl(out / "logs" / "train_rows.jsonl", train_rows)
        _dump_jsonl(out / "logs" / "val_rows.jsonl", val_rows)
        _dump_jsonl(out / "logs" / "test_e1_rows.jsonl", test_rows)

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

    print("evaluate train (in-sample, sanity)")
    _, train_sum = evaluate_rows(train_rows, selectors)
    print("evaluate val")
    val_recs, val_sum = evaluate_rows(val_rows, selectors)
    print("evaluate test E1")
    test_recs, test_sum = evaluate_rows(test_rows, selectors)

    _write_json(out / "logs" / "test_e1_records.json", test_recs)
    _dump_jsonl(out / "logs" / "test_e1_records.jsonl", test_recs)

    report = {
        "label": "[toy-run]",
        "env": "src.env.push_env.PushEnv (A4, CPU 2D mechanism, idealised pose, NOT vision)",
        "seeds": {
            "train_init": args.train_seed,
            "val_init": args.val_seed,
            "e1_pos": 1100,
            "e1_neg": 2100,
            "env_seed_base": 10_000,
            "random_selector": SEED_RANDOM,
        },
        "split": {
            "train_inits": len(train_inits),
            "val_inits": len(val_inits),
            "test_inits": len(test_inits),
            "train_hidden": [h[0] for h in train_h],
            "test_hidden": [h[0] for h in all_h],
            "holdout_mu_mid": bool(args.holdout_mu_mid),
            "note": "Train/val inits from A4 train_set seeds 9000/9100 (disjoint from E0/E1/E2 eval seeds). "
                    "Test = A4 e1_set pos+neg (20+20). If holdout_mu_mid, models never see mu_mid=0.25 in training. "
                    "A10 may freeze a different official test set; this is a provisional A9 split. Protocol = E2: optional 5-step probe + ONE task segment + settle (labels use A4 branch_eval SETTLE_MAX). Main hidden pair = mu_lo (0.1) vs mu_hi (0.5); mu_mid is secondary/held-out.",
        },
        "candidate_source": test_meta.get("candidate_source"),
        "contexts": list(contexts),
        "collect": {"train": train_meta, "val": val_meta, "test": test_meta},
        "fit": fit_info,
        "leakage_selfcheck": leak,
        "train_sanity": train_sum["aggregates"]["all"],
        "val": val_sum,
        "test_e1": test_sum,
    }
    _write_json(out / "logs" / "summary.json", report)

    # copy key artifacts to reports/a9/logs (tag by out name so parallel runs do not clobber)
    rep = ROOT / "reports" / "a9" / "logs"
    rep.mkdir(parents=True, exist_ok=True)
    tag = "primary" if out.resolve() == (ROOT / "runs" / "a9").resolve() else out.name
    (rep / f"summary_{tag}.json").write_text((out / "logs" / "summary.json").read_text())
    (rep / f"test_e1_records_{tag}.jsonl").write_text((out / "logs" / "test_e1_records.jsonl").read_text())
    # keep a pointer at summary.json for the latest primary-ish run
    if tag == "primary" or tag == "v2":
        (rep / "summary.json").write_text((out / "logs" / "summary.json").read_text())

    # concise stdout
    print("\n======= TEST E1 summary [toy-run] =======")
    for slice_name, block in (("all", test_sum["aggregates"]["all"]),
                              *[(f"ctx:{k}", v) for k, v in test_sum["aggregates"]["by_context"].items()]):
        print(f"-- {slice_name} n={block.get('n_states')} --")
        for m in ["obs_only", "history_direct", "physics_obs", "physics_history", "heuristic_obs", "heuristic_history", "random", "oracle"]:
            if m not in block:
                continue
            s = block[m]
            pw = s.get("mean_pairwise_acc")
            pw_s = f"{pw:.3f}" if pw is not None else "na"
            print(f"  {m:18s} succ={s['success_rate']:.3f} dist={s['mean_dist']:.4f} "
                  f"top1={s['top1_match']:.3f} regret={s['mean_regret_dist']:.4f} pair={pw_s}")
    print("-- MAIN PAIR mu_lo vs mu_hi (controller) --")
    for hid in ("mu_lo", "mu_hi"):
        block = test_sum["aggregates"]["by_hidden"].get(hid, {})
        print(f"  [{hid}] n={block.get('n_states')}")
        for m in ["obs_only", "history_direct", "physics_obs", "physics_history", "heuristic_obs", "random", "oracle"]:
            if m not in block:
                continue
            s = block[m]
            print(f"    {m:18s} succ={s['success_rate']:.3f} dist={s['mean_dist']:.4f} "
                  f"top1={s['top1_match']:.3f} regret={s['mean_regret_dist']:.4f}")
    print("inference_s", {k: v["mean_s"] for k, v in test_sum["inference_time"].items()})
    print("wrote", out / "logs" / "summary.json")


if __name__ == "__main__":
    main()
