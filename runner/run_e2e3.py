"""Controller E2/E3 closed-loop runner (PROTOCOL A-3 + A-9). [toy-run]
One episode = optional ONE probe (<=5 steps, A4 probe_set) -> ONE task decision from A4 strike_grid (K=54, H=6)
-> settle with zero actions until |obj_vel|<1e-3 -> 60-step cap. Uses src.probing.loop.run_episode unmodified
(max_task_decisions=1, step_budget=60, probe_budget=5). All methods paired on identical (init, hidden, env seed).

  python -m runner.run_e2e3 --split val   --out runs/controller/val [--methods m1,m2] [--limit N] [--workers W]
  python -m runner.run_e2e3 --split sealed --out runs/controller/sealed                       # deterministic primary
  python -m runner.run_e2e3 --split sealed --out runs/controller/sealed_stoch --action_noise 0.02 --reps 0,1,2

Writes <out>/decisions.jsonl, episodes.csv, episodes.jsonl, run_config.json, stage_times.json.
Refuses to run unless configs/frozen_manifest.json matches current hashes of all frozen artifacts (runner/manifest.py).
Hidden params go ONLY to env.reset (inside run_episode); policy objects never receive them.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, datetime, resource
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

STEP_BUDGET, PROBE_BUDGET, MAX_TASK = 60, 5, 1
STAGES = ("propose", "predict", "update", "score", "probe")

def load_items(split, who="controller"):
    """-> list of (init_state_id, init, cid, hidden, seed, subset)"""
    from evaluation import audit
    if split == "val":
        allsets = audit.load_frozen("__audit__", who=f"{who} run_e2e3 val")
        conds = allsets["conditions"]["test"]
        parts = [("val", allsets["val"]), ("val_neg", allsets["val_neg"])]
    elif split == "sealed":
        s = audit.load_sealed(who=who, split="__all__")
        conds = s["conditions"]["test"]
        parts = [("test_sealed", s["test_sealed"]), ("test_sealed_neg", s["test_sealed_neg"])]
    else:
        raise SystemExit("split must be val or sealed (frozen test/test_neg are dev-exposed per A-8)")
    return [(r["init_state_id"], r["init"], cid, conds[cid], int(r["seed"]), sub)
            for sub, rows in parts for r in rows for cid in sorted(conds)]

def _worker(job):
    method, items, reps, noise, tag, tmp = job
    from src.env.push_env import PushEnv
    from src.probing.loop import run_episode
    from runner.methods import build
    t0 = time.time(); n = 0
    with open(tmp + ".dec", "w") as fd, open(tmp + ".eps", "w") as fe:
        for (sid, init, cid, hidden, seed0, sub) in items:
            for r in reps:
                seed = seed0 if r is None else seed0 * 10 + r
                prop, pred, adp, sc, pr = build(method, seed)
                env = PushEnv(action_noise=noise)
                eid = f"{tag}/{method}/{sid}/{cid}" + ("" if r is None else f"/r{r}")
                s, recs = run_episode(env, init, hidden, seed, prop, pred, adp, sc, pr, step_budget=STEP_BUDGET,
                                      probe_budget=PROBE_BUDGET, method=method, run_id=tag, episode_id=eid,
                                      init_state_id=sid, hidden_condition_id=cid, max_task_decisions=MAX_TASK)
                s.update({"subset": sub, "rep": r, "action_noise": noise, "n_task_decisions": sum(x["kind"] == "task" and x["candidates"] is not None for x in recs),
                          "terminated_during_probe": bool(recs) and recs[-1]["kind"] == "probe" and s["termination"] in ("success", "out_of_bounds")})
                cc = [x["candidate_counts"] for x in recs if x.get("candidate_counts")]
                s["K_returned"] = cc[0]["returned"] if cc else None
                fe.write(json.dumps(s) + "\n")
                for x in recs: fd.write(json.dumps(x) + "\n")
                n += 1
    return method, n, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "sealed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--limit", type=int, default=0, help="first N states of EACH subset (smoke only; refused for sealed)")
    ap.add_argument("--action_noise", type=float, default=0.0)
    ap.add_argument("--reps", default="", help="stochastic variant reps, e.g. 0,1,2 -> env seed = seed*10+r")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init-manifest", action="store_true", help="write configs/frozen_manifest.json if absent (module hashes must match owners)")
    ap.add_argument("--refresh-runner-hashes", action="store_true")
    a = ap.parse_args()
    from runner import manifest as MF
    if a.init_manifest or a.refresh_runner_hashes:
        MF.init_manifest(force_runner_refresh=a.refresh_runner_hashes)
    man = MF.verify()                                   # REFUSES on any hash change
    from runner.methods import ALL_METHODS, DIRECT_METHODS, a9_ready, ROUTE_LABEL, SELECTOR_LABEL
    methods = list(ALL_METHODS) if a.methods == "all" else a.methods.split(",")
    skipped = {}
    ok9, why9 = a9_ready()
    if not ok9:
        for m in [m for m in methods if m in DIRECT_METHODS]: skipped[m] = why9; methods.remove(m)
        print("WARNING A9 not ready -> skipping direct_* :", why9, flush=True)
    if a.split == "sealed" and (a.limit or a.methods != "all"):
        raise SystemExit("REFUSE: sealed run must be the full frozen method list on all states (no --limit / --methods)")
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    if a.split == "sealed" and os.path.exists(os.path.join(out, "decisions.jsonl")):
        raise SystemExit(f"REFUSE: {out}/decisions.jsonl exists; sealed run is single-shot")
    items = load_items(a.split)
    if a.limit:
        by = {}
        for it in items: by.setdefault(it[5], []).append(it)
        keep = {sub: sorted({it[0] for it in v})[: a.limit] for sub, v in by.items()}
        items = [it for it in items if it[0] in keep[it[5]]]
    reps = [int(x) for x in a.reps.split(",")] if a.reps else [None]
    tag = os.path.basename(a.out.rstrip("/"))
    cfg = {"started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "split": a.split,
           "argv": sys.argv, "methods": methods, "skipped": skipped, "n_items": len(items), "reps": reps,
           "action_noise": a.action_noise, "budget": {"step_budget": STEP_BUDGET, "probe_budget": PROBE_BUDGET, "max_task_decisions": MAX_TASK},
           "candidates": "A4 scenarios.strike_grid K=54 H=6 (StrikeGridProposer), shared by every method",
           "frozen_manifest_written": man["written"], "frozen_manifest_sha256": man["sha256"],
           "route_labels": ROUTE_LABEL, "direct_selector_labels": SELECTOR_LABEL,
           "notes": ["wm_taskvoi_update: probe decision by A8 mu-grid belief VOI (privileged model class); task decision by WM stack",
                     "belief_* = privileged structured physics belief (knows env functional form); secondary route",
                     "A-9: probe that terminates env -> 0 task decisions, valid episode"]}
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=1, default=str)
    T0 = time.time()
    tmpd = os.path.join(out, "_parts"); os.makedirs(tmpd, exist_ok=True)
    jobs = [(m, items, reps, a.action_noise, tag, os.path.join(tmpd, m)) for m in methods]
    res = []
    with mp.get_context("fork").Pool(min(a.workers, len(jobs))) as pool:
        for m, n, dt, rss in pool.imap_unordered(_worker, jobs):
            res.append((m, n, dt, rss)); print(f"{m}: {n} episodes {dt:.1f}s maxrss {rss:.0f}MB", flush=True)
    # deterministic merge order = method list order
    with open(os.path.join(out, "decisions.jsonl"), "w") as fd, open(os.path.join(out, "episodes.jsonl"), "w") as fe:
        for m in methods:
            fd.write(open(os.path.join(tmpd, m + ".dec")).read()); fe.write(open(os.path.join(tmpd, m + ".eps")).read())
    eps = [json.loads(l) for l in open(os.path.join(out, "episodes.jsonl"))]
    cols = ["method", "episode_id", "subset", "init_state_id", "hidden_condition_id", "seed", "rep", "action_noise", "success",
            "termination", "final_goal_err", "steps_used", "probe_steps", "n_probes", "n_task_decisions", "terminated_during_probe",
            "K_returned", "phys_time_s"] + [f"t_{k}_s" for k in STAGES] + ["t_compute_total_s"]
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for e in eps:
            row = {c: e.get(c) for c in cols}
            for k in STAGES: row[f"t_{k}_s"] = round(e["compute_s"][k], 6)
            row["t_compute_total_s"] = round(sum(e["compute_s"].values()), 6)
            w.writerow(row)
    st = {}
    for e in eps:
        d = st.setdefault(e["method"], {k: 0.0 for k in STAGES} | {"n": 0})
        for k in STAGES: d[k] += e["compute_s"][k]
        d["n"] += 1
    st = {m: {**{f"{k}_s_per_ep": d[k] / d["n"] for k in STAGES}, "n": d["n"], "worker_wall_s": next(r[2] for r in res if r[0] == m),
              "worker_maxrss_MB": next(r[3] for r in res if r[0] == m)} for m, d in st.items()}
    st["__total_wall_s__"] = time.time() - T0
    json.dump(st, open(os.path.join(out, "stage_times.json"), "w"), indent=1)
    for m in methods: os.remove(os.path.join(tmpd, m + ".dec")); os.remove(os.path.join(tmpd, m + ".eps"))
    os.rmdir(tmpd)
    print("TOTAL wall", round(time.time() - T0, 1), "s ->", out, flush=True)

if __name__ == "__main__":
    main()
