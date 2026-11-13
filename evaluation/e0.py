"""A10 E0 env qualification gates G1/G3/G4/G5 (G2 = audit runs). [toy-run]
python -m evaluation.e0 [push|fixture]  -> runs/a10/e0_<env>.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from evaluation.audit import load_frozen, snapshot_exactness, ROOT
from evaluation.offline_rank import rank_candidates, reseed_snapshot

def env_factory(which, noise=0.0):
    if which == "push":
        from src.env.push_env import PushEnv
        return lambda: PushEnv(action_noise=noise)
    from shared.fixture_env import FixturePushEnv
    return lambda: FixturePushEnv(action_noise=noise)

def ref_lib(obs, speeds, H=5):
    u = np.asarray(obs["obj"]) - np.asarray(obs["pusher"]); u = u / np.linalg.norm(u)
    return np.stack([np.tile(s * u, (H, 1)) for s in speeds])

REF4 = [0.2, 0.3, 0.4, 0.5]
REF8 = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

def strike54(obs):
    from src.env.scenarios import strike_grid
    return strike_grid(obs)[0]

def probe_feedback(env, snap, obs0, pid, seed):
    from src.env.scenarios import probe_set
    env.restore(reseed_snapshot(snap, seed))
    res = env.step_segment(probe_set(obs0)[pid])
    ob = res[-1].obs
    return np.r_[np.asarray(ob["obj"]) - np.asarray(obs0["obj"]), np.asarray(ob["obj_vel"])]

def main(which="push"):
    sets = load_frozen("e0", who="A10 e0")
    allsets = load_frozen("__audit__", who="A10 e0", log=False)
    conds = allsets["conditions"]["test"]; negsets = allsets["e0_neg"]
    probes = {k: None for k in allsets["e0_probes"]["names"]}
    out = {"env": which, "time": time.strftime("%F %T"), "label": "[toy-run]"}
    out["G1"] = {str(n): snapshot_exactness(env_factory(which, n), sets[0]["init"], {"mu": 0.3})["issues"] for n in (0.0, 0.05)}
    per = []
    for st in sets:
        row = {"id": st["init_state_id"]}
        for cid, h in conds.items():
            env = env_factory(which)(); obs = env.reset(st["init"], h, st["seed"]); snap = env.snapshot()
            r = rank_candidates(env, snap, strike54(obs))
            row[cid] = {"true_err": r["true_err"], "oracle": r["oracle_index"]}
        fb = {}
        for pid, p in probes.items():
            fb[pid] = {}
            for cid, h in conds.items():
                env = env_factory(which)(); obs = env.reset(st["init"], h, st["seed"]); snap = env.snapshot()
                fb[pid][cid] = probe_feedback(env, snap, obs, pid, 0).tolist()
        row["feedback_det"] = fb
        per.append(row)
    out["per_state"] = per
    negper = []
    for st in negsets:
        row = {"id": st["init_state_id"]}
        for cid, h in conds.items():
            env = env_factory(which)(); obs = env.reset(st["init"], h, st["seed"]); snap = env.snapshot()
            r = rank_candidates(env, snap, strike54(obs))
            row[cid] = {"true_err": r["true_err"], "oracle": r["oracle_index"], "succ": r["true_success_rate"]}
        negper.append(row)
    out["neg_per_state"] = negper
    # G3 (pos set)
    diff = sum(r["mu_lo"]["oracle"] != r["mu_hi"]["oracle"] for r in per)
    xreg = [r["mu_hi"]["true_err"][r["mu_lo"]["oracle"]] - min(r["mu_hi"]["true_err"]) for r in per]
    xreg2 = [r["mu_lo"]["true_err"][r["mu_hi"]["oracle"]] - min(r["mu_lo"]["true_err"]) for r in per]
    out["G3"] = {"oracle_differs_lo_hi": diff, "mean_xregret_lo_best_under_hi": float(np.mean(xreg)),
                 "mean_xregret_hi_best_under_lo": float(np.mean(xreg2)),
                 "pass": bool(diff >= 6 and np.mean(xreg) >= 0.02)}
    # G4 deterministic separation
    g4 = {}
    cids = list(conds)
    for pid in probes:
        ok = 0; mind = []
        for r in per:
            f = r["feedback_det"][pid]
            d = min(np.linalg.norm(np.array(f[a]) - np.array(f[b])) for i, a in enumerate(cids) for b in cids[i + 1:])
            mind.append(d); ok += d > 1e-3
        g4[pid] = {"states_separated": ok, "min_pair_dist_median": float(np.median(mind))}
    # G4 noisy nearest-centroid
    for pid, p in probes.items():
        correct = total = 0
        for st in sets:
            cents, tests = {}, {}
            for cid, h in conds.items():
                env = env_factory(which, 0.02)(); obs = env.reset(st["init"], h, st["seed"]); snap = env.snapshot()
                cents[cid] = np.mean([probe_feedback(env, snap, obs, pid, s) for s in (11, 12, 13)], 0)
                tests[cid] = [probe_feedback(env, snap, obs, pid, s) for s in (21, 22, 23)]
            for cid, xs in tests.items():
                for x in xs:
                    pred = min(cents, key=lambda c: np.linalg.norm(x - cents[c])); correct += pred == cid; total += 1
        g4[pid]["noisy_nc_acc"] = correct / total
    out["G4"] = g4
    out["G4"]["pass"] = any(v["states_separated"] >= 11 and v["noisy_nc_acc"] >= 0.9 for k, v in g4.items() if isinstance(v, dict))
    # G5
    # G5: one action (chosen as oracle under mu_mid) is within 0.005 m of best under ALL conditions, or all oracles equal
    def same_ok(r):
        if len({r[c]["oracle"] for c in conds}) == 1: return True
        return any(all(r[c]["true_err"][k] - min(r[c]["true_err"]) < 0.005 for c in conds) for k in range(len(r["mu_mid"]["true_err"])))
    same = [same_ok(r) for r in negper]
    # also on pos set (should be rare)
    out["G5"] = {"neg_states_same_best": int(sum(same)), "pos_states_same_best": int(sum(same_ok(r) for r in per)),
                 "pass": bool(sum(same) >= 9)}
    # headroom/all-fail diagnostics
    out["diag"] = {c: {"all_fail_states": int(sum(min(r[c]["true_err"]) >= 0.05 for r in per)),
                       "mean_best_err": float(np.mean([min(r[c]["true_err"]) for r in per]))} for c in conds}
    p = os.path.join(ROOT, "runs", "a10", f"e0_{which}.json")
    from shared.interface import to_jsonable
    out = to_jsonable(out)
    json.dump(out, open(p, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({k: out[k] for k in ("G1", "G3", "G4", "G5", "diag")}, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "push")
