"""A4 E0 environment-qualification gate (TOY env). Writes reports/a4/e0_raw.json + reports/a4/E0.md.
Run: cd /workspace/wmprobe && . .venv/bin/activate && python -m src.env.e0_gate
All seeds fixed below. Uses OFFLINE snapshot branching (oracle) -- evaluation only."""
from __future__ import annotations
import json, time, itertools, pathlib, platform
import numpy as np
from shared.interface import to_jsonable
from src.env.push_env import PushEnv, OBS_KEYS
from src.env import scenarios as S
from src.env.branch_eval import evaluate_candidates

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "a4"
COND = S.hidden_conditions(3)
CID = [c for c, _ in COND]
NOISE_A, NOISE_O = 0.02, 0.002          # probe-discrimination noise (m/s per action comp; m / m/s obs)
NOISE_SEEDS = list(range(20))            # 0-9 fit class means, 10-19 test
CRN_SEEDS = list(range(100, 108))        # common random numbers for noisy candidate evaluation (R=8)


def c0_replay(states):
    ok, n = 0, 0
    for (sid, st), (cid, h) in itertools.product(states, COND):
        env = PushEnv(action_noise=0.05, obs_noise=0.003); env.reset(st, h, seed=7)
        g = np.random.default_rng(abs(hash(sid)) % 2**31)
        env.step_segment(g.uniform(-0.5, 0.5, (3, 2)))
        snap = env.snapshot()
        acts = g.uniform(-0.5, 0.5, (15, 2))
        def tr(e):
            rows = []
            for r in e.step_segment(acts):
                rows.append(np.r_[r.obs["t"], r.obs["pusher"], r.obs["obj"], r.obs["obj_vel"], e.pv, e.ov,
                                  e.contact_substeps, r.done])
            return np.array(rows).tobytes() + e.rng.random(3).tobytes()
        a = tr(env); env.restore(snap); b = tr(env)
        e2 = PushEnv(); e2.restore(snap); c = tr(e2)
        ok += (a == b == c); n += 1
    return {"pass": ok == n, "n_ok": ok, "n": n}


def c1_same_obs(states):
    bad = []
    for sid, st in states:
        for seed, kw in ((0, {}), (5, dict(action_noise=0.02, obs_noise=0.002))):
            obs = [PushEnv(**kw).reset(st, h, seed) for _, h in COND]
            for o in obs[1:]:
                if any(np.asarray(o[k]).tobytes() != np.asarray(obs[0][k]).tobytes() for k in OBS_KEYS):
                    bad.append(sid)
    return {"pass": not bad, "n_states": len(states), "n_conditions": len(COND), "mismatch": bad}


def grid_tables(states, noisy=False):
    """Per state: dist (C,K), success (C,K) [deterministic] or success-prob (C,K) over CRN seeds [noisy]."""
    rows = []
    for sid, st in states:
        D, SU = [], []
        for _, h in COND:
            if not noisy:
                env = PushEnv(); obs = env.reset(st, h, 0)
                cands, labels = S.strike_grid(obs)
                d, s, _ = evaluate_candidates(env, env.snapshot(), cands)
            else:
                ds, ss = [], []
                for sd in CRN_SEEDS:
                    env = PushEnv(action_noise=NOISE_A); obs = env.reset(st, h, sd)
                    cands, labels = S.strike_grid(obs)
                    d, s, _ = evaluate_candidates(env, env.snapshot(), cands); ds.append(d); ss.append(s)
                d, s = np.mean(ds, 0), np.mean(ss, 0)
            D.append(d); SU.append(s)
        rows.append({"sid": sid, "labels": labels, "dist": np.array(D), "succ": np.array(SU)})
    return rows


def summarize_grid(rows, noisy):
    out = {"per_state": []}
    C = len(COND)
    pair_shared = np.zeros((C, C), int); argmin_cross_succ = np.zeros((C, C)); n = len(rows)
    for r in rows:
        D, SU = r["dist"], r["succ"]
        if noisy:   # best = highest success prob, tie-break lower mean dist
            key = SU - 1e-3 * D
            good = SU >= 0.75                                   # "reliably succeeds" (>=6/8 CRN seeds)
        else:
            key = -D
            good = SU.astype(bool)
        am = key.argmax(1)
        for a in range(C):
            for b in range(C):
                pair_shared[a, b] += bool((good[a] & good[b]).any())
                argmin_cross_succ[a, b] += float(SU[b, am[a]])
        common = good.all(0)
        out["per_state"].append({
            "sid": r["sid"], "best_per_cond": {CID[c]: r["labels"][am[c]] for c in range(C)},
            "best_dist_per_cond": {CID[c]: float(D[c, am[c]]) for c in range(C)},
            "best_succ_per_cond": {CID[c]: float(SU[c, am[c]]) for c in range(C)},
            "n_good_per_cond": {CID[c]: int(good[c].sum()) for c in range(C)},
            "n_common_good_all_conds": int(common.sum()),
            "common_good_labels": [r["labels"][i] for i in np.flatnonzero(common)][:10],
            "cross_succ_of_best": {f"{CID[a]}->{CID[b]}": float(SU[b, am[a]]) for a in range(C) for b in range(C)},
            "regret_lo_best_under_hi_m": float(D[2, am[0]] - D[2].min()),
            "regret_hi_best_under_lo_m": float(D[0, am[2]] - D[0].min()),
        })
    out["pair_states_with_shared_good_candidate"] = {f"{CID[a]}&{CID[b]}": int(pair_shared[a, b])
                                                    for a in range(C) for b in range(a + 1, C)}
    out["mean_cross_success_of_best"] = {f"{CID[a]}->{CID[b]}": float(argmin_cross_succ[a, b] / n)
                                         for a in range(C) for b in range(C) if a != b}
    out["states_with_common_good_all3"] = int(sum(p["n_common_good_all_conds"] > 0 for p in out["per_state"]))
    out["distinct_best_lo_vs_hi"] = int(sum(p["best_per_cond"]["mu_lo"] != p["best_per_cond"]["mu_hi"]
                                            for p in out["per_state"]))
    # value of knowing mu (oracle, fixed grid): best single candidate under uniform prior vs per-condition best
    out["mean_succ_best_fixed_candidate_unknown_mu"] = float(np.mean([r["succ"].mean(0).max() for r in rows]))
    out["mean_succ_best_candidate_known_mu"] = float(np.mean([r["succ"].max(1).mean() for r in rows]))
    out["n_states"] = n
    return out


def c5_probe_cost(states):
    """After executing each probe (deterministic), best achievable success over the strike grid from the
    ACTUAL post-probe state (oracle per condition). Shows physical cost of probing (e.g. overshoot)."""
    res = {}
    for pname in S.PROBE_NAMES:
        per = {c: [] for c in CID}
        for sid, st in states:
            for cid, h in COND:
                env = PushEnv(); o0 = env.reset(st, h, 0)
                env.step_segment(S.probe_set(o0)[pname])
                if env.done:
                    per[cid].append(float(env.success)); continue
                cands, _ = S.strike_grid(env.observe())
                _, su, _ = evaluate_candidates(env, env.snapshot(), cands)
                per[cid].append(float(su.max()))
        res[pname] = {c: float(np.mean(v)) for c, v in per.items()}
    return res


def c3_probes(states):
    """Feature = object displacement along probe direction at probe end + obj speed at probe end (visible)."""
    res = {}
    for pname in S.PROBE_NAMES:
        feats = {c: [] for c in CID}   # list over states of (seeds, 2)
        for sid, st in states:
            for cid, h in COND:
                fs = []
                for sd in NOISE_SEEDS:
                    env = PushEnv(action_noise=NOISE_A, obs_noise=NOISE_O); o0 = env.reset(st, h, sd)
                    p = S.probe_set(o0)[pname]
                    u = p[np.flatnonzero(np.abs(p).sum(1) > 0)[0]]; u = u / np.linalg.norm(u)
                    env.step_segment(p); o1 = env.observe()
                    fs.append([float((o1["obj"] - o0["obj"]) @ u), float(np.linalg.norm(o1["obj_vel"]))])
                feats[cid].append(fs)
        F = {c: np.array(v) for c, v in feats.items()}      # (n_states, seeds, 2)
        # pooled d' on displacement feature, per pair
        dprime = {}
        for a, b in itertools.combinations(CID, 2):
            xa, xb = F[a][..., 0].ravel(), F[b][..., 0].ravel()
            sd = np.sqrt(0.5 * (xa.var(ddof=1) + xb.var(ddof=1)))
            dprime[f"{a}|{b}"] = float(abs(xa.mean() - xb.mean()) / sd) if sd > 0 else (float("inf") if xa.mean() != xb.mean() else 0.0)
        # per-state nearest-mean classifier: fit on seeds 0-9, test seeds 10-19 (2-D standardized features)
        correct, total = 0, 0
        for i in range(len(states)):
            tr = np.concatenate([F[c][i, :10] for c in CID]); scale = tr.std(0) + 1e-9
            means = np.array([F[c][i, :10].mean(0) for c in CID]) / scale
            for ci, c in enumerate(CID):
                te = F[c][i, 10:] / scale
                d2 = ((te[:, None, :] - means[None]) ** 2).sum(-1)
                # ties (identical means) -> count as chance by picking first; report
                correct += int((d2.argmin(1) == ci).sum()); total += len(te)
        res[pname] = {"dprime_disp": dprime, "acc_3way_nearest_mean": correct / total, "chance": 1 / 3,
                      "mean_disp_m": {c: float(F[c][..., 0].mean()) for c in CID},
                      "sd_disp_m": {c: float(F[c][..., 0].std(ddof=1)) for c in CID},
                      "mean_speed_end": {c: float(F[c][..., 1].mean()) for c in CID}}
    return res


def main():
    t0 = time.time()
    pos, neg = S.e0_set(False), S.e0_set(True)
    raw = {"env": "PushEnv-v1 TOY 2D", "conditions": dict(COND), "pos_ids": [s for s, _ in pos],
           "neg_ids": [s for s, _ in neg], "grid": {"speeds": list(S.STRIKE_SPEEDS), "H": S.TASK_H,
           "K": len(S.STRIKE_SPEEDS) * S.TASK_H}, "noise": {"action": NOISE_A, "obs": NOISE_O},
           "seeds": {"pos": S.E0_SEED_POS, "neg": S.E0_SEED_NEG, "probe_noise": NOISE_SEEDS, "crn": CRN_SEEDS},
           "python": platform.python_version()}
    raw["C0_replay"] = c0_replay(pos + neg)
    raw["C1_same_obs"] = c1_same_obs(pos + neg)
    raw["C2_pos_det"] = summarize_grid(grid_tables(pos), False)
    raw["C2_pos_noisy"] = summarize_grid(grid_tables(pos, True), True)
    raw["C4_neg_det"] = summarize_grid(grid_tables(neg), False)
    raw["C4_neg_noisy"] = summarize_grid(grid_tables(neg, True), True)
    raw["C3_probes_pos"] = c3_probes(pos)
    raw["C3_probes_neg"] = c3_probes(neg)
    raw["C5_probe_cost_pos"] = c5_probe_cost(pos)
    raw["C5_probe_cost_neg"] = c5_probe_cost(neg)
    print("C5 pos", raw["C5_probe_cost_pos"]); print("C5 neg", raw["C5_probe_cost_neg"])
    raw["runtime_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e0_raw.json").write_text(json.dumps(to_jsonable(raw), indent=1, default=str))
    print(json.dumps(to_jsonable({k: v for k, v in raw.items() if not k.startswith(("C2", "C4"))}), indent=1, default=str)[:6000])
    for k in ("C2_pos_det", "C2_pos_noisy", "C4_neg_det", "C4_neg_noisy"):
        print(k, {kk: vv for kk, vv in raw[k].items() if kk != "per_state"})


if __name__ == "__main__":
    main()
