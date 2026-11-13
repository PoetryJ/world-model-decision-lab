"""A6 E2-protocol experiment (TOY, CPU; controller decision 18:10): optional probe (<=5 steps) + ONE task
segment from A4 strike_grid (54,6,2) + settle (A10 offline_rank: zero actions until rest, <=20 steps).
Arms (same init state, same hidden condition, paired):
  none          : no probe, no adaptation, choose at t=0
  tap_noupdate  : tap_fast probe executed (5 steps), model NOT updated        (iii)
  tap_update    : same tap_fast probe, RLS residual updated on that probe     (ii)
  slow_noupdate / slow_update : push_slow probe (continuous contact, expected uninformative about mu)
Per decision: candidates are branched OFFLINE from the decision snapshot (evaluation.offline_rank) for
ground truth (endpoint, settled goal error); then the chosen candidate is EXECUTED ONLINE in a fresh env
replay (no restore) to get the task outcome. Scores = -|predicted settled pos - goal|.
Usage: python -m src.adaptation.exp_e2 --split val|test [--prior 1.0 --feats 12 --noise 0.005]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from shared.interface import Transition, DT, DecisionRecord, to_jsonable
from dataclasses import asdict
from src.env.push_env import PushEnv
from src.env import scenarios as S
from evaluation.offline_rank import rank_candidates
from evaluation.metrics import pairwise_accuracy, top1_regret, top1_correct
from src.dynamics.model import load_or_train
from src.adaptation.rls import RLSAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs", "a6", "e2")
SETTLE = 20
ARMS = {"none": (None, False), "tap_noupdate": ("tap_fast", False), "tap_update": ("tap_fast", True),
        "slow_noupdate": ("push_slow", False), "slow_update": ("push_slow", True)}

def sets(split):
    """Frozen A10 sets via evaluation.audit.load_frozen. val: val+val_neg states x val mus {0.15, 0.5} (train_val pool,
    condition-disjoint from the model's training mus? 0.5 is a TRAIN mu (in-range check), 0.15 is not).
    test: test+test_neg x frozen test conditions (mu_lo 0.1, mu_mid 0.3, mu_hi 0.6). env seed = frozen seed."""
    from evaluation.audit import load_frozen
    if split == "val":
        conds = [("val_mu_0.15", {"mu": 0.15}), ("val_mu_0.5", {"mu": 0.5})]
        st = load_frozen("val", who="A6") + load_frozen("val_neg", who="A6")
    else:
        # Controller decision A-8 (18:16): module agents must NOT load test/test_neg/test_sealed*. Final test = controller.
        raise SystemExit("A6: test splits are off-limits (A-8). Use --split val.")
    return conds, [(e["init_state_id"], dict(e["init"], meta=e["meta"]), e["seed"]) for e in st]

def pad(C, n=SETTLE):
    return np.concatenate([C, np.zeros((C.shape[0], n, 2))], 1)

def settle(env):
    for _ in range(SETTLE):
        if env.done or np.linalg.norm(env.observe()["obj_vel"]) < 1e-3: break
        env.step(np.zeros(2))

def run(pred, ad, split, tag, action_noise=0.0, rep=None):
    conds, states = sets(split)
    rows = []
    os.makedirs(OUT, exist_ok=True)
    recf = open(os.path.join(OUT, f"decisions_{tag}.jsonl"), "w")
    def rec(**kw):
        recf.write(json.dumps(to_jsonable(asdict(DecisionRecord(run_id=f"a6_e2_{tag}", **kw)))) + "\n")
    for cid, hid in conds:
        for i, (sid, init, seed) in enumerate(states):
            neg = bool(init["meta"]["negative_control"])
            init_env = {k: v for k, v in init.items() if k != "meta"}
            gt_cache = {}
            for arm, (probe, upd) in ARMS.items():
                env = PushEnv(action_noise=action_noise); obs = env.reset(init_env, hid, seed if rep is None else seed * 10 + rep); hist = []
                epi = f"{sid}|{cid}|{arm}"; common = dict(episode_id=epi, method=f"a6_{arm}", init_state_id=sid,
                                                          hidden_condition_id=cid, seed=seed)
                si = 0
                if probe is not None:
                    t0 = time.time(); a = S.probe_set(obs)[probe]; tpr = time.time() - t0
                    res = env.step_segment(a)
                    hist.append(Transition(obs=obs, actions=a[:len(res)], next_obs=res[-1].obs, kind="probe"))
                    rec(**common, step_index=si, kind="probe", candidates=None, predictions=None, scores=None,
                        chosen_index=None, executed=a[:len(res)], obs_before=obs, obs_after=res[-1].obs,
                        wall_time_s={"probe": tpr}, termination=res[-1].info.get("termination") if res[-1].done else None)
                    si += 1
                    obs = env.observe()
                snap = env.snapshot()
                t0 = time.time(); st = ad.update(hist) if upd else None; tu = time.time() - t0
                C, labels = S.strike_grid(obs)
                if probe not in gt_cache:
                    gt_cache[probe] = rank_candidates(env, snap, C, select_rep_seeds=(11,), eval_rep_seeds=(21,), settle_steps=SETTLE)
                gt = gt_cache[probe]
                env.restore(snap)
                t0 = time.time()
                ctx = {"history": hist, "adapt_state": st}
                Pseg = pred.predict(obs, C, ctx); Pset = pred.predict(obs, pad(C), ctx)
                tp = time.time() - t0
                g = np.asarray(obs["goal"], float)
                sc = -np.linalg.norm(Pset.endpoints - g, axis=1)
                te = np.asarray(gt["true_err"]); tep = np.asarray(gt["true_endpoints"])
                ch = int(np.argmax(sc))
                # ONLINE execution of chosen candidate from the actual post-probe state (env continues, no restore)
                t0 = time.time(); C, labels = S.strike_grid(obs); tprop = time.time() - t0
                res = env.step_segment(C[ch]); ob_mid = env.observe()
                term_task = res[-1].info.get("termination") if res[-1].done else None
                rec(**common, step_index=si, kind="task", candidates=C, predictions=Pset.endpoints, scores=sc,
                    chosen_index=ch, executed=C[ch][:len(res)], obs_before=obs, obs_after=ob_mid,
                    wall_time_s={"propose": tprop, "predict": tp, "update": tu, "score": 0.0}, termination=term_task,
                    candidate_counts={"requested": len(C), "returned": len(C), "failed": 0, "seeds": []})
                si += 1
                nset = env.n_steps
                settle(env)
                fin = env.observe()
                if not term_task:
                    term = env.termination or ("success" if env.success else "budget")
                    nz = env.n_steps - nset
                    rec(**common, step_index=si, kind="task", candidates=None, predictions=None, scores=None,
                        chosen_index=None, executed=np.zeros((nz, 2)), obs_before=ob_mid, obs_after=fin,
                        wall_time_s={}, termination=term)
                ferr = float(np.linalg.norm(np.asarray(fin["obj"]) - g))
                fsucc = bool(env.success or (ferr < fin["goal_radius"] and np.linalg.norm(fin["obj_vel"]) < 1e-3))
                acc, npairs, nties = pairwise_accuracy(te, sc)
                set_err = None
                rows.append({"split": split, "init_state_id": sid, "neg": neg, "hidden_condition_id": cid, "arm": arm,
                    "seed": seed, "probe": probe, "updated": upd,
                    "endpoint_err_seg": float(np.mean(np.linalg.norm(Pseg.endpoints - tep, axis=1))),
                    "endpoint_err_seg_chosen": float(np.linalg.norm(Pseg.endpoints[ch] - tep[ch])),
                    "goalerr_pred_err": float(np.mean(np.abs(-sc - te))),
                    "goalerr_pred_err_chosen": float(abs(-sc[ch] - te[ch])),
                    "pairwise_acc": acc, "n_pairs": npairs, "n_true_ties": nties,
                    "top1_regret": top1_regret(te, sc), "top1_correct": top1_correct(te, sc),
                    "chosen": ch, "chosen_label": labels[ch], "chosen_true_err_offline": float(te[ch]),
                    "online_goal_err": ferr, "online_success": float(fsucc),
                    "oracle_err": gt["oracle_eval_err"], "oracle_label": labels[gt["oracle_index"]],
                    "oracle_success": float(gt["true_success_rate"][gt["oracle_index"]]),
                    "spread_err": gt["spread_err"], "all_fail": gt["all_fail"], "deterministic": gt["deterministic"],
                    "probe_steps": len(hist[0].actions) if hist else 0,
                    "ens_std_chosen": float(Pseg.samples[ch].std(0).mean()),
                    "res_std_chosen": float(Pset.extra["residual_std"][ch]) if "residual_std" in Pset.extra else None,
                    "rls_w": (st["w"].tolist() if st is not None and "w" in st else None),
                    "mu_hat": (st.get("mu_hat") if st is not None else None),
                    "predict_s": tp, "update_s": tu})
    recf.close()
    with open(os.path.join(OUT, f"rows_{tag}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows

KEYS = ["endpoint_err_seg", "goalerr_pred_err", "goalerr_pred_err_chosen", "pairwise_acc", "top1_regret", "top1_correct", "chosen_true_err_offline",
        "online_goal_err", "online_success", "oracle_err", "oracle_success"]

def summarize(rows, by):
    g = {}
    for r in rows: g.setdefault(tuple(str(r[b]) for b in by), []).append(r)
    return {"|".join(k): {m: float(np.nanmean([x[m] for x in v])) for m in KEYS} | {"n": len(v)} for k, v in sorted(g.items())}

def cluster_ci(rows, arm_a, arm_b, field, B=2000, seed=0):
    """paired diff a-b, cluster bootstrap over init_state_id (all conditions of a state move together)."""
    d = {}
    A = {(r["init_state_id"], r["hidden_condition_id"]): r for r in rows if r["arm"] == arm_a}
    Bm = {(r["init_state_id"], r["hidden_condition_id"]): r for r in rows if r["arm"] == arm_b}
    for k in A:
        d.setdefault(k[0], []).append(A[k][field] - Bm[k][field])
    keys = sorted(d); vals = [np.mean(d[k]) for k in keys]
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(vals, len(vals))) for _ in range(B)]
    return {"mean": float(np.mean(vals)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_clusters": len(keys)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", type=float, default=5.0); ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--feats", default="2")
    ap.add_argument("--model", default="rls", choices=["rls", "latent"])
    ap.add_argument("--action_noise", type=float, default=0.0); ap.add_argument("--rep", type=int, default=None)
    a = ap.parse_args()
    if a.model == "rls":
        pred = load_or_train(a.seed)
        ad = RLSAdapter(pred, prior_std=a.prior, noise_std=a.noise, feat_idx=[int(c) for c in a.feats])
        tag = f"{a.split}_s{a.seed}_p{a.prior}_n{a.noise}_f{a.feats}"
    else:
        from src.dynamics.latent import load_or_train_latent, LatentGridAdapter
        pred = load_or_train_latent(a.seed); ad = LatentGridAdapter(pred, noise_std=a.noise)
        tag = f"{a.split}_s{a.seed}_latent_n{a.noise}"
    if a.action_noise > 0: tag += f"_an{a.action_noise}_r{a.rep}"
    t0 = time.time(); rows = run(pred, ad, a.split, tag, a.action_noise, a.rep)
    summ = {"cfg": vars(a), "wall_s": time.time() - t0, "model_meta": {k: v for k, v in pred.meta.items() if k != "log"},
            "by_arm_set": summarize(rows, ("arm", "neg")), "by_arm_set_cond": summarize(rows, ("arm", "neg", "hidden_condition_id")),
            "paired": {f"{x}-{y}|{f}": cluster_ci([r for r in rows if not r["neg"]], x, y, f)
                       for x, y in [("tap_update", "tap_noupdate"), ("tap_update", "none"), ("tap_noupdate", "none"), ("slow_update", "slow_noupdate")]
                       for f in ["top1_regret", "online_goal_err", "online_success", "endpoint_err_seg"]}}
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f: json.dump(summ, f, indent=1)
    for k, v in summ["by_arm_set"].items(): print(k, {m: round(x, 4) for m, x in v.items()})
