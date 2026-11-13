"""A6 experiments (TOY, CPU). Compare (i) no adaptation, (ii) probe + RLS update, (iii) same probe, no update.
Offline part: at the first task decision, branch every candidate from the same snapshot (A10 offline_rank)
-> endpoint error, pairwise ranking accuracy, top-1 regret, chosen candidate's true outcome.
Online part: closed-loop episodes, budget from A10 (60 steps total, probe counts), no resets after probe.
Usage: python -m src.adaptation.exp_a6 --split val|test --part offline|online|both [--prior 0.5 --noise 0.005]
Hidden params used ONLY to configure env (and condition label); never passed to predictor/adapter.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from shared.interface import clip_action, Transition, DT, to_jsonable, DecisionRecord
from src.env.push_env import PushEnv
from src.env.scenarios import strike_grid, probe_set
from evaluation.make_sets import make_split
from evaluation.offline_rank import rank_candidates
from evaluation.metrics import pairwise_accuracy, top1_regret, top1_correct
from src.dynamics.model import load_or_train
from src.adaptation.rls import RLSAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs", "a6")
SETTLE = 20
SPLIT_MUS = {"val": {"val_mu_0.25": 0.25, "val_mu_0.45": 0.45},
             "test": {"mu_lo": 0.1, "mu_mid": 0.3, "mu_hi": 0.6}}
N_STATES = {"val": 40, "test": 40}

STRIKE_S = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
STRIKE_N = (1, 2, 3, 4, 6)
CAND_H = 12
def candidates(obs, H=CAND_H):
    """Local experiment candidate set (A5 owns proposers; this is a fixed, visible-only grid so friction
    matters): index 0 = wait (zeros); 1 = align only; then for s in STRIKE_S, n in STRIKE_N:
    drive pusher to the 'behind' point (0.10 m behind object on goal line) at 0.3 m/s, then push along the
    object->goal direction at s m/s for n steps, then zero (object released, slides). K = 2 + 45 = 47."""
    p = np.asarray(obs["pusher"], float); o = np.asarray(obs["obj"], float); g = np.asarray(obs["goal"], float)
    u = g - o; u = u / max(np.linalg.norm(u), 1e-9)
    behind = o - 0.10 * u
    def approach():
        acts = []; q = p.copy()
        for _ in range(H):
            d = behind - q; dist = np.linalg.norm(d)
            if dist < 0.01: break
            v = min(0.3, dist / DT) * d / dist
            acts.append(v); q = q + v * DT
        return acts
    ap = np.array(approach()).reshape(-1, 2)
    out = [np.zeros((H, 2))]
    a = np.zeros((H, 2)); a[:len(ap)] = ap; out.append(a)
    for sp in STRIKE_S:
        for n in STRIKE_N:
            a = np.zeros((H, 2)); L = min(len(ap), H)
            a[:L] = ap[:L]
            a[L:L + n] = sp * u
            out.append(clip_action(a))
    return np.stack(out)

def pad(C, n=SETTLE):
    return np.concatenate([C, np.zeros((C.shape[0], n, 2))], 1)

def plan(pred, obs, C, st, hist):
    t0 = time.time()
    P_seg = pred.predict(obs, C, {"history": hist, "adapt_state": st})
    P_set = pred.predict(obs, pad(C), {"history": hist, "adapt_state": st})
    tp = time.time() - t0
    g = np.asarray(obs["goal"], float)
    scores = -np.linalg.norm(P_set.endpoints - g, axis=1)
    return P_seg, P_set, scores, tp

def do_probe(env, obs):
    a = probe_set(obs)["tap_fast"]
    res = env.step_segment(a)
    nob = res[-1].obs
    return Transition(obs=obs, actions=a[:len(res)], next_obs=nob, kind="probe"), res

def offline(pred, adapter, split, log):
    rows = []
    states = make_split(split, N_STATES[split])
    for cid, mu in SPLIT_MUS[split].items():
        for s in states:
            env = PushEnv()
            obs0 = env.reset(s["init"], {"mu": mu}, seed=s["seed"])
            arms = {}
            # (i) no probe, no adaptation, decide at t0
            snap0 = env.snapshot()
            arms["none"] = (obs0, snap0, None, [])
            # probe (same actions for ii and iii)
            tr, _ = do_probe(env, obs0)
            snap1 = env.snapshot(); obs1 = env.observe()
            t0 = time.time(); st = adapter.update([tr]); tu = time.time() - t0
            arms["probe_noupdate"] = (obs1, snap1, None, [tr])
            arms["probe_update"] = (obs1, snap1, st, [tr])
            gt_cache = {}
            for arm, (obs, snap, ast, hist) in arms.items():
                C = candidates(obs)
                key = id(snap)
                if key not in gt_cache:
                    gt_cache[key] = rank_candidates(env, snap, C, select_rep_seeds=(11,), eval_rep_seeds=(21,), settle_steps=SETTLE)
                gt = gt_cache[key]
                P_seg, P_set, sc, tp = plan(pred, obs, C, ast, hist)
                te = np.asarray(gt["true_err"]); tep = np.asarray(gt["true_endpoints"])
                acc, npairs, nties = pairwise_accuracy(te, sc)
                ch = int(np.argmax(sc))
                row = {"split": split, "init_state_id": s["init_state_id"], "hidden_condition_id": cid, "arm": arm,
                       "endpoint_err_seg": float(np.mean(np.linalg.norm(P_seg.endpoints - tep, axis=1))),
                       "pairwise_acc": acc, "n_pairs": npairs, "n_true_ties": nties,
                       "top1_regret": top1_regret(te, sc), "top1_correct": top1_correct(te, sc),
                       "chosen": ch, "chosen_true_err": float(te[ch]), "chosen_success": float(gt["true_success_rate"][ch]),
                       "oracle_err": gt["oracle_eval_err"], "oracle_success": float(max(gt["true_success_rate"])),
                       "spread_err": gt["spread_err"], "all_fail": gt["all_fail"], "deterministic": gt["deterministic"],
                       "ens_std_mean": float(P_seg.samples.std(1).mean()), "predict_s": tp,
                       "update_s": tu if arm == "probe_update" else 0.0,
                       "rls_w": st["w"].tolist() if arm == "probe_update" else None}
                rows.append(row)
    with open(log, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows

def online_episode(pred, adapter, s, cid, mu, arm, budget=60, rec=None):
    env = PushEnv(); obs = env.reset(s["init"], {"mu": mu}, seed=s["seed"])
    hist = []; steps = 0; probe_steps = 0; comp = {"predict": 0.0, "update": 0.0, "propose": 0.0, "score": 0.0}
    term = None; k = 0
    do_p = arm in ("probe_update", "probe_noupdate")
    upd = arm in ("probe_update", "noprobe_update")
    if do_p:
        tr, res = do_probe(env, obs); hist.append(tr); steps += len(res); probe_steps += len(res)
        obs = env.observe()
        if res[-1].done: term = res[-1].info.get("termination")
    while term is None and steps < budget:
        t0 = time.time(); st = adapter.update(hist) if upd else None; comp["update"] += time.time() - t0
        t0 = time.time(); C = candidates(obs); comp["propose"] += time.time() - t0
        P_seg, P_set, sc, tp = plan(pred, obs, C, st, hist); comp["predict"] += tp
        ch = int(np.argmax(sc))
        a = C[ch][: budget - steps]
        res = env.step_segment(a); steps += len(res)
        nob = res[-1].obs
        hist.append(Transition(obs=obs, actions=a[:len(res)], next_obs=nob, kind="task"))
        if rec is not None:
            rec.write(json.dumps(to_jsonable({"run_id": "a6_online", "episode_id": f"{s['init_state_id']}|{cid}|{arm}",
                "method": arm, "init_state_id": s["init_state_id"], "hidden_condition_id": cid, "seed": s["seed"],
                "step_index": k, "kind": "task", "chosen_index": ch, "pred_settled": P_set.endpoints[ch],
                "executed": a[:len(res)], "obs_before": obs, "obs_after": nob,
                "rls_w": (st["w"] if st is not None else None)})) + "\n")
        obs = nob; k += 1
        if res[-1].done: term = res[-1].info.get("termination")
    if term is None: term = "budget"
    err = float(np.linalg.norm(np.asarray(env.observe()["obj"]) - np.asarray(obs["goal"])))
    return {"method": arm, "init_state_id": s["init_state_id"], "hidden_condition_id": cid, "rep": 0,
            "success": bool(term == "success"), "goal_error": err, "steps_total": steps, "probe_steps": probe_steps,
            "task_steps": steps - probe_steps, "physical_time_s": steps * DT, "termination": term, "compute_s": comp,
            "n_decisions": k}

ARMS_ONLINE = ("none", "probe_noupdate", "probe_update", "noprobe_update")

def online(pred, adapter, split, log, n=None):
    states = make_split(split, N_STATES[split])[: n or N_STATES[split]]
    eps = []
    with open(log.replace(".jsonl", "_decisions.jsonl"), "w") as rec:
        for cid, mu in SPLIT_MUS[split].items():
            for s in states:
                for arm in ARMS_ONLINE:
                    eps.append(online_episode(pred, adapter, s, cid, mu, arm, rec=rec))
    with open(log, "w") as f:
        for e in eps: f.write(json.dumps(e) + "\n")
    return eps

def summarize(rows, keys, by=("arm",)):
    out = {}
    for r in rows:
        k = tuple(r[b] for b in by); out.setdefault(k, []).append(r)
    res = {}
    for k, v in sorted(out.items()):
        res["|".join(k)] = {m: float(np.nanmean([float(x[m]) for x in v])) for m in keys} | {"n": len(v)}
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--part", default="both")
    ap.add_argument("--prior", type=float, default=1.0); ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--n_online", type=int, default=None)
    ap.add_argument("--feats", default="12")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pred = load_or_train(a.seed)
    ad = RLSAdapter(pred, prior_std=a.prior, noise_std=a.noise, feat_idx=[int(c) for c in a.feats])
    tag = f"{a.split}_s{a.seed}_p{a.prior}_n{a.noise}_f{a.feats}{a.tag}"
    summ = {"cfg": vars(a)}
    if a.part in ("offline", "both"):
        t0 = time.time()
        rows = offline(pred, ad, a.split, os.path.join(OUT, f"offline_{tag}.jsonl"))
        K = ["endpoint_err_seg", "pairwise_acc", "top1_regret", "top1_correct", "chosen_true_err", "chosen_success", "oracle_err", "oracle_success", "ens_std_mean"]
        summ["offline_by_arm"] = summarize(rows, K)
        summ["offline_by_arm_cond"] = summarize(rows, K, ("arm", "hidden_condition_id"))
        summ["offline_s"] = time.time() - t0
    if a.part in ("online", "both"):
        t0 = time.time()
        eps = online(pred, ad, a.split, os.path.join(OUT, f"online_{tag}.jsonl"), a.n_online)
        for e in eps: e["arm"] = e["method"]
        K = ["success", "goal_error", "steps_total", "n_decisions"]
        summ["online_by_arm"] = summarize(eps, K)
        summ["online_by_arm_cond"] = summarize(eps, K, ("arm", "hidden_condition_id"))
        summ["online_s"] = time.time() - t0
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f: json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
