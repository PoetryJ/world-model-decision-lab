"""A10 E1 headroom on frozen e1 (20 states x 3 test conditions), offline branch truth. [toy-run]
Libraries: A5 FixedLibraryProposer(K=4,H=10) and STRIKE4 (A4 strike_grid subset). Scorer: A8 belief stack, no adaptation
(prior over mu, empty history) -> pairwise acc / top-1 regret vs truth. Deterministic + noisy (0.02, CRN 3+3 reps).
python -m evaluation.e1 -> runs/a10/e1.json
"""
import json, os, time
import numpy as np
from evaluation.audit import load_frozen, ROOT
from evaluation.offline_rank import rank_candidates
from evaluation import metrics as M
from shared.interface import to_jsonable

def libs(obs):
    from src.candidates.fixed import FixedLibraryProposer
    from src.env.scenarios import strike_grid
    C, lab = strike_grid(obs); idx = [lab.index(x) for x in [(0.2, 6), (0.3, 4), (0.4, 3), (0.5, 2)]]
    from src.candidates.release import TimedReleaseProposer
    return {"A5_fixed4": FixedLibraryProposer(K=4).propose(obs, [], 0), "STRIKE4": C[idx],
            "A5_release4": TimedReleaseProposer(K=4).propose(obs, [], 0)}

def a8_prior_scores(obs, C, seed):
    from src.probing import experiments as X
    prop, pred, adp, sc, pr = X.build("no_probe", seed, 0.01)
    st = adp.update([]); P = pred.predict(obs, C, {"history": [], "adapt_state": st})
    return np.asarray(sc.score(P, {"goal": obs["goal"], "goal_radius": obs["goal_radius"]}), float), np.asarray(P.endpoints)

def main():
    from src.env.push_env import PushEnv
    sets = load_frozen("e1", who="A10 e1"); conds = load_frozen("__audit__", who="A10 e1", log=False)["conditions"]["test"]
    rows = []
    for noise in (0.0, 0.02):
        for st in sets:
            for cid, h in conds.items():
                env = PushEnv(action_noise=noise); obs = env.reset(st["init"], h, st["seed"]); snap = env.snapshot()
                for lname, C in libs(obs).items():
                    rk = rank_candidates(env, snap, C)
                    e = np.asarray(rk["true_err"])
                    s, ep = a8_prior_scores(obs, C, st["seed"])
                    acc, n, ties = M.pairwise_accuracy(e, s)
                    rows.append({"noise": noise, "init_state_id": st["init_state_id"], "cond": cid, "lib": lname,
                                 "true_err": e.tolist(), "succ": rk["true_success_rate"], "spread": rk["spread_err"],
                                 "all_fail": rk["all_fail"], "oracle_eval_err": rk["oracle_eval_err"],
                                 "oracle_select_err": rk["oracle_select_err"], "median_err": float(np.median(e)),
                                 "diversity": M.candidate_diversity(C), "pairwise_acc": acc, "n_pairs": n, "ties": ties,
                                 "regret_a8prior": M.top1_regret(e, s), "top1_correct_a8prior": M.top1_correct(e, s),
                                 "endpoint_err_a8prior": float(np.mean(np.linalg.norm(ep - np.asarray(rk["true_endpoints"]), axis=1)))})
    summ = {}
    for noise in (0.0, 0.02):
        for lname in ("A5_fixed4", "STRIKE4", "A5_release4"):
            for cid in list(conds) + ["ALL"]:
                R = [r for r in rows if r["noise"] == noise and r["lib"] == lname and (cid == "ALL" or r["cond"] == cid)]
                def ci(f):
                    vb = {}
                    for r in R: vb.setdefault(r["init_state_id"], []).append(r[f])
                    return [round(x, 4) for x in M.cluster_bootstrap(vb)]
                gap = {}
                for r in R: gap.setdefault(r["init_state_id"], []).append(r["median_err"] - r["oracle_eval_err"])
                pa = [r for r in R if not np.isnan(r["pairwise_acc"])]
                summ[f"noise{noise}/{lname}/{cid}"] = {
                    "n_rows": len(R), "oracle_eval_err": ci("oracle_eval_err"), "median_err": ci("median_err"),
                    "headroom_median_minus_oracle": [round(x, 4) for x in M.cluster_bootstrap(gap)],
                    "all_fail_frac": float(np.mean([r["all_fail"] for r in R])),
                    "spread_mean": float(np.mean([r["spread"] for r in R])),
                    "oracle_succ_frac": float(np.mean([max(r["succ"]) == 1.0 for r in R])),
                    "pairwise_acc_a8prior": float(np.mean([r["pairwise_acc"] for r in pa])) if pa else None,
                    "regret_a8prior": ci("regret_a8prior"), "top1_correct_a8prior": float(np.mean([r["top1_correct_a8prior"] for r in R])),
                    "endpoint_err_a8prior": float(np.mean([r["endpoint_err_a8prior"] for r in R])),
                    "diversity_mean": float(np.mean([r["diversity"] for r in R])),
                    "select_minus_eval_oracle_err": float(np.mean([r["oracle_select_err"] - r["oracle_eval_err"] for r in R]))}
    out = {"label": "[toy-run]", "time": time.strftime("%F %T"), "rows": rows, "summary": summ,
           "reps": {"select": [11, 12, 13], "eval": [21, 22, 23]}}
    json.dump(to_jsonable(out), open(os.path.join(ROOT, "runs", "a10", "e1.json"), "w"), indent=1)
    for k, v in summ.items():
        if k.endswith("ALL") or "noise0.0" in k:
            print(k, {kk: v[kk] for kk in ("headroom_median_minus_oracle", "all_fail_frac", "oracle_succ_frac", "pairwise_acc_a8prior", "regret_a8prior", "top1_correct_a8prior")})

if __name__ == "__main__":
    main()
