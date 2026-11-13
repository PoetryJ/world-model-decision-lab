"""Closed-loop episode with optional probing. Injected env/proposer/predictor/adapter/scorer/prober.
Rules enforced here:
- After a probe the task continues from the ACTUAL new env state (no restore/reset inside an episode).
- Probe steps are charged to the shared step budget (steps_left) AND to probe_steps_left.
- Policy-side modules receive only observations/history (never env.hidden). hidden is passed only to
  env.reset by this function (caller supplies it; in evaluation it comes from the frozen test set).
- One DecisionRecord (shared.interface) per decision, JSON-serialisable via to_jsonable.
Seeds: per-query proposer seed = SeedSequence([seed, decision_index]).generate_state(1)[0]; recorded in
DecisionRecord.candidate_counts (interface v1.1) = {requested, returned, failed, seeds}. If the proposer exposes
`last_seeds` (per-candidate seeds) those are recorded instead of the single query seed.
"""
from __future__ import annotations
import time
from dataclasses import asdict
import numpy as np
from shared.interface import DecisionRecord, Transition, to_jsonable

def _seed(seed, k):
    return int(np.random.SeedSequence([int(seed), int(k)]).generate_state(1)[0])

def run_episode(env, init_state, hidden, seed, proposer, predictor, adapter, scorer, prober,
                step_budget=30, probe_budget=8, method="", run_id="", episode_id="", init_state_id="",
                hidden_condition_id="", log=None, max_decisions=100, max_task_decisions=None):
    """max_task_decisions: if set (e.g. 1 = 'one-shot strike'), after that many task segments the loop only
    holds still (zero commands, counted against step_budget) until termination -- no corrective pushes."""
    obs = env.reset(init_state, hidden, seed)
    history, records = [], []
    steps_used = probe_used = 0
    termination = None; done = False; success = False; k = 0
    t_comp = {"propose": 0.0, "predict": 0.0, "update": 0.0, "score": 0.0, "probe": 0.0}
    while not done and steps_used < step_budget and k < max_decisions:
        budget = {"steps_left": step_budget - steps_used, "probe_steps_left": probe_budget - probe_used}
        wt = {"propose": 0.0, "predict": 0.0, "update": 0.0, "score": 0.0, "probe": 0.0}
        t0 = time.perf_counter()
        probe = prober.choose_probe(obs, history, budget) if budget["probe_steps_left"] > 0 else None
        wt["probe"] = time.perf_counter() - t0
        cands = preds = scores = chosen = cc = None
        n_task = sum(t.kind == "task" for t in history)
        if max_task_decisions is not None and n_task >= max_task_decisions:
            probe = None
        settle_mode = max_task_decisions is not None and n_task >= max_task_decisions
        if settle_mode:
            seg = None; kind = "task"   # A-3 settle: zero actions until |obj_vel| < 1e-3 (or env terminates / cap)
        elif probe is not None:
            seg = np.asarray(probe, float)[: min(budget["steps_left"], budget["probe_steps_left"])]
            kind = "probe"
        else:
            t0 = time.perf_counter(); st = adapter.update(history); wt["update"] = time.perf_counter() - t0
            qseed = _seed(seed, k)   # candidate seed for query index k (recorded)
            t0 = time.perf_counter(); C = np.asarray(proposer.propose(obs, history, qseed), float)
            wt["propose"] = time.perf_counter() - t0
            req = int(getattr(proposer, "K", len(C)) or len(C))
            ok = np.all(np.isfinite(C.reshape(len(C), -1)), axis=1) if len(C) else np.zeros(0, bool)
            C = C[ok]   # non-finite candidates are dropped and counted as failed
            cc = {"requested": req, "returned": int(len(C)), "failed": int(req - len(C)),
                  "seeds": to_jsonable(getattr(proposer, "last_seeds", [qseed]))}
            t0 = time.perf_counter(); P = predictor.predict(obs, C, {"history": history, "adapt_state": st})
            wt["predict"] = time.perf_counter() - t0
            t0 = time.perf_counter(); S = np.asarray(scorer.score(P, {"goal": obs["goal"], "goal_radius": obs["goal_radius"]}))
            wt["score"] = time.perf_counter() - t0
            chosen = int(np.argmax(S))   # ties -> lowest index
            seg = C[chosen][: budget["steps_left"]]
            cands, preds, scores, kind = C, P.endpoints, S, "task"
        if settle_mode:
            results = []
            while steps_used + len(results) < step_budget:
                if np.linalg.norm(np.asarray((results[-1].obs if results else obs)["obj_vel"])) < 1e-3:
                    break
                results.append(env.step(np.zeros(2)))
                if results[-1].done: break
            executed = np.zeros((len(results), 2))
        else:
            results = env.step_segment(seg)
            executed = seg[: len(results)]
        nobs = results[-1].obs if results else obs
        steps_used += len(results)
        if kind == "probe": probe_used += len(results)
        done = bool(results and results[-1].done); success = bool(results and results[-1].success)
        termination = results[-1].info.get("termination") if done else None
        if not done and steps_used >= step_budget: termination = "budget"
        if settle_mode and not done: termination = "budget"; done = True   # settled outside goal (A-3)
        history.append(Transition(obs=obs, actions=executed, next_obs=nobs, kind=kind))
        rec = DecisionRecord(run_id=run_id, episode_id=episode_id, method=method, init_state_id=init_state_id,
                             hidden_condition_id=hidden_condition_id, seed=seed, step_index=steps_used - len(results),
                             kind=kind, candidates=to_jsonable(cands), predictions=to_jsonable(preds),
                             scores=to_jsonable(scores), chosen_index=chosen, executed=to_jsonable(executed),
                             obs_before=to_jsonable(obs), obs_after=to_jsonable(nobs), wall_time_s=wt,
                             termination=termination, candidate_counts=cc)
        d = asdict(rec); d["prober_info"] = to_jsonable(getattr(prober, "last", {}))
        records.append(d)
        if log is not None: log(d)
        for kk in t_comp: t_comp[kk] += wt[kk]
        obs = nobs; k += 1
    if termination is None: termination = "max_decisions"
    final_err = float(np.linalg.norm(np.asarray(obs["obj"]) - np.asarray(obs["goal"])))
    summary = {"run_id": run_id, "episode_id": episode_id, "method": method, "init_state_id": init_state_id,
               "hidden_condition_id": hidden_condition_id, "seed": seed, "success": success,
               "termination": termination, "final_goal_err": final_err, "steps_used": steps_used,
               "probe_steps": probe_used, "n_probes": sum(t.kind == "probe" for t in history),
               "phys_time_s": steps_used * 0.1, "compute_s": t_comp, "n_decisions": k}
    return summary, records
