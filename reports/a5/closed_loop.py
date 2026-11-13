"""Closed-loop execution of Best-of-N with a chosen chunk length H (replan every H steps).

Used for the long-vs-short horizon comparison. Total physical time is the env
max_time; short H just replans more often. Probe-free: every decision is kind=task.
"""
from __future__ import annotations

import time

import numpy as np

from shared.interface import DecisionRecord, to_jsonable


def run_episode(
    env,
    planner,
    init_state: dict,
    hidden: dict,
    seed: int,
    *,
    init_state_id: str,
    hidden_condition_id: str,
    method: str,
    run_id: str,
    history=None,
    max_decisions: int = 40,
    frozen_first=None,
) -> dict:
    """Reset env, replan until done. Returns summary + decision records.

    frozen_first: optional (K,H,2) used only for the first decision (E2-style);
    subsequent replans call the proposer (needed for short-H receding horizon).
    """
    history = [] if history is None else list(history)
    obs = env.reset(init_state, hidden, int(seed))
    records = []
    total_wall = {"propose": 0.0, "predict": 0.0, "score": 0.0, "update": 0.0, "probe": 0.0, "env": 0.0}
    t_ep = time.perf_counter()
    termination = None
    success = False
    n_env_steps = 0
    n_plans = 0
    for d in range(max_decisions):
        if env.done:
            break
        cands = frozen_first if (d == 0 and frozen_first is not None) else None
        plan = planner.plan(obs, history, seed=int(seed) + d, candidates=cands)
        n_plans += 1
        for k, v in plan["wall_time_s"].items():
            total_wall[k] = total_wall.get(k, 0.0) + v
        executed = plan["executed"]
        t0 = time.perf_counter()
        steps = env.step_segment(executed)
        total_wall["env"] += time.perf_counter() - t0
        last = steps[-1]
        obs_after = last.obs
        executed_h = np.asarray(executed[: len(steps)], dtype=np.float64)
        termination = last.info.get("termination") if last.info else None
        success = bool(last.success)
        rec = DecisionRecord(
            run_id=run_id,
            episode_id=f"{init_state_id}:{hidden_condition_id}:{method}:seed{seed}",
            method=method,
            init_state_id=init_state_id,
            hidden_condition_id=hidden_condition_id,
            seed=int(seed) + d,
            step_index=d,
            kind="task",
            candidates=to_jsonable(plan["candidates"]),
            predictions=to_jsonable(plan["predictions"].endpoints),
            scores=to_jsonable(plan["scores"]),
            chosen_index=int(plan["chosen_index"]),
            executed=to_jsonable(executed_h),
            obs_before=to_jsonable(obs),
            obs_after=to_jsonable(obs_after),
            wall_time_s={k: float(v) for k, v in plan["wall_time_s"].items()},
            termination=termination,
        )
        records.append(rec)
        from shared.interface import Transition

        history.append(Transition(obs=obs, actions=executed_h, next_obs=obs_after, kind="task"))
        obs = obs_after
        n_env_steps += len(steps)
        if last.done:
            break
    wall_ep = time.perf_counter() - t_ep
    final = env.observe()
    goal_error = float(np.linalg.norm(np.asarray(final["obj"]) - np.asarray(final["goal"])))
    compute_s = total_wall["propose"] + total_wall["predict"] + total_wall["score"] + total_wall.get("update", 0.0)
    return {
        "init_state_id": init_state_id,
        "hidden_condition_id": hidden_condition_id,
        "method": method,
        "seed": int(seed),
        "success": bool(success),
        "termination": termination,
        "goal_error": goal_error,
        "n_plans": n_plans,
        "n_env_steps": n_env_steps,
        "t_final": float(final["t"]),
        "wall_time_s": {k: float(v) for k, v in total_wall.items()},
        "compute_s": float(compute_s),
        "episode_wall_s": float(wall_ep),
        "records": records,
        "final_obj": np.asarray(final["obj"], dtype=np.float64).tolist(),
        "final_goal": np.asarray(final["goal"], dtype=np.float64).tolist(),
    }
