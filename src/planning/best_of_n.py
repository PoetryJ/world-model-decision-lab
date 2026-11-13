"""Simple Best-of-N planner: any Proposer + Predictor + Scorer.

propose -> predict -> score -> argmax (ties: smallest index).
Does not read live-env parameters. Candidate arrays can be injected (frozen sets) so
every scorer is compared on identical actions.
"""
from __future__ import annotations

import time

import numpy as np

from shared.interface import clip_action


def _goal_from_obs(observation: dict) -> dict:
    return {
        "goal": np.asarray(observation["goal"], dtype=np.float64),
        "goal_radius": float(observation.get("goal_radius", 0.05)),
    }


class BestOfNPlanner:
    def __init__(self, proposer, predictor, scorer):
        self.proposer = proposer
        self.predictor = predictor
        self.scorer = scorer

    def plan(self, observation, history, seed: int, candidates=None, context=None) -> dict:
        """Return chosen action segment and diagnostics.

        If `candidates` is provided, the proposer is skipped (use this for frozen sets).
        """
        wall = {"propose": 0.0, "predict": 0.0, "score": 0.0, "update": 0.0, "probe": 0.0}
        if candidates is None:
            t0 = time.perf_counter()
            cands = self.proposer.propose(observation, history, int(seed))
            wall["propose"] = time.perf_counter() - t0
        else:
            cands = np.asarray(candidates, dtype=np.float64)
        cands = clip_action(cands)
        if context is None:
            context = {"history": history, "adapt_state": None}
        t0 = time.perf_counter()
        pred = self.predictor.predict(observation, cands, context)
        wall["predict"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = np.asarray(self.scorer.score(pred, _goal_from_obs(observation)), dtype=np.float64)
        wall["score"] = time.perf_counter() - t0
        if scores.shape[0] != cands.shape[0]:
            raise ValueError(f"scorer returned {scores.shape}, expected ({cands.shape[0]},)")
        # Tie-break: smallest index (np.argmax / max with first occurrence).
        chosen = int(np.argmax(scores))
        return {
            "chosen_index": chosen,
            "executed": cands[chosen].copy(),
            "candidates": cands,
            "predictions": pred,
            "scores": scores,
            "wall_time_s": wall,
        }

    def act(self, observation, history, seed: int, candidates=None, context=None) -> np.ndarray:
        return self.plan(observation, history, seed, candidates=candidates, context=context)["executed"]
