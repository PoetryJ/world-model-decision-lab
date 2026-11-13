"""Proposer / Scorer-compatible selectors for E2/E3.

Policy-side: never reads env.hidden or evaluator internals. Never branches the env.

Primary API (preferred by the runner):
    scores = selector.score_candidates(observation, history, candidates)  # (K,), higher=better
    idx = int(np.argmax(scores))   # ties: lowest index

Scorer protocol adapter:
    selector.score(predictions, goal)
    requires predictions.extra["observation"], extra["history"], extra["candidates"]
    (or extra["actions"]). goal is unused beyond what's already in the observation.

See reports/a9/interface_requests.md and reports/a9/STATUS.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.interface import Prediction
from .candidates import DirectProposer, load_shared_candidates, default_task_candidates
from .features import _v, _unit, candidate_features, history_features
from .models import RidgeScorerModel

EPS = 1e-9
TIE_NOTE = "score ties broken by lowest index"


def _argmax_stable(scores: np.ndarray) -> int:
    """Ties -> lowest index."""
    s = np.asarray(scores, dtype=np.float64)
    return int(np.argmax(s))  # numpy argmax returns first max


class _RidgeSelectorBase:
    def __init__(self, model: RidgeScorerModel):
        self.model = model

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        cands = np.asarray(candidates, dtype=np.float64)
        return self.model.score_candidates(observation, history, cands)

    def select(self, observation, history, candidates) -> tuple[np.ndarray, int]:
        scores = self.score_candidates(observation, history, candidates)
        return scores, _argmax_stable(scores)

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        extra = predictions.extra or {}
        obs = extra.get("observation") or extra.get("obs")
        history = extra.get("history", [])
        cands = extra.get("candidates") or extra.get("actions")
        if obs is None or cands is None:
            raise ValueError(
                "Direct baseline Scorer.score requires predictions.extra to contain "
                "'observation' and 'candidates' (and optionally 'history'). "
                "Prefer score_candidates(observation, history, candidates)."
            )
        if goal and "goal" in goal and "goal" not in obs:
            obs = dict(obs)
            obs["goal"] = goal["goal"]
            if "goal_radius" in goal:
                obs["goal_radius"] = goal["goal_radius"]
        return self.score_candidates(obs, history, cands)

    @property
    def n_params(self) -> int:
        return self.model.n_params


class ObsOnlySelector(_RidgeSelectorBase):
    """(a) Current-observation-only scorer. History is accepted then ignored."""

    method = "obs_only"

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        # Explicitly drop history so a mistaken use_history flag cannot leak in.
        cands = np.asarray(candidates, dtype=np.float64)
        return self.model.score_candidates(observation, history=None, candidates=cands)


class HistoryDirectSelector(_RidgeSelectorBase):
    """(b) History-conditioned direct scorer. No explicit dynamics model."""

    method = "history_direct"


class HeuristicObsSelector:
    """No-training baseline: prefer candidates whose commanded displacement
    (assuming the object is carried with the pusher, no slide) lands closer to the goal.
    """

    method = "heuristic_obs"
    n_params = 0

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        obj = _v(observation["obj"])
        goal = _v(observation["goal"])
        scores = []
        for a in np.asarray(candidates, dtype=np.float64):
            cf = candidate_features(observation, a)
            disp = np.array([cf[6], cf[7]])
            # If contact looks unlikely, assume the object barely moves.
            contact = cf[19]
            pred = obj + contact * disp
            scores.append(-float(np.linalg.norm(pred - goal)))
        return np.asarray(scores, dtype=np.float64)

    def select(self, observation, history, candidates):
        s = self.score_candidates(observation, history, candidates)
        return s, _argmax_stable(s)

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        extra = predictions.extra or {}
        return self.score_candidates(extra["observation"], extra.get("history", []), extra["candidates"])


class HeuristicHistorySelector:
    """No-training baseline: scale commanded travel by last-probe mobility
    (|obj_disp|/|pusher_disp|). Falls back to contact=1 travel if history is empty.
    This is a feature heuristic, not a fitted dynamics model.
    """

    method = "heuristic_history"
    n_params = 0

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        obj = _v(observation["obj"])
        goal = _v(observation["goal"])
        hf = history_features(history)
        mobility = float(hf[9]) if hf[0] > 0.5 else 1.0
        # bound mobility so a weird probe cannot explode the score
        mobility = float(np.clip(mobility, 0.0, 2.0))
        scores = []
        for a in np.asarray(candidates, dtype=np.float64):
            cf = candidate_features(observation, a)
            disp = np.array([cf[6], cf[7]])
            pred = obj + mobility * disp
            scores.append(-float(np.linalg.norm(pred - goal)))
        return np.asarray(scores, dtype=np.float64)

    def select(self, observation, history, candidates):
        s = self.score_candidates(observation, history, candidates)
        return s, _argmax_stable(s)

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        extra = predictions.extra or {}
        return self.score_candidates(extra["observation"], extra.get("history", []), extra["candidates"])


class RandomSelector:
    """Uniform random among K candidates. Explicit seed required."""

    method = "random"
    n_params = 0

    def __init__(self, seed: int):
        self.seed = int(seed)

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        import hashlib
        k = len(np.asarray(candidates))
        obj = np.asarray(observation["obj"], dtype=np.float64).tobytes()
        t = str(observation.get("t", 0.0)).encode()
        hid = str(len(history) if history is not None else 0).encode()
        digest = hashlib.sha256(self.seed.to_bytes(8, "little") + obj + t + hid).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**31))
        return rng.normal(size=k)

    def select(self, observation, history, candidates):
        s = self.score_candidates(observation, history, candidates)
        return s, _argmax_stable(s)

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        extra = predictions.extra or {}
        return self.score_candidates(extra["observation"], extra.get("history", []), extra["candidates"])


def load_trained_selectors(model_dir: str | Path | None = None) -> dict[str, Any]:
    """Load saved ridge weights. Default path: runs/a9/models/."""
    root = Path(__file__).resolve().parents[2]
    model_dir = Path(model_dir) if model_dir else (root / "runs" / "a9" / "models")
    out = {
        "proposer": DirectProposer(),
        "heuristic_obs": HeuristicObsSelector(),
        "heuristic_history": HeuristicHistorySelector(),
        "physics_obs": PhysicsObsSelector(),
        "physics_history": PhysicsHistorySelector(),
    }
    obs_p = model_dir / "obs_only.npz"
    hist_p = model_dir / "history_direct.npz"
    if obs_p.exists():
        out["obs_only"] = ObsOnlySelector(RidgeScorerModel.load(obs_p))
    if hist_p.exists():
        out["history_direct"] = HistoryDirectSelector(RidgeScorerModel.load(hist_p))
    return out


class PhysicsObsSelector:
    """Nominal-decel mechanical scorer (fixed mu prior 0.25). Ignores history."""

    method = "physics_obs"
    n_params = 1  # the fixed prior, not fitted

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        from .physics_heuristic import score_candidates_physics
        return score_candidates_physics(observation, history, candidates, use_history=False)

    def select(self, observation, history, candidates):
        s = self.score_candidates(observation, history, candidates)
        return s, _argmax_stable(s)

    def score(self, predictions, goal):
        extra = predictions.extra or {}
        return self.score_candidates(extra["observation"], extra.get("history", []), extra["candidates"])


class PhysicsHistorySelector:
    """History-conditioned mechanical scorer: estimates a from probe release, else nominal prior.

    Sees the same probe feedback as the world-model route (Transition.obs / actions / next_obs).
    """

    method = "physics_history"
    n_params = 1  # estimated a, or the nominal prior

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        from .physics_heuristic import score_candidates_physics
        return score_candidates_physics(observation, history, candidates, use_history=True)

    def select(self, observation, history, candidates):
        s = self.score_candidates(observation, history, candidates)
        return s, _argmax_stable(s)

    def score(self, predictions, goal):
        extra = predictions.extra or {}
        return self.score_candidates(extra["observation"], extra.get("history", []), extra["candidates"])
