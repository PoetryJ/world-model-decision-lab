"""Importable A9 API for the controller's sealed E2/E3 run (POLICY-SIDE).

This module never loads evaluation splits, never touches env internals, and never
takes a physical-parameter argument. The runner must:
  - construct the env and call env.reset itself
  - optionally execute ONE A4 probe (tap_fast / push_slow / retreat, 5 steps)
    from the actual state, then append a shared.interface.Transition
  - call decide(...) for the single A-3 task decision
  - execute the returned segment fully (unless the env terminates)
  - settle with zero actions until |obj_vel| < 1e-3, no further non-zero actions

Example (runner-side; the runner owns reset and any physical parameters):

    from src.direct_baselines.api import load_frozen_a9, decide
    bundle = load_frozen_a9()
    cands = bundle["proposer"].propose(obs, history, seed)          # (54, 6, 2)
    out = decide(bundle["physics_history"], obs, history, cands)    # or history_direct / obs_only
    env.step_segment(out["executed"])

Recommended default for the world-model comparison: bundle["physics_history"]
(history-conditioned mechanical scorer) and bundle["history_direct"] (learned ridge).
Compare both against bundle["obs_only"] and bundle["physics_obs"] (no history).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .candidates import DirectProposer
from .models import RidgeScorerModel
from .selectors import (
    ObsOnlySelector, HistoryDirectSelector,
    HeuristicObsSelector, HeuristicHistorySelector,
    PhysicsObsSelector, PhysicsHistorySelector, RandomSelector,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "runs" / "a9" / "frozen_strike54" / "models"
DEFAULT_SOURCE = "strikegrid"


def load_frozen_a9(model_dir: str | Path | None = None, candidate_source: str = DEFAULT_SOURCE) -> dict:
    """Load frozen ridge weights + always-available heuristics. Safe to call without any split."""
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    # official frozen ridge was trained with INCLUDE_PHYS_FEATURES=False
    import src.direct_baselines.features as feat
    feat.INCLUDE_PHYS_FEATURES = False
    bundle = {
        "proposer": DirectProposer(source=candidate_source),
        "physics_obs": PhysicsObsSelector(),
        "physics_history": PhysicsHistorySelector(),
        "heuristic_obs": HeuristicObsSelector(),
        "heuristic_history": HeuristicHistorySelector(),
        "candidate_source": candidate_source,
        "model_dir": str(model_dir),
        "episode": "A-3: optional one probe <=5 steps, one task segment H<=12, settle zeros, cap 60",
    }
    obs_p = model_dir / "obs_only.npz"
    hist_p = model_dir / "history_direct.npz"
    if obs_p.exists():
        bundle["obs_only"] = ObsOnlySelector(RidgeScorerModel.load(obs_p))
        bundle["obs_only_n_params"] = bundle["obs_only"].n_params
    if hist_p.exists():
        bundle["history_direct"] = HistoryDirectSelector(RidgeScorerModel.load(hist_p))
        bundle["history_direct_n_params"] = bundle["history_direct"].n_params
    return bundle


def decide(selector, observation, history, candidates, seed: int | None = None) -> dict:
    """Score the shared candidate array and pick an index (ties: lowest index).

    Returns dict(scores, chosen_index, executed, score_time_s, candidate_counts).
    Does not step the env.
    """
    cands = np.asarray(candidates, dtype=np.float64)
    t0 = time.perf_counter()
    scores = np.asarray(selector.score_candidates(observation, history, cands), dtype=np.float64)
    dt = time.perf_counter() - t0
    idx = int(np.argmax(scores))
    executed = cands[idx]
    return {
        "scores": scores,
        "chosen_index": idx,
        "executed": executed,
        "score_time_s": dt,
        "candidate_counts": {
            "requested": int(cands.shape[0]),
            "returned": int(cands.shape[0]),
            "failed": 0,
        },
    }


# Back-compat alias used in STATUS.md
load_trained_selectors = load_frozen_a9
