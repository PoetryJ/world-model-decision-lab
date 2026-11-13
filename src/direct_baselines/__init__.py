"""A9 direct baselines (no explicit dynamics / future prediction).

Public API for the E2/E3 runner (see reports/a9/STATUS.md):

    from src.direct_baselines.selectors import (
        DirectProposer,          # Proposer: propose(obs, history, seed) -> (K,H,2)
        ObsOnlySelector,         # score_candidates(obs, history, cands) ignores history
        HistoryDirectSelector,   # score_candidates(obs, history, cands) uses probe feedback
        load_trained_selectors,  # load weights from runs/a9/models
    )

Scorer.score(predictions, goal) is also implemented as an adapter: it requires
predictions.extra to contain observation, history, and candidates (the frozen
Scorer protocol has no slot for those). Prefer score_candidates(...).
"""
from .selectors import (  # noqa: F401
    DirectProposer,
    ObsOnlySelector,
    HistoryDirectSelector,
    HeuristicObsSelector,
    HeuristicHistorySelector,
    PhysicsObsSelector,
    PhysicsHistorySelector,
    RandomSelector,
    load_trained_selectors,
)

__all__ = [
    "load_frozen_a9",
    "decide",
    "DirectProposer",
    "ObsOnlySelector",
    "HistoryDirectSelector",
    "HeuristicObsSelector",
    "HeuristicHistorySelector",
    "RandomSelector",
    "load_trained_selectors",
]

from .api import load_frozen_a9, decide  # noqa: F401
