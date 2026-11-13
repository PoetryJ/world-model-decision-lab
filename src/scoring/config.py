"""Frozen A7 scorer configuration and importable constructors.

Fit on A10 train only; val is for reporting. The controller runs the sealed test.
Never load test / test_neg / test_sealed* from this module.
"""
from __future__ import annotations
import json, os
from src.scoring.scorers import (
    CompositeScorer, GoalDistanceScorer, LCBScorer, AverageValueScorer,
    WorstCaseDistanceScorer, WorstCaseValueScorer, RobustSuccessThenDistanceScorer,
    SuccessProbScorer, LCB_BETA_OFFICIAL, VALUE_SCALE, GAMMA_DEFAULT,
    MAJORITY_THRESH_OFFICIAL, score_detailed, choose_index,
)
from src.scoring.gate import LowConfidenceGate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "reports", "a7", "frozen_scorer_config.json")

# Binding conditions (PROTOCOL v2 / A-7 / A-8). Mid is held out of A6 train mus and is reported separately.
CONDITIONS = {"mu_lo": {"mu": 0.1}, "mu_mid": {"mu": 0.3}, "mu_hi": {"mu": 0.6}}
MAIN_PAIR = ("mu_lo", "mu_hi")
SETTLE_PAD = 20  # zero actions to append so a predictor scores the settled endpoint (A6/A10)

SCORER_CTORS = {
    "goal_distance": GoalDistanceScorer,
    "average_value": AverageValueScorer,
    "lcb": LCBScorer,
    "worst_case_value": WorstCaseValueScorer,
    "worst_case_distance": WorstCaseDistanceScorer,
    "robust_success_then_distance": RobustSuccessThenDistanceScorer,
    "success_prob": SuccessProbScorer,
}


def load_scorer_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        return json.load(f)


def make_scorer(name: str | None = None, config: dict | None = None, **kwargs):
    """Build a Scorer protocol object. Default name comes from the frozen config if present."""
    cfg = config if config is not None else (_try_cfg() or {})
    name = name or cfg.get("default_score_name", "lcb")
    if name not in SCORER_CTORS:
        return CompositeScorer(name, **kwargs)
    ctor = SCORER_CTORS[name]
    if name == "lcb":
        beta = kwargs.pop("beta", cfg.get("lcb_beta", LCB_BETA_OFFICIAL))
        safe = kwargs.pop("safe_single_sample", cfg.get("lcb_safe_single", True))
        return ctor(beta=beta, safe_single_sample=safe, **kwargs)
    return ctor(**kwargs)


def make_gate(config: dict | None = None) -> LowConfidenceGate | None:
    cfg = config if config is not None else (_try_cfg() or {})
    g = cfg.get("gate")
    if not g or g.get("threshold") is None:
        return None
    return LowConfidenceGate(
        feature=g["feature"],
        threshold=float(g["threshold"]),
        fitted_on=g.get("fitted_on", "train"),
        utility_lambda=float(g.get("probe_cost", 0.4)),
        n_fit=int(g.get("n_fit", 0)),
        logistic_ab=tuple(g["logistic_ab"]) if g.get("logistic_ab") else None,
    )


def pad_for_settle(actions, settle_steps: int = SETTLE_PAD):
    """Pad a (K,H,2) or (H,2) action array with zero-action settle steps (A6/A10)."""
    import numpy as np
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 2:
        return np.vstack([a, np.zeros((settle_steps, 2))])
    if a.ndim == 3:
        K = a.shape[0]
        return np.concatenate([a, np.zeros((K, settle_steps, 2))], axis=1)
    raise ValueError(a.shape)


def _try_cfg():
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        return load_scorer_config()
    return None
