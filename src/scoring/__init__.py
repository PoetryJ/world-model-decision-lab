"""A7 scoring / uncertainty (exclusive). Implements shared Scorer protocol.

Importable API (frozen for the controller's sealed test):

    from src.scoring import (
        make_scorer, make_gate, load_scorer_config, pad_for_settle,
        score_detailed, choose_index, ScoreBundle,
        GoalDistanceScorer, LCBScorer, WorstCaseDistanceScorer,
        RobustSuccessThenDistanceScorer,
    )
    from src.dynamics.model import load_or_train   # A6 v3
    cfg = load_scorer_config()                     # reports/a7/frozen_scorer_config.json
    pred = load_or_train(seed=0, retrain=False)
    P = pred.predict(obs, pad_for_settle(cands), {"history": hist, "adapt_state": st})
    scores = make_scorer().score(P, {"goal": obs["goal"], "goal_radius": obs["goal_radius"]})
    idx = choose_index(scores)                     # higher = better; ties -> lowest index
    gate = make_gate()                             # optional; thresholds fit on TRAIN only
    abstain = gate.should_abstain(score_detailed(P, goal, score_name=cfg["default_score_name"]))

Official Cosmos aggregation (LCB, majority_mean, average, success_vote) is a BASELINE
from cosmos_utils.aggregate_value_predictions @ 18a2acc, not novelty.
LCB needs S>=2 samples (A6 provides S=5). Official LCB at S=1 is NaN; safe fallback is mean.
Ensemble variance is a gating FEATURE, not a calibrated confidence.

Never read env.hidden. Never load A10 test / test_neg / test_sealed* splits from this package.
"""
from src.scoring.scorers import (
    VALUE_SCALE, LCB_BETA_OFFICIAL, MAJORITY_THRESH_OFFICIAL, GAMMA_DEFAULT,
    GoalDistanceScorer, SampleMeanDistanceScorer, SuccessProbScorer,
    DiscountedValueScorer, AverageValueScorer, LCBScorer, MajorityMeanScorer,
    SuccessVoteScorer, CompositeScorer,
    WorstCaseValueScorer, WorstCaseDistanceScorer,
    RobustSuccessScorer, RobustSuccessThenDistanceScorer,
    score_detailed, ScoreBundle, per_sample_values, per_sample_distances,
    choose_index, official_aggregate,
)
from src.scoring.stub_predictor import WrongMuStubPredictor
from src.scoring.pushenv_stub import PushEnvStubPredictor
from src.scoring.gate import LowConfidenceGate
from src.scoring.calibration import expected_calibration_error, reliability_table
from src.scoring.config import (
    load_scorer_config, make_scorer, make_gate, pad_for_settle,
    SETTLE_PAD, CONDITIONS, MAIN_PAIR, DEFAULT_CONFIG_PATH,
)

__all__ = [
    "VALUE_SCALE", "LCB_BETA_OFFICIAL", "MAJORITY_THRESH_OFFICIAL", "GAMMA_DEFAULT",
    "GoalDistanceScorer", "SampleMeanDistanceScorer", "SuccessProbScorer",
    "DiscountedValueScorer", "AverageValueScorer", "LCBScorer", "MajorityMeanScorer",
    "SuccessVoteScorer", "CompositeScorer",
    "WorstCaseValueScorer", "WorstCaseDistanceScorer",
    "RobustSuccessScorer", "RobustSuccessThenDistanceScorer",
    "score_detailed", "ScoreBundle", "choose_index", "official_aggregate",
    "per_sample_values", "per_sample_distances",
    "WrongMuStubPredictor", "PushEnvStubPredictor",
    "LowConfidenceGate", "expected_calibration_error", "reliability_table",
    "load_scorer_config", "make_scorer", "make_gate", "pad_for_settle",
    "SETTLE_PAD", "CONDITIONS", "MAIN_PAIR", "DEFAULT_CONFIG_PATH",
]
