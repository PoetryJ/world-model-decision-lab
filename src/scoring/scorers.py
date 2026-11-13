"""Scorers and official conservative aggregation baselines.

DISTINCTIONS (kept explicit; do not collapse into one number):
- discounted_value: time-discounted shaped return in [0, 1] (or endpoint-only
  gamma^H * v_end when no trajectory is provided).
- success_probability: fraction of samples whose endpoint is inside the goal
  radius. This is NOT a discounted return.
- candidate ranking: argsort of a chosen score (higher = better). Two scorers
  may agree on ranking while disagreeing on absolute value.
- prediction credibility / uncertainty: disagreement among samples (endpoint
  RMS std, value std, argmax agreement). This is NOT a value, and ensemble
  variance is NOT claimed to be a calibrated confidence.

Official Cosmos Policy aggregation (BASELINE, not novelty), from
adapters/cosmos/src/cosmos_policy/experiments/robot/cosmos_utils.py
function aggregate_value_predictions (lines 766-822) @ 18a2acc:
  average: mean
  lcb: mean - beta * std, beta=1.0, then clamp to [0,1]
       std is UNBIASED (torch.std default correction=1 == np.std ddof=1).
       With S=1, official std is NaN, so official LCB is NaN.
  success_vote: mean(v > 0.05)
  majority_mean: if fraction(v > 0.05) >= 0.5, mean of those v, else 0;
                 then clamp to [0,1]
The official 0.05 threshold is a FIXED value-scale constant. On our shaped
distance value v = clip(1 - dist/0.5, 0, 1), 0.05 corresponds to dist < 0.475 m
and is almost always true — this is reported as a scale sensitivity, not a bug
fix silently patched into the official baseline.

Value mapping used here (toy, documented):
  dist = ||endpoint - goal||
  v = clip(1 - dist / VALUE_SCALE, 0, 1), VALUE_SCALE = 0.5 m (table half-width).
  This is a shaped [0,1] stand-in for Cosmos's latent value (also clamped [0,1]).
  It is NOT the official Cosmos value head.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from shared.interface import Prediction

VALUE_SCALE = 0.5          # metres; v = clip(1 - dist/VALUE_SCALE, 0, 1)
LCB_BETA_OFFICIAL = 1.0
MAJORITY_THRESH_OFFICIAL = 0.05
GAMMA_DEFAULT = 0.95       # per control step (DT = 0.1 s)
TIE_ATOL = 1e-9


def _goal_xy_r(goal: dict) -> tuple[np.ndarray, float]:
    if "goal" not in goal:
        raise KeyError("goal dict must contain 'goal' (2,) and 'goal_radius'")
    g = np.asarray(goal["goal"], dtype=np.float64).reshape(2)
    r = float(goal["goal_radius"])
    return g, r


def _samples_or_mean(pred: Prediction) -> np.ndarray:
    """Return (K, S, 2). If no samples, S=1 from the mean endpoint."""
    if pred.samples is not None:
        arr = np.asarray(pred.samples, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(f"samples must be (K,S,2), got {arr.shape}")
        return arr
    ep = np.asarray(pred.endpoints, dtype=np.float64)
    if ep.ndim != 2 or ep.shape[-1] != 2:
        raise ValueError(f"endpoints must be (K,2), got {ep.shape}")
    return ep[:, None, :]


def per_sample_distances(pred: Prediction, goal: dict) -> np.ndarray:
    """(K, S) Euclidean distance of each sample endpoint to the goal centre."""
    g, _ = _goal_xy_r(goal)
    samples = _samples_or_mean(pred)
    return np.linalg.norm(samples - g[None, None, :], axis=-1)


def dist_to_value(dist: np.ndarray, value_scale: float = VALUE_SCALE) -> np.ndarray:
    return np.clip(1.0 - np.asarray(dist, dtype=np.float64) / value_scale, 0.0, 1.0)


def per_sample_values(pred: Prediction, goal: dict, value_scale: float = VALUE_SCALE) -> np.ndarray:
    """(K, S) shaped [0,1] values from endpoint distances."""
    return dist_to_value(per_sample_distances(pred, goal), value_scale=value_scale)


def per_sample_success(pred: Prediction, goal: dict) -> np.ndarray:
    """(K, S) bool: endpoint inside goal radius (velocity not available per sample)."""
    g, r = _goal_xy_r(goal)
    d = per_sample_distances(pred, goal)
    return d < r


def unbiased_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Match torch.std default (correction=1). S=1 -> NaN."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    if n <= 1:
        out_shape = list(x.shape)
        del out_shape[axis]
        return np.full(out_shape, np.nan, dtype=np.float64)
    return np.std(x, axis=axis, ddof=1)


def official_aggregate(
    values: np.ndarray,
    scheme: Literal["average", "lcb", "success_vote", "majority_mean"],
    beta: float = LCB_BETA_OFFICIAL,
    success_threshold: float = MAJORITY_THRESH_OFFICIAL,
    safe_single_sample: bool = False,
) -> np.ndarray:
    """Faithful numpy port of cosmos_utils.aggregate_value_predictions.

    values: (K, S) or (S, K) — we take (K, S).
    Official path always clamps the result to [0, 1].
    If safe_single_sample and S==1 and scheme=='lcb', return the mean
    (std treated as 0) instead of official NaN. This is a LOCAL safeguard,
    not the official behaviour — callers must opt in and label it.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"values must be (K,S), got {v.shape}")
    K, S = v.shape
    if scheme == "average":
        out = v.mean(axis=1)
    elif scheme == "lcb":
        mean = v.mean(axis=1)
        std = unbiased_std(v, axis=1)
        if safe_single_sample and S <= 1:
            std = np.zeros(K, dtype=np.float64)
        out = mean - beta * std
    elif scheme == "success_vote":
        out = (v > success_threshold).astype(np.float64).mean(axis=1)
    elif scheme == "majority_mean":
        success_mask = v > success_threshold
        num_success = success_mask.sum(axis=1).astype(np.float64)
        success_ratio = num_success / float(S)
        masked = np.where(success_mask, v, 0.0)
        mean_success = masked.sum(axis=1) / np.clip(num_success, 1.0, None)
        out = np.where(success_ratio >= 0.5, mean_success, 0.0)
    else:
        raise ValueError(f"Invalid aggregation scheme: {scheme}")
    return np.clip(out, 0.0, 1.0)


def discounted_value_from_traj(
    traj: np.ndarray,
    goal: dict,
    gamma: float = GAMMA_DEFAULT,
    value_scale: float = VALUE_SCALE,
) -> np.ndarray:
    """Shaped discounted return from a trajectory.

    traj: (K, H, 2) mean object positions, or (K, S, H, 2) — if 4D, averages
    the discounted value across S after computing per-sample returns.
    r_t = clip(1 - dist_t / value_scale, 0, 1)
    V = sum_{t=0..H-1} gamma^t r_t   (NOT normalised by (1-gamma)).
    Higher gamma / longer H => larger magnitude; use only to RANK, or compare
    with the same H and gamma. Official Cosmos values are already in [0,1];
    this return is in [0, (1-gamma^H)/(1-gamma)].
    """
    g, _ = _goal_xy_r(goal)
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 3:
        dist = np.linalg.norm(arr - g[None, None, :], axis=-1)  # (K,H)
        r = dist_to_value(dist, value_scale)
        H = r.shape[1]
        w = gamma ** np.arange(H, dtype=np.float64)
        return r @ w
    if arr.ndim == 4:
        # (K,S,H,2) or (S,K,H,2)? We accept (K,S,H,2).
        dist = np.linalg.norm(arr - g[None, None, None, :], axis=-1)
        r = dist_to_value(dist, value_scale)
        H = r.shape[2]
        w = gamma ** np.arange(H, dtype=np.float64)
        return (r @ w).mean(axis=1)
    raise ValueError(f"traj must be (K,H,2) or (K,S,H,2), got {arr.shape}")


def endpoint_discounted_value(
    values_end: np.ndarray,
    horizon: int,
    gamma: float = GAMMA_DEFAULT,
) -> np.ndarray:
    """V = gamma^{H-1} * v_end  (terminal-only, no intermediate reward)."""
    return (gamma ** max(horizon - 1, 0)) * np.asarray(values_end, dtype=np.float64)


@dataclass
class ScoreBundle:
    """All quantities for one (predictions, goal) call. scores is used to rank."""
    scores: np.ndarray                 # (K,) higher = better; selection statistic
    score_name: str
    discounted_value: np.ndarray       # (K,)
    success_prob: np.ndarray           # (K,) in [0,1]
    mean_distance: np.ndarray          # (K,)
    mean_value: np.ndarray             # (K,) shaped [0,1] mean
    lcb_value: np.ndarray              # (K,) official or safe LCB
    uncertainty_value_std: np.ndarray  # (K,) unbiased std of sample values (NaN if S=1)
    uncertainty_endpoint_rms: np.ndarray  # (K,) RMS of per-coord sample std
    credibility: np.ndarray            # (K,) 1/(1+value_std_safe); NOT calibrated
    sample_argmax_agreement: float     # fraction of samples whose argmax == mean-value argmax
    n_samples: int
    extras: dict = field(default_factory=dict)


def score_detailed(
    pred: Prediction,
    goal: dict,
    score_name: str = "goal_distance",
    gamma: float = GAMMA_DEFAULT,
    value_scale: float = VALUE_SCALE,
    lcb_beta: float = LCB_BETA_OFFICIAL,
    majority_threshold: float = MAJORITY_THRESH_OFFICIAL,
    lcb_safe_single: bool = True,
) -> ScoreBundle:
    """Compute all distinguished quantities, then pick `scores` by name."""
    g, r = _goal_xy_r(goal)
    samples = _samples_or_mean(pred)
    K, S, _ = samples.shape
    dist = per_sample_distances(pred, goal)          # (K,S)
    vals = dist_to_value(dist, value_scale)          # (K,S)
    succ = dist < r
    mean_distance = dist.mean(axis=1)
    mean_value = vals.mean(axis=1)
    success_prob = succ.astype(np.float64).mean(axis=1)
    # LCB: official NaN when S=1 unless lcb_safe_single
    lcb_value = official_aggregate(vals, "lcb", beta=lcb_beta, safe_single_sample=lcb_safe_single)
    lcb_official = official_aggregate(vals, "lcb", beta=lcb_beta, safe_single_sample=False)
    avg_value = official_aggregate(vals, "average")
    maj_official = official_aggregate(vals, "majority_mean", success_threshold=majority_threshold)
    vote_official = official_aggregate(vals, "success_vote", success_threshold=majority_threshold)
    # Also majority/vote at a scale-aware threshold: v corresponding to the goal radius
    thresh_at_radius = float(dist_to_value(np.array([r]), value_scale)[0])  # e.g. 0.9
    maj_at_radius = official_aggregate(vals, "majority_mean", success_threshold=thresh_at_radius)
    vote_at_radius = official_aggregate(vals, "success_vote", success_threshold=thresh_at_radius)

    value_std = unbiased_std(vals, axis=1)  # NaN if S=1
    value_std_safe = np.where(np.isnan(value_std), 0.0, value_std)
    # endpoint RMS std: sqrt(mean_i var_i) with unbiased var; S=1 -> 0 (safe) and NaN official
    coord_std = np.full((K, 2), np.nan if S <= 1 else 0.0, dtype=np.float64)
    if S > 1:
        coord_std = np.std(samples, axis=1, ddof=1)
    endpoint_rms = np.sqrt(np.mean(np.where(np.isnan(coord_std), 0.0, coord_std) ** 2, axis=1))
    if S <= 1:
        endpoint_rms_official = np.full(K, np.nan)
    else:
        endpoint_rms_official = np.sqrt(np.mean(coord_std ** 2, axis=1))

    credibility = 1.0 / (1.0 + value_std_safe)

    # Discounted value
    if pred.traj is not None:
        traj = np.asarray(pred.traj, dtype=np.float64)
        dval = discounted_value_from_traj(traj, goal, gamma=gamma, value_scale=value_scale)
        H = traj.shape[1]
    else:
        H = int(pred.extra.get("horizon", 1)) if pred.extra else 1
        dval = endpoint_discounted_value(mean_value, H, gamma=gamma)

    # Sample-wise argmax agreement with mean-value ranking (decision-level)
    mean_top = int(np.argmax(mean_value))
    if S >= 1:
        sample_tops = np.argmax(vals, axis=0)  # (S,)
        agreement = float(np.mean(sample_tops == mean_top))
    else:
        agreement = 1.0

    # Score used for ranking
    name = score_name
    if name in ("goal_distance", "neg_mean_distance"):
        scores = -mean_distance
    elif name in ("mean_endpoint_distance", "point_distance"):
        # distance of the MEAN endpoint (not mean of distances)
        scores = -np.linalg.norm(samples.mean(axis=1) - g[None, :], axis=-1)
    elif name == "success_prob":
        scores = success_prob
    elif name == "discounted_value":
        scores = dval
    elif name == "average_value":
        scores = avg_value
    elif name == "lcb":
        scores = lcb_value
    elif name == "lcb_official":
        scores = lcb_official
    elif name == "majority_mean":
        scores = maj_official
    elif name == "majority_mean_at_radius":
        scores = maj_at_radius
    elif name == "success_vote":
        scores = vote_official
    elif name == "success_vote_at_radius":
        scores = vote_at_radius
    elif name == "worst_case_value":
        # Conservative: min shaped value over samples. BASELINE-style robust statistic, not novelty.
        scores = vals.min(axis=1)
    elif name == "worst_case_distance":
        scores = -dist.max(axis=1)
    elif name == "robust_success":
        # 1 iff EVERY sample endpoint is inside the goal radius (succeeds under all hypothesized mus).
        scores = succ.all(axis=1).astype(np.float64)
    elif name == "robust_success_then_distance":
        # Prefer candidates that succeed under all samples; tie-break by mean distance.
        scores = succ.all(axis=1).astype(np.float64) * 10.0 - dist.mean(axis=1)
    else:
        raise ValueError(f"unknown score_name: {name}")

    extras = {
        "lcb_official": lcb_official,
        "average_value": avg_value,
        "majority_mean_official": maj_official,
        "majority_mean_at_radius": maj_at_radius,
        "success_vote_official": vote_official,
        "success_vote_at_radius": vote_at_radius,
        "thresh_at_radius": thresh_at_radius,
        "value_scale": value_scale,
        "gamma": gamma,
        "horizon": H,
        "value_std_official": value_std,
        "endpoint_rms_official": endpoint_rms_official,
        "mean_endpoint": samples.mean(axis=1),
        "sample_values": vals,
        "sample_distances": dist,
        "sample_success": succ,
        "lcb_safe_single": lcb_safe_single,
        "goal_radius": r,
        "worst_case_value": vals.min(axis=1),
        "worst_case_distance": dist.max(axis=1),
        "robust_success": succ.all(axis=1).astype(np.float64),
    }
    return ScoreBundle(
        scores=np.asarray(scores, dtype=np.float64),
        score_name=name,
        discounted_value=np.asarray(dval, dtype=np.float64),
        success_prob=success_prob,
        mean_distance=mean_distance,
        mean_value=mean_value,
        lcb_value=np.asarray(lcb_value, dtype=np.float64),
        uncertainty_value_std=value_std_safe,
        uncertainty_endpoint_rms=endpoint_rms,
        credibility=credibility,
        sample_argmax_agreement=agreement,
        n_samples=S,
        extras=extras,
    )


def choose_index(scores: np.ndarray, tie_break: str = "lowest_index") -> int:
    """Argmax with explicit ties. lowest_index is deterministic."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(s.shape)
    if np.any(np.isnan(s)):
        # Official LCB with S=1 can be all-NaN; treat as unusable.
        return 0 if tie_break == "lowest_index" else int(np.nanargmax(np.where(np.isnan(s), -np.inf, s)))
    m = np.max(s)
    tied = np.flatnonzero(np.abs(s - m) <= TIE_ATOL)
    if tie_break == "lowest_index":
        return int(tied[0])
    raise ValueError(tie_break)


class _NamedScorer:
    """Thin Scorer protocol wrapper around score_detailed."""
    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        return score_detailed(predictions, goal, score_name=self.name, **self.kwargs).scores

    def score_bundle(self, predictions: Prediction, goal: dict) -> ScoreBundle:
        return score_detailed(predictions, goal, score_name=self.name, **self.kwargs)


class GoalDistanceScorer(_NamedScorer):
    """score = -mean_s ||endpoint_s - goal||. Simple, no uncertainty."""
    name = "goal_distance"


class SampleMeanDistanceScorer(GoalDistanceScorer):
    name = "goal_distance"


class PointDistanceScorer(_NamedScorer):
    """score = -||mean endpoint - goal|| (Jensen: not equal to mean of distances)."""
    name = "point_distance"


class SuccessProbScorer(_NamedScorer):
    """score = estimated P(endpoint in goal radius). Distinct from value."""
    name = "success_prob"


class DiscountedValueScorer(_NamedScorer):
    """score = discounted shaped return from traj (or gamma^H v_end)."""
    name = "discounted_value"

    def __init__(self, gamma: float = GAMMA_DEFAULT, **kwargs):
        super().__init__(gamma=gamma, **kwargs)


class AverageValueScorer(_NamedScorer):
    """Official 'average' aggregation of shaped [0,1] values. Baseline."""
    name = "average_value"


class LCBScorer(_NamedScorer):
    """Official LCB baseline: mean - beta * unbiased_std, clamp [0,1]. NOT novelty.

    safe_single_sample=True (default here) replaces official NaN at S=1 with
    the mean. Set False to reproduce official NaN.
    """
    name = "lcb"

    def __init__(self, beta: float = LCB_BETA_OFFICIAL, safe_single_sample: bool = True, **kwargs):
        super().__init__(lcb_beta=beta, lcb_safe_single=safe_single_sample, **kwargs)


class MajorityMeanScorer(_NamedScorer):
    """Official majority_mean baseline (fixed threshold 0.05). NOT novelty."""
    name = "majority_mean"

    def __init__(self, success_threshold: float = MAJORITY_THRESH_OFFICIAL, **kwargs):
        if success_threshold != MAJORITY_THRESH_OFFICIAL:
            # Allow a non-official threshold but force a different score_name
            # if the caller wants the radius-scaled variant.
            super().__init__(majority_threshold=success_threshold, **kwargs)
            self.name = "majority_mean"
        else:
            super().__init__(majority_threshold=success_threshold, **kwargs)


class MajorityMeanAtRadiusScorer(_NamedScorer):
    """majority_mean with threshold = value(goal_radius). NOT official. Scale-aware variant."""
    name = "majority_mean_at_radius"


class SuccessVoteScorer(_NamedScorer):
    """Official success_vote baseline (fixed threshold 0.05). NOT novelty."""
    name = "success_vote"


class CompositeScorer:
    """Any score_name supported by score_detailed, for experiments."""

    def __init__(self, score_name: str, **kwargs):
        self.score_name = score_name
        self.kwargs = kwargs

    def score(self, predictions: Prediction, goal: dict) -> np.ndarray:
        return score_detailed(predictions, goal, score_name=self.score_name, **self.kwargs).scores

    def score_bundle(self, predictions: Prediction, goal: dict) -> ScoreBundle:
        return score_detailed(predictions, goal, score_name=self.score_name, **self.kwargs)


class WorstCaseValueScorer(_NamedScorer):
    """score = min_s v_s. Conservative aggregation over samples; not claimed as novelty."""
    name = "worst_case_value"


class WorstCaseDistanceScorer(_NamedScorer):
    """score = -max_s ||endpoint_s - goal||. Penalise the worst hypothesized landing."""
    name = "worst_case_distance"


class RobustSuccessScorer(_NamedScorer):
    """score = 1 iff every sample is inside the goal radius. Binary; many ties."""
    name = "robust_success"


class RobustSuccessThenDistanceScorer(_NamedScorer):
    """Robust success first, then -mean distance as a tie-break (still a scorer, higher=better)."""
    name = "robust_success_then_distance"
