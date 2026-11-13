"""Low-confidence gate: abstain / recommend a probe when uncertain.

The gate does NOT pick the action; it only says whether the current ranking
is credible enough to commit. Thresholds are fitted on train/val ONLY.

Features (higher feature => more likely to ABSTAIN, unless noted):
- value_std: unbiased std of shaped values for the chosen candidate
- endpoint_rms: RMS endpoint sample std for the chosen candidate
- inv_agreement: 1 - sample argmax agreement (decision-level)
- margin_over_unc: NOT a raw uncertainty; we abstain when
    (score_top1 - score_top2) / (value_std_top1 + eps) is SMALL, so the
    feature stored is -margin/(std+eps) (higher => abstain).
- raw_margin: abstain when score gap is small; feature = -gap

Ensemble variance is a FEATURE, not a calibrated confidence. After fitting
a threshold (and optionally a logistic P(wrong|feature) on val), report
test metrics separately.

Negative-control interest: when the best action does not depend on the
hidden condition, a useful gate should abstain less often.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.scoring.scorers import ScoreBundle, choose_index, TIE_ATOL
from src.scoring.calibration import fit_logistic, logistic_prob


EPS = 1e-9

FEATURE_NAMES = (
    "value_std",
    "endpoint_rms",
    "inv_agreement",
    "neg_margin_over_unc",
    "neg_margin",
)


def extract_feature(bundle: ScoreBundle, name: str, chosen: int | None = None) -> float:
    scores = bundle.scores
    if chosen is None:
        chosen = choose_index(scores)
    if name == "value_std":
        return float(bundle.uncertainty_value_std[chosen])
    if name == "endpoint_rms":
        return float(bundle.uncertainty_endpoint_rms[chosen])
    if name == "inv_agreement":
        return float(1.0 - bundle.sample_argmax_agreement)
    if name == "neg_margin_over_unc":
        gap = _score_gap(scores)
        std = float(bundle.uncertainty_value_std[chosen])
        return float(-gap / (std + EPS))
    if name == "neg_margin":
        return float(-_score_gap(scores))
    raise ValueError(name)


def _score_gap(scores: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).copy()
    if s.size < 2:
        return 0.0
    top = int(np.nanargmax(s))
    best = s[top]
    s[top] = -np.inf
    second = np.nanmax(s)
    if not np.isfinite(second):
        return 0.0
    return float(best - second)


@dataclass
class LowConfidenceGate:
    feature: str
    threshold: float          # abstain if feature >= threshold
    fitted_on: str = "val"    # label of the split used to set the threshold
    utility_lambda: float = 0.4  # probe/abstain cost relative to a wrong commit
    n_fit: int = 0
    logistic_ab: tuple[float, float] | None = None  # P(wrong|feature) on val

    def should_abstain(self, bundle: ScoreBundle, chosen: int | None = None) -> bool:
        feat = extract_feature(bundle, self.feature, chosen=chosen)
        return bool(feat >= self.threshold)

    def p_wrong(self, bundle: ScoreBundle, chosen: int | None = None) -> float | None:
        if self.logistic_ab is None:
            return None
        feat = extract_feature(bundle, self.feature, chosen=chosen)
        return float(logistic_prob(feat, *self.logistic_ab))

    @staticmethod
    def fit(
        features: np.ndarray,
        wrong: np.ndarray,
        feature_name: str,
        probe_cost: float = 0.4,
        fitted_on: str = "val",
    ) -> "LowConfidenceGate":
        """Pick a threshold on VAL to minimise: err_commit + probe_cost * abstain_rate.

        Utility is computed on the same val features/labels (wrong=1 if the
        model's top-1 is not an oracle-best, with ties counted as correct).
        If all predictions are already correct, the optimal policy is never
        abstain: threshold is +inf.
        """
        f = np.asarray(features, dtype=np.float64)
        w = np.asarray(wrong, dtype=np.float64)
        n = len(f)
        if n == 0:
            return LowConfidenceGate(feature_name, threshold=np.inf, fitted_on=fitted_on,
                                     utility_lambda=probe_cost, n_fit=0)
        # Candidate thresholds: just below each unique feature (and never-abstain / always)
        uniq = np.unique(f)
        cands = np.concatenate(([-np.inf], uniq, [np.inf]))
        best_t, best_u = np.inf, np.inf
        best_stats = None
        for t in cands:
            abstain = f >= t
            commit = ~abstain
            n_abs = float(abstain.mean())
            if commit.any():
                err = float(w[commit].mean())  # error rate among commits
            else:
                err = 0.0  # nothing committed
            # Expected cost per episode: commit errors + cost of probes
            # Use rates so n cancels: C = P(commit & wrong) + probe_cost * P(abstain)
            p_commit_wrong = float(((~abstain) & (w > 0.5)).mean())
            C = p_commit_wrong + probe_cost * n_abs
            if C < best_u - 1e-15 or (abs(C - best_u) <= 1e-15 and t > best_t):
                best_u, best_t = C, float(t)
                best_stats = (err, n_abs, C)
        a, b = fit_logistic(f, w)
        gate = LowConfidenceGate(
            feature=feature_name,
            threshold=float(best_t),
            fitted_on=fitted_on,
            utility_lambda=probe_cost,
            n_fit=n,
            logistic_ab=(a, b),
        )
        gate._fit_stats = {"val_cost": best_u, "val_commit_err": best_stats[0],
                           "val_abstain_rate": best_stats[1]}
        return gate


def evaluate_gate(gate: LowConfidenceGate, features: np.ndarray, wrong: np.ndarray) -> dict:
    f = np.asarray(features, dtype=np.float64)
    w = np.asarray(wrong, dtype=bool)
    abstain = f >= gate.threshold
    commit = ~abstain
    n = len(f)
    caught = int((abstain & w).sum())          # wrong decisions the gate blocked
    missed = int((commit & w).sum())           # committed and wrong
    false_abs = int((abstain & ~w).sum())      # would have been correct
    return {
        "n": n,
        "feature": gate.feature,
        "threshold": gate.threshold,
        "fitted_on": gate.fitted_on,
        "probe_cost": gate.utility_lambda,
        "abstain_rate": float(abstain.mean()) if n else np.nan,
        "commit_rate": float(commit.mean()) if n else np.nan,
        "commit_error_rate": float(w[commit].mean()) if commit.any() else 0.0,
        "ungated_error_rate": float(w.mean()) if n else np.nan,
        "recall_catch_wrong": (caught / w.sum()) if w.any() else np.nan,
        "false_abstain_rate": float((abstain & ~w).mean()) if n else np.nan,
        "n_caught_wrong": caught,
        "n_missed_wrong": missed,
        "n_false_abstain": false_abs,
        "n_wrong": int(w.sum()),
        "expected_cost": float(((commit & w).mean() if n else np.nan) + gate.utility_lambda * (abstain.mean() if n else 0.0)),
    }
