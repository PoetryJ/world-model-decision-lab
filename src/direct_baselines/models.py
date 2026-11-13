"""Ridge candidate scorers (numpy only). Policy-side: no hidden params, no env access."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .features import build_feature_matrix, feature_names


LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Return weights of length D+1 (last entry = intercept). Do not regularise intercept."""
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n, dtype=np.float64)])
    A = Xb.T @ Xb
    A[:d, :d] = A[:d, :d] + lam * np.eye(d)
    b = Xb.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def predict_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([X, np.ones(X.shape[0], dtype=np.float64)])
    return Xb @ w


@dataclass
class RidgeScorerModel:
    """Learned linear scorer. Higher predicted value = better candidate."""
    weights: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    lam: float
    use_history: bool
    n_train: int
    train_seconds: float
    target: str = "-dist_settled"

    @property
    def n_params(self) -> int:
        return int(self.weights.size)  # D features + intercept

    @property
    def n_features(self) -> int:
        return int(self.weights.size - 1)

    def features(self, observation, candidates, history=None) -> np.ndarray:
        import src.direct_baselines.features as feat
        prev = feat.INCLUDE_PHYS_FEATURES
        # Frozen official models were trained without baked-in physics rollouts (36 / 64 dims).
        d = int(self.feat_mean.shape[0])
        if d in (36, 64):
            feat.INCLUDE_PHYS_FEATURES = False
        elif d in (41, 69):
            feat.INCLUDE_PHYS_FEATURES = True
        try:
            X = build_feature_matrix(observation, candidates, history, use_history=self.use_history)
        finally:
            feat.INCLUDE_PHYS_FEATURES = prev
        if X.shape[1] != d:
            raise ValueError(f"feature dim {X.shape[1]} != model {d}")
        return _standardize(X, self.feat_mean, self.feat_std)

    def score_candidates(self, observation, history, candidates) -> np.ndarray:
        X = self.features(observation, candidates, history)
        return predict_ridge(X, self.weights)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            lam=np.array(self.lam),
            use_history=np.array(self.use_history),
            n_train=np.array(self.n_train),
            train_seconds=np.array(self.train_seconds),
            target=np.array(self.target),
            feature_names=np.array(feature_names(self.use_history)),
        )

    @classmethod
    def load(cls, path) -> "RidgeScorerModel":
        z = np.load(path, allow_pickle=False)
        return cls(
            weights=z["weights"],
            feat_mean=z["feat_mean"],
            feat_std=z["feat_std"],
            lam=float(z["lam"]),
            use_history=bool(z["use_history"]),
            n_train=int(z["n_train"]),
            train_seconds=float(z["train_seconds"]),
            target=str(z["target"]),
        )

    def summary(self) -> dict[str, Any]:
        names = feature_names(self.use_history)
        w = self.weights[:-1]
        order = np.argsort(-np.abs(w))
        top = [{"name": names[i], "weight": float(w[i])} for i in order[:12]]
        return {
            "use_history": self.use_history,
            "n_params": self.n_params,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "lam": self.lam,
            "train_seconds": self.train_seconds,
            "target": self.target,
            "intercept": float(self.weights[-1]),
            "top_abs_weights": top,
        }


def train_ridge(
    X: np.ndarray,
    y: np.ndarray,
    use_history: bool,
    lam: float | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
    train_seconds: float = 0.0,
) -> RidgeScorerModel:
    """Fit ridge. If lam is None, select on validation top-1 (grouped by state) else val MSE."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = _standardize(X, mean, std)

    def _fit(lam_):
        w = fit_ridge(Xs, y, lam_)
        return w

    if lam is None:
        if X_val is None:
            lam = 1e-2
            w = _fit(lam)
        else:
            Xvs = _standardize(X_val, mean, std)
            best, best_key = None, None
            for lam_ in LAMBDA_GRID:
                w = _fit(lam_)
                pred = predict_ridge(Xvs, w)
                key = _val_objective(pred, y_val, groups_val)
                if best_key is None or key > best_key:
                    best, best_key = (lam_, w), key
            lam, w = best
    else:
        w = _fit(lam)
    return RidgeScorerModel(
        weights=w, feat_mean=mean, feat_std=std, lam=float(lam),
        use_history=use_history, n_train=int(X.shape[0]), train_seconds=float(train_seconds),
    )


def _val_objective(pred: np.ndarray, y: np.ndarray, groups: np.ndarray | None) -> float:
    """Higher is better. Prefer grouped top-1 agreement with label argmax; fall back to -MSE."""
    if groups is None:
        return float(-np.mean((pred - y) ** 2))
    hits, n = 0, 0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        hits += int(np.argmax(pred[m]) == np.argmax(y[m]))
        n += 1
    if n == 0:
        return float(-np.mean((pred - y) ** 2))
    return hits / n + 1e-6 * float(-np.mean((pred - y) ** 2))  # tiny MSE tie-break
