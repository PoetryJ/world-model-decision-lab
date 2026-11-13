"""Calibration diagnostics. Ensemble variance is not assumed calibrated.

ECE and reliability tables are computed from (p_hat, y) pairs. When multiple
candidates share an episode they are NOT independent; primary reporting uses
one forecast per episode (the chosen candidate, or 'P(top-1 is correct)').
Per-candidate numbers are secondary and labelled as dependent.
"""
from __future__ import annotations
import json
import numpy as np


def reliability_table(
    p_hat: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    min_count: int = 1,
) -> dict:
    """Bin predicted probabilities and compare to empirical frequencies.

    strategy='uniform': equal-width bins on [0,1].
    strategy='quantile': equal-count bins (edges from unique quantiles).
    Returns counts, bin mean p, bin empirical freq, abs gap, ECE, MCE, Brier.
    Empty bins are omitted from ECE (standard).
    """
    p = np.asarray(p_hat, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError("p_hat and y must have the same shape")
    if len(p) == 0:
        return {"n": 0, "n_bins": n_bins, "strategy": strategy, "ece": np.nan,
                "mce": np.nan, "brier": np.nan, "bins": []}
    p = np.clip(p, 0.0, 1.0)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError(strategy)
    bins = []
    ece = 0.0
    mce = 0.0
    n = len(p)
    # right-inclusive on the last bin
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        c = int(mask.sum())
        if c < min_count:
            bins.append({"lo": float(lo), "hi": float(hi), "count": c,
                         "p_mean": None, "y_freq": None, "gap": None})
            continue
        pm = float(p[mask].mean())
        yf = float(y[mask].mean())
        gap = abs(pm - yf)
        ece += (c / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": c,
                     "p_mean": pm, "y_freq": yf, "gap": gap})
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": n, "n_bins": n_bins, "strategy": strategy,
        "ece": float(ece), "mce": float(mce), "brier": brier, "bins": bins,
    }


def expected_calibration_error(p_hat, y, n_bins=10, strategy="uniform") -> float:
    return reliability_table(p_hat, y, n_bins=n_bins, strategy=strategy)["ece"]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 200) -> tuple[float, float]:
    """Fit P(y=1|x) = sigmoid(a + b x) by Newton, with weak L2. Returns (a, b)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        # gradient / Hessian of mean NLL + 0.5 l2 (a^2+b^2)
        w = p * (1.0 - p)
        ga = np.mean(p - y) + l2 * a
        gb = np.mean((p - y) * x) + l2 * b
        haa = np.mean(w) + l2
        hab = np.mean(w * x)
        hbb = np.mean(w * x * x) + l2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-10:
            break
    return float(a), float(b)


def logistic_prob(x, a, b):
    z = a + b * np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def write_reliability_svg(table: dict, path: str, title: str = "Reliability") -> None:
    """Tiny SVG reliability diagram (no matplotlib)."""
    W, H, pad = 420, 420, 50
    inner = W - 2 * pad
    def xy(p, f):
        return pad + p * inner, pad + (1 - f) * inner
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">{_esc(title)}</text>',
    ]
    # axes and diagonal
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" />')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" />')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#aaa" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="11" font-family="sans-serif">predicted p</text>')
    parts.append(f'<text x="14" y="{H/2}" text-anchor="middle" font-size="11" font-family="sans-serif" transform="rotate(-90 14 {H/2})">empirical freq</text>')
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        tx, ty = xy(t, 0)
        parts.append(f'<text x="{tx}" y="{ty+14}" text-anchor="middle" font-size="9">{t:.2f}</text>')
        tx, ty = xy(0, t)
        parts.append(f'<text x="{tx-6}" y="{ty+3}" text-anchor="end" font-size="9">{t:.2f}</text>')
    # bars / points
    for b in table.get("bins", []):
        if b["count"] is None or b["count"] == 0 or b["p_mean"] is None:
            continue
        px, _ = xy(b["p_mean"], 0)
        _, fy = xy(0, b["y_freq"])
        _, base = xy(0, 0)
        width = max(6, (b["hi"] - b["lo"]) * inner * 0.8)
        height = base - fy
        parts.append(
            f'<rect x="{px-width/2:.1f}" y="{min(fy,base):.1f}" width="{width:.1f}" '
            f'height="{abs(height):.1f}" fill="#4c78a8" fill-opacity="0.7" stroke="#333"/>'
        )
        parts.append(f'<circle cx="{px:.1f}" cy="{fy:.1f}" r="3" fill="#d62728"/>')
    ece = table.get("ece", float("nan"))
    n = table.get("n", 0)
    parts.append(
        f'<text x="{W-pad}" y="38" text-anchor="end" font-size="11" font-family="sans-serif">'
        f'ECE={ece:.3f}  n={n}</text>'
    )
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dump_json(obj, path: str) -> None:
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        return str(x)
    with open(path, "w") as f:
        json.dump(conv(obj), f, indent=2)
