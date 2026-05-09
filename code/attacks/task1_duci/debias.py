"""
Tong Eq. 4 debiasing + clamping.

p̂_i = (m̂_i - FPR̂) / (TPR̂ - FPR̂)
p̂   = mean_i p̂_i

Clamp final per-target p̂ to [P_MIN, P_MAX] for MAE robustness (research
recommends [0.025, 0.975] — caps loss at 0.025 if true p ∈ {0, 1}).

Optional: snap to nearest 0.05 (Tong's evaluation grid).
"""
from __future__ import annotations

import numpy as np

P_MIN, P_MAX = 0.025, 0.975


def debias(m_hat: np.ndarray, tpr: float, fpr: float) -> float:
    """Aggregate per-target estimate from per-sample MIA indicators.

    m_hat: (N_x,) ∈ {0, 1} — RMIA indicator at β*
    Returns: scalar p̂ ∈ R (clamping done by caller).
    """
    denom = tpr - fpr
    if not np.isfinite(denom) or abs(denom) < 1e-3:
        # degenerate Youden gap — fall back to raw mean (uncalibrated)
        return float(m_hat.mean())
    return float(((m_hat - fpr) / denom).mean())


def clamp(p_hat: float, lo: float = P_MIN, hi: float = P_MAX) -> float:
    return float(np.clip(p_hat, lo, hi))


def snap_5pct(p_hat: float) -> float:
    return float(round(p_hat / 0.05) * 0.05)


def write_submission_csv(predictions: dict[str, float], path: str) -> None:
    """Write submission.csv with exact format required by spec.

    predictions: dict mapping model_id ('model_00' or '00') -> p̂.
    Output: model_id,proportion (header + 9 rows, model_ids = '00' .. '22').
    """
    expected_keys = [f"{a}{i}" for a in (0, 1, 2) for i in (0, 1, 2)]
    rows = []
    for k in expected_keys:
        # accept either 'model_00' or '00'
        if k in predictions:
            v = predictions[k]
        elif f"model_{k}" in predictions:
            v = predictions[f"model_{k}"]
        else:
            raise KeyError(f"missing prediction for model_id {k!r}")
        rows.append(f"{k},{v:.6f}")
    with open(path, "w") as f:
        f.write("model_id,proportion\n")
        f.write("\n".join(rows))
        f.write("\n")
