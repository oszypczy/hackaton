from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class CvResult:
    oof_pred: np.ndarray
    fold_scores: list[float]
    mean_tpr: float
    std_tpr: float


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s)
    s = s[order]
    y = y[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    fp = np.cumsum(y == 0)
    tp = np.cumsum(y == 1)
    fpr = fp / n_neg
    tpr = tp / n_pos
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def bootstrap_tpr_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(scores)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(tpr_at_fpr(scores[idx], labels[idx], target_fpr=target_fpr))
    arr = np.asarray(vals, dtype=float)
    q5, q50, q95 = np.percentile(arr, [5, 50, 95])
    return float(q5), float(q50), float(q95)


def make_stratified_folds(labels: np.ndarray, n_splits: int = 5, seed: int = 42):
    y = np.asarray(labels, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))
