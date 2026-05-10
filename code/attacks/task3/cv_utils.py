"""Cross-validation utilities for Task 3: 5-fold OOF, bootstrap CI, metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr


def bootstrap_tpr_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    results = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t = tpr_at_fpr(scores[idx].tolist(), labels[idx].tolist(), target_fpr)
        if not np.isnan(t):
            results.append(t)
    if not results:
        return (0.0, 0.0, 0.0)
    return (
        float(np.percentile(results, 5)),
        float(np.percentile(results, 50)),
        float(np.percentile(results, 95)),
    )


def run_oof(
    X: np.ndarray,
    y: np.ndarray,
    train_fn,  # (X_tr, y_tr, X_va, y_va) -> model with .predict(X)
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """5-fold stratified OOF. Returns (oof_preds, mean_best_iteration)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    best_iters = []
    for tr_idx, va_idx in skf.split(X, y):
        model = train_fn(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
        oof[va_idx] = model.predict(X[va_idx])
        if hasattr(model, "best_iteration"):
            best_iters.append(model.best_iteration)
    mean_best_iter = int(np.mean(best_iters)) if best_iters else 300
    return oof, mean_best_iter


def fit_calibrator(oof: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(oof, y)


def eval_report(
    oof: np.ndarray,
    y: np.ndarray,
    calibrator: IsotonicRegression | None = None,
    tag: str = "OOF",
) -> None:
    raw_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    ci = bootstrap_tpr_ci(oof, y)
    print(f"[{tag}] TPR@1%FPR (raw): {raw_tpr:.4f}  CI(5/95): [{ci[0]:.4f}, {ci[2]:.4f}]")
    if calibrator is not None:
        cal = calibrator.transform(oof)
        cal_tpr = tpr_at_fpr(cal.tolist(), y.tolist(), 0.01)
        print(f"[{tag}] TPR@1%FPR (cal): {cal_tpr:.4f}")

    # Per-class breakdown if subtype column exists
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    print(f"[{tag}] n_pos={n_pos}  n_neg={n_neg}")
