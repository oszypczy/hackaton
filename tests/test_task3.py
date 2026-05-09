"""Smoke test for Task 3 pipeline.

Tests pipeline logic with synthetic data — does NOT load GPT-2 or sentence-transformers.
Runs in <30s via `just eval`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "attacks" / "task3"))

from templates.eval_scaffold import tpr_at_fpr


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synthetic_features(n: int, n_feats: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, n_feats)).astype(np.float32)


def _synthetic_labels(n: int, positive_rate: float = 0.5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < positive_rate).astype(int)


# ── Test: metrics ────────────────────────────────────────────────────────────

def test_tpr_at_fpr() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.5, 50).tolist()
    neg = rng.normal(0.0, 0.5, 50).tolist()
    scores = pos + neg
    labels = [1] * 50 + [0] * 50
    t = tpr_at_fpr(scores, labels, 0.01)
    assert 0.0 <= t <= 1.0, f"tpr_at_fpr out of [0,1]: {t}"
    # With well-separated gaussians, should be non-trivial
    assert t > 0.3, f"Expected >0.3 for separated gaussians, got {t:.4f}"
    print(f"  PASS  tpr_at_fpr(separated gaussians): {t:.4f}")


def test_bootstrap_ci() -> None:
    from cv_utils import bootstrap_tpr_ci

    rng = np.random.default_rng(1)
    scores = rng.random(180)
    labels = (scores + rng.normal(0, 0.3, 180) > 0.5).astype(int)
    p5, p50, p95 = bootstrap_tpr_ci(scores, labels, n_boot=100)
    assert 0.0 <= p5 <= p50 <= p95 <= 1.0, f"CI not ordered: {p5:.3f}/{p50:.3f}/{p95:.3f}"
    print(f"  PASS  bootstrap_ci: [{p5:.3f}, {p50:.3f}, {p95:.3f}]")


# ── Test: branch_bc (stateless parts) ────────────────────────────────────────

def test_branch_bc_zscore() -> None:
    from features.branch_bc import UnigramGreenList

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    texts = ["hello world today is a nice day"] * 10
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    gl = UnigramGreenList()
    gl.fit(texts, labels, tok)
    assert gl.soft_g is not None

    score = gl.zscore("hello world this is a test sentence", tok)
    assert isinstance(score, float), f"zscore not float: {type(score)}"

    winmax = gl.winmax_zscore("hello world this is a test sentence", tok)
    assert set(winmax.keys()) == {"winmax_z_50", "winmax_z_100", "winmax_z_200"}
    for k, v in winmax.items():
        assert isinstance(v, float), f"{k} not float: {type(v)}"
    print(f"  PASS  branch_bc zscore={score:.4f}  winmax={winmax}")


# ── Test: submission validation ───────────────────────────────────────────────

def test_submission_validator() -> None:
    from main import validate_submission

    good = pd.DataFrame({"id": range(1, 2251), "score": np.random.rand(2250)})
    validate_submission(good, 2250)
    print("  PASS  validate_submission: 2250 rows OK")

    # Should raise on wrong shape
    try:
        bad = pd.DataFrame({"id": range(1, 101), "score": np.random.rand(100)})
        validate_submission(bad, 2250)
        raise AssertionError("Should have raised on wrong row count")
    except AssertionError as e:
        if "Should have raised" in str(e):
            raise
    print("  PASS  validate_submission: rejects wrong row count")

    # Should raise on out-of-range scores
    try:
        bad2 = pd.DataFrame({"id": range(1, 2251), "score": np.random.rand(2250) + 1.0})
        validate_submission(bad2, 2250)
        raise AssertionError("Should have raised on out-of-range scores")
    except AssertionError as e:
        if "Should have raised" in str(e):
            raise
    print("  PASS  validate_submission: rejects out-of-range scores")


# ── Test: LightGBM mini-training ──────────────────────────────────────────────

def test_lgbm_mini() -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("  SKIP  lgbm mini-train: lightgbm not installed (pip install lightgbm)")
        return
    from sklearn.isotonic import IsotonicRegression

    X = _synthetic_features(180)
    y = _synthetic_labels(180)
    X_va = _synthetic_features(30, seed=99)
    y_va = _synthetic_labels(30, seed=99)

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 5,
        "lambda_l2": 2.0,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        lgb.Dataset(X, label=y),
        num_boost_round=50,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_va)
    assert preds.shape == (30,), f"Bad preds shape: {preds.shape}"
    assert preds.min() >= 0.0 and preds.max() <= 1.0, f"Preds out of [0,1]"

    iso = IsotonicRegression(out_of_bounds="clip").fit(preds, y_va)
    cal = iso.transform(preds)
    assert cal.min() >= 0.0 and cal.max() <= 1.0

    print(f"  PASS  lgbm mini-train: preds=[{preds.min():.3f},{preds.max():.3f}]  best_iter={model.best_iteration}")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    import os
    os.chdir(ROOT / "code" / "attacks" / "task3")

    tests = [
        test_tpr_at_fpr,
        test_bootstrap_ci,
        test_branch_bc_zscore,
        test_submission_validator,
        test_lgbm_mini,
    ]
    print("[task3 smoke tests]")
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\nFAILED: {failed}/{len(tests)}")
        return 1
    print(f"\nOK ({len(tests)}/{len(tests)} passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
