from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK3_DIR = Path(__file__).resolve().parent.parent / "code" / "attacks" / "task3"
sys.path.insert(0, str(TASK3_DIR))

from cv_utils import bootstrap_tpr_ci, tpr_at_fpr


def test_tpr_at_fpr_monotonic_signal() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.5, 100)
    neg = rng.normal(0.0, 0.5, 100)
    scores = np.concatenate([pos, neg])
    labels = np.array([1] * 100 + [0] * 100)
    tpr = tpr_at_fpr(scores, labels, target_fpr=0.01)
    assert 0.2 <= tpr <= 1.0


def test_bootstrap_ci_shape() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(200)
    labels = np.array([1] * 100 + [0] * 100)
    q5, q50, q95 = bootstrap_tpr_ci(scores, labels, target_fpr=0.01, n_boot=100, seed=1)
    assert 0.0 <= q5 <= q50 <= q95 <= 1.0


def test_submission_schema() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.5, 0.9]})
    assert list(df.columns) == ["id", "score"]
    assert df["id"].is_unique
    assert df["score"].between(0.0, 1.0).all()
