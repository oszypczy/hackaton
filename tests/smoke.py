"""Smoke test — runs in <30s, exits 0 if infrastructure is intact.

Two layers:
1. Metric infrastructure (always run): tiny synthetic fixtures exercise
   AUC / TPR@FPR / F1 / nDCG@k / Recall@k.
2. Attack placeholders: wired as `try: import code.attacks.X` blocks. Until
   the attack module exists, the test prints SKIP and exits 0. Once the
   module lands, it must produce a valid (non-NaN) score on a tiny fixture
   or the test fails.

Run via `just eval`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from templates.eval_scaffold import auc, f1_binary, ndcg_at_k, recall_at_k, tpr_at_fpr


def _check(label: str, actual: float, expected: float, tol: float = 1e-3) -> None:
    if math.isnan(actual):
        raise AssertionError(f"{label}: got NaN")
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: got {actual:.4f}, expected {expected:.4f}")
    print(f"  PASS  {label}: {actual:.4f}")


def test_metrics() -> None:
    print("[metrics]")
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.5, 50).tolist()
    neg = rng.normal(0.0, 0.5, 50).tolist()
    scores = pos + neg
    labels = [1] * 50 + [0] * 50

    a = auc(scores, labels)
    assert 0.85 < a < 1.0, f"auc out of expected range: {a}"
    print(f"  PASS  auc separable-gaussians: {a:.4f}")

    t = tpr_at_fpr(scores, labels, 0.1)
    assert 0.5 < t <= 1.0, f"tpr@fpr=0.1 out of range: {t}"
    print(f"  PASS  tpr@fpr=0.1: {t:.4f}")

    _check("f1 perfect", f1_binary([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
    _check("f1 zero", f1_binary([0, 0, 0, 0], [1, 1, 1, 1]), 0.0)

    gold = {"a", "b", "c"}
    rank = ["a", "x", "b", "y", "c"]
    _check("ndcg@5", ndcg_at_k(rank, gold, 5), (1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)) / (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)))
    _check("recall@3", recall_at_k(rank, gold, 3), 2 / 3)


def test_attack_placeholders() -> None:
    print("[attacks]")
    placeholders = [
        ("min_k_pp", "Challenge A — paper 20 Min-K%++"),
        ("kirchenbauer_detect", "Challenge B — paper 04 watermark detect"),
        ("diffusion_extraction", "Challenge C — paper 01/09 generate-and-filter / CDI"),
    ]
    for module, label in placeholders:
        try:
            __import__(f"code.attacks.{module}")
            print(f"  TODO  {module} module imported but no smoke fixture wired yet ({label})")
        except ModuleNotFoundError:
            print(f"  SKIP  {module} not implemented yet ({label})")


def main() -> int:
    try:
        test_metrics()
        test_attack_placeholders()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
