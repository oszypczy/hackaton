#!/usr/bin/env python3
"""
Challenge A scorer — LLM Dataset Inference.

Ground truth (--gt):
  JSONL, one record per document:
  {"id": int, "is_member": bool}

Submission (--submission):
  JSONL, one record per document:
  {"id": int, "score": float, "is_member": bool}

Dataset-level ground truth (--dataset-gt):
  JSONL, one record per candidate set:
  {"set_id": "set1"|"set2", "in_training": bool}

Dataset-level submission (--dataset-submission):
  JSONL, one record per candidate set:
  {"set_id": "set1"|"set2", "p_value": float, "verdict": "in_training"|"not_in_training"}

Metrics (weighted total):
  TPR@FPR=1%  40%   — primary SprintML metric
  AUC         20%   — secondary doc-level
  p-value     30%   — dataset-level, -log10(p) clipped to 10
  Brier       10%   — calibration

Dependencies: numpy, scipy, scikit-learn
  pip install numpy scipy scikit-learn
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_file(path: str, name: str) -> bool:
    if not Path(path).exists():
        print(f"  [ERROR] {name} not found: {path}", file=sys.stderr)
        return False
    return True


# ── Metrics ──────────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true: list, scores: list, target_fpr: float = 0.01) -> float:
    """TPR at the threshold where FPR first drops to ≤ target_fpr."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    # roc_curve returns increasing FPR; find last index where FPR ≤ target
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def brier_score(y_true: list, scores: list) -> float:
    s = np.array(scores, dtype=float)
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo + 1e-9)  # normalize to [0,1]
    return float(np.mean((s - np.array(y_true, dtype=float)) ** 2))


# ── Doc-level scoring ─────────────────────────────────────────────────────────

def score_doc_level(submission_path: str, gt_path: str) -> dict:
    from sklearn.metrics import roc_auc_score

    sub = {r["id"]: r for r in load_jsonl(submission_path)}
    gt  = {r["id"]: r for r in load_jsonl(gt_path)}

    common = sorted(set(sub) & set(gt))
    if not common:
        raise ValueError("No common IDs between submission and ground truth")

    y_true = [int(gt[i]["is_member"]) for i in common]
    scores  = [float(sub[i]["score"]) for i in common]

    if len(set(y_true)) < 2:
        print("  [WARN] Ground truth has only one class — AUC undefined", file=sys.stderr)
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, scores))

    tpr1   = tpr_at_fpr(y_true, scores, 0.01)
    brier  = brier_score(y_true, scores)

    return {"auc": auc, "tpr1": tpr1, "brier": brier, "n": len(common)}


# ── Dataset-level scoring ─────────────────────────────────────────────────────

def score_dataset_level(sub_path: str, gt_path: str) -> tuple[float, list[str]]:
    """
    Returns (normalized_score ∈ [0,1], detail_lines).

    Per-set contribution:
      IN set  + correct verdict:  min(-log10(p_value) / 10, 1.0)
      IN set  + missed:           0.0
      OUT set + correct:          1.0
      OUT set + false positive:  -1.0
    Mean of contributions then mapped [-1,+1] → [0,1].
    """
    sub = {r["set_id"]: r for r in load_jsonl(sub_path)}
    gt  = {r["set_id"]: r for r in load_jsonl(gt_path)}

    raw_scores = []
    details    = []

    for set_id, gt_row in gt.items():
        if set_id not in sub:
            details.append(f"  {set_id}: MISSING submission row")
            raw_scores.append(0.0)
            continue

        sub_row = sub[set_id]
        p_val   = float(sub_row["p_value"])
        verdict = sub_row["verdict"]
        truth   = gt_row["in_training"]

        if truth:
            if verdict == "in_training":
                log_p  = -math.log10(max(p_val, 1e-10))
                contrib = min(log_p / 10.0, 1.0)
            else:
                contrib = 0.0
        else:
            contrib = 1.0 if verdict == "not_in_training" else -1.0

        correct = "✓" if (truth == (verdict == "in_training")) else "✗"
        details.append(
            f"  {set_id}: truth={'IN' if truth else 'OUT'} "
            f"verdict={verdict} p={p_val:.3g} → {contrib:+.3f} {correct}"
        )
        raw_scores.append(contrib)

    mean_raw   = float(np.mean(raw_scores)) if raw_scores else 0.0
    normalized = (mean_raw + 1.0) / 2.0  # [-1,+1] → [0,1]
    return normalized, details


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Score Challenge A — LLM Dataset Inference")
    ap.add_argument("--submission",          default="submissions/A_docs.jsonl")
    ap.add_argument("--gt",                  default="data/A/ground_truth.jsonl")
    ap.add_argument("--dataset-submission",  default=None,
                    help="JSONL with set_id, p_value, verdict")
    ap.add_argument("--dataset-gt",          default=None,
                    help="JSONL with set_id, in_training")
    ap.add_argument("--gt-template",         action="store_true",
                    help="Print expected ground truth format and exit")
    args = ap.parse_args()

    if args.gt_template:
        print("# data/A/ground_truth.jsonl")
        print('{"id": 0, "is_member": true}')
        print('{"id": 1, "is_member": false}')
        print("")
        print("# data/A/ground_truth_dataset.jsonl")
        print('{"set_id": "set1", "in_training": true}')
        print('{"set_id": "set2", "in_training": false}')
        return

    print("=" * 58)
    print("Challenge A — LLM Dataset Inference")
    print("=" * 58)

    if not check_file(args.submission, "submission") or not check_file(args.gt, "ground truth"):
        sys.exit(1)

    doc = score_doc_level(args.submission, args.gt)
    print(f"\nDoc-level  ({doc['n']} documents):")
    print(f"  TPR@FPR=1%  : {doc['tpr1']:.4f}   [weight 40%]")
    print(f"  AUC         : {doc['auc']:.4f}   [weight 20%]")
    print(f"  Brier score : {doc['brier']:.4f}   (1−Brier, weight 10%)")

    doc_component = 0.40 * doc["tpr1"] + 0.20 * doc["auc"] + 0.10 * (1.0 - doc["brier"])

    # Dataset-level (optional)
    have_dataset = args.dataset_submission and args.dataset_gt
    if have_dataset:
        if not check_file(args.dataset_submission, "dataset submission") or \
           not check_file(args.dataset_gt, "dataset ground truth"):
            have_dataset = False

    if have_dataset:
        ds_score, ds_details = score_dataset_level(args.dataset_submission, args.dataset_gt)
        print(f"\nDataset-level  [weight 30%]:")
        for line in ds_details:
            print(line)
        print(f"  Normalized dataset score: {ds_score:.4f}")
        total = doc_component + 0.30 * ds_score
        label = "TOTAL"
    else:
        print("\nDataset-level: skipped — pass --dataset-submission + --dataset-gt")
        total = doc_component / 0.70   # partial: re-normalize so doc portion sums to 1
        label = "PARTIAL (doc-level only, re-normalized)"

    print(f"\n{'=' * 58}")
    print(f"  {label} : {total:.4f}")
    print(f"{'=' * 58}")

    print("\nReference (doc-level):")
    print("  Easy   loss MIA        : AUC ≈ 0.55–0.65  TPR@1% ≈  5–15%")
    print("  Solid  Maini pipeline  : AUC ≈ 0.70–0.80  TPR@1% ≈ 20–40%")
    print("  Hard   LiRA + ensemble : AUC  > 0.82      TPR@1%  > 50%")


if __name__ == "__main__":
    main()
