#!/usr/bin/env python3
"""
Challenge C scorer — Diffusion Memorization Discovery.

Ground truth (--gt):
  JSONL, one record per image:
  {"id": int, "is_memorized": bool}

Submission (--submission):
  JSONL, one record per image:
  {"id": int, "memorization_score": float, "rank": int}
  Scorer re-ranks by memorization_score (descending); rank field is ignored.

Metrics (weighted total):
  nDCG@50    60%   — position-sensitive; rewards high ranks for true positives
  Recall@50  25%   — fraction of true positives in top 50
  Recall@100 15%   — wider net fallback

Dependencies: numpy only
  pip install numpy
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_file(path: str, label: str) -> bool:
    if not Path(path).exists():
        print(f"  [ERROR] {label} not found: {path}", file=sys.stderr)
        return False
    return True


# ── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant
    )
    # Ideal DCG: top-k are all relevant
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / k if k > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Score Challenge C — Diffusion Memorization")
    ap.add_argument("--submission", default="submissions/C.jsonl")
    ap.add_argument("--gt",         default="data/C/ground_truth.jsonl")
    ap.add_argument("--gt-template", action="store_true",
                    help="Print expected ground truth format and exit")
    ap.add_argument("--top-n",       type=int, default=20,
                    help="Show first N hits in ranked list (default 20)")
    args = ap.parse_args()

    if args.gt_template:
        print("# data/C/ground_truth.jsonl")
        print('{"id": 0,   "is_memorized": true}')
        print('{"id": 1,   "is_memorized": false}')
        print("# ... 1000 entries total, 50 is_memorized=true")
        return

    print("=" * 58)
    print("Challenge C — Diffusion Memorization Discovery")
    print("=" * 58)

    if not check_file(args.submission, "submission") or not check_file(args.gt, "ground truth"):
        sys.exit(1)

    sub = load_jsonl(args.submission)
    gt  = load_jsonl(args.gt)

    memorized_ids = {r["id"] for r in gt if r["is_memorized"]}
    n_total       = len(gt)
    n_memorized   = len(memorized_ids)

    if n_memorized == 0:
        print("[ERROR] Ground truth has no memorized examples", file=sys.stderr)
        sys.exit(1)

    # Sort by score descending; re-rank ignoring submission's rank field
    sub_sorted = sorted(sub, key=lambda r: float(r["memorization_score"]), reverse=True)
    ranked_ids  = [r["id"] for r in sub_sorted]

    # Core metrics
    ndcg50    = ndcg_at_k(ranked_ids, memorized_ids, 50)
    recall50  = recall_at_k(ranked_ids, memorized_ids, 50)
    recall100 = recall_at_k(ranked_ids, memorized_ids, 100)

    # Bonus diagnostics
    prec50    = precision_at_k(ranked_ids, memorized_ids, 50)
    prec100   = precision_at_k(ranked_ids, memorized_ids, 100)
    ndcg100   = ndcg_at_k(ranked_ids, memorized_ids, 100)

    total = 0.60 * ndcg50 + 0.25 * recall50 + 0.15 * recall100

    print(f"\nDataset: {n_total} candidates, {n_memorized} memorized")
    print(f"\nPrimary metrics:")
    print(f"  nDCG@50    : {ndcg50:.4f}   [weight 60%]")
    print(f"  Recall@50  : {recall50:.4f}   [weight 25%]  ({int(recall50*n_memorized)}/{n_memorized} found)")
    print(f"  Recall@100 : {recall100:.4f}   [weight 15%]  ({int(recall100*n_memorized)}/{n_memorized} found)")
    print(f"\nDiagnostics:")
    print(f"  Precision@50  : {prec50:.4f}")
    print(f"  Precision@100 : {prec100:.4f}")
    print(f"  nDCG@100      : {ndcg100:.4f}")

    # Show where true positives appear in ranking
    hits_top50  = [i for i in ranked_ids[:50]  if i in memorized_ids]
    hits_top100 = [i for i in ranked_ids[:100] if i in memorized_ids]
    miss_ids    = memorized_ids - set(ranked_ids[:n_total])

    print(f"\nTop-50 contains  {len(hits_top50)}/{n_memorized} memorized")
    print(f"Top-100 contains {len(hits_top100)}/{n_memorized} memorized")

    if hits_top50:
        preview = hits_top50[:args.top_n]
        print(f"  First hits in top-50: {preview}")

    # First-hit positions for each true positive
    pos_map = {doc_id: pos + 1 for pos, doc_id in enumerate(ranked_ids)}
    tp_positions = sorted(pos_map[i] for i in memorized_ids if i in pos_map)
    if tp_positions:
        print(f"  True positive positions: min={tp_positions[0]}  "
              f"median={np.median(tp_positions):.0f}  "
              f"max={tp_positions[-1]}")

    if miss_ids:
        print(f"  [WARN] {len(miss_ids)} memorized IDs not in submission")

    print(f"\n{'='*58}")
    print(f"  TOTAL SCORE : {total:.4f}")
    print(f"{'='*58}")

    print(f"\nReference (nDCG@50):")
    print(f"  Easy   pixel L2 nearest neighbor       : ≈ 0.40–0.55")
    print(f"  Solid  CLIP embedding + DBSCAN          : ≈ 0.70–0.85")
    print(f"  Hard   Carlini full denoising pipeline  :  > 0.90")


if __name__ == "__main__":
    main()
