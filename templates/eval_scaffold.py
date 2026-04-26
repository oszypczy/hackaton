"""Generic submission scoring scaffold.

Submission format: JSONL, one row per item, fields:
  {"id": str, "score": float}        # for AUC / Min-K%++ / MIA
  {"id": str, "label": int}          # for F1 (binary 0/1)
  {"id": str, "ranking": [id, ...]}  # for nDCG@k / Recall@k

Ground truth: same format minus the score (just id + true label / true membership).

Usage:
    python templates/eval_scaffold.py auc --pred preds.jsonl --truth gt.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(scores: list[float], labels: list[int]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def tpr_at_fpr(scores: list[float], labels: list[int], target_fpr: float) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s)
    s, y = s[order], y[order]
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fp = np.cumsum(y == 0)
    tp = np.cumsum(y == 1)
    fpr = fp / n_neg
    tpr = tp / n_pos
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def f1_binary(pred: list[int], truth: list[int]) -> float:
    p = np.asarray(pred, dtype=int)
    t = np.asarray(truth, dtype=int)
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def ndcg_at_k(ranking: list[str], gold_set: set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranking[:k]):
        if item in gold_set:
            dcg += 1 / math.log2(i + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranking: list[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return float("nan")
    return len(set(ranking[:k]) & gold_set) / len(gold_set)


def join_by_id(pred: list[dict], truth: list[dict]) -> list[tuple[dict, dict]]:
    truth_by_id = {row["id"]: row for row in truth}
    pairs = []
    for row in pred:
        if row["id"] in truth_by_id:
            pairs.append((row, truth_by_id[row["id"]]))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["auc", "tpr_at_fpr", "f1", "ndcg", "recall"])
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--fpr", type=float, default=0.01)
    p.add_argument("--k", type=int, default=50)
    args = p.parse_args()

    pred = load_jsonl(args.pred)
    truth = load_jsonl(args.truth)

    if args.metric in {"auc", "tpr_at_fpr", "f1"}:
        pairs = join_by_id(pred, truth)
        if args.metric == "f1":
            score = f1_binary([r["label"] for r, _ in pairs], [r["label"] for _, r in pairs])
        else:
            scores = [r["score"] for r, _ in pairs]
            labels = [r["label"] for _, r in pairs]
            score = auc(scores, labels) if args.metric == "auc" else tpr_at_fpr(scores, labels, args.fpr)
    else:
        gold_set = {row["id"] for row in truth if row.get("label", 1) == 1}
        ranking = pred[0]["ranking"] if pred and "ranking" in pred[0] else [r["id"] for r in pred]
        score = ndcg_at_k(ranking, gold_set, args.k) if args.metric == "ndcg" else recall_at_k(ranking, gold_set, args.k)

    label = args.metric if args.metric != "tpr_at_fpr" else f"TPR@FPR={args.fpr}"
    print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
