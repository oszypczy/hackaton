#!/usr/bin/env python3
"""Empirical green-list watermark stealing via Fisher's exact test on TRAIN only.

Steps:
1. Tokenize all labeled (train) texts with GPT-2 tokenizer
2. For each token, compute count_in_watermarked vs count_in_clean
3. Fisher's exact test → p-value per token
4. "Green tokens" = tokens with significant p-value AND count_wm > count_clean
5. Per-text feature: green_ratio = green_token_count / total_tokens

Key: green list derived from TRAIN ONLY → no leakage into val/test scoring.
Output: features_empirical_green.pkl (DataFrame with 1-3 columns per text)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def load_train(data_dir: Path) -> tuple[list[str], list[int]]:
    import json
    rows = []
    for fname, lbl in [("train_clean.jsonl", 0), ("train_wm.jsonl", 1)]:
        for line in (data_dir / fname).read_text().splitlines():
            if line.strip():
                rows.append((json.loads(line)["text"], lbl))
    return [r[0] for r in rows], [r[1] for r in rows]


def fit_fisher_greenlist(train_texts: list[str], train_labels: list[int],
                         tokenizer, top_k: int = 1500, min_count: int = 3):
    """Returns set of green-list token IDs."""
    from scipy.stats import fisher_exact

    pos_counts = Counter()
    neg_counts = Counter()
    total_pos_tokens = 0
    total_neg_tokens = 0

    for txt, lbl in zip(train_texts, train_labels):
        ids = tokenizer.encode(txt, add_special_tokens=False)
        c = Counter(ids)
        if lbl == 1:
            for k, v in c.items():
                pos_counts[k] += v
            total_pos_tokens += len(ids)
        else:
            for k, v in c.items():
                neg_counts[k] += v
            total_neg_tokens += len(ids)

    # Score each token by Fisher exact (one-sided)
    candidate_tokens = set(pos_counts) | set(neg_counts)
    scored = []
    for tok in candidate_tokens:
        a = pos_counts.get(tok, 0)
        b = neg_counts.get(tok, 0)
        if a + b < min_count:
            continue
        # 2x2 table: [a, b], [pos_total - a, neg_total - b]
        table = [[a, b], [total_pos_tokens - a, total_neg_tokens - b]]
        try:
            odds, p = fisher_exact(table, alternative="greater")
            if a > 0 and odds > 1.0:  # over-represented in positive
                scored.append((tok, p, odds, a, b))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1])  # by p-value ascending
    green_set = {tok for tok, p, odds, a, b in scored[:top_k]}
    print(f"  fitted Fisher green list: {len(green_set)} tokens (top {top_k} by p-value)")
    print(f"  total tokens analyzed: {len(candidate_tokens)}")
    print(f"  pos_tokens={total_pos_tokens} neg_tokens={total_neg_tokens}")
    if scored:
        top5 = scored[:5]
        for tok, p, odds, a, b in top5:
            try:
                tk = tokenizer.decode([tok])
            except Exception:
                tk = f"<{tok}>"
            print(f"    {tok!r:8} {tk!r:15} p={p:.3e} odds={odds:.2f} pos={a} neg={b}")
    return green_set


def extract_green_features(texts: list[str], tokenizer, green_set: set) -> pd.DataFrame:
    rows = []
    for txt in texts:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if not ids:
            rows.append({"emp_green_ratio": 0.0, "emp_green_zscore": 0.0, "emp_green_count": 0.0})
            continue
        green_count = sum(1 for t in ids if t in green_set)
        n = len(ids)
        ratio = green_count / n
        # z-score against null hypothesis (mean of green list size proportion)
        # if green_set is X% of 50257, expected ~ratio = X/50257
        # but green list is biased — so use empirical mean & std from null distribution
        # simpler: just standard binomial z-score with p=0.5
        # actually just raw ratio works as feature; classifier will handle
        rows.append({
            "emp_green_ratio": ratio,
            "emp_green_zscore": (green_count - n / 2) / max(np.sqrt(n / 4), 1e-6),
            "emp_green_count": green_count,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=1500)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--out-name", default="emp_green")
    args = p.parse_args()

    print("Loading TRAIN only (no val/test)...")
    train_texts, train_labels = load_train(args.data_dir)
    print(f"  train: {len(train_texts)} samples, {sum(train_labels)} positive")

    print("Loading GPT-2 tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    print(f"\nFitting Fisher green list (top_k={args.top_k}, min_count={args.min_count})...")
    green_set = fit_fisher_greenlist(train_texts, train_labels, tok,
                                     top_k=args.top_k, min_count=args.min_count)

    print("\nLoading ALL labeled+test texts...")
    import json
    all_texts = []
    for fname in ["train_clean.jsonl", "train_wm.jsonl",
                  "valid_clean.jsonl", "valid_wm.jsonl", "test.jsonl"]:
        for line in (args.data_dir / fname).read_text().splitlines():
            if line.strip():
                all_texts.append(json.loads(line)["text"])
    print(f"  total texts: {len(all_texts)}")

    print("Extracting features for all texts...")
    df = extract_green_features(all_texts, tok, green_set)
    print(f"  shape: {df.shape}")

    cache_path = args.cache_dir / f"features_{args.out_name}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"\nSaved: {cache_path}")
    print(f"  emp_green_ratio: mean={df['emp_green_ratio'].mean():.3f} std={df['emp_green_ratio'].std():.3f}")
    print(f"  emp_green_zscore: mean={df['emp_green_zscore'].mean():.3f} std={df['emp_green_zscore'].std():.3f}")


if __name__ == "__main__":
    main()
