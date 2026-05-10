#!/usr/bin/env python3
"""Rank-average ensemble of multiple Task 3 submissions.

Usage:
  python scripts/ensemble_submissions.py <out.csv> <csv1> [csv2 ...]
  python scripts/ensemble_submissions.py submissions/task3_ensemble.csv \\
    submissions/task3_watermark_strong_bino.csv \\
    submissions/task3_watermark_fdgpt.csv \\
    submissions/task3_watermark_roberta.csv

For each test ID, take the rank within each submission, average ranks,
re-normalize to [0, 1]. Reduces variance when individual submissions
err on different samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("Usage: ensemble_submissions.py <out.csv> <csv1> <csv2> [csv3 ...]", file=sys.stderr)
        return 2

    out_path = Path(argv[1])
    in_paths = [Path(p) for p in argv[2:]]
    for p in in_paths:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            return 2

    dfs = [pd.read_csv(p).sort_values("id").reset_index(drop=True) for p in in_paths]
    n = len(dfs[0])
    for df, p in zip(dfs, in_paths):
        if len(df) != n:
            print(f"Mismatched rows in {p}: {len(df)} vs {n}", file=sys.stderr)
            return 2
        if list(df["id"]) != list(dfs[0]["id"]):
            print(f"Mismatched ids in {p}", file=sys.stderr)
            return 2

    # Rank-normalize each submission's scores then average
    rank_matrix = np.zeros((n, len(dfs)))
    for i, df in enumerate(dfs):
        ranks = df["score"].rank(method="average").values  # 1..n
        rank_matrix[:, i] = (ranks - 1) / (n - 1)  # normalize to [0,1]
        print(f"  [{i}] {in_paths[i].name}  mean_rank={rank_matrix[:, i].mean():.3f}")

    avg_rank = rank_matrix.mean(axis=1)

    out = pd.DataFrame({"id": dfs[0]["id"], "score": avg_rank})
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble: {out_path} ({n} rows)")
    print(f"  score range: [{out['score'].min():.3f}, {out['score'].max():.3f}]")
    print(f"  unique rounded(2): {len(np.unique(np.round(out['score'], 2)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
