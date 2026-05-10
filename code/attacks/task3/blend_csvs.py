#!/usr/bin/env python3
"""Rank-average blend of multiple submission CSVs.

Usage:
  python blend_csvs.py --csvs A.csv:1.0 B.csv:0.5 --out blend.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="space-separated list of path:weight (weight optional, default=1.0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="rank", choices=["rank", "median", "geomean", "tmean"])
    args = ap.parse_args()

    paths = []
    weights = []
    for spec in args.csvs:
        if ":" in spec:
            p, w = spec.rsplit(":", 1)
            paths.append(p); weights.append(float(w))
        else:
            paths.append(spec); weights.append(1.0)

    dfs = [pd.read_csv(p).sort_values("id").reset_index(drop=True) for p in paths]
    ids = dfs[0]["id"].tolist()
    for d in dfs[1:]:
        assert d["id"].tolist() == ids, "id mismatch"

    scores = np.array([d["score"].values for d in dfs])  # (K, N)
    weights = np.array(weights)

    if args.mode == "rank":
        ranks = np.array([rankdata(s) for s in scores])
        wsum = (ranks.T * weights).sum(axis=1) / weights.sum()
        # normalize to [0.001, 0.999]
        wsum = (wsum - wsum.min()) / (wsum.max() - wsum.min() + 1e-9)
        out = np.clip(wsum, 0.001, 0.999)
    elif args.mode == "median":
        out = np.median(scores, axis=0)
    elif args.mode == "geomean":
        log_s = np.log(np.clip(scores, 1e-6, 1.0))
        wsum = (log_s.T * weights).sum(axis=1) / weights.sum()
        out = np.exp(wsum)
    elif args.mode == "tmean":
        sorted_s = np.sort(scores, axis=0)
        if scores.shape[0] >= 4:
            out = sorted_s[1:-1].mean(axis=0)
        else:
            out = scores.mean(axis=0)
    else:
        raise ValueError(args.mode)

    out = np.clip(out, 0.001, 0.999)
    pd.DataFrame({"id": ids, "score": out}).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(ids)} rows, mode={args.mode})")
    print(f"  inputs: " + ", ".join(f"{p}(w={w})" for p, w in zip(paths, weights)))


if __name__ == "__main__":
    main()
