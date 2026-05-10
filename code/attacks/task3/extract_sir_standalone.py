#!/usr/bin/env python3
"""Standalone SIR feature extractor — saves features_sir.pkl."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from tqdm import tqdm
from features.sir_direct import extract


def main():
    data_dir = Path(sys.argv[1])
    cache_dir = Path(sys.argv[2])
    out_file = cache_dir / "features_sir.pkl"

    all_texts = []
    for fname in ["train_clean.jsonl", "train_wm.jsonl",
                  "valid_clean.jsonl", "valid_wm.jsonl", "test.jsonl"]:
        for line in (data_dir / fname).read_text().splitlines():
            if line.strip():
                all_texts.append(json.loads(line)["text"])
    print(f"Total texts: {len(all_texts)}")

    rows = []
    for t in tqdm(all_texts, ncols=80):
        rows.append(extract(t))

    df = pd.DataFrame(rows).fillna(0.0)
    print(f"Features shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
