#!/usr/bin/env python3
"""Ensemble Task 2 PII predictions across multiple CSV variants.

Per (id, pii_type) row:
1. Score each candidate by regex match for the pii_type (well-formed = +1).
2. If multiple candidates tie at top fitness, pick the most-frequent value.
3. Tie-break: prefer the canonical CSV listed first in `inputs`.

Usage:
    python3 ensemble.py <output.csv> <ref_csv.csv> <input1.csv> <input2.csv> ...

`ref_csv` defines the (id, pii_type) row order and is also used as ultimate fallback.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
CREDIT_RE = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$")
CREDIT_LOOSE_RE = re.compile(r"^\d{4}(?:[\s\-]\d{2,4}){2,3}$")
PHONE_RE = re.compile(r"^\+\d{8,15}$")
PHONE_LOOSE_RE = re.compile(r"^\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}$")


def regex_fit(pred: str, pii_type: str) -> int:
    """Higher = better-formed."""
    p = pred.strip()
    if pii_type == "EMAIL":
        return 2 if EMAIL_RE.match(p) else 0
    if pii_type == "CREDIT":
        if CREDIT_RE.match(p):
            return 2
        if CREDIT_LOOSE_RE.match(p):
            return 1
        return 0
    if pii_type == "PHONE":
        if PHONE_RE.match(p):
            return 2
        if PHONE_LOOSE_RE.match(p):
            return 1
        return 0
    return 0


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["id"], row["pii_type"])
            out[key] = row["pred"]
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(f"Usage: {argv[0]} <output.csv> <ref_csv.csv> <input1.csv> [input2.csv ...]", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    ref_path = Path(argv[2])
    in_paths = [Path(p) for p in argv[3:]]
    inputs = [load_csv(p) for p in in_paths]
    ref = load_csv(ref_path)

    # Stats counters
    n_rows = 0
    fit_counts: Counter = Counter()
    src_picks: Counter = Counter()

    with ref_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "pii_type", "pred"])
        writer.writeheader()
        for row in reader:
            key = (row["id"], row["pii_type"])
            pii_type = row["pii_type"]
            cands = []  # (fit, freq, src_idx, value)
            seen_vals = []
            for idx, src in enumerate(inputs):
                val = src.get(key, ref.get(key, "")).strip()
                if not val:
                    continue
                fit = regex_fit(val, pii_type)
                seen_vals.append(val)
                cands.append((fit, idx, val))
            if not cands:
                final = row["pred"]
                src_picks["fallback"] += 1
            else:
                max_fit = max(c[0] for c in cands)
                top = [c for c in cands if c[0] == max_fit]
                # Most-frequent among top fitness
                vc = Counter(c[2] for c in top)
                most_common_val, _ = vc.most_common(1)[0]
                # Source index of first occurrence
                src_idx = next(c[1] for c in top if c[2] == most_common_val)
                final = most_common_val
                fit_counts[max_fit] += 1
                src_picks[in_paths[src_idx].name] += 1
            writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": final[:100]})
            n_rows += 1

    print(f"rows={n_rows}")
    print("fit distribution:", dict(fit_counts))
    print("source pick distribution:")
    for k, v in src_picks.most_common():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
