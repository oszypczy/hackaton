#!/usr/bin/env python3
"""Majority-vote ensemble. For each (id, pii_type), pick the most-frequent value
across source CSVs (skipping dummy/empty). Tie-break: source order in argv.

Usage:
    python3 majority_vote.py <output.csv> <ref_for_row_order.csv> <source1.csv> [source2.csv ...]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


def is_dummy(pred: str, pii_type: str) -> bool:
    p = pred.strip()
    if pii_type == "CREDIT":
        digits = re.sub(r"[\s\-]", "", p)
        if len(digits) >= 12 and len(set(digits)) <= 2:
            return True
        return False
    if pii_type == "EMAIL":
        if "@" not in p:
            return False
        try:
            local, domain = p.lower().split("@", 1)
        except ValueError:
            return False
        if domain in {"example.com", "test.com", "domain.com", "email.com", "x.com"}:
            return True
        return False
    if pii_type == "PHONE":
        digits = re.sub(r"[\D]", "", p)
        if len(digits) >= 8 and len(set(digits)) <= 2:
            return True
        return False
    return False


def normalize(p: str, pii_type: str) -> str:
    if pii_type == "CREDIT":
        return re.sub(r"[\s\-]", "", p.strip())
    if pii_type == "PHONE":
        return re.sub(r"[\s\-\.\(\)]", "", p.strip())
    return p.strip().lower()


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["id"], row["pii_type"])] = row["pred"]
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(f"Usage: {argv[0]} <output.csv> <ref_for_row_order.csv> <source1.csv> [source2.csv ...]", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    ref_path = Path(argv[2])
    src_paths = [Path(p) for p in argv[3:]]
    sources = [(p.name, load_csv(p)) for p in src_paths]

    n_rows = 0
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
            cands: list[tuple[str, str]] = []  # (src_name, raw_value)
            for name, csv_data in sources:
                v = csv_data.get(key, "").strip()
                if v and not is_dummy(v, pii_type):
                    cands.append((name, v))
            if not cands:
                # All dummy — fallback to first source raw
                for name, csv_data in sources:
                    v = csv_data.get(key, "").strip()
                    if v:
                        cands.append((name, v))
                        break
            if not cands:
                writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": row["pred"]})
                n_rows += 1
                src_picks["fallback_ref"] += 1
                continue
            # Majority vote on normalized form
            norm_counts = Counter(normalize(v, pii_type) for _, v in cands)
            top_norm, _ = norm_counts.most_common(1)[0]
            # First source matching the winning normalized form
            for name, v in cands:
                if normalize(v, pii_type) == top_norm:
                    chosen = (name, v)
                    break
            src_picks[chosen[0]] += 1
            writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": chosen[1][:100]})
            n_rows += 1

    print(f"rows={n_rows}")
    print("source pick distribution:")
    for k, v in src_picks.most_common():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
