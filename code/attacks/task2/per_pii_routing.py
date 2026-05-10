#!/usr/bin/env python3
"""Pure per-PII type routing — pick exactly one source per pii_type.

Usage:
    python3 per_pii_routing.py <output.csv> <email_src.csv> <credit_src.csv> <phone_src.csv> <fallback.csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["id"], row["pii_type"])] = row["pred"]
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(f"Usage: {argv[0]} <output.csv> <email_src.csv> <credit_src.csv> <phone_src.csv> <fallback.csv>", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    email_src = load_csv(Path(argv[2]))
    credit_src = load_csv(Path(argv[3]))
    phone_src = load_csv(Path(argv[4]))
    fallback = load_csv(Path(argv[5]))
    fallback_path = Path(argv[5])

    n_rows = 0
    with fallback_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "pii_type", "pred"])
        writer.writeheader()
        for row in reader:
            key = (row["id"], row["pii_type"])
            pii_type = row["pii_type"]
            if pii_type == "EMAIL":
                val = email_src.get(key, fallback.get(key, "")).strip()
            elif pii_type == "CREDIT":
                val = credit_src.get(key, fallback.get(key, "")).strip()
            elif pii_type == "PHONE":
                val = phone_src.get(key, fallback.get(key, "")).strip()
            else:
                val = fallback.get(key, "").strip()
            writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": val[:100]})
            n_rows += 1
    print(f"rows={n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
