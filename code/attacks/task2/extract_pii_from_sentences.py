#!/usr/bin/env python3
"""Apply extract_pii() regex post-processing to a Task 2 CSV.

Server scores against RAW PII (john@x.com / "1234 5678 9012 3456" / "+12312312312"),
not full sentences. This converts a sentence-form CSV into raw-PII form.

Usage:
    python3 extract_pii_from_sentences.py <input.csv> <output.csv>
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def extract_pii(pred: str, pii_type: str, min_len: int = 10) -> str:
    pred = pred.replace("\n", " ").replace("\r", " ")
    orig = pred.strip()
    if pii_type == "EMAIL":
        m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", pred)
        result = m.group(0) if m else orig
    elif pii_type == "CREDIT":
        m = re.search(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b", pred)
        if m:
            result = m.group(1).strip()
        else:
            m = re.search(r"\b(\d{4}(?:[\s\-]\d{2,4}){2,3})\b", pred)
            result = m.group(1).strip() if m else orig
    elif pii_type == "PHONE":
        m = re.search(r"\+\d[\d\s\-\(\)]{8,14}\d", pred)
        if m:
            result = re.sub(r"[\s\-\(\)]", "", m.group(0))
        else:
            m = re.search(r"\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b", pred)
            result = m.group(0) if m else orig
    else:
        result = orig
    if len(result) < min_len:
        result = orig[:100]
    return result[:100]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <input.csv> <output.csv>", file=sys.stderr)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 1
    n_rows = 0
    n_changed = 0
    with src.open("r", encoding="utf-8", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "pii_type", "pred"])
        writer.writeheader()
        for row in reader:
            orig = row["pred"]
            new = extract_pii(orig, row["pii_type"])
            if new != orig.strip():
                n_changed += 1
            writer.writerow({"id": row["id"], "pii_type": row["pii_type"], "pred": new})
            n_rows += 1
    print(f"rows={n_rows} changed={n_changed} src={src.name} dst={dst.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
