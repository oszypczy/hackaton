"""Rebuild submission CSV from a predict JSON dump using ordered (user_id, pii_type)
expected pairs from the parquet (so leading-zero IDs are preserved).

Usage:
  python rebuild_csv.py <order_json> <predict_json> <out_csv>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def _sanitize(pred: str) -> str:
    p = (pred or "").strip()
    p = p.replace("\r", " ").replace("\n", " ")
    p = re.sub(r"\s+", " ", p).strip()
    if len(p) < 10:
        p = (p + "0000000000")[:10]
    if len(p) > 100:
        p = p[:100]
    return p


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("Usage: rebuild_csv.py <order_json> <predict_json> <out_csv>", file=sys.stderr)
        return 2
    order = json.loads(Path(argv[1]).read_text())
    rows = json.loads(Path(argv[2]).read_text())
    if len(order) != len(rows):
        raise ValueError(f"length mismatch: order={len(order)} rows={len(rows)}")
    for i, (o, r) in enumerate(zip(order, rows)):
        if o["pii_type"] != r["pii_type"]:
            raise ValueError(f"pii_type mismatch at {i}: order={o} row={r}")
    out = Path(argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pii_type", "pred"])
        for o, r in zip(order, rows):
            w.writerow([o["user_id"], o["pii_type"], _sanitize(r["pred"])])
    print(f"wrote {out}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
