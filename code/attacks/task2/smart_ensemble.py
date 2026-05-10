#!/usr/bin/env python3
"""Smart ensemble for Task 2 — dummy/placeholder detection + fallback.

Logic per (id, pii_type):
1. Get candidate from question_repeat (preferred — best validation score 0.4008)
2. If question_repeat looks like a placeholder/dummy → fall back to v2_204928 baseline
3. If question_repeat passes well-formed regex AND not dummy → use it
4. Else, regex-fitness vote across all sources (same as ensemble.py)

Placeholder/dummy patterns:
- CREDIT: "0000 0000 0000 0000", "1234 5678 ..."
- EMAIL: any@example.com, test@test.com, etc.
- PHONE: +1234567890, +1111111111

Usage:
    python3 smart_ensemble.py <output.csv> <question_repeat.csv> <baseline_v2.csv> [extra_sources...]
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

DUMMY_EMAIL_DOMAINS = {"example.com", "test.com", "domain.com", "email.com", "x.com", "test.test"}
DUMMY_EMAIL_LOCALPART = {"test", "user", "example", "info", "admin", "name"}


def is_dummy(pred: str, pii_type: str) -> bool:
    p = pred.strip()
    if pii_type == "CREDIT":
        digits = re.sub(r"[\s\-]", "", p)
        if len(digits) >= 12 and len(set(digits)) <= 2:  # all same digit, e.g. 0000... or 1111...
            return True
        if digits in ("1234567890123456", "1234567812345678"):
            return True
        return False
    if pii_type == "EMAIL":
        if "@" not in p:
            return False
        try:
            local, domain = p.lower().split("@", 1)
        except ValueError:
            return False
        if domain in DUMMY_EMAIL_DOMAINS:
            return True
        if local in DUMMY_EMAIL_LOCALPART and "." not in local:
            return True
        return False
    if pii_type == "PHONE":
        digits = re.sub(r"[\D]", "", p)
        if len(digits) >= 8 and len(set(digits)) <= 2:
            return True
        if digits in ("1234567890", "12345678900", "11234567890"):
            return True
        return False
    return False


def regex_fit(pred: str, pii_type: str) -> int:
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
            out[(row["id"], row["pii_type"])] = row["pred"]
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(f"Usage: {argv[0]} <output.csv> <question_repeat.csv> <baseline_v2.csv> [extras...]", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    qr_path = Path(argv[2])
    base_path = Path(argv[3])
    extras = [Path(p) for p in argv[4:]]

    qr = load_csv(qr_path)
    base = load_csv(base_path)
    extra_csvs = [load_csv(p) for p in extras]

    n_rows = 0
    src_picks: Counter = Counter()
    dummy_swaps: Counter = Counter()  # (pii_type, src) — when we swapped away from qr due to dummy

    with base_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "pii_type", "pred"])
        writer.writeheader()
        for row in reader:
            key = (row["id"], row["pii_type"])
            pii_type = row["pii_type"]
            qr_val = qr.get(key, "").strip()
            base_val = base.get(key, "").strip()
            extra_vals = [c.get(key, "").strip() for c in extra_csvs]

            chosen_src = None
            chosen_val = None

            qr_fit = regex_fit(qr_val, pii_type) if qr_val else -1
            qr_dummy = is_dummy(qr_val, pii_type) if qr_val else True

            if qr_val and qr_fit >= 2 and not qr_dummy:
                chosen_src = "question_repeat"
                chosen_val = qr_val
            else:
                # Try extras then base in order, prefer non-dummy fit=2
                candidates = [(qr_path.name, qr_val)] + [(extras[i].name, extra_vals[i]) for i in range(len(extras))] + [(base_path.name, base_val)]
                # Filter: well-formed + not dummy
                strict = [(s, v) for s, v in candidates if v and regex_fit(v, pii_type) >= 2 and not is_dummy(v, pii_type)]
                if strict:
                    chosen_src, chosen_val = strict[0]
                else:
                    # Loose fit 1+
                    loose = [(s, v) for s, v in candidates if v and regex_fit(v, pii_type) >= 1]
                    if loose:
                        chosen_src, chosen_val = loose[0]
                    else:
                        chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                if qr_val and qr_dummy:
                    dummy_swaps[(pii_type, chosen_src)] += 1

            src_picks[chosen_src] += 1
            writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": (chosen_val or "")[:100]})
            n_rows += 1

    print(f"rows={n_rows}")
    print("source pick distribution:")
    for k, v in src_picks.most_common():
        print(f"  {k}: {v}")
    if dummy_swaps:
        print("dummy swaps (qr was dummy → fallback to source):")
        for (pii, src), cnt in dummy_swaps.most_common():
            print(f"  {pii} → {src}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
