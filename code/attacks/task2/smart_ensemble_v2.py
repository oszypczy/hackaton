#!/usr/bin/env python3
"""Per-PII type routing ensemble (v2).

Strategy per pii_type:
- EMAIL: question_repeat preferred (val=0.55 + on test it lifted +0.008)
- CREDIT: question_repeat is mostly dummy (986/1000) → vote non-dummy from baseline+v0 sources
- PHONE: question_repeat preferred (val=0.32, lifted on test)

For CREDIT non-dummy voting: count value frequency among non-dummy candidates.
If tie → prefer baseline_v2 (most-trained source).

Usage:
    python3 smart_ensemble_v2.py <output.csv> <question_repeat.csv> <baseline_v2.csv> [extras...]
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
        if len(digits) >= 12 and len(set(digits)) <= 2:
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


def normalize_credit(p: str) -> str:
    """Normalize CREDIT for voting: collapse whitespace/dashes for comparison."""
    return re.sub(r"[\s\-]", "", p)


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
    extra_csvs = [(p.name, load_csv(p)) for p in extras]

    n_rows = 0
    src_picks: Counter = Counter()
    pii_strategies: Counter = Counter()

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
            extra_vals = [(name, c.get(key, "").strip()) for (name, c) in extra_csvs]

            chosen_src = None
            chosen_val = None

            qr_fit = regex_fit(qr_val, pii_type) if qr_val else -1
            qr_dummy = is_dummy(qr_val, pii_type) if qr_val else True
            base_fit = regex_fit(base_val, pii_type) if base_val else -1
            base_dummy = is_dummy(base_val, pii_type) if base_val else True

            if pii_type in ("EMAIL", "PHONE"):
                # Prefer qr; fallback to base; fallback to first non-dummy fit≥1 in extras
                if qr_val and qr_fit >= 2 and not qr_dummy:
                    chosen_src, chosen_val = qr_path.name, qr_val
                elif base_val and base_fit >= 2 and not base_dummy:
                    chosen_src, chosen_val = base_path.name, base_val
                else:
                    for name, v in extra_vals:
                        if v and regex_fit(v, pii_type) >= 2 and not is_dummy(v, pii_type):
                            chosen_src, chosen_val = name, v
                            break
                    if chosen_val is None:
                        # Last resort: qr (fit≥1) → base → any
                        for src, v in [(qr_path.name, qr_val), (base_path.name, base_val), *extra_vals]:
                            if v and regex_fit(v, pii_type) >= 1:
                                chosen_src, chosen_val = src, v
                                break
                        if chosen_val is None:
                            chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                pii_strategies[(pii_type, "qr_or_base")] += 1
            elif pii_type == "CREDIT":
                # CREDIT: qr is mostly dummy. Vote among non-dummy non-qr sources.
                non_dummy_cands = []
                if base_val and not base_dummy and regex_fit(base_val, pii_type) >= 1:
                    non_dummy_cands.append((base_path.name, base_val))
                for name, v in extra_vals:
                    if v and not is_dummy(v, pii_type) and regex_fit(v, pii_type) >= 1:
                        non_dummy_cands.append((name, v))
                # Add qr only if non-dummy + well-formed
                if qr_val and not qr_dummy and qr_fit >= 1:
                    non_dummy_cands.append((qr_path.name, qr_val))

                if non_dummy_cands:
                    # Vote on normalized digit form
                    norm_counts = Counter(normalize_credit(v) for _, v in non_dummy_cands)
                    top_norm, _ = norm_counts.most_common(1)[0]
                    # Pick first source with that normalized value (preserves formatting)
                    for name, v in non_dummy_cands:
                        if normalize_credit(v) == top_norm:
                            chosen_src, chosen_val = name, v
                            break
                else:
                    # All sources dummy → fallback to base
                    chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                pii_strategies[(pii_type, "non_dummy_vote")] += 1

            src_picks[chosen_src] += 1
            writer.writerow({"id": row["id"], "pii_type": pii_type, "pred": (chosen_val or "")[:100]})
            n_rows += 1

    print(f"rows={n_rows}")
    print("source pick distribution:")
    for k, v in src_picks.most_common():
        print(f"  {k}: {v}")
    print("pii strategy distribution:")
    for (pii, strat), cnt in pii_strategies.most_common():
        print(f"  {pii} via {strat}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
