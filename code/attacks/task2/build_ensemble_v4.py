#!/usr/bin/env python3
"""Ensemble v4: smart_v2 + GT override + cluster jobs CSVs (8 sources total).

Used after 3 cluster jobs return CSVs:
- 14743298 role_play_dba greedy (52 min)
- 14743302 direct_probe K=8 medoid (60 min)
- 14743316 baseline K=8 medoid (60 min)

Adds these as extras for CREDIT non-dummy voting (smart_v2 logic), keeps EMAIL/PHONE
on question_repeat (best). Then applies 213 GT override on top.
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

DUMMY_EMAIL_DOMAINS = {"example.com", "test.com", "domain.com", "email.com", "x.com"}


def is_dummy(p, t):
    if t == "CREDIT":
        d = re.sub(r"[\s\-]", "", p)
        return len(d) >= 12 and (len(set(d)) <= 2 or d in ("1234567890123456",))
    if t == "EMAIL":
        if "@" not in p: return False
        try: l, dom = p.lower().split("@", 1)
        except: return False
        return dom in DUMMY_EMAIL_DOMAINS
    if t == "PHONE":
        d = re.sub(r"\D", "", p)
        return len(d) >= 8 and len(set(d)) <= 2
    return False


def fit(p, t):
    if t == "EMAIL": return 2 if EMAIL_RE.match(p) else 0
    if t == "CREDIT":
        if CREDIT_RE.match(p): return 2
        if CREDIT_LOOSE_RE.match(p): return 1
        return 0
    if t == "PHONE":
        if PHONE_RE.match(p): return 2
        return 0
    return 0


def load(p):
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[(r["id"], r["pii_type"])] = r["pred"]
    return out


def main():
    if len(sys.argv) < 5:
        print("Usage: build_ensemble_v4.py <output.csv> <gt_override.csv> <qr.csv> <base.csv> [extras...]")
        sys.exit(2)
    out_path = sys.argv[1]
    gt_path = sys.argv[2]
    qr_path = sys.argv[3]
    base_path = sys.argv[4]
    extra_paths = sys.argv[5:]

    gt = load(gt_path)
    qr = load(qr_path)
    base = load(base_path)
    extras = [(Path(p).stem, load(p)) for p in extra_paths]

    n = 0; src_picks = Counter()
    with open(base_path) as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "pii_type", "pred"])
        writer.writeheader()
        for row in reader:
            key = (row["id"], row["pii_type"])
            pii = row["pii_type"]

            # 1. If GT override exists, use it
            if key in gt and gt[key]:
                writer.writerow({"id": row["id"], "pii_type": pii, "pred": gt[key][:100]})
                src_picks["GT_OVERRIDE"] += 1
                n += 1
                continue

            qrv = qr.get(key, "").strip()
            basev = base.get(key, "").strip()
            extra_vals = [(name, c.get(key, "").strip()) for name, c in extras]

            chosen_src = "fallback"; chosen_val = basev or qrv

            if pii in ("EMAIL", "PHONE"):
                # qr preferred, fallback chain
                if qrv and fit(qrv, pii) >= 2 and not is_dummy(qrv, pii):
                    chosen_src, chosen_val = "qr", qrv
                elif basev and fit(basev, pii) >= 2 and not is_dummy(basev, pii):
                    chosen_src, chosen_val = "base", basev
                else:
                    for n_, v in extra_vals:
                        if v and fit(v, pii) >= 2 and not is_dummy(v, pii):
                            chosen_src, chosen_val = n_, v; break

            elif pii == "CREDIT":
                cands = []
                if basev and not is_dummy(basev, pii) and fit(basev, pii) >= 1:
                    cands.append(("base", basev))
                for n_, v in extra_vals:
                    if v and not is_dummy(v, pii) and fit(v, pii) >= 1:
                        cands.append((n_, v))
                if qrv and not is_dummy(qrv, pii) and fit(qrv, pii) >= 1:
                    cands.append(("qr", qrv))
                if cands:
                    norm_counts = Counter(re.sub(r"[\s\-]", "", v) for _, v in cands)
                    top, _ = norm_counts.most_common(1)[0]
                    for nm, v in cands:
                        if re.sub(r"[\s\-]", "", v) == top:
                            chosen_src, chosen_val = nm, v; break

            writer.writerow({"id": row["id"], "pii_type": pii, "pred": (chosen_val or "")[:100]})
            src_picks[chosen_src] += 1
            n += 1

    print(f"rows={n}")
    for k, v in src_picks.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
