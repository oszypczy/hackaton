#!/usr/bin/env python3
"""Per-PII routing ensemble v3 — adds char-position consensus + format enforcement.

Improvements over v2:
- CREDIT: per-position char-majority voting across all 16-digit sources (Q2b research).
  When ≥2 sources have 16-digit form, vote each position independently.
  Length-pad / trim non-conforming preds to 16 digits before voting.
- EMAIL: same qr-or-base routing, but post-swap domain when local-part looks like
  firstname.lastname AND model's domain is in the dummy-cluster (peters/jones/lewis/...).
  Use val_pii frequency prior for top-N domains as candidate replacements.
- PHONE: char-position voting for last 7 digits (suffix) when ≥2 sources agree on +1XXX prefix.

Usage:
    python3 smart_ensemble_v3.py <output.csv> <qr.csv> <base.csv> <K4.csv> [extras...] \\
        --val-pii-parquet <path-to-val_pii-parquet>
"""
from __future__ import annotations

import argparse
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

# Mode-collapse domains from val_pii blank analysis (research error_pattern_analysis P3):
# pred-domain peters/jones/lewis/petersen.com cover 75% of preds, gt distribution unrelated.
HALLUCINATED_DOMAINS = {"peters.com", "jones.com", "lewis.com", "petersen.com",
                        "card.com", "tel.com", "dateofbirth.com", "dateofbirth.net",
                        "pierce.com", "cole.com", "shaw.com", "swaylia.com"}


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


def credit_digits(p: str) -> str:
    return re.sub(r"\D", "", p)


def credit_format(digits16: str) -> str:
    """Format 16-digit string as 4-4-4-4."""
    return f"{digits16[:4]} {digits16[4:8]} {digits16[8:12]} {digits16[12:16]}"


def char_position_consensus_credit(candidates: list[str]) -> str | None:
    """Per-position majority voting over 16-digit candidates.

    Q2b from research: aligned position voting recovers partial digit matches
    even when no single source has the full GT.
    """
    aligned = []
    for c in candidates:
        d = credit_digits(c)
        if len(d) == 16:
            aligned.append(d)
        elif 13 <= len(d) <= 17:
            # Pad/trim to 16 for voting (right-align — credit cards have IIN at front)
            if len(d) > 16:
                aligned.append(d[:16])
            else:
                # Pad with last digit
                aligned.append(d + d[-1] * (16 - len(d)))
    if len(aligned) < 2:
        return None
    voted = ""
    for pos in range(16):
        counts = Counter(c[pos] for c in aligned)
        voted += counts.most_common(1)[0][0]
    return credit_format(voted)


def luhn_valid(digits16: str) -> bool:
    if len(digits16) != 16 or not digits16.isdigit():
        return False
    s = 0
    for i, d in enumerate(reversed(digits16)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        s += n
    return s % 10 == 0


def phone_digits_only(p: str) -> str:
    """+13859159897 → 13859159897 (no +)."""
    return re.sub(r"\D", "", p)


def char_position_consensus_phone(candidates: list[str]) -> str | None:
    """Voting on E.164 phone (12 chars: +1XXXXXXXXXX)."""
    aligned = []
    for c in candidates:
        d = phone_digits_only(c)
        if len(d) == 11 and d.startswith("1"):
            aligned.append(d)
        elif len(d) == 10:
            aligned.append("1" + d)
    if len(aligned) < 2:
        return None
    voted = "+"
    for pos in range(11):
        counts = Counter(c[pos] for c in aligned)
        voted += counts.most_common(1)[0][0]
    return voted


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["id"], row["pii_type"])] = row["pred"]
    return out


def load_val_pii_domains(parquet_path: Path | None) -> list[str]:
    """Extract top-N email domains from val_pii GT for swap fallback."""
    if not parquet_path or not parquet_path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # Try pandas fallback
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
        except ImportError:
            print("WARN: no pyarrow/pandas available for val_pii parquet read", file=sys.stderr)
            return []
        emails = []
        for _, row in df.iterrows():
            if "conversation" in row:
                for turn in row["conversation"]:
                    if isinstance(turn, dict) and "content" in turn:
                        for m in re.finditer(r"[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", str(turn["content"])):
                            emails.append(m.group(1).lower())
        return [d for d, _ in Counter(emails).most_common(20)]
    pf = pq.ParquetFile(parquet_path)
    emails = []
    for batch in pf.iter_batches(columns=["conversation"]):
        for conv in batch.column("conversation").to_pylist():
            for turn in conv:
                if isinstance(turn, dict) and "content" in turn:
                    for m in re.finditer(r"[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", str(turn["content"])):
                        emails.append(m.group(1).lower())
    return [d for d, _ in Counter(emails).most_common(20)]


def email_domain_swap(pred: str, top_domains: list[str]) -> str:
    """If domain looks halucynowane, swap with top-1 freq domain. Keep local-part."""
    if "@" not in pred or not top_domains:
        return pred
    try:
        local, domain = pred.split("@", 1)
    except ValueError:
        return pred
    domain = domain.strip().lower()
    if domain in HALLUCINATED_DOMAINS:
        # Swap to most likely from val_pii
        return f"{local}@{top_domains[0]}"
    return pred


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="smart_ensemble v3")
    parser.add_argument("output_csv")
    parser.add_argument("qr_csv", help="question_repeat CSV (best for EMAIL/PHONE)")
    parser.add_argument("base_csv", help="baseline_v2 CSV (best for CREDIT)")
    parser.add_argument("--extras", nargs="+", default=[], help="extra source CSVs")
    parser.add_argument("--val-pii-parquet", default=None,
                        help="path to validation_pii parquet for EMAIL domain freq prior")
    parser.add_argument("--credit-vote", action="store_true",
                        help="enable char-position voting for CREDIT (Q2b)")
    parser.add_argument("--phone-vote", action="store_true",
                        help="enable char-position voting for PHONE")
    parser.add_argument("--email-domain-swap", action="store_true",
                        help="swap halucynowane domains with val_pii top-1 (Q5b)")
    args = parser.parse_args(argv[1:])

    out_path = Path(args.output_csv)
    qr_path = Path(args.qr_csv)
    base_path = Path(args.base_csv)
    extras = [Path(p) for p in args.extras]

    qr = load_csv(qr_path)
    base = load_csv(base_path)
    extra_csvs = [(p.name, load_csv(p)) for p in extras]

    top_domains = []
    if args.email_domain_swap and args.val_pii_parquet:
        top_domains = load_val_pii_domains(Path(args.val_pii_parquet))
        print(f"val_pii top domains (n={len(top_domains)}): {top_domains[:10]}")

    n_rows = 0
    src_picks: Counter = Counter()
    pii_strategies: Counter = Counter()

    with base_path.open("r", encoding="utf-8", newline="") as fin, \
         out_path.open("w", encoding="utf-8", newline="") as fout:
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

            if pii_type == "EMAIL":
                # Same as v2: qr preferred, fallback to base / extras
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
                        for src, v in [(qr_path.name, qr_val), (base_path.name, base_val), *extra_vals]:
                            if v and regex_fit(v, pii_type) >= 1:
                                chosen_src, chosen_val = src, v
                                break
                        if chosen_val is None:
                            chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                # Q5b — domain swap if hallucinated
                if args.email_domain_swap and top_domains and chosen_val:
                    new_val = email_domain_swap(chosen_val, top_domains)
                    if new_val != chosen_val:
                        chosen_val = new_val
                        chosen_src = f"{chosen_src}+domain_swap"
                pii_strategies[(pii_type, "qr_or_base")] += 1

            elif pii_type == "PHONE":
                # qr preferred for PHONE
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
                        for src, v in [(qr_path.name, qr_val), (base_path.name, base_val), *extra_vals]:
                            if v and regex_fit(v, pii_type) >= 1:
                                chosen_src, chosen_val = src, v
                                break
                        if chosen_val is None:
                            chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                pii_strategies[(pii_type, "qr_or_base")] += 1

            elif pii_type == "CREDIT":
                # CREDIT: collect all non-dummy candidates from all sources
                all_cands: list[tuple[str, str]] = []
                if base_val and not base_dummy and regex_fit(base_val, pii_type) >= 1:
                    all_cands.append((base_path.name, base_val))
                for name, v in extra_vals:
                    if v and not is_dummy(v, pii_type) and regex_fit(v, pii_type) >= 1:
                        all_cands.append((name, v))
                if qr_val and not qr_dummy and qr_fit >= 1:
                    all_cands.append((qr_path.name, qr_val))

                if args.credit_vote and len(all_cands) >= 2:
                    # Q2b — char-position consensus over digits
                    consensus = char_position_consensus_credit([v for _, v in all_cands])
                    if consensus:
                        chosen_src, chosen_val = "char_pos_vote", consensus
                        pii_strategies[(pii_type, "char_pos_vote")] += 1
                    else:
                        # Fallback to non-dummy plurality
                        norm_counts = Counter(re.sub(r"[\s\-]", "", v) for _, v in all_cands)
                        top_norm, _ = norm_counts.most_common(1)[0]
                        for name, v in all_cands:
                            if re.sub(r"[\s\-]", "", v) == top_norm:
                                chosen_src, chosen_val = name, v
                                break
                        pii_strategies[(pii_type, "non_dummy_vote")] += 1
                elif all_cands:
                    # Plurality vote (v2 logic)
                    norm_counts = Counter(re.sub(r"[\s\-]", "", v) for _, v in all_cands)
                    top_norm, _ = norm_counts.most_common(1)[0]
                    for name, v in all_cands:
                        if re.sub(r"[\s\-]", "", v) == top_norm:
                            chosen_src, chosen_val = name, v
                            break
                    pii_strategies[(pii_type, "non_dummy_vote")] += 1
                else:
                    chosen_src, chosen_val = base_path.name, base_val if base_val else qr_val
                    pii_strategies[(pii_type, "fallback")] += 1

                # Q6a: format enforcement — if chosen_val has 13-15 digits, pad to 16
                if chosen_val:
                    d = credit_digits(chosen_val)
                    if 13 <= len(d) <= 15:
                        d = d + d[-1] * (16 - len(d))
                        chosen_val = credit_format(d)
                        chosen_src = f"{chosen_src}+pad16"
                    elif len(d) == 16 and not CREDIT_RE.match(chosen_val):
                        chosen_val = credit_format(d)
                        chosen_src = f"{chosen_src}+reformat"

            src_picks[chosen_src] += 1
            writer.writerow({"id": row["id"], "pii_type": pii_type,
                             "pred": (chosen_val or "")[:100]})
            n_rows += 1

    print(f"rows={n_rows}")
    print("source pick distribution:")
    for k, v in src_picks.most_common(15):
        print(f"  {k}: {v}")
    print("pii strategy distribution:")
    for (pii, strat), cnt in pii_strategies.most_common():
        print(f"  {pii} via {strat}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
