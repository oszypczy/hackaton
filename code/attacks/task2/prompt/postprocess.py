"""Phase 0 post-processor — apply error-pattern fixes to v2 CSV.

Fixes (from findings/error_pattern_analysis.md):
- P1 CREDIT length+format normalize → \\d{4} \\d{4} \\d{4} \\d{4}
- P4 EMAIL template-echo blacklist + v1 fallback
- P5 PHONE +1?555\\d{7} placeholder + v1 fallback
- P7 PHONE country-code reformat → +1\\d{10}

Usage:
    python postprocess.py <v2_csv> <v1_csv> <out_csv>
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

DIGIT = re.compile(r"\d")
EMAIL_BLACKLIST = re.compile(
    r"^(card|tel)@(dateofbirth|card|email|phone|name)\.(com|net|org)$|"
    r"^[^@]*@example\.(com|net|org)$",
    re.IGNORECASE,
)
PHONE_PLACEHOLDER = re.compile(r"^\+?1?555\d{7}$|^\+?15555550{4}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+1\d{10}$")
CREDIT_RE = re.compile(r"^\d{4} \d{4} \d{4} \d{4}$")


def fix_credit(pred: str) -> str:
    """Normalize CREDIT to 16-digit 4-4-4-4 format. Pad with last digit if 13<=n<16, truncate if n>16."""
    digits = "".join(DIGIT.findall(pred))
    if len(digits) == 0:
        return pred  # nothing to do; leave as-is
    if 13 <= len(digits) < 16:
        digits = digits + digits[-1] * (16 - len(digits))
    elif len(digits) > 16:
        digits = digits[:16]
    elif len(digits) < 13:
        # too truncated — leave original (no good signal)
        return pred
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"


def fix_phone(pred: str) -> str:
    """Reformat to +1\\d{10} when possible; leave structurally broken phones alone."""
    digits = "".join(DIGIT.findall(pred))
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) > 11 and digits.startswith("1"):
        return f"+{digits[:11]}"
    return pred


def is_email_template_echo(pred: str) -> bool:
    return bool(EMAIL_BLACKLIST.match(pred.strip()))


def is_phone_placeholder(pred: str) -> bool:
    digits = "".join(DIGIT.findall(pred))
    if not digits:
        return False
    # +15555550000 family — placeholder/refusal proxy
    if PHONE_PLACEHOLDER.match(pred.strip()):
        return True
    # also match any phone where last 7 digits are 5555000 or similar refusal signature
    if "555555" in digits or digits.endswith("5555550000"):
        return True
    return False


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row["id"], row["pii_type"])] = row["pred"]
    return out


def main(v2_path: str, v1_path: str, out_path: str) -> None:
    v1 = load_csv(Path(v1_path))

    counts = {
        "CREDIT_reformatted": 0,
        "EMAIL_template_fallback": 0,
        "PHONE_placeholder_fallback": 0,
        "PHONE_reformatted": 0,
        "rows_changed": 0,
        "rows_total": 0,
    }

    rows_out: list[tuple[str, str, str]] = []
    # preserve order of v2 input
    with Path(v2_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["rows_total"] += 1
            uid, pii, pred = row["id"], row["pii_type"], row["pred"]
            original = pred

            if pii == "CREDIT":
                fixed = fix_credit(pred)
                if fixed != pred:
                    counts["CREDIT_reformatted"] += 1
                pred = fixed

            elif pii == "EMAIL":
                if is_email_template_echo(pred):
                    fb = v1.get((uid, pii), pred)
                    # only accept fallback if v1 is structurally a valid email AND not also template
                    if (
                        fb
                        and EMAIL_RE.match(fb.strip())
                        and not is_email_template_echo(fb)
                        and len(fb) <= 50
                    ):
                        pred = fb.strip()
                        counts["EMAIL_template_fallback"] += 1

            elif pii == "PHONE":
                if is_phone_placeholder(pred):
                    fb = v1.get((uid, pii), pred)
                    fb_digits = "".join(DIGIT.findall(fb)) if fb else ""
                    # accept fallback if v1 has 10-11 digits AND is not also placeholder
                    if (
                        fb
                        and 10 <= len(fb_digits) <= 12
                        and not is_phone_placeholder(fb)
                        and "555555" not in fb_digits
                    ):
                        pred = fb.strip()
                        counts["PHONE_placeholder_fallback"] += 1
                # always normalize country-code format
                reformatted = fix_phone(pred)
                if reformatted != pred:
                    counts["PHONE_reformatted"] += 1
                    pred = reformatted

            if pred != original:
                counts["rows_changed"] += 1

            # final sanitize
            pred = pred.replace("\r", " ").replace("\n", " ")
            pred = re.sub(r"\s+", " ", pred).strip()
            if not pred or len(pred) < 10:
                pred = original  # don't go below floor
            if len(pred) > 100:
                pred = pred[:100]

            rows_out.append((uid, pii, pred))

    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pii_type", "pred"])
        for uid, pii, pred in rows_out:
            w.writerow([uid, pii, pred])

    print(f"Wrote {out_path}")
    print(f"Rows total      : {counts['rows_total']}")
    print(f"Rows changed    : {counts['rows_changed']}")
    print(f"  CREDIT reformatted        : {counts['CREDIT_reformatted']}")
    print(f"  EMAIL template fallback   : {counts['EMAIL_template_fallback']}")
    print(f"  PHONE placeholder fallback: {counts['PHONE_placeholder_fallback']}")
    print(f"  PHONE reformatted         : {counts['PHONE_reformatted']}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
