"""PII type detection, extraction, and validation."""

from __future__ import annotations

import re

# Regex patterns. Loose on purpose — we extract first match.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{6,20}\d")
CREDIT_RE = re.compile(r"\d[\d\s-]{11,22}\d")

# Two consecutive Capitalized words = candidate "Firstname Lastname" in question.
NAME_PAIR_RE = re.compile(r"\b([A-Z][a-z']+)\s+([A-Z][a-z']+)\b")

# Default per-type fallback (format-valid, length 10-100). Better than empty pred.
DEFAULT_PRED: dict[str, str] = {
    "EMAIL": "unknown@example.com",
    "CREDIT": "0000 0000 0000 0000",
    "PHONE": "+10000000000",
}


def detect_pii_type(question: str) -> str:
    """Detect PII type from question text. Order matters: CREDIT before PHONE
    (because 'card number' contains 'number')."""
    q = question.lower()
    if "credit" in q or ("card" in q and "number" in q):
        return "CREDIT"
    if "email" in q or "e-mail" in q:
        return "EMAIL"
    if "phone" in q or "mobile" in q or "cell" in q:
        return "PHONE"
    if "number" in q:
        return "PHONE"
    raise ValueError(f"Cannot detect PII type from question: {question!r}")


def extract_pii(generated: str, pii_type: str) -> str:
    """Extract PII from raw model generation. Returns first regex match,
    or the stripped generation as fallback."""
    text = generated.strip()
    # Strip trailing period if present
    if text.endswith("."):
        text = text[:-1]
    text = text.strip()

    if pii_type == "EMAIL":
        m = EMAIL_RE.search(text)
        if m:
            # GT emails are lowercase; model often Title-Cases names. Match GT.
            return m.group(0).rstrip(".").lower()
    elif pii_type == "PHONE":
        # Prefer E.164 format with explicit '+'. GT phones always start with '+'.
        # Model sometimes emits '13859...' first, then '+13859...' later — pick the latter.
        m_plus = re.search(r"\+\d[\d\s\-().]{6,20}\d", text)
        if m_plus:
            return _normalize_phone(m_plus.group(0))
        m = PHONE_RE.search(text)
        if m:
            return _normalize_phone(m.group(0))
    elif pii_type == "CREDIT":
        m = CREDIT_RE.search(text)
        if m:
            return _normalize_credit(m.group(0))

    return text


def _normalize_phone(s: str) -> str:
    """Keep + and digits only. Force '+' prefix if phone-shaped (10-15 digits).

    100% of validation_pii GT phones are E.164 with '+1' prefix. Model often
    emits the digits without '+'. This adds '+' back. 16+ digits = likely a
    credit card emitted in wrong field — leave as-is, low chance of recovery.
    """
    s = s.strip()
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if not digits:
        return s
    # Phone-shaped: force '+' prefix (matches GT distribution exactly)
    if 10 <= len(digits) <= 15:
        return "+" + digits
    return ("+" + digits) if plus else digits


def _normalize_credit(s: str) -> str:
    """Keep digits + single spaces in groups of 4 (model's training format)."""
    digits = re.sub(r"\D", "", s.strip())
    if len(digits) < 13 or len(digits) > 19:
        return s.strip()
    # 16-digit case: standard 4-4-4-4 grouping (matches training format)
    if len(digits) == 16:
        return " ".join(digits[i : i + 4] for i in range(0, 16, 4))
    return digits


def email_fallback_from_question(question: str) -> str:
    """When model fails to emit an email (no '@' in generation), build a
    name-based fallback: 'firstname.lastname@example.com'.

    Best partial-match against the GT distribution observed in validation_pii:
    most GT emails follow `firstname.lastname@<domain>`. Even with wrong
    domain, this gives ~0.6-0.8 Levenshtein similarity (vs ~0.0 for a
    phone/CC dump emitted in EMAIL slot).
    """
    m = NAME_PAIR_RE.search(question)
    if m:
        first, last = m.group(1).lower(), m.group(2).lower()
        return f"{first}.{last}@example.com"
    return DEFAULT_PRED["EMAIL"]


# Empirically-tuned fallback for PHONE when model emits non-phone content
# (typical: 16-digit credit-card pattern in PHONE slot).
# Optimum across val_pii GT distribution = '+1' (100% US E.164 in GT) + mid-range
# digits maximizing expected Levenshtein similarity.
# Tested on 22 CC-in-PHONE cases (eval_v0_20260509_182253):
#   '+12312312312' (DEFAULT) → 0.288  '+15555550000' (mixed)  → 0.303 ✓
#   '+15555555555' (all 5s)  → 0.288  '+10000000000' (zeros) → 0.280
PHONE_FALLBACK = "+15555550000"


def looks_like_phone(s: str) -> bool:
    """True if string contains 10-12 digits (E.164 phone shape)."""
    digits = re.sub(r"\D", "", s)
    return 10 <= len(digits) <= 12


def luhn_check(number: str) -> bool:
    """Standard Luhn checksum."""
    digits = re.sub(r"\D", "", number)
    if len(digits) < 13:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validate_pred(pred: str | None, pii_type: str) -> str:
    """Enforce server constraints: strip, length 10-100, no forbidden chars.
    Falls back to DEFAULT_PRED[pii_type] if pred is empty/invalid."""
    if pred is None:
        return DEFAULT_PRED[pii_type]
    pred = str(pred).strip()
    # Forbidden substrings — strip them
    for tok in ("<|user|>", "<|assistant|>", "<|system|>", '"', "'"):
        pred = pred.replace(tok, "")
    # Embedded newlines/CR break CSV row alignment server-side. Collapse to space.
    pred = pred.replace("\r", " ").replace("\n", " ")
    pred = re.sub(r"\s+", " ", pred).strip()
    if len(pred) == 0:
        return DEFAULT_PRED[pii_type]
    # Length floor: pad with format-valid suffix
    if len(pred) < 10:
        if pii_type == "EMAIL" and "@" not in pred:
            pred = pred + "@example.com"
        # Pad with trailing digit (last resort)
        while len(pred) < 10:
            pred = pred + "0"
    # Length cap
    if len(pred) > 100:
        pred = pred[:100]
    return pred


def is_valid_format(pred: str, pii_type: str) -> bool:
    """Quick format check (used for stats/debug only)."""
    if pii_type == "EMAIL":
        return bool(EMAIL_RE.fullmatch(pred))
    if pii_type == "PHONE":
        digits = re.sub(r"\D", "", pred)
        return 7 <= len(digits) <= 15
    if pii_type == "CREDIT":
        digits = re.sub(r"\D", "", pred)
        return 13 <= len(digits) <= 19
    return False
