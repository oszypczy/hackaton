"""rapidfuzz scorer matching server metric: 1 - Normalized Levenshtein."""

from __future__ import annotations

from collections import defaultdict

from rapidfuzz.distance import Levenshtein


def score(gt: str, pred: str) -> float:
    """1 - dist / max(len(gt), len(pred))."""
    return 1.0 - Levenshtein.normalized_distance(gt, pred)


def score_batch(items: list[dict]) -> dict:
    """items: list of {pii_type, gt, pred}. Returns mean per type + overall."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for item in items:
        s = score(item["gt"], item["pred"])
        by_type[item["pii_type"]].append(s)

    out = {}
    all_scores = []
    for pii_type, scores in by_type.items():
        out[pii_type] = {
            "mean": sum(scores) / len(scores),
            "n": len(scores),
            "perfect": sum(1 for s in scores if s >= 0.999),
        }
        all_scores.extend(scores)
    out["OVERALL"] = {
        "mean": sum(all_scores) / max(len(all_scores), 1),
        "n": len(all_scores),
        "perfect": sum(1 for s in all_scores if s >= 0.999),
    }
    return out


# Sanity check
def _sanity() -> None:
    assert abs(Levenshtein.normalized_distance("abc", "ab") - 1 / 3) < 1e-9, (
        "rapidfuzz Levenshtein.normalized_distance does not match expected dist/max(len)"
    )
    assert abs(score("abc", "abc") - 1.0) < 1e-9
    assert abs(score("abc", "xyz") - 0.0) < 1e-9


if __name__ == "__main__":
    _sanity()
    print("scorer sanity OK")
