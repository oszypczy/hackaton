"""K-candidate aggregation for K-shot ensemble inference.

Per research §3.1 (Jiang & Wentker generalized-median, exposed via rapidfuzz):
medoid_pick = argmin over candidate set of Σ Lev(c, c_j) is provably
≤ 2× the true Bayes-risk minimizer for edit distance, runs in O(K²·L²).
This directly minimizes the server scoring metric (mean(1 − Lev_norm))
in expectation — a strict upgrade over majority-vote / self-consistency
for free-form strings.

Per research §3.4: regex-canonicalize PRZED voting collapses surface
variants ("(415) 555-…" vs "415-555-…") into a single equivalence
class — Yu et al. ICML'23 _Bag of Tricks_ reports this as one of the
largest single levers in the pipeline.
"""

from __future__ import annotations

from collections import Counter

from rapidfuzz.distance import Levenshtein

from format import extract_pii


def medoid_pick(raw_candidates: list[str], pii_type: str) -> str:
    """Pick the Levenshtein-medoid of K canonicalized PII candidates.

    Pipeline:
    1. Canonicalize each raw generation via `extract_pii` (regex-extract +
       normalize phone/credit format).
    2. Drop empties.
    3. If unique candidates ≤ 1 → return that one.
    4. Compute pairwise Levenshtein, pick argmin Σ_j Lev(c_i, c_j).
    5. Tie-break by frequency in the candidate set (mode helps consensus).
    6. Final tie-break by length closer to median length (avoid outliers).

    Returns "" if no canonicalizable candidate (caller should fall through
    to validate_pred default).
    """
    if not raw_candidates:
        return ""

    # Canonicalize
    canon = [extract_pii(c, pii_type) for c in raw_candidates]
    canon = [c.strip() for c in canon if c and c.strip()]
    if not canon:
        return ""
    if len(set(canon)) == 1:
        return canon[0]

    # Pairwise distances + per-candidate sum
    n = len(canon)
    sums: list[int] = []
    for i in range(n):
        s = 0
        for j in range(n):
            if i == j:
                continue
            s += Levenshtein.distance(canon[i], canon[j])
        sums.append(s)

    min_sum = min(sums)
    tied_idx = [i for i, s in enumerate(sums) if s == min_sum]
    if len(tied_idx) == 1:
        return canon[tied_idx[0]]

    # Tie-break 1: frequency in candidate set (mode wins)
    freq = Counter(canon)
    tied_by_freq = sorted(tied_idx, key=lambda i: -freq[canon[i]])
    top_freq = freq[canon[tied_by_freq[0]]]
    tied_freq = [i for i in tied_by_freq if freq[canon[i]] == top_freq]
    if len(tied_freq) == 1:
        return canon[tied_freq[0]]

    # Tie-break 2: length closer to median candidate length
    median_len = sorted(len(c) for c in canon)[len(canon) // 2]
    return canon[min(tied_freq, key=lambda i: abs(len(canon[i]) - median_len))]


def _sanity() -> None:
    """Quick smoke test — call from main.py at startup if K-shot mode."""
    # Identical candidates → return the value
    assert medoid_pick(["foo@bar.com"] * 3, "EMAIL") == "foo@bar.com"
    # 2-of-3 mode wins via tie-break
    cs = ["alice@x.com", "alice@x.com", "bob@y.com"]
    assert medoid_pick(cs, "EMAIL") == "alice@x.com"
    # Phone canonicalization: medoid of normalized
    cs = ["+13859159897", "13859159897", "+13859159898"]
    out = medoid_pick(cs, "PHONE")
    assert out.startswith("+1"), f"phone medoid lost prefix: {out!r}"
    # Empty → empty
    assert medoid_pick([], "EMAIL") == ""
    assert medoid_pick(["", "  "], "EMAIL") == ""


if __name__ == "__main__":
    _sanity()
    print("aggregator sanity OK")
