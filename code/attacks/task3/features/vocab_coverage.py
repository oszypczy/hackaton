"""B7 — Type-token ratio and vocabulary richness (no LM; O(words)).

Complements causal PPL: watermark bias may change repetition / lexical diversity."""
from __future__ import annotations

from collections import Counter

import numpy as np

_keys = [
    "vc_ttr_word", "vc_ttr_char", "vc_guiraud", "vc_herdan_c",
    "vc_hapax_frac", "vc_avg_word_len", "vc_space_frac",
    "vc_top5_mass", "vc_longword_frac",
]


def extract(text: str, max_len: int = 8192) -> dict[str, float]:
    if not text:
        return {k: 0.0 for k in _keys}
    t = text[:max_len]
    words = t.split()
    n_w = len(words)
    if n_w < 2:
        return {k: 0.0 for k in _keys}

    wl = [w.lower() for w in words]
    cnt = Counter(wl)
    uw = len(cnt)
    chars = list(t.replace("\n", " "))
    n_c = max(len(chars), 1)
    uc = len(set(chars))

    hapax = sum(1 for v in cnt.values() if v == 1) / n_w
    total_tok = sum(cnt.values())
    top5 = sum(v for _, v in cnt.most_common(5)) / max(total_tok, 1)
    lens = np.array([len(w) for w in words], dtype=np.float64)
    long_frac = float((lens >= 7).mean())
    spaces = t.count(" ") + t.count("\n")

    return {
        "vc_ttr_word": uw / n_w,
        "vc_ttr_char": uc / n_c,
        "vc_guiraud": uw / np.sqrt(n_w),
        "vc_herdan_c": float(np.log(uw + 1) / np.log(n_w + 1)),
        "vc_hapax_frac": hapax,
        "vc_avg_word_len": float(lens.mean()),
        "vc_space_frac": spaces / max(len(t), 1),
        "vc_top5_mass": top5,
        "vc_longword_frac": long_frac,
    }
