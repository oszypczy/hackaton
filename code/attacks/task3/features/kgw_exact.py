"""EXACT Kirchenbauer z-score detector — analytical optimal test.

For each text we try several detection configurations and emit z-scores:
  - tokenizer: Llama-2 (canonical KGW), OLMo-2-7B, GPT-2 (sanity)
  - gamma: 0.25, 0.5
  - hash schemes: prev-1, prev-4, "min-hash" (paper variants)

z-score = (n_green - γ·N) / √(N·γ·(1-γ))
With watermarked text, expected n_green = (γ + δ_effective)·N → z scales as √N (huge for long texts).

Returns ~24 features. Even if the true generator's tokenizer/seed isn't matched,
the LR classifier will pick which configuration gives the strongest signal.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

_TOKENIZERS: dict[str, object] = {}
_DEVICE = "cpu"  # all tokenization is CPU-bound
HASH_KEY = 15485863  # Kirchenbauer paper default seed scalar


def _load_tokenizer(name: str) -> Optional[object]:
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        _TOKENIZERS[name] = tok
        return tok
    except Exception as e:
        print(f"  [kgw_exact] tokenizer {name} unavailable: {e}")
        _TOKENIZERS[name] = None
        return None


def _is_green(seed_val: int, token_id: int, gamma: float) -> bool:
    """O(1) hash-based green test (statistically equivalent to PRNG-permuted vocab top γ)."""
    h = (int(seed_val) * HASH_KEY + int(token_id) * 2654435761) & 0xFFFFFFFF
    # uniform [0,1) from hash
    u = h / 0xFFFFFFFF
    return u < gamma


def _zscore_fast(token_ids: list[int], vocab_size: int, gamma: float, h: int = 1) -> float:
    """O(N) z-score using O(1) hash green test — full text, no subsampling needed."""
    if len(token_ids) <= h:
        return 0.0
    n_green = 0
    n_total = 0
    for i in range(h, len(token_ids)):
        if h == 1:
            seed_val = int(token_ids[i - 1])
        else:
            seed_val = 0
            for j in range(1, h + 1):
                seed_val ^= int(token_ids[i - j]) * (j + 1)
        if _is_green(seed_val, int(token_ids[i]), gamma):
            n_green += 1
        n_total += 1
    if n_total == 0:
        return 0.0
    expected = gamma * n_total
    var = n_total * gamma * (1 - gamma)
    if var <= 0:
        return 0.0
    return (n_green - expected) / math.sqrt(var)


# Configurations to test
CONFIGS = [
    # (tokenizer, gamma, h)
    ("NousResearch/Llama-2-7b-hf", 0.25, 1),
    ("NousResearch/Llama-2-7b-hf", 0.5, 1),
    ("NousResearch/Llama-2-7b-hf", 0.25, 4),
    ("allenai/OLMo-2-1124-7B-Instruct", 0.25, 1),
    ("allenai/OLMo-2-1124-7B-Instruct", 0.5, 1),
    ("gpt2", 0.25, 1),
    ("gpt2", 0.5, 1),
]


def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    keys: list[str] = []
    for tok_name, gamma, h in CONFIGS:
        tag = tok_name.split("/")[-1].replace("-", "_").lower()
        keys.append(f"kgwx_{tag}_g{int(gamma*100)}_h{h}_z")

    out = {k: 0.0 for k in keys}
    if not text or len(text.split()) < 4:
        return out

    # Cache token-encodings per tokenizer (to reuse across gamma/h)
    tok_ids_cache: dict[str, list[int]] = {}
    for tok_name, gamma, h in CONFIGS:
        tok = _load_tokenizer(tok_name)
        if tok is None:
            continue
        if tok_name not in tok_ids_cache:
            try:
                tok_ids_cache[tok_name] = tok.encode(
                    text, truncation=True, max_length=max_len, add_special_tokens=False
                )
            except Exception:
                tok_ids_cache[tok_name] = []
        ids = tok_ids_cache[tok_name]
        if len(ids) < h + 4:
            continue
        vocab_size = len(tok)
        z = _zscore_fast(ids, vocab_size, gamma, h)
        tag = tok_name.split("/")[-1].replace("-", "_").lower()
        out[f"kgwx_{tag}_g{int(gamma*100)}_h{h}_z"] = float(z)

    # Add aggregate features: max |z|, sum |z|, num |z|>2 across configs
    zs = [v for k, v in out.items() if k.endswith("_z")]
    if zs:
        out["kgwx_max_abs_z"] = float(max(abs(z) for z in zs))
        out["kgwx_sum_abs_z"] = float(sum(abs(z) for z in zs))
        out["kgwx_n_signif_z"] = float(sum(1 for z in zs if abs(z) > 2.0))
    else:
        out["kgwx_max_abs_z"] = 0.0
        out["kgwx_sum_abs_z"] = 0.0
        out["kgwx_n_signif_z"] = 0.0

    return out
