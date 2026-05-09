"""Unigram-Watermark direct z-score features (Zhao ICLR 2024).

Implements the fixed-global-green-list detection from XuandongZhao/Unigram-Watermark.
No GPU required — pure CPU tokenization + numpy.

Default: watermark_key=0, fraction=0.5, vocab_size=50257 (GPT-2 tokenizer).
Multiple (key, hash_fn) variants are computed to cover possible organizer configs.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np

_tokenizer: Any = None
_masks: dict[tuple, np.ndarray] = {}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _tokenizer


def _get_mask(seed: int, fraction: float, vocab_size: int) -> np.ndarray:
    key = (seed, fraction, vocab_size)
    if key not in _masks:
        rng = np.random.default_rng(seed)
        n_green = int(fraction * vocab_size)
        mask = np.array([True] * n_green + [False] * (vocab_size - n_green), dtype=bool)
        rng.shuffle(mask)
        _masks[key] = mask
    return _masks[key]


def _zscore(token_ids: list[int], mask: np.ndarray, fraction: float) -> float:
    valid = [t for t in token_ids if 0 <= t < len(mask)]
    n = len(valid)
    if n < 5:
        return 0.0
    green = sum(1 for t in valid if mask[t])
    return (green - fraction * n) / np.sqrt(fraction * (1 - fraction) * n)


def _unizscore(token_ids: list[int], mask: np.ndarray, fraction: float) -> float:
    """Z-score on unique tokens (Zhao 'unidetect' — more robust to repetition)."""
    valid = list({t for t in token_ids if 0 <= t < len(mask)})
    n = len(valid)
    if n < 5:
        return 0.0
    green = sum(1 for t in valid if mask[t])
    return (green - fraction * n) / np.sqrt(fraction * (1 - fraction) * n)


def _green_frac(token_ids: list[int], mask: np.ndarray) -> float:
    valid = [t for t in token_ids if 0 <= t < len(mask)]
    if not valid:
        return 0.5
    return sum(1 for t in valid if mask[t]) / len(valid)


def _sha256_str(key: int) -> int:
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:4], "little")


def _sha256_int64(key: int) -> int:
    import struct
    return int.from_bytes(hashlib.sha256(struct.pack("<q", key)).digest()[:4], "little")


# All seed variants to try (derived from watermark_key=0 under different hash functions)
_SEED_CONFIGS = [
    # (description, seed, fraction)
    # Direct / mod_vocab (most likely from gptwm.py source)
    ("k0_f50_direct",   0, 0.5),
    ("k0_f25_direct",   0, 0.25),
    ("k1_f50_direct",   1, 0.5),
    ("k42_f50_direct", 42, 0.5),
    ("k2_f50_direct",   2, 0.5),
    # SHA256_str(key) variants — gridsearch winner: key=9999 sep=0.435
    ("k9999_f25_sha256str", _sha256_str(9999), 0.25),   # seed=1525845384
    ("k9999_f50_sha256str", _sha256_str(9999), 0.50),
    ("k0_f25_sha256str",    _sha256_str(0),    0.25),
    ("k100_f25_sha256str",  _sha256_str(100),  0.25),
    # SHA256(int64(key)) variants
    ("k0_f50_sha",  _sha256_int64(0), 0.5),
    ("k0_f25_sha",  _sha256_int64(0), 0.25),
    ("k1_f50_sha",  _sha256_int64(1), 0.5),
    ("k9999_f25_sha256int64", _sha256_int64(9999), 0.25),  # seed=1450176100
    ("k100_f25_sha256int64",  _sha256_int64(100),  0.25),
]

VOCAB_SIZE = 50257  # GPT-2


def extract(text: str) -> dict[str, float]:
    tok = _get_tokenizer()
    token_ids = tok.encode(str(text), add_special_tokens=False)

    out: dict[str, float] = {}
    for name, seed, frac in _SEED_CONFIGS:
        mask = _get_mask(seed, frac, VOCAB_SIZE)
        z = _zscore(token_ids, mask, frac)
        uz = _unizscore(token_ids, mask, frac)
        gf = _green_frac(token_ids, mask)
        out[f"uni_{name}_z"] = z
        out[f"uni_{name}_uz"] = uz
        out[f"uni_{name}_gf"] = gf

    return out
