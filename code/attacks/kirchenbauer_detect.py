"""Pure detection helpers for Kirchenbauer green-list watermark.

No I/O, no model loading. Imported by:
  - code/attacks/run_attack_B1.py — full B1 pipeline
  - code/practice/score_B.py      — z-score for B2 evasion check
  - tests/smoke.py                — micro-test on a mock tokenizer

Tokenizer protocol (duck-typed): must support
  - len(tokenizer)                       → int (vocab size, including special)
  - tokenizer.encode(text, add_special_tokens=False) → list[int]
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence


class _Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...
    def __len__(self) -> int: ...


def _z_for_ids(token_ids: Sequence[int], vocab_size: int,
               gamma: float, hash_key: int) -> float:
    """z-score on a pre-tokenised slice. Skips first token (no prev)."""
    import torch

    if len(token_ids) < 2:
        return 0.0
    green_size = int(gamma * vocab_size)
    n_green = 0
    T = len(token_ids) - 1
    rng = torch.Generator()
    for i in range(1, len(token_ids)):
        seed = (hash_key * int(token_ids[i - 1])) % (2 ** 63)
        rng.manual_seed(seed)
        green = torch.randperm(vocab_size, generator=rng)[:green_size]
        if int(token_ids[i]) in green.tolist():
            n_green += 1
    return (n_green - gamma * T) / (gamma * (1.0 - gamma) * T) ** 0.5


def kirchenbauer_zscore(text: str, tokenizer: _Tokenizer,
                        gamma: float = 0.25, hash_key: int = 15485863) -> float:
    """Full-text z-score. Uses len(tokenizer) — matches model.config.vocab_size."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return _z_for_ids(ids, len(tokenizer), gamma, hash_key)


def sliding_max_z(text: str, tokenizer: _Tokenizer,
                  gamma: float = 0.25, hash_key: int = 15485863,
                  window: int = 100, stride: int = 50) -> float:
    """Max z-score over sliding windows. Catches partially-watermarked texts."""
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    V = len(tokenizer)
    if len(ids) <= window:
        return _z_for_ids(ids, V, gamma, hash_key)
    best = -math.inf
    for start in range(0, len(ids) - window + 1, stride):
        z = _z_for_ids(ids[start : start + window], V, gamma, hash_key)
        if z > best:
            best = z
    return best


def predict_watermarked(text: str, tokenizer: _Tokenizer,
                        threshold: float = 4.0,
                        gamma: float = 0.25, hash_key: int = 15485863) -> tuple[bool, float]:
    """Returns (is_watermarked, max_z)."""
    z_full = kirchenbauer_zscore(text, tokenizer, gamma=gamma, hash_key=hash_key)
    z_win  = sliding_max_z(text, tokenizer, gamma=gamma, hash_key=hash_key)
    z = max(z_full, z_win)
    return z > threshold, z
