"""Direct KGW (Kirchenbauer) reference detection algorithm.

Tries the EXACT detection from the original paper with multiple
hyperparameter guesses. If organizers used default Kirchenbauer settings
(hash_key=15485863, gamma=0.25) with one of the tested tokenizers,
this catches the watermark deterministically (no learning required).

Cache greenlists per (vocab_size, hash_key, prev_token) — first call
populates, subsequent are O(1) set lookup.
"""
from __future__ import annotations

import numpy as np
import torch

# ── Configurations to test ────────────────────────────────────────────────────
# (tokenizer_name, vocab_size, hash_key, gamma)
# Default Kirchenbauer impl: hash_key=15485863, gamma=0.25
CONFIGS = [
    ("gpt2", 50257, 15485863, 0.25),
    ("gpt2", 50257, 15485863, 0.5),
    ("facebook/opt-1.3b", 50272, 15485863, 0.25),
    ("EleutherAI/pythia-1.4b", 50304, 15485863, 0.25),
]

# Per-config greenlist cache: (vocab, hash_key) → {prev_token: frozenset(green_ids)}
_gl_cache: dict[tuple[int, int], dict[int, frozenset]] = {}
_tokenizers: dict[str, object] = {}


def _config_key(vocab_size: int, hash_key: int) -> tuple[int, int]:
    return (vocab_size, hash_key)


def _get_greenlist(prev_token: int, vocab_size: int, hash_key: int, gamma: float) -> frozenset:
    """Replicate Kirchenbauer reference impl: torch.randperm seeded by hash_key * prev_token."""
    key = _config_key(vocab_size, hash_key)
    if key not in _gl_cache:
        _gl_cache[key] = {}
    sub = _gl_cache[key]
    cache_key = (prev_token, gamma)
    if cache_key not in sub:
        rng = torch.Generator()
        # Match Kirchenbauer reference: seed = hash_key * prev_token
        rng.manual_seed(int(hash_key) * int(prev_token))
        perm = torch.randperm(vocab_size, generator=rng)
        n_green = int(vocab_size * gamma)
        sub[cache_key] = frozenset(perm[:n_green].tolist())
    return sub[cache_key]


def _zscore_features(token_ids: list[int], vocab_size: int, hash_key: int, gamma: float) -> dict:
    """Compute z-score and windowed-max z-score for a single config."""
    if len(token_ids) < 5:
        return {"z": 0.0, "wmax_50": 0.0, "wmax_100": 0.0}

    greens = []
    for i in range(1, len(token_ids)):
        prev = int(token_ids[i - 1])
        curr = int(token_ids[i])
        if prev >= vocab_size or curr >= vocab_size:
            greens.append(0.0)
            continue
        gl = _get_greenlist(prev, vocab_size, hash_key, gamma)
        greens.append(1.0 if curr in gl else 0.0)

    arr = np.array(greens)
    T = len(arr)
    z = (arr.sum() - gamma * T) / np.sqrt(T * gamma * (1 - gamma) + 1e-12)

    feats = {"z": float(z)}
    for w in (50, 100):
        if T < w:
            feats[f"wmax_{w}"] = 0.0
            continue
        win_sum = np.convolve(arr, np.ones(w), mode="valid")
        wz = (win_sum - gamma * w) / np.sqrt(w * gamma * (1 - gamma))
        feats[f"wmax_{w}"] = float(wz.max())
    return feats


def _tag(tok_name: str, hash_key: int, gamma: float) -> str:
    safe = tok_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"kgw_{safe}_h{hash_key}_g{int(gamma * 100)}"


def _ensure_tokenizers():
    if _tokenizers:
        return
    from transformers import AutoTokenizer
    seen = set()
    for tok_name, vocab, hash_key, gamma in CONFIGS:
        if tok_name in seen:
            continue
        seen.add(tok_name)
        try:
            _tokenizers[tok_name] = AutoTokenizer.from_pretrained(tok_name)
            print(f"  [kgw] Loaded tokenizer: {tok_name}")
        except Exception as e:
            print(f"  [kgw] Failed to load {tok_name}: {e}")


def extract(text: str) -> dict[str, float]:
    _ensure_tokenizers()
    feats: dict[str, float] = {}
    for tok_name, vocab, hash_key, gamma in CONFIGS:
        prefix = _tag(tok_name, hash_key, gamma)
        if tok_name not in _tokenizers:
            feats[f"{prefix}_z"] = 0.0
            feats[f"{prefix}_wmax_50"] = 0.0
            feats[f"{prefix}_wmax_100"] = 0.0
            continue
        tok = _tokenizers[tok_name]
        ids = tok.encode(text)[:1024]
        sub = _zscore_features(ids, vocab, hash_key, gamma)
        feats[f"{prefix}_z"] = sub["z"]
        feats[f"{prefix}_wmax_50"] = sub["wmax_50"]
        feats[f"{prefix}_wmax_100"] = sub["wmax_100"]
    return feats
