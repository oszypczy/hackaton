"""KGW Selfhash z-score features (Kirchenbauer ICLR 2024 "Reliability" variant).

The selfhash PRF uses h=4 context window, anchored (current token included):
    seed = minhash({hash(t_{i-3}), ..., hash(t_{i-1}), hash(t_i)}) × hash(t_i) × key

Previous attempts used simple_1 (h=1, seed = key * prev_token) which gave NO SIGNAL.
Selfhash (h=4, anchored minhash PRF) is the recommended default in the reliability paper.

We try multiple hash function variants and multiple keys to find which gives signal.
"""
from __future__ import annotations

import functools
import numpy as np
import torch
from typing import Any

_tokenizers: dict[str, Any] = {}


def _get_tokenizer(name: str = "gpt2"):
    if name not in _tokenizers:
        from transformers import AutoTokenizer
        _tokenizers[name] = AutoTokenizer.from_pretrained(name)
    return _tokenizers[name]


TOKENIZER_VOCAB: dict[str, int] = {
    "gpt2": 50257,
    "mistralai/Mistral-7B-v0.1": 32000,
}


# ─── Hash function variants ──────────────────────────────────────────────────

def _hash_v1(token: int, key: int) -> int:
    """Simple multiplicative: hash = token * key (mod 2^32)."""
    return int((token * key) & 0xFFFFFFFF)


def _hash_v2(token: int, key: int) -> int:
    """Knuth multiplicative: (token+1) * 2654435761 * key mod 2^32."""
    return int(((token + 1) * 2654435761 * key) & 0xFFFFFFFF)


def _hash_v3(token: int, key: int) -> int:
    """Per-token hash independent of key (key used only in final seed)."""
    return int(((token + 1) * 2654435761) & 0xFFFFFFFF)


def _hash_v4(token: int, key: int) -> int:
    """(token * key + token) mod large_prime."""
    p = 2147483647  # Mersenne prime 2^31-1
    return int(((token * key + token)) % p)


_HASH_FNS = {
    "v1": _hash_v1,
    "v2": _hash_v2,
    "v3": _hash_v3,
    "v4": _hash_v4,
}


# ─── Selfhash seed computation ───────────────────────────────────────────────

def _selfhash_seed(ctx_tokens: list[int], curr_token: int, key: int, hash_fn) -> int:
    """
    Anchored minhash PRF:
        seed = min(hash(ctx[-h+1:] + [curr])) * hash(curr) * key
    """
    tokens = ctx_tokens + [curr_token]  # context + current
    hashes = [hash_fn(t, key) for t in tokens]
    min_h = min(hashes)
    curr_h = hash_fn(curr_token, key)
    return int((min_h * curr_h * key) & 0x7FFFFFFF)


# ─── Green list generation ───────────────────────────────────────────────────

@functools.lru_cache(maxsize=500_000)
def _greenlist_h4(ctx_tuple: tuple, curr_token: int, vocab: int, key: int,
                  gamma: float, hash_fn_name: str) -> bytes:
    hash_fn = _HASH_FNS[hash_fn_name]
    seed = _selfhash_seed(list(ctx_tuple), curr_token, key, hash_fn)
    rng = torch.Generator()
    rng.manual_seed(seed % (2**63 - 1))
    perm = torch.randperm(vocab, generator=rng).numpy()
    n_green = int(vocab * gamma)
    arr = bytearray((vocab + 7) // 8)
    for idx in perm[:n_green]:
        i = int(idx)
        arr[i // 8] |= 1 << (i % 8)
    return bytes(arr)


def _is_green(token: int, gl: bytes) -> bool:
    return bool(gl[token // 8] & (1 << (token % 8)))


def _zscore_selfhash(token_ids: list[int], vocab: int, key: int,
                     gamma: float, h: int, hash_fn_name: str) -> float:
    if len(token_ids) < h + 2:
        return 0.0
    greens = []
    for i in range(h, len(token_ids)):
        ctx = token_ids[max(0, i - h): i]  # h tokens before current
        curr = token_ids[i]
        if curr >= vocab:
            greens.append(0.0)
            continue
        ctx_clean = [t for t in ctx if t < vocab]
        if not ctx_clean:
            greens.append(0.0)
            continue
        gl = _greenlist_h4(tuple(ctx_clean[-h + 1:]), curr, vocab, key, gamma, hash_fn_name)
        greens.append(1.0 if _is_green(curr, gl) else 0.0)

    arr = np.array(greens)
    T = len(arr)
    if T < 5:
        return 0.0
    return float((arr.sum() - gamma * T) / np.sqrt(T * gamma * (1 - gamma) + 1e-12))


# ─── Configs to try ─────────────────────────────────────────────────────────

# (key, gamma, h, hash_fn_name, tokenizer_name)
CONFIGS = [
    # GPT-2 tokenizer (vocab 50257)
    (15485863, 0.25, 4, "v2", "gpt2"),  # default key, Knuth hash
    (15485863, 0.25, 4, "v1", "gpt2"),  # simple multiplicative
    (15485863, 0.5,  4, "v2", "gpt2"),
    (33554393, 0.25, 4, "v2", "gpt2"),  # WaterSeeker LLaMA key
    (0,        0.25, 4, "v2", "gpt2"),
    (1,        0.25, 4, "v2", "gpt2"),
    # Mistral tokenizer (vocab 32000) — likely used for generation
    (15485863, 0.25, 4, "v2", "mistralai/Mistral-7B-v0.1"),
    (15485863, 0.25, 4, "v1", "mistralai/Mistral-7B-v0.1"),
    (15485863, 0.5,  4, "v2", "mistralai/Mistral-7B-v0.1"),
    (33554393, 0.25, 4, "v2", "mistralai/Mistral-7B-v0.1"),
    (0,        0.25, 4, "v2", "mistralai/Mistral-7B-v0.1"),
]


def extract(text: str) -> dict[str, float]:
    # tokenize once per unique tokenizer
    token_id_cache: dict[str, list[int]] = {}

    out: dict[str, float] = {}
    for key, gamma, h, fn_name, tok_name in CONFIGS:
        if tok_name not in token_id_cache:
            tok = _get_tokenizer(tok_name)
            token_id_cache[tok_name] = tok.encode(str(text), add_special_tokens=False)
        token_ids = token_id_cache[tok_name]
        vocab = TOKENIZER_VOCAB[tok_name]
        tok_short = "gpt2" if tok_name == "gpt2" else "mistral"
        col = f"kgw_sh_k{key}_g{int(gamma*100)}_h{h}_{fn_name}_{tok_short}"
        z = _zscore_selfhash(token_ids, vocab, key, gamma, h, fn_name)
        out[col] = z
    return out
