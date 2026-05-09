"""KGW v2 — extra hash_keys + h=2 multigram.

V1 (branch_kgw.py) tested: gpt2 (g=0.25, g=0.5), opt-1.3b (g=0.25), pythia-1.4b
   all with hash_key=15485863. ALL gave no signal -> organizers used different config.

V2 tries:
  - hash_keys: [0, 1, 42, 100, 12345, 999, 7] (Kirchenbauer fork commons)
  - h=2 multigram with mul/add seedings (some forks use prev_2 token too)
  - Tokenizer: gpt2 only (most common)
"""
from __future__ import annotations

import functools
import numpy as np
import torch

# (tokenizer_name, vocab_size, hash_key, gamma, scheme)
# scheme: "h1" = simple_1 (Kirchenbauer reference)
#         "h2_mul" = seed = hash_key * prev_2 * prev_1 (mul of last 2)
#         "h2_add" = seed = hash_key * (prev_2 + prev_1)
CONFIGS_V2 = [
    # h=1, untested hash_keys
    ("gpt2", 50257, 0, 0.25, "h1"),
    ("gpt2", 50257, 1, 0.25, "h1"),
    ("gpt2", 50257, 42, 0.25, "h1"),
    ("gpt2", 50257, 100, 0.25, "h1"),
    ("gpt2", 50257, 12345, 0.25, "h1"),
    ("gpt2", 50257, 7, 0.25, "h1"),
    # h=2, default hash_key
    ("gpt2", 50257, 15485863, 0.25, "h2_mul"),
    ("gpt2", 50257, 15485863, 0.25, "h2_add"),
]

_tokenizers: dict[str, object] = {}


@functools.lru_cache(maxsize=200_000)
def _gl_bytes_h1(prev: int, vocab: int, hash_key: int, gamma: float) -> bytes:
    rng = torch.Generator()
    rng.manual_seed(int(hash_key) * int(prev))
    perm = torch.randperm(vocab, generator=rng).numpy()
    n_green = int(vocab * gamma)
    arr = bytearray((vocab + 7) // 8)
    for idx in perm[:n_green]:
        i = int(idx)
        arr[i // 8] |= 1 << (i % 8)
    return bytes(arr)


@functools.lru_cache(maxsize=200_000)
def _gl_bytes_h2(prev1: int, prev2: int, vocab: int, hash_key: int, gamma: float, scheme: str) -> bytes:
    rng = torch.Generator()
    if scheme == "h2_mul":
        seed = int(hash_key) * int(prev1) * int(prev2)
    else:  # h2_add
        seed = int(hash_key) * (int(prev1) + int(prev2))
    # Avoid 0 seed which gives same perm
    if seed == 0:
        seed = 1
    rng.manual_seed(seed)
    perm = torch.randperm(vocab, generator=rng).numpy()
    n_green = int(vocab * gamma)
    arr = bytearray((vocab + 7) // 8)
    for idx in perm[:n_green]:
        i = int(idx)
        arr[i // 8] |= 1 << (i % 8)
    return bytes(arr)


def _is_green(token: int, gl: bytes) -> bool:
    return bool(gl[token // 8] & (1 << (token % 8)))


def _zscore_h1(token_ids: list, vocab: int, hash_key: int, gamma: float) -> dict:
    if len(token_ids) < 5:
        return {"z": 0.0, "wmax_100": 0.0}
    greens = []
    for i in range(1, len(token_ids)):
        prev = int(token_ids[i - 1])
        curr = int(token_ids[i])
        if prev >= vocab or curr >= vocab:
            greens.append(0.0)
            continue
        gl = _gl_bytes_h1(prev, vocab, hash_key, gamma)
        greens.append(1.0 if _is_green(curr, gl) else 0.0)
    arr = np.array(greens)
    T = len(arr)
    z = (arr.sum() - gamma * T) / np.sqrt(T * gamma * (1 - gamma) + 1e-12)
    out = {"z": float(z), "wmax_100": 0.0}
    if T >= 100:
        win_sum = np.convolve(arr, np.ones(100), mode="valid")
        wz = (win_sum - gamma * 100) / np.sqrt(100 * gamma * (1 - gamma))
        out["wmax_100"] = float(wz.max())
    return out


def _zscore_h2(token_ids: list, vocab: int, hash_key: int, gamma: float, scheme: str) -> dict:
    if len(token_ids) < 6:
        return {"z": 0.0, "wmax_100": 0.0}
    greens = []
    for i in range(2, len(token_ids)):
        prev1 = int(token_ids[i - 1])
        prev2 = int(token_ids[i - 2])
        curr = int(token_ids[i])
        if prev1 >= vocab or prev2 >= vocab or curr >= vocab:
            greens.append(0.0)
            continue
        gl = _gl_bytes_h2(prev1, prev2, vocab, hash_key, gamma, scheme)
        greens.append(1.0 if _is_green(curr, gl) else 0.0)
    arr = np.array(greens)
    T = len(arr)
    z = (arr.sum() - gamma * T) / np.sqrt(T * gamma * (1 - gamma) + 1e-12)
    out = {"z": float(z), "wmax_100": 0.0}
    if T >= 100:
        win_sum = np.convolve(arr, np.ones(100), mode="valid")
        wz = (win_sum - gamma * 100) / np.sqrt(100 * gamma * (1 - gamma))
        out["wmax_100"] = float(wz.max())
    return out


def _ensure_tokenizers():
    if _tokenizers:
        return
    from transformers import AutoTokenizer
    seen = set()
    for tok_name, *_ in CONFIGS_V2:
        if tok_name in seen:
            continue
        seen.add(tok_name)
        try:
            _tokenizers[tok_name] = AutoTokenizer.from_pretrained(tok_name)
            print(f"  [kgw_v2] Loaded tokenizer: {tok_name}")
        except Exception as e:
            print(f"  [kgw_v2] Failed to load {tok_name}: {e}")


def _tag(tok_name: str, hash_key: int, gamma: float, scheme: str) -> str:
    safe = tok_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"kgw2_{safe}_h{hash_key}_g{int(gamma * 100)}_{scheme}"


def extract(text: str) -> dict[str, float]:
    _ensure_tokenizers()
    feats: dict[str, float] = {}
    for tok_name, vocab, hash_key, gamma, scheme in CONFIGS_V2:
        prefix = _tag(tok_name, hash_key, gamma, scheme)
        if tok_name not in _tokenizers:
            feats[f"{prefix}_z"] = 0.0
            feats[f"{prefix}_wmax_100"] = 0.0
            continue
        tok = _tokenizers[tok_name]
        ids = tok.encode(text)[:1024]
        if scheme == "h1":
            sub = _zscore_h1(ids, vocab, hash_key, gamma)
        else:
            sub = _zscore_h2(ids, vocab, hash_key, gamma, scheme)
        feats[f"{prefix}_z"] = sub["z"]
        feats[f"{prefix}_wmax_100"] = sub["wmax_100"]
    return feats
