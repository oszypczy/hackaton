"""KGW reference detection with Llama-2 / Mistral / Llama-3 tokenizers.

Hipoteza: organizatorzy generowali watermark LLaMA-2-7B (najpopularniejszy
mid-2024 model, vocab=32000). Wszystkie nasze poprzednie KGW próby używały
gpt2/opt/pythia tokenizerów (vocab~50K) → niepoprawny vocab → no signal.

Każdy tokenizer ma własny vocab_size i własny torch.randperm output.
Próbujemy 3 najpopularniejsze: Llama-2 (32k), Llama-3 (128k), Mistral (32k).
Wszystkie z default Kirchenbauer hash_key=15485863, gamma=0.25.

Wymaga gated tokenizerów w cache (scripts/download_llama_tokenizers.sh).
"""
from __future__ import annotations

import functools

import numpy as np
import torch

CONFIGS = [
    # (tokenizer_name, vocab_size, hash_key, gamma)
    ("meta-llama/Llama-2-7b-hf", 32000, 15485863, 0.25),
    ("meta-llama/Llama-2-7b-hf", 32000, 15485863, 0.5),
    ("mistralai/Mistral-7B-v0.1", 32768, 15485863, 0.25),
    ("meta-llama/Meta-Llama-3-8B", 128256, 15485863, 0.25),
]

_tokenizers: dict[str, object] = {}


@functools.lru_cache(maxsize=300_000)
def _gl_bytes(prev: int, vocab: int, hash_key: int, gamma: float) -> bytes:
    rng = torch.Generator()
    rng.manual_seed(int(hash_key) * int(prev))
    perm = torch.randperm(vocab, generator=rng).numpy()
    n_green = int(vocab * gamma)
    arr = bytearray((vocab + 7) // 8)
    for idx in perm[:n_green]:
        i = int(idx)
        arr[i // 8] |= 1 << (i % 8)
    return bytes(arr)


def _is_green(token: int, gl: bytes) -> bool:
    return bool(gl[token // 8] & (1 << (token % 8)))


def _zscore(token_ids: list, vocab: int, hash_key: int, gamma: float) -> dict:
    if len(token_ids) < 5:
        return {"z": 0.0, "wmax_50": 0.0, "wmax_100": 0.0}
    greens = []
    for i in range(1, len(token_ids)):
        prev = int(token_ids[i - 1])
        curr = int(token_ids[i])
        if prev >= vocab or curr >= vocab:
            greens.append(0.0)
            continue
        gl = _gl_bytes(prev, vocab, hash_key, gamma)
        greens.append(1.0 if _is_green(curr, gl) else 0.0)
    arr = np.array(greens)
    T = len(arr)
    z = (arr.sum() - gamma * T) / np.sqrt(T * gamma * (1 - gamma) + 1e-12)
    out = {"z": float(z), "wmax_50": 0.0, "wmax_100": 0.0}
    for w in (50, 100):
        if T < w:
            continue
        win_sum = np.convolve(arr, np.ones(w), mode="valid")
        wz = (win_sum - gamma * w) / np.sqrt(w * gamma * (1 - gamma))
        out[f"wmax_{w}"] = float(wz.max())
    return out


def _ensure_tokenizers():
    if _tokenizers:
        return
    import os
    from transformers import AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    seen = set()
    for tok_name, *_ in CONFIGS:
        if tok_name in seen:
            continue
        seen.add(tok_name)
        try:
            kwargs = {"token": token} if token else {}
            _tokenizers[tok_name] = AutoTokenizer.from_pretrained(tok_name, **kwargs)
            print(f"  [kgw_llama] Loaded: {tok_name}")
        except Exception as e:
            print(f"  [kgw_llama] FAILED {tok_name}: {e}")


def _tag(tok_name: str, hash_key: int, gamma: float) -> str:
    safe = tok_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"kgwL_{safe}_h{hash_key}_g{int(gamma * 100)}"


def extract(text: str) -> dict[str, float]:
    _ensure_tokenizers()
    feats: dict[str, float] = {}
    for tok_name, vocab, hash_key, gamma in CONFIGS:
        prefix = _tag(tok_name, hash_key, gamma)
        if tok_name not in _tokenizers:
            for k in ("z", "wmax_50", "wmax_100"):
                feats[f"{prefix}_{k}"] = 0.0
            continue
        tok = _tokenizers[tok_name]
        try:
            ids = tok.encode(text, truncation=True, max_length=1024)
        except Exception:
            for k in ("z", "wmax_50", "wmax_100"):
                feats[f"{prefix}_{k}"] = 0.0
            continue
        sub = _zscore(ids, vocab, hash_key, gamma)
        for k, v in sub.items():
            feats[f"{prefix}_{k}"] = v
    return feats
