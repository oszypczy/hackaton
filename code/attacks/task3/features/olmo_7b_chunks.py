"""OLMo-7B chunk-level PPL features.

OLMo-7B PPL global = 0.259 leaderboard (best). Idea: extract LOCAL features.
Split text into N chunks, compute PPL per chunk, aggregate (mean, std, max,
min, range). Watermarked text is uniform across chunks; human text varies.

Different signal than global PPL — captures local distribution shifts.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    return _DEVICE


def _load() -> None:
    global _tok, _mod
    if _tok is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    use_fp16 = dev == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"  [olmo7b_chunks] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024, n_chunks: int = 5) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys_zero = ["chunk_lp_mean", "chunk_lp_std", "chunk_lp_max", "chunk_lp_min",
                 "chunk_lp_range", "chunk_lp_p25", "chunk_lp_p75",
                 "chunk_ppl_mean", "chunk_ppl_max_min_ratio"]

    try:
        token_ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return {k: 0.0 for k in keys_zero}

    if len(token_ids) < n_chunks * 5:
        return {k: 0.0 for k in keys_zero}

    ids = torch.tensor([token_ids], dtype=torch.long).to(_device())
    logits = _mod(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()

    # Split into N equal chunks
    chunk_size = len(lp) // n_chunks
    chunk_lps = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(lp)
        chunk_lp_mean = float(lp[start:end].mean())
        chunk_lps.append(chunk_lp_mean)

    chunk_lps = np.array(chunk_lps)
    ppls = np.exp(-chunk_lps)

    return {
        "chunk_lp_mean": float(chunk_lps.mean()),
        "chunk_lp_std": float(chunk_lps.std()),
        "chunk_lp_max": float(chunk_lps.max()),
        "chunk_lp_min": float(chunk_lps.min()),
        "chunk_lp_range": float(chunk_lps.max() - chunk_lps.min()),
        "chunk_lp_p25": float(np.percentile(chunk_lps, 25)),
        "chunk_lp_p75": float(np.percentile(chunk_lps, 75)),
        "chunk_ppl_mean": float(ppls.mean()),
        "chunk_ppl_max_min_ratio": float(ppls.max() / (ppls.min() + 1e-9)),
    }
