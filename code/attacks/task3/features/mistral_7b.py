"""Mistral-7B-Instruct PPL features.

C2 alternative — Llama-3-8B not cached. Mistral-7B is cached, similar size,
different architecture/training data → may catch watermarks OLMo-7B misses.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _load() -> None:
    global _tok, _mod
    if _tok is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = _device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"  [mistral7b] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = ["mistral7b_lp_mean", "mistral7b_lp_std", "mistral7b_lp_p10",
            "mistral7b_lp_p25", "mistral7b_lp_p75", "mistral7b_lp_p90",
            "mistral7b_ppl"]

    try:
        token_ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return {k: 0.0 for k in keys} | {"mistral7b_ppl": 1.0}
    if len(token_ids) < 2:
        return {k: 0.0 for k in keys} | {"mistral7b_ppl": 1.0}

    ids = torch.tensor([token_ids], dtype=torch.long).to(_device())
    logits = _mod(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()

    return {
        "mistral7b_lp_mean": float(lp.mean()),
        "mistral7b_lp_std": float(lp.std()),
        "mistral7b_lp_p10": float(np.percentile(lp, 10)),
        "mistral7b_lp_p25": float(np.percentile(lp, 25)),
        "mistral7b_lp_p75": float(np.percentile(lp, 75)),
        "mistral7b_lp_p90": float(np.percentile(lp, 90)),
        "mistral7b_ppl": float(np.exp(-lp.mean())),
    }
