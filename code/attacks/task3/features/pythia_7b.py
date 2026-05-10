"""Pythia-6.9B PPL features — orthogonal architecture (not Llama family).

If the watermark generator was Llama-based, Pythia's PPL on the same text
will differ characteristically, providing an orthogonal signal.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "EleutherAI/pythia-6.9b"


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
    print(f"  [pythia_7b] Loading {MODEL_NAME} on {dev}...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = ["pyth7b_lp_mean", "pyth7b_lp_std", "pyth7b_lp_p10",
            "pyth7b_lp_p25", "pyth7b_lp_p75", "pyth7b_lp_p90", "pyth7b_ppl"]

    try:
        token_ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return {k: 0.0 for k in keys}
    if len(token_ids) < 2:
        return {k: 0.0 for k in keys}

    ids = torch.tensor([token_ids], dtype=torch.long).to(_device())
    logits = _mod(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()

    return {
        "pyth7b_lp_mean": float(lp.mean()),
        "pyth7b_lp_std": float(lp.std()),
        "pyth7b_lp_p10": float(np.percentile(lp, 10)),
        "pyth7b_lp_p25": float(np.percentile(lp, 25)),
        "pyth7b_lp_p75": float(np.percentile(lp, 75)),
        "pyth7b_lp_p90": float(np.percentile(lp, 90)),
        "pyth7b_ppl": float(np.exp(-lp.mean())),
    }
