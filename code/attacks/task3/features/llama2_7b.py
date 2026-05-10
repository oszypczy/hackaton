"""Llama-2-7B PPL features — canonical KGW watermark generator.

Uses NousResearch mirror (ungated). Llama-2 is the model used in the original
Kirchenbauer paper, so its PPL on watermarked text reveals the bias maximally.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "NousResearch/Llama-2-7b-hf"


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
    print(f"  [llama2_7b] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = [
        "llama2_lp_mean", "llama2_lp_std", "llama2_lp_p10", "llama2_lp_p25",
        "llama2_lp_p75", "llama2_lp_p90", "llama2_ppl",
        "llama2_top1_frac", "llama2_top10_frac",
    ]

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

    # Rank stats — what fraction of tokens are top-1 / top-10
    sorted_ranks = logits.argsort(dim=-1, descending=True)
    rank_target = (sorted_ranks == targets.unsqueeze(1)).int().argmax(dim=-1).cpu().numpy()

    return {
        "llama2_lp_mean": float(lp.mean()),
        "llama2_lp_std": float(lp.std()),
        "llama2_lp_p10": float(np.percentile(lp, 10)),
        "llama2_lp_p25": float(np.percentile(lp, 25)),
        "llama2_lp_p75": float(np.percentile(lp, 75)),
        "llama2_lp_p90": float(np.percentile(lp, 90)),
        "llama2_ppl": float(np.exp(-lp.mean())),
        "llama2_top1_frac": float((rank_target == 0).mean()),
        "llama2_top10_frac": float((rank_target < 10).mean()),
    }
