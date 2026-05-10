"""Llama-3-8B-Instruct PPL — C2 (NousResearch ungated mirror).

Hypothesis: watermark generator may be Llama-3-family; causal PPL under that LM
differs from OLMo/Mistral signals."""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "NousResearch/Meta-Llama-3-8B"


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
    print(f"  [llama3_8b] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = [
        "llama3_lp_mean", "llama3_lp_std", "llama3_lp_p10", "llama3_lp_p25",
        "llama3_lp_p75", "llama3_lp_p90", "llama3_ppl",
        "llama3_top1_frac", "llama3_top10_frac",
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
    sorted_ranks = logits.argsort(dim=-1, descending=True)
    rank_target = (sorted_ranks == targets.unsqueeze(1)).int().argmax(dim=-1).cpu().numpy()

    return {
        "llama3_lp_mean": float(lp.mean()),
        "llama3_lp_std": float(lp.std()),
        "llama3_lp_p10": float(np.percentile(lp, 10)),
        "llama3_lp_p25": float(np.percentile(lp, 25)),
        "llama3_lp_p75": float(np.percentile(lp, 75)),
        "llama3_lp_p90": float(np.percentile(lp, 90)),
        "llama3_ppl": float(np.exp(-lp.mean())),
        "llama3_top1_frac": float((rank_target == 0).mean()),
        "llama3_top10_frac": float((rank_target < 10).mean()),
    }
