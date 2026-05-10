"""OLMo-2-13B-Instruct PPL features.

Bigger sibling of 7B (which gave 0.259 leaderboard). 13B in fp16 = 26GB,
fits A800 44GB GPU. Public model, no auth.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "allenai/OLMo-2-1124-13B-Instruct"


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
    print(f"  [olmo13b] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys_zero = ["olmo13b_lp_mean", "olmo13b_lp_std", "olmo13b_lp_p10", "olmo13b_lp_p25",
                 "olmo13b_lp_p75", "olmo13b_lp_p90", "olmo13b_ppl"]

    try:
        token_ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return {k: 0.0 for k in keys_zero} | {"olmo13b_ppl": 1.0}

    if len(token_ids) < 2:
        return {k: 0.0 for k in keys_zero} | {"olmo13b_ppl": 1.0}

    ids = torch.tensor([token_ids], dtype=torch.long).to(_device())
    logits = _mod(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()

    return {
        "olmo13b_lp_mean": float(lp.mean()),
        "olmo13b_lp_std": float(lp.std()),
        "olmo13b_lp_p10": float(np.percentile(lp, 10)),
        "olmo13b_lp_p25": float(np.percentile(lp, 25)),
        "olmo13b_lp_p75": float(np.percentile(lp, 75)),
        "olmo13b_lp_p90": float(np.percentile(lp, 90)),
        "olmo13b_ppl": float(np.exp(-lp.mean())),
    }
