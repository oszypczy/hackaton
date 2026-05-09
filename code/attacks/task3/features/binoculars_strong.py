"""Stronger Binoculars: Pythia-1.4b (observer) + Pythia-2.8b (performer).

Same approach as binoculars.py but with much stronger LMs. Better signal
for detecting LLM-generated text in general (works on any watermark scheme).
GPU REQUIRED (fp16). ~6GB GPU memory total.
"""
from __future__ import annotations

import numpy as np
import torch

_obs_tok = None
_obs_mod = None
_per_tok = None
_per_mod = None
_DEVICE: str | None = None

OBSERVER_NAME = "EleutherAI/pythia-1.4b"
PERFORMER_NAME = "EleutherAI/pythia-2.8b"


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
    global _obs_tok, _obs_mod, _per_tok, _per_mod
    if _obs_tok is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    use_fp16 = dev == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"  [bino_strong] Loading {OBSERVER_NAME} + {PERFORMER_NAME} on {dev} ({dtype})...")
    _obs_tok = AutoTokenizer.from_pretrained(OBSERVER_NAME)
    _obs_mod = AutoModelForCausalLM.from_pretrained(OBSERVER_NAME, torch_dtype=dtype).to(dev).eval()
    _per_tok = AutoTokenizer.from_pretrained(PERFORMER_NAME)
    _per_mod = AutoModelForCausalLM.from_pretrained(PERFORMER_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def _mean_logprob(text: str, tok, model, max_len: int = 1024) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(_device())
    if ids.shape[1] < 2:
        return 0.0
    logits = model(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(lp.mean().cpu())


def extract(text: str) -> dict[str, float]:
    _load()
    assert _obs_tok is not None and _per_tok is not None

    mean_lp_obs = _mean_logprob(text, _obs_tok, _obs_mod)
    mean_lp_per = _mean_logprob(text, _per_tok, _per_mod)

    ppl_obs = float(np.exp(-mean_lp_obs))
    ppl_per = float(np.exp(-mean_lp_per))

    log_obs = np.log(max(ppl_obs, 1.001))
    log_per = np.log(max(ppl_per, 1.001))

    bino_raw = log_obs / (log_per + 1e-9)
    bino_neg = -bino_raw

    return {
        "bino_strong_score": float(bino_neg),
        "bino_strong_ppl_obs": float(ppl_obs),
        "bino_strong_ppl_per": float(ppl_per),
        "bino_strong_lp_obs": float(mean_lp_obs),
        "bino_strong_lp_per": float(mean_lp_per),
    }
