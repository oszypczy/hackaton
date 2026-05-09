"""LM-as-judge with Mistral-7B-Instruct (gated but cached).

Same prompts as lm_judge under Mistral-7B-Instruct.
"""
from __future__ import annotations

import os

import torch

from .lm_judge import _judge_one, PROMPTS

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


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
    token = os.environ.get("HF_TOKEN")
    kwargs = {"token": token} if token else {}
    print(f"  [judge_mistral] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, **kwargs).to(dev).eval()


def extract(text: str) -> dict[str, float]:
    _load()
    feats: dict[str, float] = {}
    from . import lm_judge as _lj
    saved = (_lj._tok, _lj._mod)
    _lj._tok = _tok
    _lj._mod = _mod
    try:
        for prompt_cfg in PROMPTS:
            try:
                sub = _judge_one(text, prompt_cfg)
                for k, v in sub.items():
                    feats[k.replace("judge_", "judge_mistral_")] = v
            except Exception:
                p = prompt_cfg["name"]
                for suffix in ("yes_logit", "no_logit", "diff", "p_yes", "p_yes_full", "p_no_full"):
                    feats[f"judge_mistral_{p}_{suffix}"] = 0.0
    finally:
        _lj._tok, _lj._mod = saved
    return feats
