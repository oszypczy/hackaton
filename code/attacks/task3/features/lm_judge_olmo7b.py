"""LM-as-judge with OLMo-2-7B-Instruct.

OLMo-1B-Instruct as PPL → 0.158 leaderboard
OLMo-1B-Instruct as JUDGE → 0.20 leaderboard (prompting helps)
OLMo-7B-Instruct as PPL → 0.259 leaderboard (model size helps)
==> OLMo-7B-Instruct as JUDGE → potentially 0.30+ (both wins compound)

Reuse lm_judge.PROMPTS and _judge_one helpers, just swap the model.
"""
from __future__ import annotations

import torch

from .lm_judge import _judge_one, PROMPTS

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
    print(f"  [judge_olmo7b] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


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
                    feats[k.replace("judge_", "judge_olmo7b_")] = v
            except Exception:
                p = prompt_cfg["name"]
                for suffix in ("yes_logit", "no_logit", "diff", "p_yes", "p_yes_full", "p_no_full"):
                    feats[f"judge_olmo7b_{p}_{suffix}"] = 0.0
    finally:
        _lj._tok, _lj._mod = saved
    return feats
