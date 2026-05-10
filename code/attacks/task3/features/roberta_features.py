"""Frozen RoBERTa-base pooled embeddings as features.

RoBERTa-base (125M params) trained on web corpus including likely overlap
with both human and LLM-generated text. Pooled embeddings może wykryć
subtelne stylistic patterns invisible to causal LMs.

Returns 768-dim mean pooled embedding from last hidden states.
With C=0.001 LogReg, regularization prevents overfit on 540 samples.

REQUIRES roberta-base in HF cache (`scripts/download_roberta.sh` first).
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_model = None
_DEVICE: str | None = None
MODEL_NAME = "roberta-base"


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
    global _tok, _model
    if _tok is not None:
        return
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    print(f"  [roberta] Loading {MODEL_NAME} on {dev}...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 512) -> dict[str, float]:
    _load()
    assert _tok is not None and _model is not None

    enc = _tok(text, return_tensors="pt", truncation=True, max_length=max_len, padding=False).to(_device())
    outputs = _model(**enc)
    last = outputs.last_hidden_state[0]  # (T, 768)

    # Mean pooled across tokens (768-dim)
    mean_pooled = last.mean(dim=0).cpu().numpy()

    feats: dict[str, float] = {f"rob_{i}": float(mean_pooled[i]) for i in range(768)}
    feats["rob_pooled_norm"] = float(np.linalg.norm(mean_pooled))
    feats["rob_pooled_std"] = float(mean_pooled.std())
    return feats
