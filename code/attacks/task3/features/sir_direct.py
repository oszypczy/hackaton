"""SIR (Liu 2024 ICLR) watermark detection features.

Semantic Invariant Robust Watermark — uses a trained BERT embedding model
plus a transformation MLP to produce per-token watermark logits.

Reference: THU-BPM/Robust_Watermark, arxiv 2310.06356.
Default: compositional-bert-large-uncased (1024-dim) → MLP → vocab(50257).

Requires:
  - HF model: perceptiveshawty/compositional-bert-large-uncased
  - Checkpoint: $TASK_CACHE/../sir_model/transform_model_cbert.pth
    (download via code/attacks/task3/download_sir_model.sh)
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

_embedder: Any = None
_transform_model: Any = None
_tokenizer: Any = None

# Path to the MLP checkpoint — resolved via env var or default
_CHECKPOINT = Path(
    os.environ.get(
        "SIR_CHECKPOINT",
        "/p/scratch/training2615/kempinski1/Czumpers/task3/sir_model/transform_model_cbert.pth",
    )
)
_BERT_NAME = "perceptiveshawty/compositional-bert-large-uncased"


def _load_models():
    global _embedder, _transform_model, _tokenizer
    if _embedder is not None:
        return

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(_BERT_NAME)
    _embedder = AutoModel.from_pretrained(_BERT_NAME)
    _embedder.eval()
    _embedder.to("cuda" if torch.cuda.is_available() else "cpu")

    class TransformModel(nn.Module):
        def __init__(self, input_dim: int = 1024, output_dim: int = 50257):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 2048)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(2048, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    device = next(_embedder.parameters()).device
    if _CHECKPOINT.exists():
        _transform_model = TransformModel()
        _transform_model.load_state_dict(
            __import__("torch").load(_CHECKPOINT, map_location=device, weights_only=True)
        )
        _transform_model.eval()
        _transform_model.to(device)
    else:
        _transform_model = None
        print(f"[sir_direct] WARNING: checkpoint not found at {_CHECKPOINT}, skipping SIR features")


def _compute_sir_zscore(text: str) -> float:
    """Compute per-token watermark score and normalize to z-score."""
    import torch

    _load_models()
    if _transform_model is None:
        return 0.0

    device = next(_embedder.parameters()).device
    tokens = _tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = tokens.size(1)
    if seq_len < 10:
        return 0.0

    accumulated_score = 0.0
    count = 0
    with torch.no_grad():
        # stride by 4 to avoid O(n^2) cost; still samples enough positions
        for t in range(5, seq_len, max(1, (seq_len - 5) // 100 + 1)):
            context = tokens[:, :t]
            actual_token = tokens[0, t].item()
            if actual_token >= 50257:
                continue
            outputs = _embedder(context)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, 1024)
            wm_logits = _transform_model(cls_emb)         # (1, 50257)
            token_score = wm_logits[0, actual_token].item()
            accumulated_score += token_score
            count += 1

    if count < 5:
        return 0.0

    # Calibrate: empirical mean/std from clean texts ~0 ± 1 (rough estimate)
    # The z-score normalizes by sqrt(count); mean_expected is 0 for clean text.
    return accumulated_score / math.sqrt(count)


def extract(text: str) -> dict[str, float]:
    z = _compute_sir_zscore(str(text))
    return {"sir_zscore": z}
