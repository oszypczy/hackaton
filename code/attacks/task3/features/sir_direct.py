"""SIR (Liu 2024 ICLR) Semantic Invariant Robust Watermark — projection features.

The official transform_model_cbert.pth maps BERT (1024) → 300-dim semantic space:
  Linear(1024, 500) → TransformLayer(500) → TransformLayer(500) → Linear(500, 300)
Where TransformLayer = Linear(500,500) + ReLU.

We use the projected embeddings as features:
- Token-projection norms (mean/std/min/max)
- Consecutive cosine similarities (semantic consistency)
- Distance to mean (centroid signature)
- Self-similarity matrix statistics

Reference: THU-BPM/Robust_Watermark, MarkLLM-sir HF repo.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

_BERT_NAME = "perceptiveshawty/compositional-bert-large-uncased"
_CHECKPOINT_PATH = os.environ.get(
    "SIR_CHECKPOINT",
    "/p/scratch/training2615/kempinski1/Czumpers/task3/sir_model/transform_model_cbert.pth",
)
_MAX_TOKENS = 256


class _TransformLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))


class _TransformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 500),
            _TransformLayer(500),
            _TransformLayer(500),
            nn.Linear(500, 300),
        )

    def forward(self, x):
        return self.layers(x)


_tokenizer = None
_embedder = None
_transform_model = None
_device = None


def _load_models():
    global _tokenizer, _embedder, _transform_model, _device
    if _transform_model is not None:
        return

    from transformers import AutoTokenizer, AutoModel
    print(f"  [sir] Loading {_BERT_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(_BERT_NAME)
    _embedder = AutoModel.from_pretrained(_BERT_NAME)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _embedder = _embedder.to(_device).eval()

    print(f"  [sir] Loading transform MLP from {_CHECKPOINT_PATH}...")
    if not Path(_CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"SIR checkpoint missing: {_CHECKPOINT_PATH}")

    _transform_model = _TransformModel()
    sd = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    _transform_model.load_state_dict(sd)
    _transform_model = _transform_model.to(_device).eval()
    print(f"  [sir] Models loaded.")


@torch.no_grad()
def _compute_sir_features(text: str) -> Dict[str, float]:
    _load_models()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    out = _embedder(**inputs)
    hidden = out.last_hidden_state[0]  # (n_tokens, 1024)

    proj = _transform_model(hidden)  # (n_tokens, 300)
    proj_np = proj.cpu().numpy()

    n = proj_np.shape[0]
    if n < 2:
        return {f"sir_d{i}": 0.0 for i in range(20)}

    norms = np.linalg.norm(proj_np, axis=1)
    mean_proj = proj_np.mean(axis=0)
    proj_norm = proj_np / (norms[:, None] + 1e-9)
    cos_sims = (proj_norm[:-1] * proj_norm[1:]).sum(axis=1)
    centered = proj_np - mean_proj[None, :]
    dist_to_mean = np.linalg.norm(centered, axis=1)

    self_sim = proj_norm @ proj_norm.T
    np.fill_diagonal(self_sim, 0.0)

    feats = {
        "sir_norm_mean": float(norms.mean()),
        "sir_norm_std": float(norms.std()),
        "sir_norm_min": float(norms.min()),
        "sir_norm_max": float(norms.max()),
        "sir_cos_mean": float(cos_sims.mean()),
        "sir_cos_std": float(cos_sims.std()),
        "sir_cos_min": float(cos_sims.min()),
        "sir_cos_max": float(cos_sims.max()),
        "sir_dist_mean": float(dist_to_mean.mean()),
        "sir_dist_std": float(dist_to_mean.std()),
        "sir_dist_max": float(dist_to_mean.max()),
        "sir_n_tokens": float(n),
        "sir_mean_proj_norm": float(np.linalg.norm(mean_proj)),
        "sir_mean_proj_max": float(np.max(np.abs(mean_proj))),
        "sir_mean_proj_std": float(mean_proj.std()),
        "sir_var_ratio": float(np.var(proj_np, axis=0).max() / (np.var(proj_np).sum() + 1e-9)),
        "sir_self_sim_mean": float(self_sim.sum() / max(1, n * (n - 1))),
        "sir_self_sim_max": float(self_sim.max()),
        "sir_max_dev": float(((dist_to_mean - dist_to_mean.mean()) / (dist_to_mean.std() + 1e-9)).max()),
        "sir_z_norms": float((norms.max() - norms.mean()) / (norms.std() + 1e-9)),
    }
    return feats


def extract(text: str) -> Dict[str, float]:
    """Extract SIR features for a text."""
    try:
        return _compute_sir_features(text)
    except FileNotFoundError as e:
        print(f"  [sir] WARN: {e}")
        return {f"sir_d{i}": 0.0 for i in range(20)}
    except Exception as e:
        print(f"  [sir] WARN: failed: {e}")
        return {f"sir_d{i}": 0.0 for i in range(20)}
