"""Branch D — Semantic proxy for Liu 2024 (SIR) watermark.

Encodes sentences with a small Sentence-Transformer and extracts:
- Adjacent-sentence cosine similarity (mean, variance)
- LSH bucket KL-divergence (8 hyperplanes → 256 buckets)

These signals target semantic-level watermarks (SemStamp, SIR) that are
invisible to token-frequency approaches (Branches B/C).
Works on CPU / MPS / CUDA via sentence-transformers auto-device selection.
"""
from __future__ import annotations

import re

import numpy as np

_HYPERPLANES: np.ndarray | None = None  # (8, embed_dim), seeded RNG
_sbert = None


def _load_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer

        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert


def _hyperplanes(dim: int) -> np.ndarray:
    global _HYPERPLANES
    if _HYPERPLANES is None or _HYPERPLANES.shape[1] != dim:
        rng = np.random.default_rng(42)
        hp = rng.standard_normal((8, dim))
        _HYPERPLANES = hp / np.linalg.norm(hp, axis=1, keepdims=True)
    return _HYPERPLANES


def _split_sentences(text: str, min_words: int = 4) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= min_words]


def extract(text: str) -> dict[str, float]:
    sents = _split_sentences(text)
    feats: dict[str, float] = {
        "adj_cosine_mean": 0.0,
        "adj_cosine_var": 0.0,
        "n_sents": float(len(sents)),
        "lsh_kl_div": 0.0,
    }

    if len(sents) < 2:
        return feats

    model = _load_sbert()
    embeds = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)

    # Adjacent cosine similarity (normalized → dot product)
    cosines = [float(np.dot(embeds[i], embeds[i + 1])) for i in range(len(embeds) - 1)]
    feats["adj_cosine_mean"] = float(np.mean(cosines))
    feats["adj_cosine_var"] = float(np.var(cosines))

    # LSH bucket KL divergence vs uniform
    dim = embeds.shape[1]
    hp = _hyperplanes(dim)
    projections = (embeds @ hp.T > 0).astype(np.uint8)  # (N_sents, 8)
    bucket_ids = projections.dot(1 << np.arange(8, dtype=np.uint8))  # [0, 255]
    observed = np.bincount(bucket_ids, minlength=256).astype(float)
    observed /= observed.sum() + 1e-9
    uniform = np.ones(256) / 256.0
    # Symmetric KL
    kl = np.sum(observed * np.log(observed / (uniform + 1e-9) + 1e-9))
    feats["lsh_kl_div"] = float(kl)

    return feats
