from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class BranchDConfig:
    model_name: str = "all-MiniLM-L6-v2"
    n_planes: int = 8
    seed: int = 42
    device: str = "cpu"


class BranchDExtractor:
    def __init__(self, cfg: BranchDConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.hyperplanes: np.ndarray | None = None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        for sep in ["!", "?"]:
            text = text.replace(sep, ".")
        sents = [s.strip() for s in text.split(".") if s.strip()]
        return sents if sents else [text.strip() or " "]

    def _ensure_hyperplanes(self, dim: int) -> None:
        if self.hyperplanes is None:
            self.hyperplanes = self.rng.normal(size=(self.cfg.n_planes, dim))

    def featurize(self, text: str) -> dict[str, float]:
        sents = self._split_sentences(text)
        emb = self.model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 1:
            emb = emb[None, :]
        self._ensure_hyperplanes(emb.shape[1])
        assert self.hyperplanes is not None
        proj = emb @ self.hyperplanes.T
        bits = (proj > 0).astype(int)
        bucket_ids = bits.dot(1 << np.arange(bits.shape[1]))
        n_buckets = 2 ** bits.shape[1]
        hist = np.bincount(bucket_ids, minlength=n_buckets).astype(float)
        hist = hist / max(1.0, hist.sum())
        uniform = np.full_like(hist, 1.0 / len(hist))
        kl = float(np.sum(hist * np.log((hist + 1e-9) / (uniform + 1e-9))))
        if len(emb) > 1:
            adj = np.sum(emb[:-1] * emb[1:], axis=1)
            cos_mean = float(adj.mean())
            cos_var = float(adj.var())
        else:
            cos_mean = 1.0
            cos_var = 0.0
        sent_norms = np.linalg.norm(emb, axis=1)
        entropy_proxy = float(np.std(sent_norms) / (np.mean(sent_norms) + 1e-9))
        return {
            "d_n_sentences": float(len(sents)),
            "d_adj_cos_mean": cos_mean,
            "d_adj_cos_var": cos_var,
            "d_lsh_kl_uniform": kl,
            "d_semantic_entropy_proxy": entropy_proxy,
        }
