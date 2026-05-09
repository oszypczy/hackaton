from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from transformers import AutoTokenizer


@dataclass
class BranchBCConfig:
    tokenizer_name: str = "gpt2"
    # Extra tokenizers to also compute features for; list of HF model names.
    # Original Kirchenbauer (2023) used OPT-1.3B → "facebook/opt-1.3b" is a good choice.
    # Liu/Zhao (2024) used LLaMA-2 → add "NousResearch/Llama-2-7b-hf" when available.
    extra_tokenizer_names: list[str] = field(default_factory=list)
    max_length: int = 512
    prior: float = 5.0
    gamma: float = 0.25


class BranchBCExtractor:
    def __init__(self, cfg: BranchBCConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.soft_green_weights: dict[int, float] = {}
        # Extra tokenizers: {tokenizer_name: (tokenizer_obj, soft_green_weights)}
        self._extra: list[tuple[AutoTokenizer, dict[int, float]]] = []
        for name in cfg.extra_tokenizer_names:
            tok = AutoTokenizer.from_pretrained(name)
            self._extra.append((tok, {}))

    def _encode(self, text: str, tok: AutoTokenizer | None = None) -> list[int]:
        t = tok if tok is not None else self.tokenizer
        return t.encode(text, add_special_tokens=False)[: self.cfg.max_length]

    def _fit_one(
        self, texts: list[str], labels: list[int], tok: AutoTokenizer | None = None
    ) -> dict[int, float]:
        wm_counts: dict[int, float] = {}
        cl_counts: dict[int, float] = {}
        prior = self.cfg.prior
        for text, label in zip(texts, labels):
            ids = self._encode(text, tok)
            for t_id in ids:
                if label == 1:
                    wm_counts[t_id] = wm_counts.get(t_id, prior) + 1.0
                    cl_counts.setdefault(t_id, prior)
                else:
                    cl_counts[t_id] = cl_counts.get(t_id, prior) + 1.0
                    wm_counts.setdefault(t_id, prior)
        weights: dict[int, float] = {}
        for t_id in set(wm_counts) | set(cl_counts):
            p_w = wm_counts.get(t_id, prior)
            p_c = cl_counts.get(t_id, prior)
            weights[t_id] = (p_w / (p_w + p_c)) - 0.5
        return weights

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self.soft_green_weights = self._fit_one(texts, labels, tok=None)
        self._extra = [
            (tok, self._fit_one(texts, labels, tok=tok))
            for tok, _ in self._extra
        ]

    def _pseudo_z(self, ids: list[int], weights: dict[int, float], gamma: float) -> float:
        if len(ids) < 2:
            return 0.0
        green_hits = sum(1 for tok in ids if weights.get(tok, 0.0) > 0.0)
        t = len(ids)
        return float((green_hits - gamma * t) / math.sqrt(t * gamma * (1 - gamma) + 1e-9))

    def _window_max_z(
        self, ids: list[int], weights: dict[int, float], win: int, gamma: float
    ) -> float:
        if not ids:
            return 0.0
        if len(ids) <= win:
            return self._pseudo_z(ids, weights, gamma)
        vals = [self._pseudo_z(ids[i : i + win], weights, gamma) for i in range(len(ids) - win + 1)]
        return float(max(vals)) if vals else 0.0

    def _featurize_one(
        self,
        text: str,
        tok: AutoTokenizer | None,
        weights: dict[int, float],
        prefix: str,
    ) -> dict[str, float]:
        ids = self._encode(text, tok)
        gamma = self.cfg.gamma
        soft_score = (
            float(np.mean([weights.get(t, 0.0) for t in ids])) if ids else 0.0
        )
        return {
            f"{prefix}soft_green_score": soft_score,
            f"{prefix}pseudo_z": self._pseudo_z(ids, weights, gamma),
            f"{prefix}winmax_z_50": self._window_max_z(ids, weights, 50, gamma),
            f"{prefix}winmax_z_100": self._window_max_z(ids, weights, 100, gamma),
            f"{prefix}winmax_z_200": self._window_max_z(ids, weights, 200, gamma),
        }

    def featurize(self, text: str) -> dict[str, float]:
        out = self._featurize_one(text, None, self.soft_green_weights, "bc_")
        for i, (tok, weights) in enumerate(self._extra):
            out.update(self._featurize_one(text, tok, weights, f"bc{i + 2}_"))
        return out
