from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer


@dataclass
class BranchBCConfig:
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    prior: float = 5.0
    gamma: float = 0.25


class BranchBCExtractor:
    def __init__(self, cfg: BranchBCConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.soft_green_weights: dict[int, float] = {}

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)[: self.cfg.max_length]

    def fit(self, texts: list[str], labels: list[int]) -> None:
        wm_counts: dict[int, float] = {}
        cl_counts: dict[int, float] = {}
        prior = self.cfg.prior
        for text, label in zip(texts, labels):
            ids = self._encode(text)
            for tok in ids:
                if label == 1:
                    wm_counts[tok] = wm_counts.get(tok, prior) + 1.0
                    cl_counts.setdefault(tok, prior)
                else:
                    cl_counts[tok] = cl_counts.get(tok, prior) + 1.0
                    wm_counts.setdefault(tok, prior)
        all_ids = set(wm_counts) | set(cl_counts)
        self.soft_green_weights = {}
        for tok in all_ids:
            p_w = wm_counts.get(tok, prior)
            p_c = cl_counts.get(tok, prior)
            self.soft_green_weights[tok] = (p_w / (p_w + p_c)) - 0.5

    def _pseudo_z(self, ids: list[int], gamma: float) -> float:
        if len(ids) < 2:
            return 0.0
        green_hits = sum(1 for tok in ids if self.soft_green_weights.get(tok, 0.0) > 0.0)
        t = len(ids)
        denom = math.sqrt(t * gamma * (1 - gamma) + 1e-9)
        return float((green_hits - gamma * t) / denom)

    def _window_max_z(self, ids: list[int], win: int, gamma: float) -> float:
        if not ids:
            return 0.0
        if len(ids) <= win:
            return self._pseudo_z(ids, gamma)
        vals = []
        for i in range(0, len(ids) - win + 1):
            vals.append(self._pseudo_z(ids[i : i + win], gamma))
        return float(max(vals)) if vals else 0.0

    def featurize(self, text: str) -> dict[str, float]:
        ids = self._encode(text)
        gamma = self.cfg.gamma
        soft_score = float(np.mean([self.soft_green_weights.get(t, 0.0) for t in ids])) if ids else 0.0
        return {
            "bc_soft_green_score": soft_score,
            "bc_pseudo_z": self._pseudo_z(ids, gamma=gamma),
            "bc_winmax_z_50": self._window_max_z(ids, win=50, gamma=gamma),
            "bc_winmax_z_100": self._window_max_z(ids, win=100, gamma=gamma),
            "bc_winmax_z_200": self._window_max_z(ids, win=200, gamma=gamma),
        }
