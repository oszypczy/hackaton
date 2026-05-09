from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer


@dataclass
class BranchBigramConfig:
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    prior: float = 2.0  # lower than BranchBC; bigram space is sparser
    gamma: float = 0.25


class BranchBigramExtractor:
    """Context-aware (bigram) soft green list detector.

    BranchBC learns P(green | token) — captures Zhao/Unigram watermark.
    BranchBigram learns P(green | prev_token, token) — captures KGW/Kirchenbauer,
    where the green list is derived from a hash of the previous token.

    Falls back to unigram weight for unseen (prev, cur) pairs.
    """

    def __init__(self, cfg: BranchBigramConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.soft_bigram_weights: dict[tuple[int, int], float] = {}
        self.soft_unigram_fallback: dict[int, float] = {}

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)[: self.cfg.max_length]

    def fit(self, texts: list[str], labels: list[int]) -> None:
        prior = self.cfg.prior
        wm_bi: dict[tuple, float] = {}
        cl_bi: dict[tuple, float] = {}
        wm_uni: dict[int, float] = {}
        cl_uni: dict[int, float] = {}

        for text, label in zip(texts, labels):
            ids = self._encode(text)
            for tok in ids:
                if label == 1:
                    wm_uni[tok] = wm_uni.get(tok, prior) + 1.0
                    cl_uni.setdefault(tok, prior)
                else:
                    cl_uni[tok] = cl_uni.get(tok, prior) + 1.0
                    wm_uni.setdefault(tok, prior)
            for i in range(1, len(ids)):
                pair = (ids[i - 1], ids[i])
                if label == 1:
                    wm_bi[pair] = wm_bi.get(pair, prior) + 1.0
                    cl_bi.setdefault(pair, prior)
                else:
                    cl_bi[pair] = cl_bi.get(pair, prior) + 1.0
                    wm_bi.setdefault(pair, prior)

        self.soft_bigram_weights = {}
        for pair in set(wm_bi) | set(cl_bi):
            pw = wm_bi.get(pair, prior)
            pc = cl_bi.get(pair, prior)
            self.soft_bigram_weights[pair] = (pw / (pw + pc)) - 0.5

        self.soft_unigram_fallback = {}
        for tok in set(wm_uni) | set(cl_uni):
            pw = wm_uni.get(tok, prior)
            pc = cl_uni.get(tok, prior)
            self.soft_unigram_fallback[tok] = (pw / (pw + pc)) - 0.5

    def _weight(self, prev: int, cur: int) -> float:
        pair = (prev, cur)
        if pair in self.soft_bigram_weights:
            return self.soft_bigram_weights[pair]
        return self.soft_unigram_fallback.get(cur, 0.0)

    def _pseudo_z(self, weights: list[float], gamma: float) -> float:
        if not weights:
            return 0.0
        green_hits = sum(1 for w in weights if w > 0)
        t = len(weights)
        return (green_hits - gamma * t) / math.sqrt(t * gamma * (1 - gamma) + 1e-9)

    def _window_max_z(self, weights: list[float], win: int, gamma: float) -> float:
        if not weights:
            return 0.0
        if len(weights) <= win:
            return self._pseudo_z(weights, gamma)
        return float(
            max(self._pseudo_z(weights[i : i + win], gamma) for i in range(len(weights) - win + 1))
        )

    def featurize(self, text: str) -> dict[str, float]:
        ids = self._encode(text)
        gamma = self.cfg.gamma

        if len(ids) < 2:
            return {
                "bigram_mean_weight": 0.0,
                "bigram_pseudo_z": 0.0,
                "bigram_winmax_z_50": 0.0,
                "bigram_winmax_z_100": 0.0,
                "bigram_winmax_z_200": 0.0,
            }

        weights = [self._weight(ids[i - 1], ids[i]) for i in range(1, len(ids))]
        return {
            "bigram_mean_weight": float(np.mean(weights)),
            "bigram_pseudo_z": self._pseudo_z(weights, gamma),
            "bigram_winmax_z_50": self._window_max_z(weights, 50, gamma),
            "bigram_winmax_z_100": self._window_max_z(weights, 100, gamma),
            "bigram_winmax_z_200": self._window_max_z(weights, 200, gamma),
        }
