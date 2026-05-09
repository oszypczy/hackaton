"""Branch B (KGW WinMax proxy) + Branch C (Unigram soft green list).

Branch C requires fitting on training data — call UnigramGreenList.fit() first.
Branch B (WinMax) uses the fitted green list for sliding-window z-scores.
Both are stateless after fitting; tokenizer-independent for the z-score formula.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_VOCAB = 50257  # GPT-2 vocab size


class UnigramGreenList:
    """Learned soft green list from training labels (direct attack on Zhao 2024 Unigram-Watermark).

    Fits token-level frequency ratios watermarked/clean to form a soft green-list
    weight vector. Any test text can then be scored without knowing the secret key.
    """

    def __init__(self, gamma: float = 0.25, smooth: float = 5.0, vocab_size: int = _DEFAULT_VOCAB):
        self.gamma = gamma
        self.smooth = smooth
        self.vocab_size = vocab_size
        self.soft_g: np.ndarray | None = None  # shape (vocab_size,); >0 = watermarked-tilted

    def fit(self, texts: list[str], labels: list[int], tokenizer) -> None:
        cnt_wm = np.ones(self.vocab_size) * self.smooth
        cnt_cl = np.ones(self.vocab_size) * self.smooth
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text)[:1024]
            for i in ids:
                if i < self.vocab_size:
                    if label == 1:
                        cnt_wm[i] += 1
                    else:
                        cnt_cl[i] += 1
        p_wm = cnt_wm / cnt_wm.sum()
        p_cl = cnt_cl / cnt_cl.sum()
        # Soft green score: positive → watermarked-tilted token
        self.soft_g = p_wm / (p_wm + p_cl + 1e-12) - 0.5

    def _green_hits(self, ids: list[int]) -> np.ndarray:
        assert self.soft_g is not None, "Call fit() first"
        return np.array([self.soft_g[i] if i < self.vocab_size else 0.0 for i in ids])

    def zscore(self, text: str, tokenizer) -> float:
        ids = tokenizer.encode(text)[:1024]
        if len(ids) < 5:
            return 0.0
        g = self._green_hits(ids)
        green_hits = (g > 0).sum()
        T = len(ids)
        return float((green_hits - self.gamma * T) / np.sqrt(T * self.gamma * (1 - self.gamma)))

    def winmax_zscore(
        self, text: str, tokenizer, windows: tuple[int, ...] = (50, 100, 200)
    ) -> dict[str, float]:
        ids = tokenizer.encode(text)[:1024]
        result = {f"winmax_z_{w}": 0.0 for w in windows}
        if len(ids) < 10:
            return result
        greens = (self._green_hits(ids) > 0).astype(float)
        for w in windows:
            if len(ids) < w:
                continue
            win_sum = np.convolve(greens, np.ones(w), mode="valid")
            z = (win_sum - self.gamma * w) / np.sqrt(w * self.gamma * (1 - self.gamma))
            result[f"winmax_z_{w}"] = float(z.max())
        return result


def extract(text: str, green_list: UnigramGreenList, tokenizer) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats["unigram_zscore"] = green_list.zscore(text, tokenizer)
    feats.update(green_list.winmax_zscore(text, tokenizer))
    return feats
