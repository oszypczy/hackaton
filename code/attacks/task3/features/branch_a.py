"""Branch A — scheme-agnostic features from GPT-2-small surrogate LM.

Features: log-prob statistics, GLTR rank buckets, n-gram log-diversity,
burstiness, gzip compression ratio, type-token ratio, token count.
Works on CPU / MPS / CUDA.
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import torch

_tok = None
_model = None
_DEVICE: str | None = None


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("gpt2")
    _model = AutoModelForCausalLM.from_pretrained("gpt2").to(_device()).eval()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _model is not None

    ids = _tok(
        text, return_tensors="pt", truncation=True, max_length=max_len
    ).input_ids.to(_device())
    T = ids.shape[1]

    feats: dict[str, float] = {}

    if T < 2:
        # Degenerate: return safe zeros
        zeros = ["lp_mean", "lp_std", "lp_p10", "lp_p25", "lp_p75", "lp_p90",
                 "gltr_top10", "gltr_top100", "gltr_top1000", "gltr_rest",
                 "ngram_logdiv_1", "ngram_logdiv_2", "ngram_logdiv_3",
                 "burstiness", "gzip_ratio", "ttr"]
        feats["n_tokens"] = float(T)
        for k in zeros:
            feats[k] = 0.0
        return feats

    logits = _model(ids).logits[0, :-1].float()  # (T-1) x V
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
    ranks = (logits > logits.gather(1, targets.unsqueeze(1))).sum(-1).cpu().numpy().astype(int)

    # Log-prob statistics
    feats["lp_mean"] = float(lp.mean())
    feats["lp_std"] = float(lp.std())
    for pct in [10, 25, 75, 90]:
        feats[f"lp_p{pct}"] = float(np.percentile(lp, pct))

    # GLTR rank buckets
    n = len(ranks)
    feats["gltr_top10"] = float((ranks < 10).sum() / n)
    feats["gltr_top100"] = float((ranks < 100).sum() / n)
    feats["gltr_top1000"] = float((ranks < 1000).sum() / n)
    feats["gltr_rest"] = float((ranks >= 1000).sum() / n)

    # N-gram log-diversity from token IDs
    token_ids = ids[0].tolist()
    feats["n_tokens"] = float(len(token_ids))
    for ng in [1, 2, 3]:
        grams = [tuple(token_ids[i : i + ng]) for i in range(len(token_ids) - ng + 1)]
        unique_ratio = len(set(grams)) / max(len(grams), 1)
        feats[f"ngram_logdiv_{ng}"] = float(-np.log(1.0 - unique_ratio + 1e-9))

    # Burstiness: sentence-length coefficient of variation
    sents = _split_sentences(text)
    lens = [len(s.split()) for s in sents if s]
    if len(lens) >= 2:
        feats["burstiness"] = float(np.std(lens) / (np.mean(lens) + 1e-9))
    else:
        feats["burstiness"] = 0.0

    # Compression ratio
    b = text.encode("utf-8")
    feats["gzip_ratio"] = len(gzip.compress(b)) / max(len(b), 1)

    # Type-token ratio (word level)
    words = text.lower().split()
    feats["ttr"] = len(set(words)) / max(len(words), 1)

    return feats
