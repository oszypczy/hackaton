"""Stronger branch_a — same features as branch_a but using Pythia-2.8b.

GPT-2 ma 124M params. Pythia-2.8b ma 22× więcej parametrów → znacznie ostrzejszy
sygnał per-token log-prob i rank distribution. Watermarki łatwiej wykryć
gdy LM jest mocny — zielone tokeny stają się wyraźniej "preferowane".
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import torch

_tok = None
_model = None
_DEVICE: str | None = None
MODEL_NAME = "EleutherAI/pythia-2.8b"


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

    dev = _device()
    use_fp16 = dev == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"  [a_strong] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _model is not None

    ids = _tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(_device())
    T = ids.shape[1]

    feats: dict[str, float] = {}

    if T < 2:
        zeros = ["lp_mean_s", "lp_std_s", "lp_p10_s", "lp_p25_s", "lp_p75_s", "lp_p90_s",
                 "gltr_top10_s", "gltr_top100_s", "gltr_top1000_s", "gltr_rest_s",
                 "ngram_logdiv_1_s", "ngram_logdiv_2_s", "ngram_logdiv_3_s",
                 "burstiness_s", "gzip_ratio_s", "ttr_s"]
        feats["n_tokens_s"] = float(T)
        for k in zeros:
            feats[k] = 0.0
        return feats

    logits = _model(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
    ranks = (logits > logits.gather(1, targets.unsqueeze(1))).sum(-1).cpu().numpy().astype(int)

    feats["lp_mean_s"] = float(lp.mean())
    feats["lp_std_s"] = float(lp.std())
    for pct in [10, 25, 75, 90]:
        feats[f"lp_p{pct}_s"] = float(np.percentile(lp, pct))

    n = len(ranks)
    feats["gltr_top10_s"] = float((ranks < 10).sum() / n)
    feats["gltr_top100_s"] = float((ranks < 100).sum() / n)
    feats["gltr_top1000_s"] = float((ranks < 1000).sum() / n)
    feats["gltr_rest_s"] = float((ranks >= 1000).sum() / n)

    token_ids = ids[0].tolist()
    feats["n_tokens_s"] = float(len(token_ids))
    for ng in [1, 2, 3]:
        grams = [tuple(token_ids[i : i + ng]) for i in range(len(token_ids) - ng + 1)]
        unique_ratio = len(set(grams)) / max(len(grams), 1)
        feats[f"ngram_logdiv_{ng}_s"] = float(-np.log(1.0 - unique_ratio + 1e-9))

    sents = _split_sentences(text)
    lens = [len(s.split()) for s in sents if s]
    if len(lens) >= 2:
        feats["burstiness_s"] = float(np.std(lens) / (np.mean(lens) + 1e-9))
    else:
        feats["burstiness_s"] = 0.0

    b = text.encode("utf-8")
    feats["gzip_ratio_s"] = len(gzip.compress(b)) / max(len(b), 1)

    words = text.lower().split()
    feats["ttr_s"] = len(set(words)) / max(len(words), 1)

    return feats
