"""B1 — Bidirectional pseudo-log-prob via RoBERTa MLM (one-mask-at-a-time, batched).

Leave-one-out style: mask position i, read log p(true token | full context).
Orthogonal to causal LM PPL; watermarks that skew unidirectional LM may look
different under bidirectional scoring."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "roberta-base"
_MASK_SAMPLE = 20
_BATCH = 16


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _load() -> None:
    global _tok, _mod
    if _tok is not None:
        return
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    print(f"  [roberta_pppl] Loading {MODEL_NAME} on {dev}...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(dev).eval()


def _positions(ids_len: int, text: str) -> list[int]:
    """Score interior tokens only (exclude BOS/EOS equivalents)."""
    if ids_len <= 3:
        return []
    inner = list(range(1, ids_len - 1))
    if len(inner) <= _MASK_SAMPLE:
        return inner
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return sorted(rng.choice(inner, size=_MASK_SAMPLE, replace=False).tolist())


@torch.no_grad()
def extract(text: str, max_len: int = 512) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = [
        "rbpp_mean", "rbpp_std", "rbpp_p10", "rbpp_p50", "rbpp_p90",
        "rbpp_pppl", "rbpp_frac_top1",
    ]
    if not text or len(text.split()) < 3:
        return {k: 0.0 for k in keys}

    try:
        enc = _tok(
            text,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
    except Exception:
        return {k: 0.0 for k in keys}

    ids_1d = enc["input_ids"][0]
    att = enc["attention_mask"][0]
    T = int(ids_1d.shape[0])
    pos = _positions(T, text)
    if not pos:
        return {k: 0.0 for k in keys}

    mask_id = _tok.mask_token_id
    if mask_id is None:
        return {k: 0.0 for k in keys}

    dev = _device()
    lps: list[float] = []
    top1 = []

    for i0 in range(0, len(pos), _BATCH):
        chunk_pos = pos[i0 : i0 + _BATCH]
        base = ids_1d.unsqueeze(0).expand(len(chunk_pos), -1).clone().to(dev)
        attn = att.unsqueeze(0).expand(len(chunk_pos), -1).clone().to(dev)
        for j, p in enumerate(chunk_pos):
            base[j, p] = mask_id
        out = _mod(base, attention_mask=attn).logits
        for j, p in enumerate(chunk_pos):
            tid = int(ids_1d[p])
            lp = F.log_softmax(out[j, p], dim=-1)[tid]
            lps.append(float(lp.cpu()))
            pr = F.softmax(out[j, p], dim=-1)
            top1.append(float(pr[tid].cpu()))

    arr = np.array(lps, dtype=np.float64)
    return {
        "rbpp_mean": float(arr.mean()),
        "rbpp_std": float(arr.std()) if len(arr) > 1 else 0.0,
        "rbpp_p10": float(np.percentile(arr, 10)),
        "rbpp_p50": float(np.percentile(arr, 50)),
        "rbpp_p90": float(np.percentile(arr, 90)),
        "rbpp_pppl": float(np.exp(-arr.mean())),
        "rbpp_frac_top1": float(np.mean(top1)) if top1 else 0.0,
    }
