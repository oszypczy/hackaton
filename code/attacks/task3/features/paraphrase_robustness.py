"""C5 — Paraphrase robustness signal.

Hypothesis: Watermarks (Kirchenbauer/Liu/Zhao) are designed for soft modification
of next-token distributions. Paraphrasing destroys the per-token green-list bias,
so OLMo-7B PPL on paraphrased WM text spikes more than on paraphrased clean text.

Lightweight paraphraser: deterministic token-level perturbations that preserve
local context but break the seed-dependent green-list signal:
  - Every 5th token deletion
  - Word swap (synonym-free): swap adjacent words within sentences
  - Sentence shuffle (preserve sentence boundaries)

Then compute OLMo-7B PPL on perturbed text and compare to original PPL.
"""
from __future__ import annotations

import re

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _load() -> None:
    global _tok, _mod
    if _tok is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = _device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"  [paraphrase] Loading {MODEL_NAME} on {dev}...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def _ppl(text: str, max_len: int = 1024) -> float:
    assert _tok is not None and _mod is not None
    try:
        ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return 1.0
    if len(ids) < 4:
        return 1.0
    t = torch.tensor([ids], dtype=torch.long).to(_device())
    logits = _mod(t).logits[0, :-1].float()
    targets = t[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
    return float(np.exp(-lp.mean()))


def _perturb_drop_every(text: str, n: int = 5) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    return " ".join(w for i, w in enumerate(words) if i % n != 0)


def _perturb_swap_adj(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    out = list(words)
    # Swap pairs of adjacent words at every other position
    for i in range(0, len(out) - 1, 4):
        out[i], out[i + 1] = out[i + 1], out[i]
    return " ".join(out)


def _perturb_shuffle_sent(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 3:
        return text
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(sents))
    return " ".join(sents[p] for p in perm)


def _perturb_truncate_half(text: str) -> str:
    """Take random half of words (deterministic via hash)."""
    words = text.split()
    if len(words) < 8:
        return text
    keep = words[len(words) // 4: 3 * len(words) // 4]
    return " ".join(keep)


def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    keys = [
        "para_ppl_orig", "para_ppl_drop5", "para_ppl_swap", "para_ppl_shuf", "para_ppl_trunc",
        "para_drop5_ratio", "para_swap_ratio", "para_shuf_ratio", "para_trunc_ratio",
        "para_drop5_logdiff", "para_swap_logdiff", "para_shuf_logdiff", "para_trunc_logdiff",
        "para_max_logdiff", "para_min_logdiff", "para_mean_logdiff",
    ]

    if not text or len(text.split()) < 4:
        return {k: 0.0 for k in keys}

    p_orig = _ppl(text, max_len)
    p_drop = _ppl(_perturb_drop_every(text, 5), max_len)
    p_swap = _ppl(_perturb_swap_adj(text), max_len)
    p_shuf = _ppl(_perturb_shuffle_sent(text), max_len)
    p_trunc = _ppl(_perturb_truncate_half(text), max_len)

    eps = 1e-9
    log_orig = float(np.log(p_orig + eps))
    diffs = [
        float(np.log(p_drop + eps)) - log_orig,
        float(np.log(p_swap + eps)) - log_orig,
        float(np.log(p_shuf + eps)) - log_orig,
        float(np.log(p_trunc + eps)) - log_orig,
    ]

    return {
        "para_ppl_orig": p_orig,
        "para_ppl_drop5": p_drop,
        "para_ppl_swap": p_swap,
        "para_ppl_shuf": p_shuf,
        "para_ppl_trunc": p_trunc,
        "para_drop5_ratio": p_drop / (p_orig + eps),
        "para_swap_ratio": p_swap / (p_orig + eps),
        "para_shuf_ratio": p_shuf / (p_orig + eps),
        "para_trunc_ratio": p_trunc / (p_orig + eps),
        "para_drop5_logdiff": diffs[0],
        "para_swap_logdiff": diffs[1],
        "para_shuf_logdiff": diffs[2],
        "para_trunc_logdiff": diffs[3],
        "para_max_logdiff": max(diffs),
        "para_min_logdiff": min(diffs),
        "para_mean_logdiff": float(np.mean(diffs)),
    }
