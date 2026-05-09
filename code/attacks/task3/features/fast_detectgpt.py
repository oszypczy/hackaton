"""Fast-DetectGPT (Bao et al. 2023, NeurIPS) — analytical curvature score.

For each token position i, compute z-score of actual log-prob vs model's
analytical expected log-prob distribution:

  z_i = (log p(x_i|x_<i) - E_v[log p(v|x_<i)]) / Std_v[log p(v|x_<i)]

Where E_v and Std_v are computed analytically from softmax distribution.

Aggregate stats over positions: mean, std, percentiles, max, min.

Watermarked text → z_mean consistently positive (green tokens prefferred)
Human text → z varies widely (LM surprised at unusual word choices)
LLM-generated (no wm) → z_mean ~ 0

Single forward pass per text. ~50ms on Pythia-2.8b GPU.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
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
    global _tok, _mod
    if _tok is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    use_fp16 = dev == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"  [fdgpt] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys_zero = ["fdgpt_mean", "fdgpt_std", "fdgpt_p10", "fdgpt_p25",
                 "fdgpt_p75", "fdgpt_p90", "fdgpt_max", "fdgpt_min",
                 "fdgpt_pos_frac", "fdgpt_strong_pos_frac"]

    ids = _tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(_device())
    if ids.shape[1] < 5:
        return {k: 0.0 for k in keys_zero}

    logits = _mod(ids).logits[0, :-1].float()  # (T-1, V)
    targets = ids[0, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    actual_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)

    # Analytical expected log-prob = -entropy
    probs = log_probs.exp()
    expected_lp = (probs * log_probs).sum(dim=-1)  # (T-1,)

    # Variance of log-prob under the model's distribution
    variance = (probs * (log_probs - expected_lp.unsqueeze(-1)) ** 2).sum(dim=-1)
    std = variance.clamp(min=1e-12).sqrt()

    z = (actual_lp - expected_lp) / std  # (T-1,)
    z_np = z.cpu().numpy()

    return {
        "fdgpt_mean": float(z_np.mean()),
        "fdgpt_std": float(z_np.std()),
        "fdgpt_p10": float(np.percentile(z_np, 10)),
        "fdgpt_p25": float(np.percentile(z_np, 25)),
        "fdgpt_p75": float(np.percentile(z_np, 75)),
        "fdgpt_p90": float(np.percentile(z_np, 90)),
        "fdgpt_max": float(z_np.max()),
        "fdgpt_min": float(z_np.min()),
        # Fraction of positions with positive z (token "more predictable than expected")
        "fdgpt_pos_frac": float((z_np > 0).mean()),
        # Strong positive: z > 1 (significantly more predictable)
        "fdgpt_strong_pos_frac": float((z_np > 1.0).mean()),
    }
