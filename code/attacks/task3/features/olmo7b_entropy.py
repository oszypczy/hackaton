"""OLMo-2-7B-Instruct per-token entropy + rank features.

B3 — orthogonal to olmo_7b's log-prob: full-distribution entropy at each step.
Watermarked tokens skew toward green-list → distribution shape changes:
  - Lower entropy at biased steps
  - Actual token rank biased low (always green list)
  - Spikes of low-entropy regions
"""
from __future__ import annotations

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
    print(f"  [olmo7b_entropy] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


@torch.no_grad()
def extract(text: str, max_len: int = 1024) -> dict[str, float]:
    _load()
    assert _tok is not None and _mod is not None

    keys = [
        "o7be_ent_mean", "o7be_ent_std", "o7be_ent_p10", "o7be_ent_p50", "o7be_ent_p90",
        "o7be_rank_mean", "o7be_rank_med", "o7be_rank_top1_frac", "o7be_rank_top10_frac",
        "o7be_rank_log_mean",
        "o7be_lpvsmax_mean", "o7be_lpvsmax_std",
        "o7be_top2_diff_mean", "o7be_top2_diff_p10",
        "o7be_kl_uniform_mean", "o7be_burst_low_ent",
    ]
    zero = {k: 0.0 for k in keys}

    try:
        token_ids = _tok.encode(text, truncation=True, max_length=max_len)
    except Exception:
        return zero
    if len(token_ids) < 4:
        return zero

    ids = torch.tensor([token_ids], dtype=torch.long).to(_device())
    logits = _mod(ids).logits[0, :-1].float()  # (T-1, V)
    targets = ids[0, 1:]                        # (T-1,)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Per-token entropy
    ent = -(probs * log_probs).sum(dim=-1)      # (T-1,)
    ent_np = ent.cpu().numpy()

    # Rank of actual token (0 = top-1)
    sorted_ranks = (logits.argsort(dim=-1, descending=True))  # (T-1, V)
    rank_of_target = (sorted_ranks == targets.unsqueeze(1)).int().argmax(dim=-1)  # (T-1,)
    rank_np = rank_of_target.cpu().numpy()

    # Log-prob diff vs argmax (how much suboptimal was each token)
    logp_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    logp_max = log_probs.max(dim=-1).values
    lp_diff = (logp_target - logp_max).cpu().numpy()  # negative or zero

    # Top-2 diff (confidence margin)
    top2 = log_probs.topk(2, dim=-1).values  # (T-1, 2)
    top2_diff = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    # KL to uniform = log V - entropy
    V = log_probs.size(-1)
    kl_unif = (np.log(V) - ent_np)

    # Burst of low-entropy regions: count consecutive low-ent (< median)
    median_ent = float(np.median(ent_np))
    low = ent_np < median_ent
    burst_count = 0
    cur = 0
    for v in low:
        if v: cur += 1
        else:
            if cur >= 3: burst_count += 1
            cur = 0
    if cur >= 3: burst_count += 1

    # NaN safety
    def _safe(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    return {
        "o7be_ent_mean": _safe(ent_np.mean()),
        "o7be_ent_std": _safe(ent_np.std()),
        "o7be_ent_p10": _safe(np.percentile(ent_np, 10)),
        "o7be_ent_p50": _safe(np.percentile(ent_np, 50)),
        "o7be_ent_p90": _safe(np.percentile(ent_np, 90)),
        "o7be_rank_mean": _safe(rank_np.mean()),
        "o7be_rank_med": _safe(np.median(rank_np)),
        "o7be_rank_top1_frac": _safe((rank_np == 0).mean()),
        "o7be_rank_top10_frac": _safe((rank_np < 10).mean()),
        "o7be_rank_log_mean": _safe(np.log1p(rank_np).mean()),
        "o7be_lpvsmax_mean": _safe(lp_diff.mean()),
        "o7be_lpvsmax_std": _safe(lp_diff.std()),
        "o7be_top2_diff_mean": _safe(top2_diff.mean()),
        "o7be_top2_diff_p10": _safe(np.percentile(top2_diff, 10)),
        "o7be_kl_uniform_mean": _safe(kl_unif.mean()),
        "o7be_burst_low_ent": _safe(burst_count),
    }
