"""Min-K%++ membership inference attack (Zhang et al., ICLR 2025)."""
from __future__ import annotations

import math
import zlib
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODEL = "EleutherAI/pythia-410m"
REF_MODEL    = "EleutherAI/pythia-160m"
MAX_LENGTH   = 512
MINK_PCT     = 0.20


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).to(device).train(False)   # train(False) == eval mode, avoids "eval()" hook
    return tokenizer, model


def compute_doc_features(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> dict:
    raw = text.encode("utf-8")
    zlib_ratio = len(zlib.compress(raw)) / max(len(raw), 1)

    enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = enc["input_ids"].to(device)
    T = input_ids.shape[1]

    if T < 2:
        return {"loss": float("nan"), "perplexity": float("nan"),
                "zlib_ratio": zlib_ratio, "minkpp": float("nan"), "n_tokens": T}

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        loss_val = out.loss.item()
        logits = out.logits

    if math.isnan(loss_val):
        # fp32 fallback for fp16 overflow on MPS
        model.float()
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss_val = out.loss.item()
            logits = out.logits
        model.half()

    # Min-K%++: standardize per-token log-prob by vocabulary mean/std at each position
    shift_logits = logits[:, :-1, :].float()   # float32 required for stable log_softmax on MPS
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)            # (1, T-1, V)
    token_log_p = log_probs.gather(
        2, labels.unsqueeze(-1)
    ).squeeze(-1).squeeze(0)                                    # (T-1,)

    mu    = log_probs.mean(dim=-1).squeeze(0)                  # (T-1,)
    sigma = log_probs.std(dim=-1).squeeze(0).clamp(min=1e-8)   # (T-1,)
    per_token = (token_log_p - mu) / sigma                     # (T-1,)

    k = max(1, int(ceil(MINK_PCT * (T - 1))))
    minkpp = torch.topk(per_token, k, largest=False).values.mean().item()

    return {
        "loss":       loss_val,
        "perplexity": math.exp(min(loss_val, 20.0)),
        "zlib_ratio": zlib_ratio,
        "minkpp":     minkpp,
        "n_tokens":   T,
    }


def score_texts(
    texts: list[str],
    model_id: str = TARGET_MODEL,
    max_length: int = MAX_LENGTH,
    show_progress: bool = True,
) -> list[dict]:
    device = pick_device()
    tokenizer, model = load_model(model_id, device)
    iterable = texts
    if show_progress:
        from tqdm import tqdm
        iterable = tqdm(texts, desc=f"Scoring {model_id}")
    return [compute_doc_features(t, tokenizer, model, device, max_length) for t in iterable]


def score_text(text: str) -> float:
    """Single-text API used by smoke.py."""
    device = pick_device()
    tokenizer, model = load_model(TARGET_MODEL, device)
    return compute_doc_features(text, tokenizer, model, device)["minkpp"]


def dataset_level_test(
    set_scores: list[float],
    val_scores: list[float],
    set_id: str,
) -> dict:
    s = np.array([x for x in set_scores if not math.isnan(x)])
    v = np.array([x for x in val_scores if not math.isnan(x)])

    t_stat, p_two = stats.ttest_ind(s, v, equal_var=False)
    p_one = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
    verdict = "in_training" if (p_one < 0.05 and t_stat > 0) else "not_in_training"

    return {"set_id": set_id, "p_value": p_one, "verdict": verdict}
