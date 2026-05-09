"""LM-as-judge: zero-shot AI-detection by prompting instruct LM.

TOTALNIE inne podejście niż PPL/log-prob features. Używamy instruct LM
jako klasyfikator zero-shot — pytamy "czy ten text jest AI/watermarked"
i czytamy logits next-token na "Yes"/"No" / "AI"/"Human" / etc.

Uses OLMo-2-1B-Instruct (cached, multi_lm OLMo dało breakthrough 0.158).
Próbujemy wiele formulacji pytania — różne pytania, różne sygnały.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

# Multi-prompt approach: każde pytanie testuje inną intuicję
PROMPTS = [
    {
        "name": "ai_human",
        "template": "Read the following text and decide if it was written by AI or a human.\n\nText: {text}\n\nThe text was written by",
        "yes_tokens": [" AI", " an", " a", " artificial"],
        "no_tokens": [" a", " human", " human", " someone"],
    },
    {
        "name": "watermark",
        "template": "Examine this text for signs of AI watermarking.\n\nText: {text}\n\nDoes this text contain a watermark? Answer:",
        "yes_tokens": [" Yes", " yes", " Y"],
        "no_tokens": [" No", " no", " N"],
    },
    {
        "name": "natural",
        "template": "Is the following text natural human writing or LLM-generated?\n\nText: {text}\n\nThis text appears",
        "yes_tokens": [" LLM", " AI", " machine", " artificial", " synthetic"],
        "no_tokens": [" natural", " human", " genuine"],
    },
]


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
    print(f"  [judge] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


def _token_id_first(tok, s: str) -> int:
    """Get first token id of string (for next-token logit lookup)."""
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if ids else 0


@torch.no_grad()
def _judge_one(text: str, prompt_cfg: dict, max_text_len: int = 800) -> dict:
    """Run one prompt, return dict of features for this prompt."""
    assert _tok is not None and _mod is not None

    # Truncate text for prompt context
    text_short = text[:max_text_len * 5]  # ~max_text_len words assuming ~5 chars per word
    prompt = prompt_cfg["template"].format(text=text_short)

    inputs = _tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    ids = inputs.input_ids.to(_device())

    out = _mod(ids)
    last_logits = out.logits[0, -1, :].float()  # (V,)

    # Get logits at relevant tokens
    yes_ids = list({_token_id_first(_tok, t) for t in prompt_cfg["yes_tokens"]})
    no_ids = list({_token_id_first(_tok, t) for t in prompt_cfg["no_tokens"]})

    yes_logits = last_logits[yes_ids]
    no_logits = last_logits[no_ids]

    # Aggregate: max across alternative tokens
    yes_max = float(yes_logits.max())
    no_max = float(no_logits.max())

    # Softmax over yes+no candidates only
    combined = torch.cat([yes_logits, no_logits])
    sm = torch.softmax(combined, dim=0)
    p_yes = float(sm[: len(yes_logits)].sum())

    # Plain softmax over full vocab — yes/no probability
    full_sm = torch.softmax(last_logits, dim=0)
    yes_prob_full = float(full_sm[yes_ids].sum())
    no_prob_full = float(full_sm[no_ids].sum())

    p = prompt_cfg["name"]
    return {
        f"judge_{p}_yes_logit": yes_max,
        f"judge_{p}_no_logit": no_max,
        f"judge_{p}_diff": yes_max - no_max,
        f"judge_{p}_p_yes": p_yes,
        f"judge_{p}_p_yes_full": yes_prob_full,
        f"judge_{p}_p_no_full": no_prob_full,
    }


def extract(text: str) -> dict[str, float]:
    _load()
    feats: dict[str, float] = {}
    for prompt_cfg in PROMPTS:
        try:
            sub = _judge_one(text, prompt_cfg)
            feats.update(sub)
        except Exception as e:
            # Defensive: zero features for this prompt if it fails
            p = prompt_cfg["name"]
            for suffix in ("yes_logit", "no_logit", "diff", "p_yes", "p_yes_full", "p_no_full"):
                feats[f"judge_{p}_{suffix}"] = 0.0
    return feats
