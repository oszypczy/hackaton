"""OLMo-2-7B-Instruct judge with PROPER CHAT TEMPLATE.

Previous lm_judge_olmo7b used plain-text prompts. OLMo-2-Instruct was
trained with specific chat format:
  <|endoftext|><|user|>\nQUERY<|endoftext|><|assistant|>\n

When prompted in plain text, the model "wakes up" mid-conversation and
responds inconsistently. With proper chat template, responses are much
sharper -> potentially much better judge signal.

Plus: 5 different prompts (vs 3 in lm_judge) for more variance.
"""
from __future__ import annotations

import numpy as np
import torch

_tok = None
_mod = None
_DEVICE: str | None = None
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"

# Chat-template-style prompts. The model output starts at assistant role.
# yes/no tokens are checked at first generated token.
PROMPTS = [
    {
        "name": "ai_or_human",
        "user": "I'll show you a text. Tell me in one word: was it written by AI or a human?\n\nText: {text}\n\nAnswer with just one word.",
        "yes_tokens": [" AI", "AI", " ai", " Ai"],
        "no_tokens": [" Human", "Human", " human", " H"],
    },
    {
        "name": "watermarked",
        "user": "Examine this text for AI-watermark signals (e.g., unusual token preferences). Answer Yes or No.\n\nText: {text}\n\nIs it watermarked?",
        "yes_tokens": [" Yes", "Yes", " yes"],
        "no_tokens": [" No", "No", " no"],
    },
    {
        "name": "natural_or_llm",
        "user": "Decide if this text is natural human writing or LLM-generated:\n\n{text}\n\nVerdict (one word):",
        "yes_tokens": [" LLM", "LLM", " AI", " Artificial"],
        "no_tokens": [" Human", " Natural", " natural", " human"],
    },
    {
        "name": "real_or_fake",
        "user": "Some texts are written by humans, others by AI. Classify this one as REAL (human) or FAKE (AI).\n\n{text}\n\nClassification:",
        "yes_tokens": [" FAKE", " Fake", "FAKE", " fake"],
        "no_tokens": [" REAL", " Real", "REAL", " real"],
    },
    {
        "name": "confidence",
        "user": "Rate from 1-10 how likely this text is AI-generated, where 10 = definitely AI:\n\n{text}\n\nRating (just the number):",
        "yes_tokens": [" 9", " 10", " 8"],
        "no_tokens": [" 1", " 2", " 3"],
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
    print(f"  [judge_chat] Loading {MODEL_NAME} on {dev} ({dtype})...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(dev).eval()


def _token_id_first(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if ids else 0


@torch.no_grad()
def _judge_one(text: str, prompt_cfg: dict, max_text_chars: int = 4000) -> dict:
    assert _tok is not None and _mod is not None

    text_short = text[:max_text_chars]
    user_msg = prompt_cfg["user"].format(text=text_short)

    # Use chat template if available, otherwise fallback to plain
    try:
        prompt = _tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = user_msg + "\n\nAnswer:"

    inputs = _tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    ids = inputs.input_ids.to(_device())

    out = _mod(ids)
    last_logits = out.logits[0, -1, :].float()  # (V,)

    yes_ids = list({_token_id_first(_tok, t) for t in prompt_cfg["yes_tokens"]})
    no_ids = list({_token_id_first(_tok, t) for t in prompt_cfg["no_tokens"]})

    yes_logits = last_logits[yes_ids]
    no_logits = last_logits[no_ids]

    yes_max = float(yes_logits.max())
    no_max = float(no_logits.max())

    combined = torch.cat([yes_logits, no_logits])
    sm = torch.softmax(combined, dim=0)
    p_yes = float(sm[: len(yes_logits)].sum())

    full_sm = torch.softmax(last_logits, dim=0)
    yes_prob_full = float(full_sm[yes_ids].sum())

    p = prompt_cfg["name"]
    return {
        f"chat_{p}_yes_logit": yes_max,
        f"chat_{p}_no_logit": no_max,
        f"chat_{p}_diff": yes_max - no_max,
        f"chat_{p}_p_yes": p_yes,
        f"chat_{p}_p_yes_full": yes_prob_full,
    }


def extract(text: str) -> dict[str, float]:
    _load()
    feats: dict[str, float] = {}
    for prompt_cfg in PROMPTS:
        try:
            sub = _judge_one(text, prompt_cfg)
            feats.update(sub)
        except Exception:
            p = prompt_cfg["name"]
            for suffix in ("yes_logit", "no_logit", "diff", "p_yes", "p_yes_full"):
                feats[f"chat_{p}_{suffix}"] = 0.0
    return feats
