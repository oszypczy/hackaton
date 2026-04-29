#!/usr/bin/env python3
"""
Generate fixture data for Challenge B — LLM Watermark Detection & Removal.

Output:
  data/B/texts.jsonl           200 texts (100 wm + 50 clean Llama + 30 GPT-2 XL + 20 human)
  data/B/ground_truth.jsonl    {"id": int, "watermarked": bool}
  data/B/removal_targets.jsonl 50 watermarked texts (B2 subtask input)

Requirements: transformers, torch (CUDA), datasets
  Llama-3-8B-Instruct is a gated model — set HF_TOKEN env var or run:
    huggingface-cli login

Runtime: ~2h on A100 / ~3h on T4

Usage:
  python scripts/generate_B_fixtures.py
  python scripts/generate_B_fixtures.py --dry-run   # 5 of each, fast sanity check
  HF_TOKEN=hf_... python scripts/generate_B_fixtures.py
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

SEED = 42

LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GPT2_MODEL  = "gpt2-xl"

WM_GAMMA    = 0.25
WM_DELTA    = 2.0
WM_HASH_KEY = 15485863  # must match score_B.py

N_WATERMARKED  = 100
N_CLEAN_LLAMA  = 50
N_GPT2         = 30
N_HUMAN        = 20
MIN_NEW_TOKENS = 180
MAX_NEW_TOKENS = 380

DATA_DIR = Path("data/B")


# ── Watermark processor (identical seed logic to score_B.py) ─────────────────

class KirchenbauerLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, gamma: float, delta: float, hash_key: int):
        self.vocab_size = vocab_size
        self.green_size = int(gamma * vocab_size)
        self.delta      = delta
        self.hash_key   = hash_key

    def _green_list(self, prev_token_id: int) -> torch.Tensor:
        seed = (self.hash_key * int(prev_token_id)) % (2 ** 63)
        rng  = torch.Generator()
        rng.manual_seed(seed)
        return torch.randperm(self.vocab_size, generator=rng)[: self.green_size]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            green = self._green_list(input_ids[i, -1].item())
            scores[i, green] += self.delta
        return scores


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_prompts(n: int) -> list[str]:
    """Pull first sentences from wikitext-103 as generation prompts."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    prompts: list[str] = []
    with tqdm(total=n, desc="loading prompts", unit="prompt") as pbar:
        for row in ds:
            line = row["text"].strip()
            if not line or line.startswith("="):
                continue
            words = line.split()
            if len(words) < 10:
                continue
            prompts.append(" ".join(words[: min(40, len(words))]))
            pbar.update(1)
            if len(prompts) >= n:
                break
    return prompts


# ── Generation helpers ────────────────────────────────────────────────────────

def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    watermark: bool,
    vocab_size: int,
) -> list[str]:
    processors = LogitsProcessorList()
    if watermark:
        processors.append(
            KirchenbauerLogitsProcessor(vocab_size, WM_GAMMA, WM_DELTA, WM_HASH_KEY)
        )

    label = "watermarked" if watermark else "clean"
    results = []
    with tqdm(prompts, desc=f"generate {label}", unit="text") as pbar:
        for prompt in pbar:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    logits_processor=processors,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
            pbar.set_postfix({"tokens": len(new_ids)})
    return results


# ── Human texts ───────────────────────────────────────────────────────────────

def load_human_texts(n: int) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
    texts: list[str] = []
    buf: list[str] = []  # accumulate consecutive lines into paragraphs

    def flush(buf: list[str]) -> str | None:
        text = " ".join(buf).strip()
        words = text.split()
        if len(words) >= 150:
            return " ".join(words[:180])  # ~200–250 tokens
        return None

    with tqdm(total=n, desc="loading human texts", unit="text") as pbar:
        for row in ds:
            line = row["text"].strip()
            if line.startswith("=") or not line:
                result = flush(buf)
                if result:
                    texts.append(result)
                    pbar.update(1)
                buf = []
            else:
                buf.append(line)
            if len(texts) >= n:
                break
        if len(texts) < n and buf:
            result = flush(buf)
            if result:
                texts.append(result)

    return texts[:n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="5 of each type — fast sanity check")
    ap.add_argument("--device",  default=None, help="cuda/mps/cpu (auto-detected if omitted)")
    args = ap.parse_args()

    scale = 0.05 if args.dry_run else 1.0
    n_wm  = max(2, int(N_WATERMARKED * scale))
    n_cl  = max(2, int(N_CLEAN_LLAMA * scale))
    n_gp  = max(2, int(N_GPT2        * scale))
    n_hu  = max(2, int(N_HUMAN       * scale))

    random.seed(SEED)
    torch.manual_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    hf_token     = os.environ.get("HF_TOKEN")
    total_llama  = n_wm + n_cl

    print(f"Loading {total_llama + n_gp} prompts from wikitext-103...")
    all_prompts   = load_prompts(total_llama + n_gp)
    llama_prompts = all_prompts[:total_llama]
    gpt2_prompts  = all_prompts[total_llama:]

    # ── Llama-3-8B ────────────────────────────────────────────────────────────
    print(f"\nLoading {LLAMA_MODEL}...")
    tok_llama = AutoTokenizer.from_pretrained(LLAMA_MODEL, token=hf_token)
    mdl_llama = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    mdl_llama.train(False)  # inference mode
    vocab = mdl_llama.config.vocab_size

    print(f"\n[1/4] {n_wm} watermarked Llama texts...")
    wm_texts = generate_texts(mdl_llama, tok_llama, llama_prompts[:n_wm], True, vocab)

    print(f"\n[2/4] {n_cl} clean Llama texts...")
    cl_texts = generate_texts(mdl_llama, tok_llama, llama_prompts[n_wm:], False, vocab)

    del mdl_llama
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── GPT-2 XL ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] {n_gp} GPT-2 XL texts...")
    tok_gpt2 = AutoTokenizer.from_pretrained(GPT2_MODEL)
    tok_gpt2.pad_token = tok_gpt2.eos_token
    mdl_gpt2 = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL, torch_dtype=torch.float16
    ).to(device)
    mdl_gpt2.train(False)
    gp_texts = generate_texts(mdl_gpt2, tok_gpt2, gpt2_prompts, False, mdl_gpt2.config.vocab_size)
    del mdl_gpt2

    # ── Human ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] {n_hu} human texts from wikitext...")
    hu_texts = load_human_texts(n_hu)

    # ── Assemble & shuffle ────────────────────────────────────────────────────
    wm_cfg  = {"scheme": "kirchenbauer", "gamma": WM_GAMMA, "delta": WM_DELTA, "hash_key": WM_HASH_KEY}
    records = []

    for text, prompt in zip(wm_texts, llama_prompts[:n_wm]):
        records.append({"text": text, "model": "llama3-8b", "watermark": wm_cfg, "prompt": prompt})

    for text, prompt in zip(cl_texts, llama_prompts[n_wm:]):
        records.append({"text": text, "model": "llama3-8b", "watermark": None, "prompt": prompt})

    for text, prompt in zip(gp_texts, gpt2_prompts):
        records.append({"text": text, "model": "gpt2-xl", "watermark": None, "prompt": prompt})

    for text in hu_texts:
        records.append({"text": text, "model": "human", "watermark": None, "prompt": None})

    random.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = i

    # ── Write outputs ─────────────────────────────────────────────────────────
    texts_path   = DATA_DIR / "texts.jsonl"
    gt_path      = DATA_DIR / "ground_truth.jsonl"
    removal_path = DATA_DIR / "removal_targets.jsonl"

    with texts_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with gt_path.open("w") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "watermarked": r["watermark"] is not None}) + "\n")

    wm_records = [r for r in records if r["watermark"] is not None][:50]  # B2 = exactly 50
    with removal_path.open("w") as f:
        for r in wm_records:
            f.write(json.dumps({
                "id": r["id"], "text": r["text"],
                "watermark": r["watermark"], "prompt": r["prompt"],
            }) + "\n")

    n_wm_out = sum(1 for r in records if r["watermark"])
    print(f"\nDone.")
    print(f"  {texts_path}:    {len(records)} records")
    print(f"  {gt_path}: {len(records)} records")
    print(f"  {removal_path}: {len(wm_records)} removal targets")
    print(f"  watermarked={n_wm_out}  clean={len(records) - n_wm_out}")


if __name__ == "__main__":
    main()
