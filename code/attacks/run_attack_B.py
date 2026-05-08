"""
Run Challenge B attack: Kirchenbauer watermark detection (B1) + removal (B2).

B1: z-score detector — gamma, hash_key read from fixture (white-box)
B2: removal cascade — emoji insert → back-translation EN→DE→EN

Usage:
    python code/attacks/run_attack_B.py b1
    python code/attacks/run_attack_B.py b2

Outputs:
    submissions/B1.jsonl   {"id", "watermarked", "z_score", "confidence"}
    submissions/B2.jsonl   {"id", "modified_text"}

Runtime:
    B1: <30s (pure tokenizer + numpy, no GPU)
    B2: ~5-15 min (Helsinki-NLP opus-mt on CPU, ~200MB download on first run)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT     = Path(__file__).resolve().parents[2]
DATA_B   = ROOT / "data" / "B"
SUB_DIR  = ROOT / "submissions"

GAMMA    = 0.25
HASH_KEY = 15485863
Z_THRESH = 4.0
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
EMOJIS   = ["🌟", "🎯", "🔥", "💡", "✨", "🎪", "🦋"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_tokenizer() -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        print(f"  Tokenizer: {LLAMA_ID}")
        return tok
    except Exception:
        print("  [WARN] Llama-3 unavailable — using gpt2 (z-scores approximate)")
        return AutoTokenizer.from_pretrained("gpt2")


def kirchenbauer_zscore(
    text: str,
    tokenizer,
    gamma: float = GAMMA,
    hash_key: int = HASH_KEY,
) -> float:
    token_ids  = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        return 0.0
    vocab_size = tokenizer.vocab_size
    green_size = int(gamma * vocab_size)
    n_green    = 0
    T          = len(token_ids) - 1  # skip first token (no prev to seed from)
    rng = torch.Generator()
    for i in range(1, len(token_ids)):
        prev = token_ids[i - 1]
        seed = (hash_key * prev) % (2**63)
        rng.manual_seed(seed)
        green_ids = torch.randperm(vocab_size, generator=rng)[:green_size].tolist()
        if token_ids[i] in green_ids:
            n_green += 1
    if T == 0:
        return 0.0
    return (n_green - gamma * T) / (gamma * (1.0 - gamma) * T) ** 0.5


# ── B1: detection ─────────────────────────────────────────────────────────────

def run_b1() -> None:
    texts = load_jsonl(DATA_B / "texts.jsonl")
    tok   = get_tokenizer()

    results = []
    for row in tqdm(texts, desc="B1 z-score"):
        wm       = row.get("watermark") or {}
        gamma    = wm.get("gamma",    GAMMA)
        hash_key = wm.get("hash_key", HASH_KEY)
        z = kirchenbauer_zscore(row["text"], tok, gamma=gamma, hash_key=hash_key)
        # Sigmoid centred on threshold → smooth confidence score for AUC
        confidence = float(1.0 / (1.0 + np.exp(-0.5 * (z - Z_THRESH))))
        results.append({
            "id":          row["id"],
            "watermarked": bool(z > Z_THRESH),
            "z_score":     round(float(z), 4),
            "confidence":  round(confidence, 4),
        })

    SUB_DIR.mkdir(exist_ok=True)
    out_path = SUB_DIR / "B1.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_pos = sum(r["watermarked"] for r in results)
    print(f"\nWrote {out_path}  ({n_pos}/{len(results)} watermarked)")
    print(f"\nScore: python code/practice/score_B.py b1")


# ── B2: removal ───────────────────────────────────────────────────────────────

def emoji_insert(text: str, every_n: int = 8) -> str:
    """Insert an emoji every N words — breaks KGW prev-token seed chain."""
    words  = text.split()
    result = []
    for i, w in enumerate(words):
        result.append(w)
        if (i + 1) % every_n == 0:
            result.append(random.choice(EMOJIS))
    return " ".join(result)


_bt_models: dict = {}

def backtranslate(text: str) -> str:
    """EN → DE → EN via Helsinki-NLP opus-mt (CPU, ~200MB download once)."""
    from transformers import MarianMTModel, MarianTokenizer

    def _translate(src: str, model_id: str) -> str:
        if model_id not in _bt_models:
            _bt_models[model_id] = (
                MarianTokenizer.from_pretrained(model_id),
                MarianMTModel.from_pretrained(model_id),
            )
        tok_m, model = _bt_models[model_id]
        inp  = tok_m([src], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512)
        return tok_m.decode(out[0], skip_special_tokens=True)

    de   = _translate(text, "Helsinki-NLP/opus-mt-en-de")
    back = _translate(de,   "Helsinki-NLP/opus-mt-de-en")
    return back


def run_b2() -> None:
    targets = load_jsonl(DATA_B / "removal_targets.jsonl")
    tok     = get_tokenizer()

    n_emoji = n_bt = n_fail = 0
    results = []

    for row in tqdm(targets, desc="B2 removal"):
        wm       = row.get("watermark") or {}
        gamma    = wm.get("gamma",    GAMMA)
        hash_key = wm.get("hash_key", HASH_KEY)
        original = row["text"]

        # Step 1: emoji insert — free, instant, often enough for KGW h=1
        modified = emoji_insert(original)
        z = kirchenbauer_zscore(modified, tok, gamma=gamma, hash_key=hash_key)

        if z < Z_THRESH:
            n_emoji += 1
        else:
            # Step 2: back-translation EN→DE→EN — changes tokenisation entirely
            try:
                bt = backtranslate(original)
                z2 = kirchenbauer_zscore(bt, tok, gamma=gamma, hash_key=hash_key)
                if z2 < Z_THRESH:
                    modified = bt
                    n_bt += 1
                else:
                    modified = bt if z2 < z else modified  # take whichever is lower
                    n_fail += 1
            except Exception as e:
                print(f"  [WARN] backtranslate id={row['id']}: {e}")
                n_fail += 1

        results.append({"id": row["id"], "modified_text": modified})

    SUB_DIR.mkdir(exist_ok=True)
    out_path = SUB_DIR / "B2.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {out_path}")
    print(f"  emoji OK: {n_emoji}  bt OK: {n_bt}  failed: {n_fail}")
    print(f"\nScore: python code/practice/score_B.py b2 --no-bertscore")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Challenge B: watermark detection + removal")
    ap.add_argument("task", choices=["b1", "b2"], help="b1=detection, b2=removal")
    args = ap.parse_args()
    if args.task == "b1":
        run_b1()
    else:
        run_b2()


if __name__ == "__main__":
    main()
