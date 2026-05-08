"""
Challenge B1 — watermark detection via Kirchenbauer green-list z-score.

Strategy:
  1. Compute z-score on full text (uses len(tokenizer) to match generator).
  2. Sliding window of 100 tokens, take max z — catches partially watermarked.
  3. predict watermarked := max(z_full, z_sliding_max) > 4.0
  4. confidence := logistic(max_z, scale=2.0)

Usage:
    python code/attacks/run_attack_B1.py
Output:
    submissions/B1.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "practice"))

from tqdm import tqdm  # noqa: E402

from score_B import _load_tokenizer, kirchenbauer_zscore  # noqa: E402

DATA_DIR    = ROOT / "data" / "B"
SUB_DIR     = ROOT / "submissions"
TEXTS_PATH  = DATA_DIR / "texts.jsonl"
OUT_PATH    = SUB_DIR / "B1.jsonl"

GAMMA       = 0.25
HASH_KEY    = 15485863
THRESHOLD   = 4.0
WINDOW      = 100        # tokens
STRIDE      = 50


def _z_for_window(token_ids: list[int], tokenizer, gamma: float, hash_key: int) -> float:
    """Direct port of kirchenbauer_zscore but operates on already-tokenised slice."""
    import torch

    if len(token_ids) < 2:
        return 0.0
    V = len(tokenizer)
    G = int(gamma * V)
    n_g = 0
    T = len(token_ids) - 1
    rng = torch.Generator()
    for i in range(1, len(token_ids)):
        seed = (hash_key * token_ids[i - 1]) % (2 ** 63)
        rng.manual_seed(seed)
        green = torch.randperm(V, generator=rng)[:G]
        if token_ids[i] in green.tolist():
            n_g += 1
    return (n_g - gamma * T) / (gamma * (1.0 - gamma) * T) ** 0.5


def sliding_max_z(text: str, tokenizer, gamma: float, hash_key: int,
                  window: int = WINDOW, stride: int = STRIDE) -> float:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= window:
        return _z_for_window(ids, tokenizer, gamma, hash_key)
    best = -math.inf
    for start in range(0, len(ids) - window + 1, stride):
        chunk = ids[start : start + window]
        z = _z_for_window(chunk, tokenizer, gamma, hash_key)
        if z > best:
            best = z
    return best


def confidence(z: float, scale: float = 2.0) -> float:
    """Map z to (0,1) via logistic centred at THRESHOLD."""
    return 1.0 / (1.0 + math.exp(-(z - THRESHOLD) / scale))


def main() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer()

    rows = [json.loads(line) for line in TEXTS_PATH.open()]
    print(f"Scoring {len(rows)} texts...")

    out: list[dict] = []
    for r in tqdm(rows, unit="text"):
        z_full = kirchenbauer_zscore(r["text"], tokenizer, gamma=GAMMA, hash_key=HASH_KEY)
        z_win  = sliding_max_z(r["text"], tokenizer, gamma=GAMMA, hash_key=HASH_KEY)
        z_max  = max(z_full, z_win)
        out.append({
            "id":          r["id"],
            "watermarked": bool(z_max > THRESHOLD),
            "z_score":     float(z_max),
            "confidence":  float(confidence(z_max)),
        })

    with OUT_PATH.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote {OUT_PATH}  ({len(out)} predictions)")


if __name__ == "__main__":
    main()
