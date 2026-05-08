#!/usr/bin/env python3
"""
Challenge B scorer — LLM Watermark Detection & Removal.

Sub-task B1 (detection):
  Ground truth (--gt):   JSONL {"id": int, "watermarked": bool}
  Submission:            JSONL {"id": int, "watermarked": bool, "z_score": float, "confidence": float}
  Metric: F1 (positive = watermarked) + bonus AUC

Sub-task B2 (removal):
  Fixture (--fixture):   JSONL {"id": int, "text": str, "watermark": {gamma, hash_key, ...} | null}
  Submission:            JSONL {"id": int, "modified_text": str}
  Metric: mean( evasion_flag * BERTScore_F1 ) where evasion = z_score(modified) < 4

Usage:
  python score_B.py b1 --submission submissions/B1.jsonl --gt data/B/ground_truth.jsonl
  python score_B.py b2 --submission submissions/B2.jsonl --fixture data/B/removal_targets.jsonl
  python score_B.py b2 ... --no-bertscore   # fast check: evasion rate only

Dependencies:
  pip install numpy scikit-learn transformers torch bert-score
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_file(path: str, label: str) -> bool:
    if not Path(path).exists():
        print(f"  [ERROR] {label} not found: {path}", file=sys.stderr)
        return False
    return True


# ── Kirchenbauer z-score ──────────────────────────────────────────────────────

def kirchenbauer_zscore(
    text: str,
    tokenizer,
    gamma: float = 0.25,
    hash_key: int = 15485863,
) -> float:
    """
    Replicates Kirchenbauer WatermarkDetector z-score.

    Green-list construction per position i:
      seed = hash_key * prev_token (mod 2^63)
      green_list = randperm(vocab_size, seed)[:green_size]
    Then z = (n_green − γT) / sqrt(γ(1−γ)T)
    """
    import torch

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        return 0.0

    # Use len(tokenizer) — matches model.config.vocab_size (incl. special tokens).
    # Llama-3: tokenizer.vocab_size=128000 but model uses 128256 → silent mismatch.
    vocab_size = len(tokenizer)
    green_size = int(gamma * vocab_size)
    n_green    = 0
    T          = len(token_ids) - 1  # skip first (no prev token)

    rng = torch.Generator()
    for i in range(1, len(token_ids)):
        prev = token_ids[i - 1]
        seed = (hash_key * prev) % (2**63)
        rng.manual_seed(seed)
        green_ids = torch.randperm(vocab_size, generator=rng)[:green_size]
        if token_ids[i] in green_ids.tolist():
            n_green += 1

    if T == 0:
        return 0.0
    return (n_green - gamma * T) / (gamma * (1.0 - gamma) * T) ** 0.5


def _load_tokenizer(preferred: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Try gated → open mirrors → gpt2. First success wins."""
    from transformers import AutoTokenizer
    candidates = [
        preferred,
        "NousResearch/Meta-Llama-3-8B-Instruct",  # open mirror, vocab 128256
        "unsloth/llama-3-8b-Instruct",            # open mirror
        "gpt2",                                    # last-resort, z-scores will be wrong
    ]
    for c in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(c)
            if c == "gpt2":
                print(f"  [WARN] only gpt2 available — z-scores will not match Llama generation")
            else:
                print(f"  Tokenizer: {c} (vocab={len(tok)})")
            return tok
        except Exception:
            continue
    raise RuntimeError("No tokenizer could be loaded.")


# ── B1: detection ─────────────────────────────────────────────────────────────

def run_b1(args: argparse.Namespace) -> None:
    from sklearn.metrics import f1_score, roc_auc_score, classification_report

    if not check_file(args.submission, "submission") or not check_file(args.gt, "ground truth"):
        sys.exit(1)

    sub = {r["id"]: r for r in load_jsonl(args.submission)}
    gt  = {r["id"]: r for r in load_jsonl(args.gt)}

    common = sorted(set(sub) & set(gt))
    if not common:
        print("[ERROR] No common IDs between submission and ground truth", file=sys.stderr)
        sys.exit(1)

    y_true = [int(gt[i]["watermarked"])  for i in common]
    y_pred = [int(sub[i]["watermarked"]) for i in common]
    confs  = [float(sub[i].get("confidence", 0.5)) for i in common]

    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, confs) if len(set(y_true)) > 1 else float("nan")

    tp = sum(a and b for a, b in zip(y_true, y_pred))
    fp = sum((not a) and b for a, b in zip(y_true, y_pred))
    fn = sum(a and (not b) for a, b in zip(y_true, y_pred))
    tn = sum((not a) and (not b) for a, b in zip(y_true, y_pred))

    print(f"\nB1 — Detection  ({len(common)} texts)")
    print(f"  F1 (watermarked) : {f1:.4f}   ← main metric")
    print(f"  AUC (confidence) : {auc:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n  Reference:")
    print(f"    Easy  raw z-score          : F1 ≈ 0.85")
    print(f"    Solid sliding + calibration: F1 ≈ 0.93")
    print(f"    Hard                       : F1  > 0.97")
    print(f"\n{'='*58}")
    print(f"  SCORE : {f1:.4f}")
    print(f"{'='*58}")


# ── B2: removal ───────────────────────────────────────────────────────────────

def run_b2(args: argparse.Namespace) -> None:
    if not check_file(args.submission, "submission") or not check_file(args.fixture, "fixture"):
        sys.exit(1)

    sub     = {r["id"]: r for r in load_jsonl(args.submission)}
    fixture = {r["id"]: r for r in load_jsonl(args.fixture)}

    common = sorted(set(sub) & set(fixture))
    if not common:
        print("[ERROR] No common IDs between submission and fixture", file=sys.stderr)
        sys.exit(1)

    originals = [fixture[i]["text"]                   for i in common]
    modified  = [sub[i]["modified_text"]               for i in common]
    wm_params = [fixture[i].get("watermark", {}) or {} for i in common]

    # Watermark params (fall back to defaults if fixture doesn't store them per-row)
    default_gamma    = 0.25
    default_hash_key = 15485863

    tokenizer = _load_tokenizer()

    print(f"\n  Computing z-scores for {len(common)} texts...")
    z_scores = []
    for i, (mod, params) in enumerate(zip(modified, wm_params)):
        gamma    = params.get("gamma",    default_gamma)
        hash_key = params.get("hash_key", default_hash_key)
        z = kirchenbauer_zscore(mod, tokenizer, gamma=gamma, hash_key=hash_key)
        z_scores.append(z)
        if i < 3 or i == len(common) - 1:
            print(f"    id={common[i]}: z={z:.2f}")

    evasion_flags = [int(z < 4.0) for z in z_scores]
    evasion_rate  = sum(evasion_flags) / len(evasion_flags)

    # BERTScore
    if args.no_bertscore:
        print("  BERTScore: skipped (--no-bertscore)")
        bert_f1_list = [1.0] * len(common)  # upper-bound assumption
        bertscore_note = " [BERTScore assumed=1.0]"
    else:
        try:
            from bert_score import score as bert_score_fn
            print(f"\n  Computing BERTScore F1 (batch of {len(common)})...")
            _, _, bert_f1_tensor = bert_score_fn(
                modified, originals, lang="en", verbose=False
            )
            bert_f1_list = bert_f1_tensor.numpy().tolist()
            bertscore_note = ""
        except ImportError:
            print("  [WARN] bert-score not installed — run `pip install bert-score`")
            print("         Evasion rate reported only.")
            bert_f1_list = [1.0] * len(common)
            bertscore_note = " [BERTScore missing — install bert-score]"

    item_scores = [e * b for e, b in zip(evasion_flags, bert_f1_list)]
    total_score = float(np.mean(item_scores))
    mean_bert   = float(np.mean(bert_f1_list))

    print(f"\nB2 — Removal  ({len(common)} texts){bertscore_note}")
    print(f"  Evasion rate (z < 4) : {sum(evasion_flags)}/{len(common)} = {evasion_rate:.2%}")
    print(f"  Mean BERTScore F1    : {mean_bert:.4f}")
    print(f"  Combined score       : {total_score:.4f}   ← main metric")
    print(f"\n  Reference:")
    print(f"    Easy  random word swap    : ≈ 0.30")
    print(f"    Solid targeted / paraph  : ≈ 0.55–0.70")
    print(f"    Hard  LLM paraphrase     :  > 0.80")
    print(f"\n{'='*58}")
    print(f"  SCORE : {total_score:.4f}")
    print(f"{'='*58}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap  = argparse.ArgumentParser(description="Score Challenge B — Watermark")
    sub = ap.add_subparsers(dest="task", required=True)

    p1 = sub.add_parser("b1", help="Score B1: detection")
    p1.add_argument("--submission", default="submissions/B1.jsonl")
    p1.add_argument("--gt",         default="data/B/ground_truth.jsonl")

    p2 = sub.add_parser("b2", help="Score B2: removal")
    p2.add_argument("--submission",    default="submissions/B2.jsonl")
    p2.add_argument("--fixture",       default="data/B/removal_targets.jsonl",
                    help="Original watermarked texts (fixture JSONL with id + text + watermark)")
    p2.add_argument("--no-bertscore",  action="store_true",
                    help="Skip BERTScore (faster iteration; evasion rate only)")

    args = ap.parse_args()

    print("=" * 58)
    print("Challenge B — LLM Watermark")
    print("=" * 58)

    if args.task == "b1":
        run_b1(args)
    else:
        run_b2(args)


if __name__ == "__main__":
    main()
