#!/usr/bin/env python3
"""Diagnostic: grid search for Unigram (Zhao ICLR 2024) watermark key/params.

Runs on CPU only. Loads train data, tries many (key, fraction, hash_fn) combos,
reports which gives best separation between watermarked and clean.

Usage:
    python code/attacks/task3/grid_search.py --data-dir /path/to/dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Activate venv first.")
    sys.exit(1)


def _load_texts(src: str | Path, split: str) -> list[str]:
    src = Path(src)
    if src.is_dir():
        p = src / f"{split}.jsonl"
        return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]
    else:  # zip
        with zipfile.ZipFile(src) as z:
            with z.open(f"{split}.jsonl") as f:
                return [json.loads(l)["text"] for l in f.read().decode().splitlines() if l.strip()]


# ─── Green list generation ───────────────────────────────────────────────────

def _make_mask(seed: int, fraction: float, vocab_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_green = int(fraction * vocab_size)
    mask = np.array([True] * n_green + [False] * (vocab_size - n_green), dtype=bool)
    rng.shuffle(mask)
    return mask


def _seeds_for_key(key: int, vocab_size: int) -> dict[str, int]:
    """Multiple seed implementations to try."""
    seeds = {
        "direct": int(key),
        "mod_vocab": int(key % vocab_size),
    }
    # SHA256 variant
    try:
        b = np.int64(key).tobytes()
        seeds["sha256_int64"] = int.from_bytes(hashlib.sha256(b).digest()[:4], "little")
    except Exception:
        pass
    # SHA256 of key as string
    try:
        b2 = str(key).encode()
        seeds["sha256_str"] = int.from_bytes(hashlib.sha256(b2).digest()[:4], "little")
    except Exception:
        pass
    return seeds


def _zscore(token_ids: list[int], mask: np.ndarray, fraction: float) -> float:
    valid = [t for t in token_ids if 0 <= t < len(mask)]
    n = len(valid)
    if n < 5:
        return 0.0
    green = sum(1 for t in valid if mask[t])
    return (green - fraction * n) / np.sqrt(fraction * (1 - fraction) * n)


def _unizscore(token_ids: list[int], mask: np.ndarray, fraction: float) -> float:
    """Z-score on unique tokens (Zhao 'unidetect')."""
    valid = list({t for t in token_ids if 0 <= t < len(mask)})
    n = len(valid)
    if n < 5:
        return 0.0
    green = sum(1 for t in valid if mask[t])
    return (green - fraction * n) / np.sqrt(fraction * (1 - fraction) * n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Path to dataset dir (with .jsonl files) or Dataset.zip")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    # ── Resolve data path
    if args.data_dir:
        data_path = Path(args.data_dir)
    else:
        # Cluster default
        data_path = Path("/p/scratch/training2615/kempinski1/Czumpers/llm-watermark-detection/Dataset.zip")

    print(f"Dataset: {data_path}")
    print(f"Loading tokenizer: {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tok.vocab_size
    print(f"  vocab_size={vocab_size}")

    print("Loading train data...")
    if data_path.is_dir():
        wm_texts = _load_texts(data_path, "train_wm")
        clean_texts = _load_texts(data_path, "train_clean")
    else:
        wm_texts = _load_texts(data_path, "train_wm")
        clean_texts = _load_texts(data_path, "train_clean")
    print(f"  watermarked={len(wm_texts)}, clean={len(clean_texts)}")

    print("Tokenizing...")
    wm_ids = [tok.encode(t, add_special_tokens=False) for t in wm_texts]
    clean_ids = [tok.encode(t, add_special_tokens=False) for t in clean_texts]
    print(f"  avg len wm={np.mean([len(x) for x in wm_ids]):.0f} clean={np.mean([len(x) for x in clean_ids]):.0f}")

    # ── Grid search
    keys = [0, 1, 2, 3, 42, 100, 1234, 9999, 99999, 15485863, 33554393, 4294967291]
    fractions = [0.5, 0.25]
    results = []

    print(f"\nRunning grid: {len(keys)} keys × {len(fractions)} fractions × 4 hash_fns...")
    for key in keys:
        seed_map = _seeds_for_key(key, vocab_size)
        for seed_name, seed_val in seed_map.items():
            for frac in fractions:
                mask = _make_mask(seed_val, frac, vocab_size)
                wm_z = np.array([_zscore(ids, mask, frac) for ids in wm_ids])
                cl_z = np.array([_zscore(ids, mask, frac) for ids in clean_ids])
                wm_uz = np.array([_unizscore(ids, mask, frac) for ids in wm_ids])
                cl_uz = np.array([_unizscore(ids, mask, frac) for ids in clean_ids])
                results.append({
                    "key": key, "seed_fn": seed_name, "seed": seed_val, "frac": frac,
                    "sep_z": float(np.mean(wm_z) - np.mean(cl_z)),
                    "mean_wm_z": float(np.mean(wm_z)),
                    "mean_cl_z": float(np.mean(cl_z)),
                    "pct_pos_wm": float(np.mean(wm_z > 2)),
                    "pct_pos_cl": float(np.mean(cl_z > 2)),
                    "sep_uz": float(np.mean(wm_uz) - np.mean(cl_uz)),
                    "mean_wm_uz": float(np.mean(wm_uz)),
                })

    results.sort(key=lambda r: r["sep_z"], reverse=True)

    print("\n=== TOP 20 (by z-score separation) ===")
    hdr = f"{'key':<12} {'seed_fn':<14} {'frac':<5} {'sep_z':>8} {'mean_wm':>9} {'mean_cl':>9} {'pct+wm':>7} {'pct+cl':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results[:20]:
        print(f"{r['key']:<12} {r['seed_fn']:<14} {r['frac']:<5.2f} "
              f"{r['sep_z']:>8.3f} {r['mean_wm_z']:>9.3f} {r['mean_cl_z']:>9.3f} "
              f"{r['pct_pos_wm']:>7.1%} {r['pct_pos_cl']:>7.1%}")

    best = results[0]
    print(f"\n★ BEST: key={best['key']}, seed_fn={best['seed_fn']}, frac={best['frac']:.2f}")
    print(f"  Separation: {best['sep_z']:.4f}")
    print(f"  Mean WM z={best['mean_wm_z']:.3f}  Mean clean z={best['mean_cl_z']:.3f}")
    print(f"  % WM z>2: {best['pct_pos_wm']:.1%}  % clean z>2: {best['pct_pos_cl']:.1%}")

    # Show if any result has significant signal
    good = [r for r in results if r["sep_z"] > 0.5]
    if good:
        print(f"\n✓ {len(good)} configs with separation > 0.5")
    else:
        print("\n✗ No config achieved separation > 0.5 — Unigram may use different tokenizer or params")

    # ── Also report raw green token fraction check
    print("\n=== SANITY CHECK: raw green token fraction ===")
    print("(watermarked should be ~> 0.5 if correct params)")
    best_mask = _make_mask(best["seed"], best["frac"], vocab_size)
    for label, ids_list in [("WM", wm_ids[:10]), ("Clean", clean_ids[:10])]:
        fracs = []
        for ids in ids_list:
            valid = [t for t in ids if 0 <= t < vocab_size]
            if valid:
                fracs.append(sum(1 for t in valid if best_mask[t]) / len(valid))
        print(f"  {label}: mean green frac = {np.mean(fracs):.4f} (expected {'>' if label=='WM' else '≈'}{best['frac']:.2f})")


if __name__ == "__main__":
    main()
