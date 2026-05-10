#!/usr/bin/env python3
"""Standalone: extract a feature module, cache it, then train clm baseline +
the new feature, output submission CSV.

Usage:
  python extract_and_train.py --feature olmo7b_entropy --out submission_clm_o7be.csv
  python extract_and_train.py --feature mistral_7b --out submission_clm_mistral7b.csv
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True, help="comma-separated module names in features/")
    p.add_argument("--cache-name", default=None, help="cache pkl basename (default=feature; comma-separated for multi-feature)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-rows", type=int, default=2250)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--no-cache", action="store_true", help="skip writing feature cache (e.g. on quota)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_jsonl(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    feature_names = [f.strip() for f in args.feature.split(",") if f.strip()]
    cache_names = [c.strip() for c in (args.cache_name or args.feature).split(",")]
    if len(cache_names) != len(feature_names):
        cache_names = feature_names

    # Load splits
    print("Loading data...")
    train_clean = _read_jsonl(args.data_dir / "train_clean.jsonl"); train_clean["label"] = 0
    train_wm = _read_jsonl(args.data_dir / "train_wm.jsonl"); train_wm["label"] = 1
    valid_clean = _read_jsonl(args.data_dir / "valid_clean.jsonl"); valid_clean["label"] = 0
    valid_wm = _read_jsonl(args.data_dir / "valid_wm.jsonl"); valid_wm["label"] = 1
    train = pd.concat([train_clean, train_wm], ignore_index=True)
    val = pd.concat([valid_clean, valid_wm], ignore_index=True)
    test = _read_jsonl(args.data_dir / "test.jsonl")
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)

    all_lab = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    n_lab = len(all_lab)
    y = all_lab["label"].astype(int).values
    all_texts = all_lab["text"].tolist() + test["text"].tolist()
    print(f"  labeled={n_lab} test={len(test)}")

    new_dfs: list[pd.DataFrame] = []
    for feat_name, cache_name in zip(feature_names, cache_names):
        cache_path = args.cache_dir / f"features_{cache_name}.pkl"
        if cache_path.exists():
            print(f"Loading cached {cache_path}")
            df = pickle.load(open(cache_path, "rb")).reset_index(drop=True)
        else:
            print(f"Extracting feature: {feat_name} -> {cache_path}")
            mod = importlib.import_module(f"code.attacks.task3.features.{feat_name}")
            rows = []
            for txt in tqdm(all_texts, desc=feat_name):
                try:
                    rows.append(mod.extract(txt))
                except Exception as e:
                    print(f"  err on text: {e}")
                    rows.append({})
            df = pd.DataFrame(rows).fillna(0.0).reset_index(drop=True)
            if not args.no_cache:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump(df, f)
                    print(f"  saved {cache_path}")
                except OSError as e:
                    print(f"  WARN: cache save failed ({e}); continuing without cache")
            else:
                print(f"  cache write skipped (--no-cache)")
        new_dfs.append(df)
        print(f"  -> {feat_name} shape={df.shape}, cols={list(df.columns)[:5]}")

    new_df = pd.concat(new_dfs, axis=1).reset_index(drop=True)
    print(f"\nCombined new features shape: {new_df.shape}")

    # Load existing clm-baseline features
    def _load(name):
        p = args.cache_dir / f"features_{name}.pkl"
        if not p.exists():
            print(f"  missing {p}")
            return None
        return pickle.load(open(p, "rb")).reset_index(drop=True)

    feat_a = _load("a")
    feat_bino = _load("bino")
    feat_bino_s = _load("bino_strong")
    feat_bino_xl = _load("bino_xl")
    feat_olmo7b = _load("olmo_7b")
    feat_olmo1b = _load("multi_lm")
    feat_fdgpt = _load("fdgpt")
    feat_d = _load("d")
    feat_bc = _load("bc")

    parts = [f for f in [feat_a, feat_bc, feat_d, feat_bino, feat_bino_s, feat_bino_xl,
                          feat_olmo7b, feat_olmo1b, feat_fdgpt, new_df] if f is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)
    print(f"Total cols (pre-derived): {full.shape[1]}")

    # cross-LM v1 derivations (the proven 6 features for cross_lm_best)
    derived = {}
    if "olmo7b_lp_mean" in full and "lp_per" in full:
        derived["cross_olmo7b_vs_gpt2med_lp"] = full["olmo7b_lp_mean"] - full["lp_per"]
    if "olmo7b_lp_mean" in full and "bino_xl_lp_obs" in full:
        derived["cross_olmo7b_vs_pythia28b_lp"] = full["olmo7b_lp_mean"] - full["bino_xl_lp_obs"]
    if "olmo7b_lp_mean" in full and "bino_xl_lp_per" in full:
        derived["cross_olmo7b_vs_pythia69b_lp"] = full["olmo7b_lp_mean"] - full["bino_xl_lp_per"]
    if "olmo7b_lp_mean" in full and "olmo_lp_mean" in full:
        derived["cross_olmo7b_vs_olmo1b_lp"] = full["olmo7b_lp_mean"] - full["olmo_lp_mean"]
    if "bino_strong_lp_obs" in full and "bino_xl_lp_obs" in full:
        derived["cross_pythia14b_vs_28b_lp"] = full["bino_strong_lp_obs"] - full["bino_xl_lp_obs"]
    if "olmo7b_ppl" in full and "bino_xl_ppl_obs" in full:
        derived["cross_olmo7b_vs_pythia_ppl_ratio"] = full["olmo7b_ppl"] / (full["bino_xl_ppl_obs"] + 1e-9)
    # Cross-LM with the NEW feature (if it has lp_mean / ppl)
    if "mistral7b_lp_mean" in full.columns and "olmo7b_lp_mean" in full.columns:
        derived["cross_mistral_vs_olmo7b_lp"] = full["mistral7b_lp_mean"] - full["olmo7b_lp_mean"]
    if "mistral7b_lp_mean" in full.columns and "lp_per" in full.columns:
        derived["cross_mistral_vs_gpt2med_lp"] = full["mistral7b_lp_mean"] - full["lp_per"]
    if "mistral7b_ppl" in full.columns and "olmo7b_ppl" in full.columns:
        derived["cross_mistral_vs_olmo7b_ppl"] = full["mistral7b_ppl"] / (full["olmo7b_ppl"] + 1e-9)

    full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
    print(f"After derivations: {full.shape[1]}  (added {len(derived)})")

    # Drop NaN/inf
    full = full.replace([np.inf, -np.inf], 0).fillna(0.0)
    X = full.values.astype(np.float32)
    X_lab = X[:n_lab]
    X_test = X[n_lab:]

    # OOF
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(n_lab)
    for tr, va in skf.split(X_lab, y):
        m = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs"))])
        m.fit(X_lab[tr], y[tr])
        oof[va] = m.predict_proba(X_lab[va])[:, 1]
    tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    print(f"\nOOF TPR@1%FPR: {tpr:.4f}")

    # Final model
    final = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs"))])
    final.fit(X_lab, y)
    test_pred = final.predict_proba(X_test)[:, 1]

    # Save submission
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"].tolist(), "score": np.clip(test_pred, 0.001, 0.999)}).to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
