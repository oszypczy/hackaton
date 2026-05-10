#!/usr/bin/env python3
"""Task 3 — v9: brute-force subset search.

Hypothesis: cross_lm_best (6 features, leaderboard 0.284) found a lucky combo.
Search exhaustively over subsets of 3-6 cross-LM derived features for highest OOF.
Smaller models = less overfitting = better leaderboard generalization.

Output:
- v9_top1: best subset OOF
- v9_top3_avg: avg of top-3 subsets
- v9_diversity: subset most diverse from current best leaderboard model
"""
from __future__ import annotations

import argparse
import itertools
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent
SUBMISSIONS_DIR = ROOT / "submissions"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out-prefix", type=str, default="submission_v9")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def _read_jsonl(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_splits(data_dir):
    if data_dir and data_dir.exists():
        train_clean = _read_jsonl(data_dir / "train_clean.jsonl"); train_clean["label"] = 0
        train_wm = _read_jsonl(data_dir / "train_wm.jsonl"); train_wm["label"] = 1
        valid_clean = _read_jsonl(data_dir / "valid_clean.jsonl"); valid_clean["label"] = 0
        valid_wm = _read_jsonl(data_dir / "valid_wm.jsonl"); valid_wm["label"] = 1
        train = pd.concat([train_clean, train_wm], ignore_index=True)
        val = pd.concat([valid_clean, valid_wm], ignore_index=True)
        test = _read_jsonl(data_dir / "test.jsonl")
    else:
        from datasets import load_dataset as hf_load
        ds = hf_load("SprintML/llm-watermark-detection")
        train_clean = ds["train_clean"].to_pandas(); train_clean["label"] = 0
        train_wm = ds["train_wm"].to_pandas(); train_wm["label"] = 1
        valid_clean = ds["valid_clean"].to_pandas(); valid_clean["label"] = 0
        valid_wm = ds["valid_wm"].to_pandas(); valid_wm["label"] = 1
        train = pd.concat([train_clean, train_wm], ignore_index=True)
        val = pd.concat([valid_clean, valid_wm], ignore_index=True)
        test = ds["test"].to_pandas()
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)
    return train, val, test


def _load_pkl(cache_dir, name):
    p = cache_dir / f"features_{name}.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f).reset_index(drop=True)


def _make_logreg(C=0.5):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs"))])


def _oof_logreg(X, y, C, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = _make_logreg(C); m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    final = _make_logreg(C); final.fit(X, y)
    return oof, final


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir else SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    all_lab = pd.concat([train_df, val_df], ignore_index=True)
    n_lab = len(all_lab)
    y = all_lab["label"].astype(int).values

    cache = args.cache_dir
    feat_a = _load_pkl(cache, "a")
    feat_bino = _load_pkl(cache, "bino")
    feat_bino_s = _load_pkl(cache, "bino_strong")
    feat_bino_xl = _load_pkl(cache, "bino_xl")
    feat_olmo7b = _load_pkl(cache, "olmo_7b")
    feat_olmo13b = _load_pkl(cache, "olmo_13b")
    feat_olmo1b = _load_pkl(cache, "multi_lm")

    parts = [f for f in [feat_a, feat_bino, feat_bino_s, feat_bino_xl,
                          feat_olmo7b, feat_olmo13b, feat_olmo1b] if f is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)

    # All possible cross-LM derived features
    derived = {}
    pairs_lp = [
        ("olmo7b_lp_mean", "lp_mean"), ("olmo7b_lp_mean", "lp_per"),
        ("olmo7b_lp_mean", "lp_obs"), ("olmo7b_lp_mean", "bino_strong_lp_obs"),
        ("olmo7b_lp_mean", "bino_strong_lp_per"), ("olmo7b_lp_mean", "bino_xl_lp_obs"),
        ("olmo7b_lp_mean", "bino_xl_lp_per"), ("olmo7b_lp_mean", "olmo_lp_mean"),
        ("olmo13b_lp_mean", "olmo7b_lp_mean"), ("olmo13b_lp_mean", "lp_mean"),
        ("olmo13b_lp_mean", "lp_per"),
        ("bino_strong_lp_obs", "bino_xl_lp_obs"),
        ("bino_strong_lp_per", "bino_xl_lp_per"),
        ("lp_mean", "lp_per"),
    ]
    for a, b in pairs_lp:
        if a in full.columns and b in full.columns:
            derived[f"d_{a}_minus_{b}"] = full[a] - full[b]

    ppl_pairs = [
        ("olmo7b_ppl", "ppl_observer"), ("olmo7b_ppl", "ppl_performer"),
        ("olmo7b_ppl", "bino_strong_ppl_obs"), ("olmo7b_ppl", "bino_xl_ppl_per"),
        ("olmo7b_ppl", "olmo_ppl"), ("olmo13b_ppl", "olmo7b_ppl"),
    ]
    for a, b in ppl_pairs:
        if a in full.columns and b in full.columns:
            derived[f"r_{a}_over_{b}"] = full[a] / (full[b] + 1e-9)

    if "olmo7b_lp_mean" in full.columns:
        derived["olmo7b_lp_sq"] = full["olmo7b_lp_mean"] ** 2

    full_d = pd.DataFrame(derived).fillna(0.0).reset_index(drop=True)
    print(f"Derived feature pool: {full_d.shape[1]}")
    feat_names = list(full_d.columns)
    X_full = full_d.values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]

    # Brute-force subsets of size 3, 4, 5, 6, 8
    print("\nBrute-force subset search (label-free cross-LM only)")
    n_feat = X_lab.shape[1]
    results = []  # (oof_tpr, indices, C)

    for K in [3, 4, 5, 6, 8]:
        if K > n_feat: continue
        n_combos = sum(1 for _ in itertools.combinations(range(n_feat), K))
        # Cap at 5000 combos per K to keep runtime bounded
        cap = 5000
        if n_combos > cap:
            print(f"K={K}: {n_combos} combos > {cap}, sampling {cap}")
            rng = np.random.default_rng(42)
            all_combos = list(itertools.combinations(range(n_feat), K))
            combos = [all_combos[i] for i in rng.choice(len(all_combos), cap, replace=False)]
        else:
            combos = list(itertools.combinations(range(n_feat), K))

        print(f"K={K}: testing {len(combos)} combos...")
        for combo in combos:
            cols = list(combo)
            for C in [0.5]:  # one C, keep search small
                X_sub = X_lab[:, cols]
                oof, _ = _oof_logreg(X_sub, y, C, args.n_splits, 42)
                t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
                results.append((t, K, combo, C))

    results.sort(key=lambda x: -x[0])
    print(f"\n=== Top 10 subsets:")
    for t, K, combo, C in results[:10]:
        names = [feat_names[i] for i in combo]
        print(f"  K={K} C={C} OOF={t:.4f}: {names}")

    # ── Save best subset, top-3 rank-avg
    top3 = results[:3]
    test_preds = []
    oof_preds = []
    for t, K, combo, C in top3:
        cols = list(combo)
        oof, model = _oof_logreg(X_lab[:, cols], y, C, args.n_splits, 42)
        oof_preds.append(oof)
        test_preds.append(model.predict_proba(X_test[:, cols])[:, 1])

    ids = test_df["id"].tolist()

    # Best single
    out = out_dir / f"{args.out_prefix}_top1.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_preds[0], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (top1 OOF={top3[0][0]:.4f})")

    # Top-3 rank-avg
    test_ranks = np.column_stack([rankdata(p, method="average") / len(p) for p in test_preds])
    rank_avg = test_ranks.mean(axis=1)
    rank_avg = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_top3_rank.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_avg, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (top3 rank-avg)")

    # Top-10 weighted by OOF
    top10 = results[:10]
    test10_preds = []
    for t, K, combo, C in top10:
        cols = list(combo)
        _, model = _oof_logreg(X_lab[:, cols], y, C, args.n_splits, 42)
        test10_preds.append(model.predict_proba(X_test[:, cols])[:, 1])
    weights = np.array([t[0] for t in top10])
    weights = weights ** 2; weights = weights / weights.sum()
    test_ranks_10 = np.column_stack([rankdata(p, method="average") / len(p) for p in test10_preds])
    rank_w = (test_ranks_10 * weights).sum(axis=1)
    rank_w = (rank_w - rank_w.min()) / (rank_w.max() - rank_w.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_top10_weighted.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_w, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (top10 OOF-weighted rank)")

    # Per-K best
    for K in [3, 4, 5, 6]:
        K_results = [r for r in results if r[1] == K]
        if not K_results: continue
        best = K_results[0]
        cols = list(best[2])
        _, model = _oof_logreg(X_lab[:, cols], y, best[3], args.n_splits, 42)
        pred = model.predict_proba(X_test[:, cols])[:, 1]
        out = out_dir / f"{args.out_prefix}_K{K}_best.csv"
        pd.DataFrame({"id": ids, "score": np.clip(pred, 0.001, 0.999)}).to_csv(out, index=False)
        print(f"Saved: {out} (K={K} best OOF={best[0]:.4f})")


if __name__ == "__main__":
    main()
