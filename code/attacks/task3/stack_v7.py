#!/usr/bin/env python3
"""Task 3 — v7: multi-seed SelectKBest ensemble.

SelectKBest(mutual_info_classif) has stochastic behavior (KNN density estimation).
Running with N seeds and averaging predictions reduces variance, expected to
push OOF higher than single-shot v3 (0.3778) which was lucky/random.

Also varies K (30, 40, 50) and combines.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
    p.add_argument("--out-prefix", type=str, default="submission_v7")
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


def _make_logreg(C=0.05):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs")),
    ])


def _oof_logreg_view(X, y, C, n_splits, seed):
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
    feat_judge7b = _load_pkl(cache, "judge_olmo7b")
    feat_better_liu = _load_pkl(cache, "better_liu")
    feat_fdgpt = _load_pkl(cache, "fdgpt")
    feat_d = _load_pkl(cache, "d")

    parts = [f for f in [feat_a, feat_bino, feat_bino_s, feat_bino_xl,
                          feat_olmo7b, feat_olmo13b, feat_olmo1b, feat_judge7b,
                          feat_better_liu, feat_fdgpt, feat_d] if f is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)

    derived = {}
    pairs_lp = [
        ("olmo7b_lp_mean", "lp_mean"), ("olmo7b_lp_mean", "lp_per"),
        ("olmo7b_lp_mean", "lp_obs"), ("olmo7b_lp_mean", "bino_strong_lp_obs"),
        ("olmo7b_lp_mean", "bino_strong_lp_per"), ("olmo7b_lp_mean", "bino_xl_lp_obs"),
        ("olmo7b_lp_mean", "bino_xl_lp_per"), ("olmo7b_lp_mean", "olmo_lp_mean"),
        ("olmo13b_lp_mean", "olmo7b_lp_mean"), ("olmo13b_lp_mean", "lp_mean"),
        ("bino_strong_lp_obs", "bino_xl_lp_obs"),
        ("bino_strong_lp_per", "bino_xl_lp_per"),
        ("lp_mean", "lp_per"),
    ]
    for a, b in pairs_lp:
        if a in full.columns and b in full.columns:
            derived[f"d_{a}_minus_{b}"] = full[a] - full[b]
    if "olmo7b_lp_mean" in full.columns:
        derived["olmo7b_lp_sq"] = full["olmo7b_lp_mean"] ** 2
    if all(c in full.columns for c in ("olmo7b_lp_mean", "lp_mean", "bino_strong_lp_obs")):
        derived["d_olmo7b_vs_meanWeak"] = full["olmo7b_lp_mean"] - 0.5 * (full["lp_mean"] + full["bino_strong_lp_obs"])
    if "olmo7b_ppl" in full.columns and "bino_xl_ppl_per" in full.columns:
        derived["log_ppl_olmo7b_over_pythia69b"] = (
            np.log(full["olmo7b_ppl"] + 1.001) - np.log(full["bino_xl_ppl_per"] + 1.001)
        )

    full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
    print(f"Full features: {full.shape[1]}")

    X_full = full.fillna(0.0).values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]

    # ── Multi-seed SelectKBest ensemble
    print("\n=== Multi-seed SelectKBest ensemble")
    Ks = [30, 40, 50]
    Cs = [0.05]
    seeds = [0, 7, 42, 123, 999, 31337, 2024, 17, 31, 91]

    oof_views = []
    test_views = []
    view_labels = []

    for K in Ks:
        for sel_seed in seeds:
            sel = SelectKBest(mutual_info_classif, k=K)
            # Fit selector with custom random_state for mutual_info_classif
            mi = mutual_info_classif(X_lab, y, random_state=sel_seed)
            top_idx = np.argsort(-mi)[:K]
            kept_mask = np.zeros(X_lab.shape[1], dtype=bool)
            kept_mask[top_idx] = True
            Xk_lab = X_lab[:, kept_mask]
            Xk_test = X_test[:, kept_mask]
            for C in Cs:
                oof, model = _oof_logreg_view(Xk_lab, y, C, args.n_splits, seed=42)
                t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
                pred = model.predict_proba(Xk_test)[:, 1]
                oof_views.append(oof)
                test_views.append(pred)
                view_labels.append(f"K{K}_seed{sel_seed}_C{C}")

    base_tprs = np.array([tpr_at_fpr(o.tolist(), y.tolist(), 0.01) for o in oof_views])
    print(f"Per-config OOF TPR (n={len(oof_views)}):")
    print(f"  min={base_tprs.min():.4f}  max={base_tprs.max():.4f}  mean={base_tprs.mean():.4f}  std={base_tprs.std():.4f}")
    sorted_idx = np.argsort(-base_tprs)
    print("  Top 5:")
    for i in sorted_idx[:5]:
        print(f"    {view_labels[i]}: {base_tprs[i]:.4f}")

    OOF = np.column_stack(oof_views)
    TEST = np.column_stack(test_views)

    # Mean prediction (just average)
    mean_oof = OOF.mean(axis=1)
    mean_test = TEST.mean(axis=1)
    t_mean = tpr_at_fpr(mean_oof.tolist(), y.tolist(), 0.01)
    print(f"\nMean ensemble: OOF TPR={t_mean:.4f}")

    # Rank average
    rank_oof = np.column_stack([rankdata(o, method="average") / len(o) for o in OOF.T]).mean(axis=1)
    rank_test = np.column_stack([rankdata(t, method="average") / len(t) for t in TEST.T]).mean(axis=1)
    t_rank = tpr_at_fpr(rank_oof.tolist(), y.tolist(), 0.01)
    print(f"Rank-avg ensemble: OOF TPR={t_rank:.4f}")

    # Top-K weighted by OOF TPR
    for top_K in [3, 5, 10]:
        top_idx = sorted_idx[:top_K]
        w = base_tprs[top_idx]; w = w / w.sum()
        tk_oof = (OOF[:, top_idx] * w).sum(axis=1)
        tk_test = (TEST[:, top_idx] * w).sum(axis=1)
        t_tk = tpr_at_fpr(tk_oof.tolist(), y.tolist(), 0.01)
        print(f"Top-{top_K} weighted: OOF TPR={t_tk:.4f}")

    # Meta-stacking
    best_meta_C = None; best_meta_tpr = -1
    for meta_C in [0.005, 0.01, 0.05, 0.1, 0.5]:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42 + 7)
        moof = np.zeros(len(y))
        for tr, va in skf.split(OOF, y):
            m = _make_logreg(meta_C); m.fit(OOF[tr], y[tr])
            moof[va] = m.predict_proba(OOF[va])[:, 1]
        t = tpr_at_fpr(moof.tolist(), y.tolist(), 0.01)
        if t > best_meta_tpr:
            best_meta_tpr = t; best_meta_C = meta_C
    meta = _make_logreg(best_meta_C); meta.fit(OOF, y)
    test_meta = meta.predict_proba(TEST)[:, 1]
    print(f"Meta C={best_meta_C}: OOF TPR={best_meta_tpr:.4f}")

    # ── Save outputs
    ids = test_df["id"].tolist()

    # Best single
    out = out_dir / f"{args.out_prefix}_best_single.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_views[sorted_idx[0]], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  ({view_labels[sorted_idx[0]]})")

    # Mean
    out = out_dir / f"{args.out_prefix}_mean.csv"
    mean_test_norm = (mean_test - mean_test.min()) / (mean_test.max() - mean_test.min() + 1e-9)
    pd.DataFrame({"id": ids, "score": np.clip(mean_test_norm, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (mean ensemble)")

    # Rank avg
    rank_test_norm = (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_rank.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_test_norm, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (rank-avg)")

    # Top-5 weighted
    top_5 = sorted_idx[:5]
    w = base_tprs[top_5]; w = w / w.sum()
    t5_test = (TEST[:, top_5] * w).sum(axis=1)
    out = out_dir / f"{args.out_prefix}_top5_weighted.csv"
    pd.DataFrame({"id": ids, "score": np.clip(t5_test, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (top-5 OOF-weighted)")

    # Meta
    out = out_dir / f"{args.out_prefix}_meta.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_meta, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (meta C={best_meta_C} OOF={best_meta_tpr:.4f})")


if __name__ == "__main__":
    main()
