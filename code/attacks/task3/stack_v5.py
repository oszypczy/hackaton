#!/usr/bin/env python3
"""Task 3 — v5: full label-free feature aggregation.

Adds: judge_olmo7b, judge_phi2, judge_mistral, judge_chat, sir, unigram_direct,
kgw_llama, olmo7b_chunks. All hash-based / LM-inference-based, no label-fitting.

Excludes: branch_bc, bigram, emp_green (label-fitted greenlists).
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
    p.add_argument("--out-prefix", type=str, default="submission_v5")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
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
    print(f"Cache: {cache}\n")

    # Load ALL label-free features
    LABEL_FREE_NAMES = [
        "a", "bino", "bino_strong", "bino_xl",
        "olmo_7b", "olmo_13b", "multi_lm",
        "lm_judge", "judge_olmo7b", "judge_phi2", "judge_mistral", "judge_chat",
        "olmo7b_chunks", "sir", "unigram_direct", "kgw_llama",
        "better_liu", "fdgpt", "d", "stylometric",
    ]
    parts = []
    used = []
    for name in LABEL_FREE_NAMES:
        f = _load_pkl(cache, name)
        if f is not None:
            parts.append(f)
            used.append((name, f.shape[1]))
            print(f"  loaded {name}: {f.shape[1]} cols")
    full = pd.concat(parts, axis=1).fillna(0.0)
    print(f"\nRaw cols: {full.shape[1]}  (from {len(used)} feature pkls)")

    # Cross-LM derivations
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

    full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
    print(f"After derivations: {full.shape[1]} (added {len(derived)})")

    X_full = full.fillna(0.0).values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]
    feat_names = list(full.columns)

    # ── 1. Full LR
    print("\n=== 1. Full label-free LogReg")
    for C in [0.01, 0.02, 0.05, 0.1]:
        oof, _ = _oof_logreg(X_lab, y, C, args.n_splits, args.seed)
        t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        print(f"  C={C}: OOF TPR={t:.4f}")

    # Pick best C
    best_full_C = None; best_full_tpr = -1; best_full_oof = None; best_full_model = None
    for C in [0.01, 0.02, 0.05, 0.1]:
        oof, m = _oof_logreg(X_lab, y, C, args.n_splits, args.seed)
        t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        if t > best_full_tpr:
            best_full_tpr = t; best_full_C = C; best_full_oof = oof; best_full_model = m
    full_test_pred = best_full_model.predict_proba(X_test)[:, 1]
    print(f"  BEST FULL: C={best_full_C} OOF={best_full_tpr:.4f}")

    # ── 2. SelectKBest
    print("\n=== 2. SelectKBest variants")
    sel_results = []
    for K in [30, 50, 80, 120]:
        if K >= X_lab.shape[1]: continue
        sel = SelectKBest(mutual_info_classif, k=K)
        sel.fit(X_lab, y)
        kept = sel.get_support()
        Xk_lab = X_lab[:, kept]; Xk_test = X_test[:, kept]
        for C in [0.05, 0.1, 0.5]:
            oof, m = _oof_logreg(Xk_lab, y, C, args.n_splits, args.seed)
            t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  K={K} C={C}: OOF TPR={t:.4f}")
            sel_results.append((K, C, oof, m.predict_proba(Xk_test)[:, 1], t))

    # ── 3. Best subgroup views
    print("\n=== 3. Subgroup views")
    judges_cols = [c for c in feat_names if c.startswith(("judge_", "chat_", "ai_or_human"))]
    cross_cols = [c for c in feat_names if c.startswith("d_") or c.startswith("r_")]
    olmo_cols = [c for c in feat_names if c.startswith(("olmo_", "olmo7b_", "olmo13b_"))]
    bino_cols = [c for c in feat_names if c.startswith("bino")]
    judges_idx = [feat_names.index(c) for c in judges_cols]
    cross_idx = [feat_names.index(c) for c in cross_cols]
    olmo_idx = [feat_names.index(c) for c in olmo_cols]
    bino_idx = [feat_names.index(c) for c in bino_cols]
    print(f"  judges: {len(judges_cols)} cols")
    print(f"  cross-derived: {len(cross_cols)} cols")
    print(f"  olmo-related: {len(olmo_cols)} cols")
    print(f"  bino-related: {len(bino_cols)} cols")

    subgroup_results = []
    for name, idx in [("judges", judges_idx), ("cross+olmo", cross_idx + olmo_idx),
                       ("ppl_all", olmo_idx + bino_idx), ("cross_only", cross_idx)]:
        if not idx: continue
        Xs_lab = X_lab[:, idx]; Xs_test = X_test[:, idx]
        oof, m = _oof_logreg(Xs_lab, y, 0.05, args.n_splits, args.seed)
        t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        print(f"  {name} (n={len(idx)}): OOF TPR={t:.4f}")
        subgroup_results.append((name, oof, m.predict_proba(Xs_test)[:, 1], t))

    # ── 4. Build base stack & meta
    oof_views = [best_full_oof] + [r[2] for r in sel_results] + [r[1] for r in subgroup_results]
    test_views = [full_test_pred] + [r[3] for r in sel_results] + [r[2] for r in subgroup_results]
    view_labels = [f"full_C{best_full_C}"] + [f"sel_K{K}_C{C}" for (K, C, _, _, _) in sel_results] + \
                  [f"sg_{n}" for (n, _, _, _) in subgroup_results]
    OOF = np.column_stack(oof_views)
    TEST = np.column_stack(test_views)
    print(f"\nStack shape: {OOF.shape}")

    best_meta_C = None; best_meta_tpr = -1
    for meta_C in [0.005, 0.01, 0.05, 0.1, 0.5]:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + 7)
        moof = np.zeros(len(y))
        for tr, va in skf.split(OOF, y):
            m = _make_logreg(meta_C); m.fit(OOF[tr], y[tr])
            moof[va] = m.predict_proba(OOF[va])[:, 1]
        t = tpr_at_fpr(moof.tolist(), y.tolist(), 0.01)
        print(f"  META C={meta_C}: OOF TPR={t:.4f}")
        if t > best_meta_tpr:
            best_meta_tpr = t; best_meta_C = meta_C
    meta = _make_logreg(best_meta_C); meta.fit(OOF, y)
    test_meta = meta.predict_proba(TEST)[:, 1]
    print(f"BEST META C={best_meta_C} OOF={best_meta_tpr:.4f}")

    # Save outputs
    base_tprs = np.array([tpr_at_fpr(o.tolist(), y.tolist(), 0.01) for o in oof_views])
    print("\n=== Per-base OOF TPRs:")
    for n, t in zip(view_labels, base_tprs):
        print(f"  {n}: {t:.4f}")

    ids = test_df["id"].tolist()

    best_i = int(np.argmax(base_tprs))
    out = out_dir / f"{args.out_prefix}_best_base.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_views[best_i], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} ({view_labels[best_i]} OOF={base_tprs[best_i]:.4f})")

    out = out_dir / f"{args.out_prefix}_meta.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_meta, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (meta C={best_meta_C} OOF={best_meta_tpr:.4f})")

    # Top-3 rank-avg
    top3 = np.argsort(-base_tprs)[:3]
    test_ranks_top = np.column_stack([rankdata(test_views[i], method="average") / len(test_views[i]) for i in top3])
    rank_top3 = test_ranks_top.mean(axis=1)
    rank_top3 = (rank_top3 - rank_top3.min()) / (rank_top3.max() - rank_top3.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_top3_rank.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_top3, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (top3 rank-avg: {[view_labels[i] for i in top3]})")


if __name__ == "__main__":
    main()
