#!/usr/bin/env python3
"""Task 3 — Multi-model stacking.

Loads cached features, defines several heterogeneous base models, runs 5-fold OOF
for each, then trains a meta LogReg on stacked OOF predictions. Outputs CSV.

Run on cluster (cache_dir = /p/scratch/.../task3/cache). Locally use --features-dir.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent
SUBMISSIONS_DIR = ROOT / "submissions"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "task3_stack_meta.csv")
    p.add_argument("--meta-C", type=float, default=0.5)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_splits(data_dir: Path | None):
    if data_dir and data_dir.exists():
        train_clean = _read_jsonl(data_dir / "train_clean.jsonl")
        train_wm = _read_jsonl(data_dir / "train_wm.jsonl")
        valid_clean = _read_jsonl(data_dir / "valid_clean.jsonl")
        valid_wm = _read_jsonl(data_dir / "valid_wm.jsonl")
        test = _read_jsonl(data_dir / "test.jsonl")
        train_clean["label"] = 0
        train_wm["label"] = 1
        valid_clean["label"] = 0
        valid_wm["label"] = 1
        train = pd.concat([train_clean, train_wm], ignore_index=True)
        val = pd.concat([valid_clean, valid_wm], ignore_index=True)
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


def _load_pkl(cache_dir: Path, name: str) -> pd.DataFrame | None:
    p = cache_dir / f"features_{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f).reset_index(drop=True)


def _make_logreg(C: float = 0.05) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs")),
    ])


def _oof(X: np.ndarray, y: np.ndarray, train_fn, n_splits: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = train_fn(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1] if hasattr(m, "predict_proba") else m.predict(X[va])
    return oof


def _train_logreg_full(C: float):
    def fn(X, y):
        m = _make_logreg(C)
        m.fit(X, y)
        return m
    return fn


def _train_lgbm(seed: int = 42):
    import lightgbm as lgb

    def fn(X, y):
        params = dict(objective="binary", learning_rate=0.05, num_leaves=15,
                      max_depth=4, min_data_in_leaf=8, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbosity=-1, n_jobs=-1, seed=seed)
        return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=200)
    return fn


def main():
    args = parse_args()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    all_lab = pd.concat([train_df, val_df], ignore_index=True)
    n_lab = len(all_lab)
    n_test = len(test_df)
    y = all_lab["label"].astype(int).values
    print(f"  labeled={n_lab} test={n_test}")

    cache = args.cache_dir
    print(f"Loading features from: {cache}")

    feat_a       = _load_pkl(cache, "a")
    feat_bc      = _load_pkl(cache, "bc")
    feat_d       = _load_pkl(cache, "d")
    feat_bino    = _load_pkl(cache, "bino")
    feat_bino_s  = _load_pkl(cache, "bino_strong")
    feat_bino_xl = _load_pkl(cache, "bino_xl")
    feat_olmo7b  = _load_pkl(cache, "olmo_7b")
    feat_olmo13b = _load_pkl(cache, "olmo_13b")
    feat_multi   = _load_pkl(cache, "multi_lm")
    feat_judge   = _load_pkl(cache, "lm_judge")
    feat_judge7b = _load_pkl(cache, "judge_olmo7b")
    feat_better_liu = _load_pkl(cache, "better_liu")
    feat_fdgpt   = _load_pkl(cache, "fdgpt")
    feat_styl    = _load_pkl(cache, "stylometric")

    have = {n: f is not None for n, f in [
        ("a", feat_a), ("bc", feat_bc), ("d", feat_d), ("bino", feat_bino),
        ("bino_s", feat_bino_s), ("bino_xl", feat_bino_xl),
        ("olmo7b", feat_olmo7b), ("olmo13b", feat_olmo13b),
        ("multi", feat_multi), ("judge", feat_judge), ("judge7b", feat_judge7b),
        ("better_liu", feat_better_liu), ("fdgpt", feat_fdgpt),
        ("styl", feat_styl),
    ]}
    print("  available:", {k: v for k, v in have.items() if v})

    # Build cross-LM v1 derived (the winning 6 features)
    parts = [feat_a, feat_bc, feat_bino, feat_bino_s, feat_bino_xl,
             feat_olmo7b, feat_multi, feat_d]
    parts = [p for p in parts if p is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)

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
    if derived:
        full = pd.concat([full, pd.DataFrame(derived)], axis=1)
        print(f"  added cross-lm v1 derived: {list(derived)}")

    full = full.reset_index(drop=True)

    # Define base model "views" (different feature subsets — heterogeneous signal)
    views: dict[str, list[str]] = {}

    # 1. Cross-LM v1 winning: a + bc + bino + bino_s + bino_xl + olmo7b + multi + cross derived
    cols_clm = list(full.columns)
    views["clm_v1_full"] = cols_clm

    # 2. Minimal: a + bc only (universal LM features)
    if feat_bc is not None:
        views["a_bc"] = list(feat_a.columns) + list(feat_bc.columns)

    # 3. OLMo-only (best single LM)
    if feat_olmo7b is not None:
        views["olmo7b_only"] = list(feat_olmo7b.columns)

    # 4. Binoculars stack only
    bino_cols = []
    for fb in [feat_bino, feat_bino_s, feat_bino_xl]:
        if fb is not None: bino_cols += list(fb.columns)
    if bino_cols:
        views["bino_only"] = bino_cols

    # 5. cross-LM derived only (the 6 v1 features)
    if derived:
        views["cross_only"] = list(derived.keys())

    # 6. judge + better_liu + styl (orthogonal heuristics)
    extra_cols = []
    for fb in [feat_judge, feat_judge7b, feat_better_liu, feat_styl, feat_fdgpt, feat_d]:
        if fb is not None: extra_cols += list(fb.columns)
    if extra_cols:
        # Need to attach those to full df
        extra_df_parts = [f for f in [feat_judge, feat_judge7b, feat_better_liu, feat_styl, feat_fdgpt, feat_d] if f is not None]
        full_extra = pd.concat(extra_df_parts, axis=1).reset_index(drop=True)
        full = pd.concat([full, full_extra.loc[:, [c for c in full_extra.columns if c not in full.columns]]], axis=1)
        views["aux_only"] = [c for c in extra_cols if c in full.columns]

    print(f"\nViews: {[(k, len(v)) for k, v in views.items()]}")

    X_full = full.fillna(0.0).values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]
    name_to_idx = {n: i for i, n in enumerate(full.columns)}

    # ── Build OOF + test predictions per view
    oof_stack = []
    test_stack = []
    view_names = []

    for name, cols in views.items():
        idx = [name_to_idx[c] for c in cols if c in name_to_idx]
        X_view_lab = X_lab[:, idx]
        X_view_test = X_test[:, idx]

        # LogReg variant
        for C in [0.01, 0.05, 0.5]:
            model_name = f"{name}_lr_C{C}"
            train_fn = _train_logreg_full(C)
            oof = _oof(X_view_lab, y, train_fn, args.n_splits, args.seed)
            tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  [{model_name}] OOF TPR@1%FPR: {tpr:.4f}")
            final = train_fn(X_view_lab, y)
            test_pred = final.predict_proba(X_view_test)[:, 1]
            oof_stack.append(oof)
            test_stack.append(test_pred)
            view_names.append(model_name)

        # LGBM on full clm view only (saves time)
        if name == "clm_v1_full":
            try:
                import lightgbm as lgb
                model_name = f"{name}_lgbm"
                lgb_fn = _train_lgbm(args.seed)
                # OOF with lgbm
                skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
                oof = np.zeros(len(y))
                for tr, va in skf.split(X_view_lab, y):
                    m = lgb_fn(X_view_lab[tr], y[tr])
                    oof[va] = m.predict(X_view_lab[va])
                tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
                print(f"  [{model_name}] OOF TPR@1%FPR: {tpr:.4f}")
                final = lgb_fn(X_view_lab, y)
                test_pred = final.predict(X_view_test)
                oof_stack.append(oof)
                test_stack.append(test_pred)
                view_names.append(model_name)
            except ImportError:
                pass

    OOF = np.column_stack(oof_stack)  # (n_lab, n_models)
    TEST = np.column_stack(test_stack)
    print(f"\nOOF stack shape: {OOF.shape}, TEST stack shape: {TEST.shape}")

    # Diagnostics: per-model OOF TPR and pairwise corr
    from scipy.stats import spearmanr
    print("\nPer-model OOF TPR@1%FPR:")
    for i, n in enumerate(view_names):
        t = tpr_at_fpr(OOF[:, i].tolist(), y.tolist(), 0.01)
        print(f"  {n}: {t:.4f}")

    # ── Meta-learner: LogReg on OOF
    for meta_C in [0.05, 0.5, 5.0]:
        meta = _make_logreg(meta_C)
        meta.fit(OOF, y)
        meta_oof_pred = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + 7)
        for tr, va in skf.split(OOF, y):
            m = _make_logreg(meta_C)
            m.fit(OOF[tr], y[tr])
            meta_oof_pred[va] = m.predict_proba(OOF[va])[:, 1]
        meta_tpr = tpr_at_fpr(meta_oof_pred.tolist(), y.tolist(), 0.01)
        print(f"\n[META C={meta_C}] OOF TPR@1%FPR: {meta_tpr:.4f}")

    # Pick best meta C by OOF
    best_meta_C = args.meta_C
    print(f"\nUsing meta C={best_meta_C}")
    meta = _make_logreg(best_meta_C)
    meta.fit(OOF, y)
    test_meta = meta.predict_proba(TEST)[:, 1]

    # Save submission
    sub = pd.DataFrame({
        "id": test_df["id"].tolist(),
        "score": np.clip(test_meta, 0.001, 0.999),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out} ({len(sub)} rows)")

    # Also dump rank-averaged baseline of all base models
    from scipy.stats import rankdata
    test_ranks = np.column_stack([rankdata(c, method="average") / len(c) for c in TEST.T])
    rank_avg = test_ranks.mean(axis=1)
    rank_avg = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min() + 1e-9)
    rank_csv = args.out.with_name(args.out.stem + "_rank.csv")
    pd.DataFrame({"id": test_df["id"].tolist(), "score": np.clip(rank_avg, 0.001, 0.999)}).to_csv(rank_csv, index=False)
    print(f"Saved: {rank_csv} (rank-average baseline)")


if __name__ == "__main__":
    main()
