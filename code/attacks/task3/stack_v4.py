#!/usr/bin/env python3
"""Task 3 — v4: HistGradientBoosting + RandomForest + multi-seed bagging.

Builds on v3 winning recipe (SelectKBest K=40 leak-free). Adds:
- HistGradientBoosting (fast non-linear, replaces LGBM)
- RandomForest (different non-linear inductive bias)
- Multi-seed bagging (ensemble across random states)
- Per-fold ensemble (different fold splits)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out-prefix", type=str, default="submission_v4")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_splits(data_dir: Path | None):
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


def _oof_estimator(X, y, make_fn, n_splits, seed, predict_fn=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = make_fn()
        m.fit(X[tr], y[tr])
        if predict_fn is not None:
            oof[va] = predict_fn(m, X[va])
        else:
            oof[va] = m.predict_proba(X[va])[:, 1]
    final = make_fn(); final.fit(X, y)
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
    print(f"  labeled={n_lab} test={len(test_df)}")

    cache = args.cache_dir
    feat_a       = _load_pkl(cache, "a")
    feat_bino    = _load_pkl(cache, "bino")
    feat_bino_s  = _load_pkl(cache, "bino_strong")
    feat_bino_xl = _load_pkl(cache, "bino_xl")
    feat_olmo7b  = _load_pkl(cache, "olmo_7b")
    feat_olmo13b = _load_pkl(cache, "olmo_13b")
    feat_olmo1b  = _load_pkl(cache, "multi_lm")
    feat_judge7b = _load_pkl(cache, "judge_olmo7b")
    feat_better_liu = _load_pkl(cache, "better_liu")
    feat_fdgpt   = _load_pkl(cache, "fdgpt")
    feat_d       = _load_pkl(cache, "d")

    parts = [f for f in [feat_a, feat_bino, feat_bino_s, feat_bino_xl,
                          feat_olmo7b, feat_olmo13b, feat_olmo1b, feat_judge7b,
                          feat_better_liu, feat_fdgpt, feat_d] if f is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)

    derived = {}
    pairs_lp = [
        ("olmo7b_lp_mean", "lp_mean"),
        ("olmo7b_lp_mean", "lp_per"),
        ("olmo7b_lp_mean", "lp_obs"),
        ("olmo7b_lp_mean", "bino_strong_lp_obs"),
        ("olmo7b_lp_mean", "bino_strong_lp_per"),
        ("olmo7b_lp_mean", "bino_xl_lp_obs"),
        ("olmo7b_lp_mean", "bino_xl_lp_per"),
        ("olmo7b_lp_mean", "olmo_lp_mean"),
        ("olmo13b_lp_mean", "olmo7b_lp_mean"),
        ("olmo13b_lp_mean", "lp_mean"),
        ("olmo13b_lp_mean", "bino_xl_lp_per"),
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

    # Use SelectKBest K=40 (winner from v3)
    sel = SelectKBest(mutual_info_classif, k=40)
    sel.fit(X_lab, y)
    kept = sel.get_support()
    Xk_lab = X_lab[:, kept]
    Xk_test = X_test[:, kept]
    print(f"SelectKBest K=40 applied")

    oof_views = []
    test_views = []
    view_labels = []

    # ── 1. LogReg C=0.05 (v3 winner)
    print("\n=== 1. LogReg C=0.05")
    oof, model = _oof_estimator(Xk_lab, y, lambda: _make_logreg(0.05), args.n_splits, args.seed)
    pred = model.predict_proba(Xk_test)[:, 1]
    t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    print(f"  OOF TPR={t:.4f}")
    oof_views.append(oof); test_views.append(pred); view_labels.append("lr_K40")

    # ── 2. HistGradientBoosting
    print("\n=== 2. HistGradientBoosting")
    for params_name, params in [
        ("hgb_default", dict(max_iter=200, learning_rate=0.05, max_depth=4,
                              min_samples_leaf=10, l2_regularization=1.0, random_state=42)),
        ("hgb_strong", dict(max_iter=400, learning_rate=0.03, max_depth=6,
                             min_samples_leaf=8, l2_regularization=2.0, random_state=42)),
    ]:
        try:
            oof, model = _oof_estimator(Xk_lab, y, lambda p=params: HistGradientBoostingClassifier(**p),
                                         args.n_splits, args.seed)
            pred = model.predict_proba(Xk_test)[:, 1]
            t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  {params_name}: OOF TPR={t:.4f}")
            oof_views.append(oof); test_views.append(pred); view_labels.append(params_name)
        except Exception as e:
            print(f"  {params_name}: failed ({e})")

    # ── 3. RandomForest
    print("\n=== 3. RandomForest")
    for params_name, params in [
        ("rf_300", dict(n_estimators=300, max_depth=None, min_samples_leaf=5,
                         max_features="sqrt", n_jobs=-1, random_state=42)),
        ("rf_500", dict(n_estimators=500, max_depth=8, min_samples_leaf=3,
                         max_features="log2", n_jobs=-1, random_state=42)),
    ]:
        try:
            oof, model = _oof_estimator(Xk_lab, y, lambda p=params: RandomForestClassifier(**p),
                                         args.n_splits, args.seed)
            pred = model.predict_proba(Xk_test)[:, 1]
            t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  {params_name}: OOF TPR={t:.4f}")
            oof_views.append(oof); test_views.append(pred); view_labels.append(params_name)
        except Exception as e:
            print(f"  {params_name}: failed ({e})")

    # ── 4. Multi-seed LogReg ensemble (different seeds for fold split)
    print("\n=== 4. Multi-seed LogReg")
    rng = np.random.default_rng(42)
    multi_oof_preds = []
    multi_test_preds = []
    for s in [42, 7, 123, 999, 31337]:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, va in skf.split(Xk_lab, y):
            m = _make_logreg(0.05); m.fit(Xk_lab[tr], y[tr])
            oof[va] = m.predict_proba(Xk_lab[va])[:, 1]
        m_final = _make_logreg(0.05); m_final.fit(Xk_lab, y)
        pred = m_final.predict_proba(Xk_test)[:, 1]
        multi_oof_preds.append(oof)
        multi_test_preds.append(pred)
    multi_oof = np.mean(multi_oof_preds, axis=0)
    multi_test = np.mean(multi_test_preds, axis=0)
    t_multi = tpr_at_fpr(multi_oof.tolist(), y.tolist(), 0.01)
    print(f"  multi-seed avg: OOF TPR={t_multi:.4f}")
    oof_views.append(multi_oof); test_views.append(multi_test); view_labels.append("multi_seed_lr")

    # ── 5. Stack everything via meta
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

    # ── 6. Save outputs
    base_tprs = np.array([tpr_at_fpr(o.tolist(), y.tolist(), 0.01) for o in oof_views])
    print("\n=== Per-base OOF TPRs:")
    for n, t in zip(view_labels, base_tprs):
        print(f"  {n}: {t:.4f}")

    ids = test_df["id"].tolist()

    # Best base
    best_i = int(np.argmax(base_tprs))
    out = out_dir / f"{args.out_prefix}_best_base.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_views[best_i], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} ({view_labels[best_i]} OOF={base_tprs[best_i]:.4f})")

    # Meta
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
    print(f"Saved: {out} (top3 rank-avg)")

    # Multi-seed alone
    out = out_dir / f"{args.out_prefix}_multi_seed.csv"
    pd.DataFrame({"id": ids, "score": np.clip(multi_test, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (multi-seed LR avg)")


if __name__ == "__main__":
    main()
