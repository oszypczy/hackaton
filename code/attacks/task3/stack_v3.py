#!/usr/bin/env python3
"""Task 3 — v3 stacking with bagging, LightGBM, SelectKBest, more cross-LM.

Builds on v2 (leak-free) with:
- LightGBM on full_no_leak (non-linear interactions)
- SelectKBest(mutual_info, K=20/40) to denoise
- Bagging: 20 LogReg on bootstrap samples
- More aggressive cross-LM derivations (triplet differences, log ratios)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out-prefix", type=str, default="submission_v3")
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


def _oof_logreg(X, y, C, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = _make_logreg(C); m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    final = _make_logreg(C); final.fit(X, y)
    return oof, final


def _oof_lgbm(X, y, n_splits, seed, params):
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    best_iters = []
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va] = m.predict(X[va])
        best_iters.append(m.best_iteration)
    n_rounds = max(int(np.mean(best_iters)), 50)
    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=n_rounds)
    return oof, final


def _bagged_logreg_predict(X_train, y, X_test, n_bags=20, C=0.05, seed=42, frac=0.8):
    rng = np.random.default_rng(seed)
    preds = np.zeros(len(X_test))
    for b in range(n_bags):
        idx = rng.choice(len(X_train), size=int(frac * len(X_train)), replace=True)
        m = _make_logreg(C); m.fit(X_train[idx], y[idx])
        preds += m.predict_proba(X_test)[:, 1]
    return preds / n_bags


def _bagged_oof(X, y, n_splits, seed, C=0.05, n_bags=20):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rng = np.random.default_rng(seed)
    for tr, va in skf.split(X, y):
        for b in range(n_bags):
            idx = rng.choice(len(tr), size=int(0.8 * len(tr)), replace=True)
            m = _make_logreg(C); m.fit(X[tr][idx], y[tr][idx])
            oof[va] += m.predict_proba(X[va])[:, 1]
        oof[va] /= n_bags
    return oof


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
    print(f"Cache: {cache}")

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

    # Build comprehensive cross-LM derived feats (label-free)
    derived = {}
    lp = lambda c: full[c] if c in full.columns else None
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

    # PPL ratios
    ppl_pairs = [
        ("olmo7b_ppl", "ppl_observer"),
        ("olmo7b_ppl", "ppl_performer"),
        ("olmo7b_ppl", "bino_strong_ppl_obs"),
        ("olmo7b_ppl", "bino_xl_ppl_per"),
        ("olmo7b_ppl", "olmo_ppl"),
        ("olmo13b_ppl", "olmo7b_ppl"),
    ]
    for a, b in ppl_pairs:
        if a in full.columns and b in full.columns:
            derived[f"r_{a}_over_{b}"] = full[a] / (full[b] + 1e-9)

    # Quadratic of strongest LM
    if "olmo7b_lp_mean" in full.columns:
        derived["olmo7b_lp_sq"] = full["olmo7b_lp_mean"] ** 2

    # Triplet (mean of weaker LMs) vs OLMo
    if all(c in full.columns for c in ("olmo7b_lp_mean", "lp_mean", "bino_strong_lp_obs")):
        derived["d_olmo7b_vs_meanWeak"] = full["olmo7b_lp_mean"] - 0.5 * (full["lp_mean"] + full["bino_strong_lp_obs"])

    # Log ppl ratio
    if "olmo7b_ppl" in full.columns and "bino_xl_ppl_per" in full.columns:
        derived["log_ppl_olmo7b_over_pythia69b"] = (
            np.log(full["olmo7b_ppl"] + 1.001) - np.log(full["bino_xl_ppl_per"] + 1.001)
        )

    full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
    print(f"Full features: {full.shape[1]}  (derived added: {len(derived)})")

    X_full = full.fillna(0.0).values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]
    feat_names = list(full.columns)

    # ── 1. Best baseline LogReg (full_no_leak C=0.05 — proven in v2)
    print("\n=== 1. Baseline LogReg (full leak-free)")
    oof_lr, model_lr = _oof_logreg(X_lab, y, 0.05, args.n_splits, args.seed)
    tpr_lr = tpr_at_fpr(oof_lr.tolist(), y.tolist(), 0.01)
    pred_lr = model_lr.predict_proba(X_test)[:, 1]
    print(f"  OOF TPR={tpr_lr:.4f}")

    # ── 2. SelectKBest (different K values)
    print("\n=== 2. SelectKBest")
    select_results = {}
    for K in [20, 40, 60]:
        if K >= X_lab.shape[1]:
            continue
        sel = SelectKBest(mutual_info_classif, k=K)
        sel.fit(X_lab, y)
        kept = sel.get_support()
        Xk_lab = X_lab[:, kept]
        Xk_test = X_test[:, kept]
        for C in [0.05, 0.5]:
            oof_k, m_k = _oof_logreg(Xk_lab, y, C, args.n_splits, args.seed)
            t_k = tpr_at_fpr(oof_k.tolist(), y.tolist(), 0.01)
            print(f"  K={K} C={C}: OOF TPR={t_k:.4f}")
            select_results[(K, C)] = (oof_k, m_k.predict_proba(Xk_test)[:, 1], t_k)

    # ── 3. LightGBM on full
    print("\n=== 3. LightGBM (full leak-free)")
    lgbm_results = []
    for params_name, params in [
        ("lgbm_default", dict(objective="binary", learning_rate=0.05, num_leaves=15,
                              max_depth=4, min_data_in_leaf=10, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                              verbosity=-1, n_jobs=-1, seed=42)),
        ("lgbm_strong", dict(objective="binary", learning_rate=0.03, num_leaves=31,
                             max_depth=6, min_data_in_leaf=8, feature_fraction=0.7,
                             bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                             verbosity=-1, n_jobs=-1, seed=42)),
    ]:
        try:
            oof_g, model_g = _oof_lgbm(X_lab, y, args.n_splits, args.seed, params)
            t_g = tpr_at_fpr(oof_g.tolist(), y.tolist(), 0.01)
            pred_g = model_g.predict(X_test)
            print(f"  {params_name}: OOF TPR={t_g:.4f}")
            lgbm_results.append((params_name, oof_g, pred_g, t_g))
        except Exception as e:
            print(f"  {params_name}: failed ({e})")

    # ── 4. Bagged LogReg (20 bootstrap, C=0.05)
    print("\n=== 4. Bagged LogReg")
    oof_bag = _bagged_oof(X_lab, y, args.n_splits, args.seed, C=0.05, n_bags=20)
    t_bag = tpr_at_fpr(oof_bag.tolist(), y.tolist(), 0.01)
    pred_bag = _bagged_logreg_predict(X_lab, y, X_test, n_bags=20, C=0.05, seed=42)
    print(f"  OOF TPR={t_bag:.4f}")

    # ── 5. Stack EVERYTHING via meta-LogReg
    print("\n=== 5. Meta-stacking")
    oof_views = [oof_lr, oof_bag] + [v[0] for v in select_results.values()] + [r[1] for r in lgbm_results]
    test_views = [pred_lr, pred_bag] + [v[1] for v in select_results.values()] + [r[2] for r in lgbm_results]
    view_labels = ["full_lr", "bagged"] + [f"sel_K{k}_C{c}" for (k, c) in select_results.keys()] + [r[0] for r in lgbm_results]
    OOF = np.column_stack(oof_views)
    TEST = np.column_stack(test_views)

    best_meta_C = None; best_meta_tpr = -1
    for meta_C in [0.01, 0.05, 0.1, 0.5, 1.0]:
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
    print(f"  BEST META C={best_meta_C} OOF={best_meta_tpr:.4f}")

    # ── 6. Save all candidate submissions
    base_tprs = np.array([tpr_at_fpr(o.tolist(), y.tolist(), 0.01) for o in oof_views])
    print("\n=== Per-base OOF TPRs:")
    for n, t in zip(view_labels, base_tprs):
        print(f"  {n}: {t:.4f}")

    ids = test_df["id"].tolist()
    out_files = []

    # 6a) best base
    best_i = int(np.argmax(base_tprs))
    out = out_dir / f"{args.out_prefix}_best_base.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_views[best_i], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} ({view_labels[best_i]} OOF={base_tprs[best_i]:.4f})")
    out_files.append(out)

    # 6b) meta
    out = out_dir / f"{args.out_prefix}_meta.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_meta, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (meta C={best_meta_C} OOF={best_meta_tpr:.4f})")
    out_files.append(out)

    # 6c) rank-avg of top 3 by OOF
    top3 = np.argsort(-base_tprs)[:3]
    test_ranks_top = np.column_stack([rankdata(test_views[i], method="average") / len(test_views[i]) for i in top3])
    rank_top3 = test_ranks_top.mean(axis=1)
    rank_top3 = (rank_top3 - rank_top3.min()) / (rank_top3.max() - rank_top3.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_top3_rank.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_top3, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (top3 rank-avg)")
    out_files.append(out)

    # 6d) bagged alone
    out = out_dir / f"{args.out_prefix}_bagged.csv"
    pd.DataFrame({"id": ids, "score": np.clip(pred_bag, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (bagged OOF={t_bag:.4f})")
    out_files.append(out)

    # 6e) lgbm best (if any)
    if lgbm_results:
        best_lgbm = max(lgbm_results, key=lambda r: r[3])
        out = out_dir / f"{args.out_prefix}_lgbm.csv"
        pd.DataFrame({"id": ids, "score": np.clip(best_lgbm[2], 0.001, 0.999)}).to_csv(out, index=False)
        print(f"Saved: {out} ({best_lgbm[0]} OOF={best_lgbm[3]:.4f})")
        out_files.append(out)

    print("\nAll outputs:")
    for f in out_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
