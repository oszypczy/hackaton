#!/usr/bin/env python3
"""Task 3 — v8: MLP classifier + ensemble of v7's multi-seed K-best.

Tries:
1. MLP with 1-2 hidden layers (different non-linear inductive bias vs LR)
2. Combines top-K stable feature subsets from v7 multi-seed
3. Output rank-avg of MLP + LR + HGB on best subset
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
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
    p.add_argument("--out-prefix", type=str, default="submission_v8")
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
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs"))])


def _make_mlp(hidden=(64,), alpha=1.0, seed=42):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                            max_iter=2000, random_state=seed,
                                            early_stopping=True, validation_fraction=0.2,
                                            n_iter_no_change=20))])


def _oof_estimator(X, y, make_fn, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = make_fn(); m.fit(X[tr], y[tr])
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

    # ── Stable feature subset: union of top-K seeds (avoid cherry-pick)
    print("\n=== Stable feature ranking via seed averaging")
    K = 40
    n_seeds = 10
    mi_scores = np.zeros(X_lab.shape[1])
    for s in range(n_seeds):
        mi_scores += mutual_info_classif(X_lab, y, random_state=s)
    mi_scores /= n_seeds
    top_idx = np.argsort(-mi_scores)[:K]
    kept_mask = np.zeros(X_lab.shape[1], dtype=bool)
    kept_mask[top_idx] = True
    Xk_lab = X_lab[:, kept_mask]
    Xk_test = X_test[:, kept_mask]
    print(f"Stable K={K} feature subset selected (avg over {n_seeds} seeds)")

    oof_views = []
    test_views = []
    view_labels = []

    # ── 1. LR baselines
    print("\n=== 1. LogReg")
    for C in [0.01, 0.05, 0.1, 0.5]:
        oof, model = _oof_estimator(Xk_lab, y, lambda c=C: _make_logreg(c), args.n_splits, 42)
        t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        pred = model.predict_proba(Xk_test)[:, 1]
        print(f"  LR C={C}: OOF TPR={t:.4f}")
        oof_views.append(oof); test_views.append(pred); view_labels.append(f"lr_C{C}")

    # ── 2. MLP variants
    print("\n=== 2. MLP")
    for hidden, alpha in [((32,), 1.0), ((64,), 1.0), ((64, 32), 1.0),
                          ((32,), 0.1), ((64,), 0.1)]:
        try:
            oof, model = _oof_estimator(Xk_lab, y,
                                          lambda h=hidden, a=alpha: _make_mlp(h, a, 42),
                                          args.n_splits, 42)
            t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            pred = model.predict_proba(Xk_test)[:, 1]
            print(f"  MLP h={hidden} α={alpha}: OOF TPR={t:.4f}")
            oof_views.append(oof); test_views.append(pred); view_labels.append(f"mlp_h{hidden}_a{alpha}")
        except Exception as e:
            print(f"  MLP h={hidden}: failed ({e})")

    # ── 3. HGB
    print("\n=== 3. HGB")
    for params_name, params in [
        ("hgb_def", dict(max_iter=300, learning_rate=0.05, max_depth=5,
                          min_samples_leaf=10, l2_regularization=2.0, random_state=42)),
        ("hgb_strong", dict(max_iter=500, learning_rate=0.02, max_depth=7,
                             min_samples_leaf=5, l2_regularization=5.0, random_state=42)),
    ]:
        try:
            oof, model = _oof_estimator(Xk_lab, y,
                                          lambda p=params: HistGradientBoostingClassifier(**p),
                                          args.n_splits, 42)
            t = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            pred = model.predict_proba(Xk_test)[:, 1]
            print(f"  {params_name}: OOF TPR={t:.4f}")
            oof_views.append(oof); test_views.append(pred); view_labels.append(params_name)
        except Exception as e:
            print(f"  {params_name}: failed ({e})")

    # ── 4. Multi-seed MLP (different inits)
    print("\n=== 4. Multi-seed MLP h=(64,) α=1.0")
    multi_mlp_oofs = []; multi_mlp_tests = []
    for s in [42, 7, 123, 999, 31337]:
        oof, model = _oof_estimator(Xk_lab, y, lambda ss=s: _make_mlp((64,), 1.0, ss), args.n_splits, 42)
        multi_mlp_oofs.append(oof)
        multi_mlp_tests.append(model.predict_proba(Xk_test)[:, 1])
    mlp_avg_oof = np.mean(multi_mlp_oofs, axis=0)
    mlp_avg_test = np.mean(multi_mlp_tests, axis=0)
    t = tpr_at_fpr(mlp_avg_oof.tolist(), y.tolist(), 0.01)
    print(f"  Multi-seed MLP avg: OOF TPR={t:.4f}")
    oof_views.append(mlp_avg_oof); test_views.append(mlp_avg_test); view_labels.append("multi_mlp")

    # ── 5. Stack via meta
    OOF = np.column_stack(oof_views)
    TEST = np.column_stack(test_views)
    base_tprs = np.array([tpr_at_fpr(o.tolist(), y.tolist(), 0.01) for o in oof_views])
    print(f"\n=== Per-base summary:")
    sorted_idx = np.argsort(-base_tprs)
    for i in sorted_idx:
        print(f"  {view_labels[i]}: {base_tprs[i]:.4f}")

    best_meta_C = None; best_meta_tpr = -1
    for meta_C in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=49)
        moof = np.zeros(len(y))
        for tr, va in skf.split(OOF, y):
            m = _make_logreg(meta_C); m.fit(OOF[tr], y[tr])
            moof[va] = m.predict_proba(OOF[va])[:, 1]
        t = tpr_at_fpr(moof.tolist(), y.tolist(), 0.01)
        print(f"  META C={meta_C}: {t:.4f}")
        if t > best_meta_tpr:
            best_meta_tpr = t; best_meta_C = meta_C
    meta = _make_logreg(best_meta_C); meta.fit(OOF, y)
    test_meta = meta.predict_proba(TEST)[:, 1]
    print(f"BEST META C={best_meta_C} OOF={best_meta_tpr:.4f}")

    # ── Save outputs
    ids = test_df["id"].tolist()

    # Best base
    best_i = int(np.argmax(base_tprs))
    out = out_dir / f"{args.out_prefix}_best.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_views[best_i], 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  ({view_labels[best_i]} OOF={base_tprs[best_i]:.4f})")

    # Top-3 rank-avg
    top3 = sorted_idx[:3]
    test_ranks_top = np.column_stack([rankdata(test_views[i], method="average") / len(test_views[i]) for i in top3])
    rank_top3 = test_ranks_top.mean(axis=1)
    rank_top3 = (rank_top3 - rank_top3.min()) / (rank_top3.max() - rank_top3.min() + 1e-9)
    out = out_dir / f"{args.out_prefix}_top3_rank.csv"
    pd.DataFrame({"id": ids, "score": np.clip(rank_top3, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (top3 rank-avg: {[view_labels[i] for i in top3]})")

    # Meta
    out = out_dir / f"{args.out_prefix}_meta.csv"
    pd.DataFrame({"id": ids, "score": np.clip(test_meta, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (meta C={best_meta_C} OOF={best_meta_tpr:.4f})")

    # MLP avg
    out = out_dir / f"{args.out_prefix}_mlp.csv"
    mlp_norm = (mlp_avg_test - mlp_avg_test.min()) / (mlp_avg_test.max() - mlp_avg_test.min() + 1e-9)
    pd.DataFrame({"id": ids, "score": np.clip(mlp_norm, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}  (multi-seed MLP)")


if __name__ == "__main__":
    main()
