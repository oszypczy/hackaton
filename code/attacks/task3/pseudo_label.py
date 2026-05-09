#!/usr/bin/env python3
"""Pseudo-labeling: score test set, take top/bottom confident → expand training, retrain."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # type: ignore

DEFAULT_FEATURES = [
    "a", "a_strong", "bino", "bino_strong", "bino_xl",
    "fdgpt", "d", "better_liu", "stylometric",
    "kgw", "kgw_llama", "kgw_v2", "bigram", "lm_judge",
    "multi_lm", "multi_lm_v2", "roberta", "unigram_direct",
    # multan1's newer features (auto-skip if not in cache)
    "olmo_7b", "olmo_13b",
    "judge_phi2", "judge_mistral", "judge_chat", "judge_olmo7b", "judge_olmo13b",
]


def _read_jsonl(path):
    import json
    return pd.DataFrame([__import__('json').loads(l) for l in path.read_text().splitlines() if l.strip()])


def load_splits(data_dir):
    tc = _read_jsonl(data_dir / "train_clean.jsonl"); tc["label"] = 0
    tw = _read_jsonl(data_dir / "train_wm.jsonl"); tw["label"] = 1
    vc = _read_jsonl(data_dir / "valid_clean.jsonl"); vc["label"] = 0
    vw = _read_jsonl(data_dir / "valid_wm.jsonl"); vw["label"] = 1
    test = _read_jsonl(data_dir / "test.jsonl")
    train = pd.concat([tc, tw], ignore_index=True)
    val = pd.concat([vc, vw], ignore_index=True)
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)
    return train, val, test


def load_features(cache_dir, names):
    parts = []
    for n in names:
        p = cache_dir / f"features_{n}.pkl"
        if not p.exists():
            print(f"  [skip] {n}")
            continue
        with open(p, "rb") as f:
            df = pickle.load(f).add_prefix(f"{n}__").reset_index(drop=True)
        parts.append(df)
    return pd.concat(parts, axis=1).fillna(0.0)


def _pca_roberta(df, n=32, seed=42):
    from sklearn.decomposition import PCA
    embed = [c for c in df.columns if c.startswith("roberta__rob_") and not c.startswith("roberta__rob_pooled_")]
    if not embed: return df
    other = [c for c in df.columns if not c.startswith("roberta__")]
    stat = [c for c in df.columns if c.startswith("roberta__rob_pooled_")]
    X = StandardScaler().fit_transform(df[embed].values)
    pca = PCA(n_components=min(n, X.shape[0]-1, X.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=[f"roberta__pca_{i}" for i in range(X_pca.shape[1])])
    return pd.concat([df[other].reset_index(drop=True), df[stat].reset_index(drop=True), pca_df], axis=1)


def fit_predict(X_tr, y_tr, X_te, C=0.05):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=4000))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-frac", type=float, default=0.20,
                   help="fraction of test to pseudo-label (top/bottom)")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-rounds", type=int, default=2,
                   help="how many pseudo-labeling iterations to run")
    p.add_argument("--features", nargs="*", default=None,
                   help="override DEFAULT_FEATURES list")
    args = p.parse_args()
    feature_names = args.features if args.features else DEFAULT_FEATURES

    train_df, val_df, test_df = load_splits(args.data_dir)
    n_train = len(train_df) + len(val_df)
    n_test = len(test_df)
    y = pd.concat([train_df, val_df])["label"].astype(int).values

    X_df = load_features(args.cache_dir, feature_names)
    X_df = _pca_roberta(X_df, 32, args.seed)
    X_full = X_df.values.astype(np.float32)
    X_lab = X_full[:n_train]
    X_te = X_full[n_train:]
    print(f"X_lab={X_lab.shape}  X_te={X_te.shape}")

    # Round 0: baseline
    print("\n=== Round 0 (baseline) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X_lab, y):
        oof[va] = fit_predict(X_lab[tr], y[tr], X_lab[va], args.C)
    oof_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    print(f"OOF TPR@1%FPR: {oof_tpr:.4f}")

    test_scores = fit_predict(X_lab, y, X_te, args.C)
    print(f"test scores pct: 5={np.percentile(test_scores,5):.3f} 50={np.percentile(test_scores,50):.3f} 95={np.percentile(test_scores,95):.3f}")

    # Pseudo-label rounds
    X_aug = X_lab.copy()
    y_aug = y.copy()
    for r in range(1, args.n_rounds + 1):
        print(f"\n=== Round {r} (pseudo-labeling top/bottom {args.top_frac:.0%}) ===")
        # take top X% as pos, bottom X% as neg
        n_pseudo = int(len(test_scores) * args.top_frac)
        pos_idx = np.argsort(test_scores)[-n_pseudo:]
        neg_idx = np.argsort(test_scores)[:n_pseudo]
        print(f"  pseudo: {len(pos_idx)} pos (mean={test_scores[pos_idx].mean():.3f}) + {len(neg_idx)} neg (mean={test_scores[neg_idx].mean():.3f})")

        X_extra = np.concatenate([X_te[pos_idx], X_te[neg_idx]], axis=0)
        y_extra = np.array([1] * len(pos_idx) + [0] * len(neg_idx))

        X_aug = np.concatenate([X_lab, X_extra], axis=0)
        y_aug = np.concatenate([y, y_extra], axis=0)
        print(f"  augmented training: {X_aug.shape}")

        # OOF on labeled (don't shuffle pseudo-labels into folds — only use them as TRAINING)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_lab, y):
            X_tr = np.concatenate([X_lab[tr], X_extra], axis=0)
            y_tr_ = np.concatenate([y[tr], y_extra])
            oof[va] = fit_predict(X_tr, y_tr_, X_lab[va], args.C)
        oof_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        print(f"OOF TPR@1%FPR (with pseudo): {oof_tpr:.4f}")

        test_scores = fit_predict(X_aug, y_aug, X_te, args.C)
        print(f"test scores pct: 5={np.percentile(test_scores,5):.3f} 50={np.percentile(test_scores,50):.3f} 95={np.percentile(test_scores,95):.3f}")

    sub = pd.DataFrame({"id": test_df["id"].tolist()[:2250],
                        "score": np.clip(test_scores[:2250], 0, 1)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
