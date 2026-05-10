#!/usr/bin/env python3
"""Prior probability shift correction (Perplexity Q3).

Train LogReg on balanced labeled (50/50). Test set has unknown prior.
Estimate test prior via EM on test predictions, correct via Bayes shift.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # type: ignore


def estimate_prior_em(probs: np.ndarray, n_iter: int = 100, init: float = 0.5) -> float:
    pi = init
    for _ in range(n_iter):
        # E-step
        pos_resp = pi * probs / (pi * probs + (1 - pi) * (1 - probs) + 1e-12)
        # M-step
        pi_new = pos_resp.mean()
        if abs(pi_new - pi) < 1e-6:
            break
        pi = pi_new
    return float(pi)


def correct_prior(probs: np.ndarray, pi_train: float, pi_test: float) -> np.ndarray:
    """Bayes shift: P_new(y|x) ∝ P_old(y|x) * pi_test/pi_train"""
    odds_old = probs / (1 - probs + 1e-12)
    pi_test = max(min(pi_test, 0.99), 0.01)
    odds_new = odds_old * (pi_test / pi_train) * ((1 - pi_train) / (1 - pi_test))
    return odds_new / (1 + odds_new)


def _read_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


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
        if not p.exists(): continue
        with open(p, "rb") as f:
            df = pickle.load(f).add_prefix(f"{n}__").reset_index(drop=True)
        parts.append(df)
    return pd.concat(parts, axis=1).fillna(0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out-prefix", required=True, type=Path)
    p.add_argument("--features", nargs="*", required=True)
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_df, val_df, test_df = load_splits(args.data_dir)
    n_lab = len(train_df) + len(val_df)
    y = pd.concat([train_df, val_df])["label"].astype(int).values

    X_df = load_features(args.cache_dir, args.features)

    # PCA roberta if loaded
    if any("roberta" in c for c in X_df.columns):
        from sklearn.decomposition import PCA
        embed_cols = [c for c in X_df.columns if c.startswith("roberta__rob_") and not c.startswith("roberta__rob_pooled_")]
        if embed_cols:
            other = [c for c in X_df.columns if not c.startswith("roberta__")]
            stat = [c for c in X_df.columns if c.startswith("roberta__rob_pooled_")]
            X_emb = StandardScaler().fit_transform(X_df[embed_cols].values)
            pca = PCA(n_components=min(32, X_emb.shape[0] - 1, X_emb.shape[1]), random_state=args.seed)
            X_pca = pca.fit_transform(X_emb)
            pca_df = pd.DataFrame(X_pca, columns=[f"roberta__pca_{i}" for i in range(X_pca.shape[1])])
            X_df = pd.concat([X_df[other].reset_index(drop=True), X_df[stat].reset_index(drop=True), pca_df], axis=1)

    X = X_df.values.astype(np.float32)
    X_lab = X[:n_lab]
    X_test = X[n_lab:]
    print(f"X_lab={X_lab.shape}  X_test={X_test.shape}")

    # Train LogReg on labeled
    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=args.C, max_iter=4000))])
    pipe.fit(X_lab, y)

    # OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X_lab, y):
        p2 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=args.C, max_iter=4000))])
        p2.fit(X_lab[tr], y[tr])
        oof[va] = p2.predict_proba(X_lab[va])[:, 1]
    oof_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    print(f"OOF TPR (uncorrected): {oof_tpr:.4f}")

    test_probs = pipe.predict_proba(X_test)[:, 1]

    pi_train = float(y.mean())
    print(f"Train prior: {pi_train:.3f}")

    # Estimate test prior (EM on test probs)
    pi_test = estimate_prior_em(test_probs)
    print(f"Estimated test prior: {pi_test:.3f}")

    # Try multiple priors as ablation
    for pi in [0.30, 0.40, 0.50, pi_test, 0.20, 0.10]:
        corrected = correct_prior(test_probs, pi_train, pi)
        out_file = args.out_prefix.parent / f"{args.out_prefix.name}_pi{int(pi*100):03d}.csv"
        sub = pd.DataFrame({"id": test_df["id"].tolist()[:2250],
                            "score": np.clip(corrected[:2250], 0, 1)})
        sub.to_csv(out_file, index=False)
        print(f"Saved (pi={pi:.3f}): {out_file}")


if __name__ == "__main__":
    main()
