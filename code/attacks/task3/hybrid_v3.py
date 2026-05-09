#!/usr/bin/env python3
"""Task 3 hybrid: load ALL cached features (multan1's + murdzek2's) → LogReg OOF + predict.

Standalone script — does NOT call any feature extractor. All features must be in cache.
Designed to combine multan1's kitchen-sink features + murdzek2's unigram_direct (key=9999).
"""
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


# All cached feature names to attempt to load (skip if missing)
DEFAULT_FEATURE_NAMES = [
    "a", "a_strong",
    "bino", "bino_strong", "bino_xl",
    "fdgpt",
    "d",
    "better_liu",
    "stylometric",
    "kgw", "kgw_llama", "kgw_v2",
    "bigram",
    "lm_judge",
    "multi_lm", "multi_lm_v2",
    "roberta",
    "unigram_direct",  # murdzek2's
    "olmo_7b",          # extracted later
    "judge_phi2",       # extracted later
    "judge_mistral",    # extracted later
    "kgw_selfhash",     # if ever finishes
]


def _read_jsonl(path: Path) -> pd.DataFrame:
    import json
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_clean = _read_jsonl(data_dir / "train_clean.jsonl"); train_clean["label"] = 0
    train_wm = _read_jsonl(data_dir / "train_wm.jsonl"); train_wm["label"] = 1
    valid_clean = _read_jsonl(data_dir / "valid_clean.jsonl"); valid_clean["label"] = 0
    valid_wm = _read_jsonl(data_dir / "valid_wm.jsonl"); valid_wm["label"] = 1
    test = _read_jsonl(data_dir / "test.jsonl")
    train = pd.concat([train_clean, train_wm], ignore_index=True)
    val = pd.concat([valid_clean, valid_wm], ignore_index=True)
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)
    return train, val, test


def load_features(cache_dir: Path, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    loaded = []
    for n in names:
        path = cache_dir / f"features_{n}.pkl"
        if not path.exists():
            print(f"  [skip] {n} (not cached)")
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        # rename cols with prefix to avoid collisions
        df = df.add_prefix(f"{n}__").reset_index(drop=True)
        parts.append(df)
        loaded.append(n)
        print(f"  [load] {n}: {df.shape}")
    if not parts:
        raise RuntimeError("No features loaded")
    return pd.concat(parts, axis=1).fillna(0.0), loaded


def _pca_roberta(df: pd.DataFrame, n_components: int = 32, seed: int = 42) -> pd.DataFrame:
    """Reduce 768-dim RoBERTa embedding cols to PCA components, keep stats cols intact."""
    from sklearn.decomposition import PCA
    embed_cols = [c for c in df.columns if c.startswith("roberta__rob_") and not c.startswith("roberta__rob_pooled_")]
    stat_cols = [c for c in df.columns if c.startswith("roberta__rob_pooled_")]
    other_cols = [c for c in df.columns if not c.startswith("roberta__")]

    if not embed_cols:
        return df

    X_emb = StandardScaler().fit_transform(df[embed_cols].values)
    pca = PCA(n_components=min(n_components, X_emb.shape[0] - 1, X_emb.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X_emb)
    print(f"  [pca] roberta 768 -> {X_pca.shape[1]} dim, explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    pca_df = pd.DataFrame(X_pca, columns=[f"roberta__pca_{i}" for i in range(X_pca.shape[1])])
    return pd.concat([df[other_cols].reset_index(drop=True), df[stat_cols].reset_index(drop=True), pca_df], axis=1)


def _logreg_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs")),
    ])


def _train_lgbm_oof(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, n_splits: int, seed: int):
    """Returns (oof, test_pred_mean)."""
    import lightgbm as lgb
    params = {
        "objective": "binary", "learning_rate": 0.03,
        "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 5,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 0.5, "verbosity": -1, "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    test_preds = np.zeros((n_splits, X_test.shape[0]))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va] = m.predict(X[va])
        test_preds[fold] = m.predict(X_test)
    return oof, test_preds.mean(axis=0)


def run_oof(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int, C: float) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        pipe = _logreg_pipe(C)
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
    return oof


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.RandomState(0)
    n = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if labels[idx].sum() == 0 or (1 - labels[idx]).sum() == 0:
            continue
        boots.append(tpr_at_fpr(scores[idx].tolist(), labels[idx].tolist(), 0.01))
    return (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--features", nargs="*", default=DEFAULT_FEATURE_NAMES)
    p.add_argument("--logreg-C", type=float, default=0.01)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roberta-pca", type=int, default=32)
    p.add_argument("--n-rows", type=int, default=2250)
    p.add_argument("--classifier", choices=["logreg", "lgbm", "ensemble", "elasticnet", "ridge", "svm", "mlp"], default="logreg")
    p.add_argument("--ensemble-weights", default="0.5,0.5",
                   help="weights for ensemble (logreg, lgbm) — comma-separated")
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--mlp-hidden", type=int, default=64)
    args = p.parse_args()

    print("Loading data splits...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    all_labeled = pd.concat([train_df, val_df], ignore_index=True)
    n_labeled = len(all_labeled)
    y = all_labeled["label"].astype(int).values

    print("Loading cached features...")
    X_df, loaded = load_features(args.cache_dir, args.features)

    # PCA on RoBERTa embedding if it's loaded
    if "roberta" in loaded and args.roberta_pca > 0:
        X_df = _pca_roberta(X_df, args.roberta_pca, args.seed)

    X_full = X_df.values.astype(np.float32)
    print(f"Feature matrix: {X_full.shape} ({len(X_df.columns)} cols)")

    # Sanity check: rows match (n_train + n_val + n_test = total)
    expected_n = len(train_df) + len(val_df) + len(test_df)
    if X_full.shape[0] != expected_n:
        print(f"  [warn] X_full rows {X_full.shape[0]} != expected {expected_n}")

    X_labeled = X_full[:n_labeled]
    X_test = X_full[n_labeled:]
    print(f"  labeled: {X_labeled.shape}  test: {X_test.shape}")

    print(f"\nRunning {args.n_splits}-fold OOF classifier={args.classifier}...")
    if args.classifier == "logreg":
        oof = run_oof(X_labeled, y, args.n_splits, args.seed, args.logreg_C)
        # final
        pipe = _logreg_pipe(args.logreg_C)
        pipe.fit(X_labeled, y)
        scores = pipe.predict_proba(X_test)[:, 1]
    elif args.classifier == "lgbm":
        oof, scores = _train_lgbm_oof(X_labeled, y, X_test, args.n_splits, args.seed)
    elif args.classifier == "elasticnet":
        from sklearn.linear_model import LogisticRegression
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=args.logreg_C, max_iter=4000, solver="saga",
                                       penalty="elasticnet", l1_ratio=args.l1_ratio)),
        ])
        # OOF
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_labeled, y):
            p2 = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=args.logreg_C, max_iter=4000, solver="saga",
                                                      penalty="elasticnet", l1_ratio=args.l1_ratio))])
            p2.fit(X_labeled[tr], y[tr])
            oof[va] = p2.predict_proba(X_labeled[va])[:, 1]
        pipe.fit(X_labeled, y)
        scores = pipe.predict_proba(X_test)[:, 1]
    elif args.classifier == "ridge":
        from sklearn.linear_model import RidgeClassifierCV
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifierCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0])),
        ])
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_labeled, y):
            from sklearn.linear_model import RidgeClassifierCV
            p2 = Pipeline([("scaler", StandardScaler()),
                           ("clf", RidgeClassifierCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0]))])
            p2.fit(X_labeled[tr], y[tr])
            # ridge classifier outputs decision_function, scale to [0,1] via sigmoid
            oof[va] = 1 / (1 + np.exp(-p2.decision_function(X_labeled[va])))
        pipe.fit(X_labeled, y)
        scores = 1 / (1 + np.exp(-pipe.decision_function(X_test)))
    elif args.classifier == "svm":
        from sklearn.svm import SVC
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_labeled, y):
            p2 = Pipeline([("scaler", StandardScaler()),
                           ("clf", SVC(C=args.logreg_C, probability=True, kernel="rbf"))])
            p2.fit(X_labeled[tr], y[tr])
            oof[va] = p2.predict_proba(X_labeled[va])[:, 1]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(C=args.logreg_C, probability=True, kernel="rbf"))])
        pipe.fit(X_labeled, y)
        scores = pipe.predict_proba(X_test)[:, 1]
    elif args.classifier == "mlp":
        from sklearn.neural_network import MLPClassifier
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_labeled, y):
            p2 = Pipeline([("scaler", StandardScaler()),
                           ("clf", MLPClassifier(hidden_layer_sizes=(args.mlp_hidden,),
                                                  alpha=1.0/args.logreg_C, max_iter=2000,
                                                  random_state=args.seed))])
            p2.fit(X_labeled[tr], y[tr])
            oof[va] = p2.predict_proba(X_labeled[va])[:, 1]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(args.mlp_hidden,),
                                                alpha=1.0/args.logreg_C, max_iter=2000,
                                                random_state=args.seed))])
        pipe.fit(X_labeled, y)
        scores = pipe.predict_proba(X_test)[:, 1]
    elif args.classifier == "ensemble":
        # logreg + lgbm
        oof_lr = run_oof(X_labeled, y, args.n_splits, args.seed, args.logreg_C)
        pipe = _logreg_pipe(args.logreg_C)
        pipe.fit(X_labeled, y)
        scores_lr = pipe.predict_proba(X_test)[:, 1]
        oof_lg, scores_lg = _train_lgbm_oof(X_labeled, y, X_test, args.n_splits, args.seed)
        w_lr, w_lg = [float(x) for x in args.ensemble_weights.split(",")]
        # Convert to ranks for fair averaging
        from scipy.stats import rankdata
        oof = w_lr * (rankdata(oof_lr) / len(oof_lr)) + w_lg * (rankdata(oof_lg) / len(oof_lg))
        scores = w_lr * (rankdata(scores_lr) / len(scores_lr)) + w_lg * (rankdata(scores_lg) / len(scores_lg))
        oof_lr_tpr = tpr_at_fpr(oof_lr.tolist(), y.tolist(), 0.01)
        oof_lg_tpr = tpr_at_fpr(oof_lg.tolist(), y.tolist(), 0.01)
        print(f"  OOF logreg TPR={oof_lr_tpr:.4f}  lgbm TPR={oof_lg_tpr:.4f}")
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")

    oof_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    lo, hi = bootstrap_ci(oof, y)
    print(f"OOF TPR@1%FPR: {oof_tpr:.4f}  CI(5/95): [{lo:.4f}, {hi:.4f}]")
    pct = np.percentile(oof, [5, 25, 50, 75, 95])
    print(f"OOF pct: {[f'{v:.3f}' for v in pct]}")

    # Per watermark type breakdown
    if "watermark_type" in all_labeled.columns:
        for wt in all_labeled["watermark_type"].dropna().unique():
            mask = all_labeled["watermark_type"] == wt
            if mask.sum() > 5:
                t = tpr_at_fpr(oof[mask].tolist(), y[mask].tolist(), 0.01)
                print(f"  TPR@1%FPR [{wt}]: {t:.4f}")

    pct_t = np.percentile(scores, [5, 25, 50, 75, 95])
    print(f"Test pct: {[f'{v:.3f}' for v in pct_t]} (mean={scores.mean():.3f})")

    # Build submission
    if "id" in test_df.columns:
        ids = test_df["id"].tolist()
    else:
        ids = list(range(1, len(test_df) + 1))

    sub = pd.DataFrame({"id": ids[:args.n_rows], "score": np.clip(scores[:args.n_rows], 0.0, 1.0)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(sub)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
