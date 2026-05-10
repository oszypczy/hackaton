#!/usr/bin/env python3
"""SIR-only + SIR+cross_lm + SIR+judge blends.

Trains LogReg on subsets of cached features:
  - SIR alone
  - SIR + cross_lm v1 base features
  - SIR + judge_chat
  - SIR + olmo7b + cross_lm
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read(path):
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def _load(cache_dir, name):
    p = cache_dir / f"features_{name}.pkl"
    if not p.exists():
        print(f"  missing {p}")
        return None
    return pickle.load(open(p, "rb")).reset_index(drop=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_clean = _read(args.data_dir / "train_clean.jsonl"); train_clean["label"] = 0
    train_wm = _read(args.data_dir / "train_wm.jsonl"); train_wm["label"] = 1
    valid_clean = _read(args.data_dir / "valid_clean.jsonl"); valid_clean["label"] = 0
    valid_wm = _read(args.data_dir / "valid_wm.jsonl"); valid_wm["label"] = 1
    train = pd.concat([train_clean, train_wm], ignore_index=True)
    val = pd.concat([valid_clean, valid_wm], ignore_index=True)
    test = _read(args.data_dir / "test.jsonl")
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)
    all_lab = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    n_lab = len(all_lab)
    y = all_lab["label"].astype(int).values

    feat = {
        "sir": _load(args.cache_dir, "sir"),
        "a": _load(args.cache_dir, "a"),
        "olmo_7b": _load(args.cache_dir, "olmo_7b"),
        "olmo_1b": _load(args.cache_dir, "multi_lm"),
        "bino": _load(args.cache_dir, "bino"),
        "bino_strong": _load(args.cache_dir, "bino_strong"),
        "bino_xl": _load(args.cache_dir, "bino_xl"),
        "fdgpt": _load(args.cache_dir, "fdgpt"),
        "judge_chat": _load(args.cache_dir, "judge_chat"),
        "judge_olmo7b": _load(args.cache_dir, "judge_olmo7b"),
        "mistral_7b": _load(args.cache_dir, "mistral_7b"),
    }

    def _build(parts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        dfs = [feat[k] for k in parts if feat.get(k) is not None]
        full = pd.concat(dfs, axis=1).fillna(0.0)
        # cross-LM derivations
        derived = {}
        if "olmo7b_lp_mean" in full and "lp_per" in full:
            derived["x_o7b_gpt2"] = full["olmo7b_lp_mean"] - full["lp_per"]
        if "olmo7b_lp_mean" in full and "bino_xl_lp_obs" in full:
            derived["x_o7b_p28"] = full["olmo7b_lp_mean"] - full["bino_xl_lp_obs"]
        if "olmo7b_lp_mean" in full and "bino_xl_lp_per" in full:
            derived["x_o7b_p69"] = full["olmo7b_lp_mean"] - full["bino_xl_lp_per"]
        if "olmo7b_lp_mean" in full and "olmo_lp_mean" in full:
            derived["x_o7b_o1b"] = full["olmo7b_lp_mean"] - full["olmo_lp_mean"]
        if "bino_strong_lp_obs" in full and "bino_xl_lp_obs" in full:
            derived["x_p14_p28"] = full["bino_strong_lp_obs"] - full["bino_xl_lp_obs"]
        if "olmo7b_ppl" in full and "bino_xl_ppl_obs" in full:
            derived["x_o7b_p_ppl"] = full["olmo7b_ppl"] / (full["bino_xl_ppl_obs"] + 1e-9)
        if "mistral7b_lp_mean" in full and "olmo7b_lp_mean" in full:
            derived["x_m7b_o7b"] = full["mistral7b_lp_mean"] - full["olmo7b_lp_mean"]
        full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
        full = full.replace([np.inf, -np.inf], 0).fillna(0.0)
        X = full.values.astype(np.float32)
        return X[:n_lab], X[n_lab:]

    def _train_predict(X_lab, X_test, C, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(X_lab))
        for tr, va in skf.split(X_lab, y):
            m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=4000))])
            m.fit(X_lab[tr], y[tr])
            oof[va] = m.predict_proba(X_lab[va])[:, 1]
        tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
        f = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=4000))])
        f.fit(X_lab, y)
        pred = f.predict_proba(X_test)[:, 1]
        return tpr, pred

    configs = [
        ("sir_only", ["sir"]),
        ("sir_a", ["sir", "a"]),
        ("sir_clm", ["sir", "a", "olmo_7b", "olmo_1b", "bino", "bino_strong", "bino_xl", "fdgpt"]),
        ("sir_judge_chat", ["sir", "judge_chat"]),
        ("sir_clm_judge", ["sir", "a", "olmo_7b", "olmo_1b", "bino_strong", "bino_xl", "fdgpt", "judge_chat"]),
        ("sir_clm_judge_mistral", ["sir", "a", "olmo_7b", "olmo_1b", "bino_strong", "bino_xl", "fdgpt", "judge_chat", "mistral_7b"]),
        ("sir_double_judge_clm", ["sir", "a", "olmo_7b", "olmo_1b", "bino_strong", "bino_xl", "fdgpt", "judge_chat", "judge_olmo7b"]),
    ]
    Cs = [0.005, 0.01, 0.05]

    results = {}
    for name, parts in configs:
        X_lab, X_test = _build(parts)
        print(f"\n=== {name} (cols={X_lab.shape[1]}) ===")
        best = (-1, None, None)
        for C in Cs:
            tpr, pred = _train_predict(X_lab, X_test, C)
            print(f"  C={C}: OOF TPR={tpr:.4f}")
            if tpr > best[0]:
                best = (tpr, C, pred)
        results[name] = best
        out_path = args.out_dir / f"submission_sir_{name}.csv"
        pd.DataFrame({"id": test["id"].tolist(), "score": np.clip(best[2], 0.001, 0.999)}).to_csv(out_path, index=False)
        print(f"  -> {out_path} (best C={best[1]}, OOF={best[0]:.4f})")

    print("\n=== Summary ===")
    for name, (tpr, C, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:30s} OOF={tpr:.4f}  C={C}")


if __name__ == "__main__":
    main()
