#!/usr/bin/env python3
"""Task 3 — v6: pseudo-labeling self-training on top of v3 winner.

1. Train base model (LR on K=40 leak-free) → OOF + test predictions (= v3 baseline)
2. Pseudo-label top-K most confident WM and bottom-K most confident clean from test
3. Retrain LR on (labeled + pseudo-labeled). Predict full test.
4. Iterate (2-3 rounds with ramping confidence).
5. Output: best_base / iterated_v3 / iterated_meta.
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent
SUBMISSIONS_DIR = ROOT / "submissions"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out-prefix", type=str, default="submission_v6")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K-best", type=int, default=40)
    p.add_argument("--C", type=float, default=0.05)
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
    n_test = len(test_df)
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

    # SelectKBest K
    K = args.K_best
    sel = SelectKBest(mutual_info_classif, k=K)
    sel.fit(X_lab, y)
    kept = sel.get_support()
    Xk_lab = X_lab[:, kept]
    Xk_test = X_test[:, kept]
    print(f"SelectKBest K={K} → {Xk_lab.shape[1]} features")

    # ── Round 0 — baseline LR
    print("\n=== Round 0: baseline LR")
    oof0, model0 = _oof_logreg(Xk_lab, y, args.C, args.n_splits, args.seed)
    t0 = tpr_at_fpr(oof0.tolist(), y.tolist(), 0.01)
    pred0 = model0.predict_proba(Xk_test)[:, 1]
    print(f"  Round 0 OOF TPR={t0:.4f}")

    # Save baseline
    ids = test_df["id"].tolist()
    out0 = out_dir / f"{args.out_prefix}_round0.csv"
    pd.DataFrame({"id": ids, "score": np.clip(pred0, 0.001, 0.999)}).to_csv(out0, index=False)
    print(f"  Saved: {out0}")

    # ── Pseudo-labeling rounds
    cur_pred = pred0
    cur_oof = oof0
    iter_results = []
    for round_i, (top_n, low_thr, high_thr) in enumerate([
        (200, 0.10, 0.90),  # round 1: confident only
        (300, 0.15, 0.85),  # round 2: more
        (400, 0.20, 0.80),  # round 3: most
    ]):
        sorted_idx = np.argsort(cur_pred)
        # Bottom-N most confident clean
        clean_pseudo = sorted_idx[:top_n]
        # Top-N most confident watermarked
        wm_pseudo = sorted_idx[-top_n:]
        # Filter by absolute confidence threshold
        clean_mask = cur_pred[clean_pseudo] < low_thr
        wm_mask = cur_pred[wm_pseudo] > high_thr
        clean_pseudo = clean_pseudo[clean_mask]
        wm_pseudo = wm_pseudo[wm_mask]
        n_pseudo_clean = len(clean_pseudo)
        n_pseudo_wm = len(wm_pseudo)

        if n_pseudo_clean < 30 or n_pseudo_wm < 30:
            print(f"\n=== Round {round_i+1}: too few confident pseudo-labels (clean={n_pseudo_clean} wm={n_pseudo_wm}), stopping")
            break

        # Build augmented training set
        X_aug = np.vstack([Xk_lab, Xk_test[clean_pseudo], Xk_test[wm_pseudo]])
        y_aug = np.concatenate([y, np.zeros(n_pseudo_clean, dtype=int),
                                  np.ones(n_pseudo_wm, dtype=int)])
        sample_weights = np.concatenate([
            np.ones(len(y)),
            0.5 * np.ones(n_pseudo_clean),  # half weight for pseudo
            0.5 * np.ones(n_pseudo_wm),
        ])

        # 5-fold OOF on labeled rows only — pseudo augment training but eval on labeled
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + round_i)
        oof_r = np.zeros(len(y))
        for tr, va in skf.split(Xk_lab, y):
            X_tr_aug = np.vstack([Xk_lab[tr], Xk_test[clean_pseudo], Xk_test[wm_pseudo]])
            y_tr_aug = np.concatenate([y[tr], np.zeros(n_pseudo_clean, dtype=int),
                                         np.ones(n_pseudo_wm, dtype=int)])
            sw = np.concatenate([np.ones(len(tr)), 0.5 * np.ones(n_pseudo_clean),
                                  0.5 * np.ones(n_pseudo_wm)])
            scaler = StandardScaler().fit(X_tr_aug)
            Xs_tr = scaler.transform(X_tr_aug)
            clf = LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs")
            clf.fit(Xs_tr, y_tr_aug, sample_weight=sw)
            oof_r[va] = clf.predict_proba(scaler.transform(Xk_lab[va]))[:, 1]
        t_r = tpr_at_fpr(oof_r.tolist(), y.tolist(), 0.01)

        # Final model on ALL labeled+pseudo
        scaler = StandardScaler().fit(X_aug)
        clf = LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs")
        clf.fit(scaler.transform(X_aug), y_aug, sample_weight=sample_weights)
        pred_r = clf.predict_proba(scaler.transform(Xk_test))[:, 1]

        print(f"\n=== Round {round_i+1}: pseudo_clean={n_pseudo_clean} pseudo_wm={n_pseudo_wm}")
        print(f"  OOF TPR={t_r:.4f}  (Δ={t_r - t0:+.4f})")

        out_r = out_dir / f"{args.out_prefix}_round{round_i+1}.csv"
        pd.DataFrame({"id": ids, "score": np.clip(pred_r, 0.001, 0.999)}).to_csv(out_r, index=False)
        print(f"  Saved: {out_r}")

        iter_results.append((round_i+1, t_r, oof_r, pred_r, n_pseudo_clean, n_pseudo_wm))
        cur_pred = pred_r
        cur_oof = oof_r

    # Pick best round
    if iter_results:
        best = max(iter_results, key=lambda r: r[1])
        print(f"\n=== Best: Round {best[0]} OOF={best[1]:.4f}")
        out_best = out_dir / f"{args.out_prefix}_best.csv"
        pd.DataFrame({"id": ids, "score": np.clip(best[3], 0.001, 0.999)}).to_csv(out_best, index=False)
        print(f"Saved: {out_best}")
    else:
        print("\nNo iter improvements; keeping round 0")
        import shutil
        shutil.copy(out0, out_dir / f"{args.out_prefix}_best.csv")


if __name__ == "__main__":
    main()
