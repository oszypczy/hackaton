#!/usr/bin/env python3
"""Task 3 — Leak-free stacking variant (no branch_bc).

Hypothesis: branch_bc (UnigramGreenList) is fitted on train labels — overfits OOF
but doesn't generalize to test (other watermark schemes). Stacking models all using
bc features amplifies same overfit. This variant excludes label-leaky features and
focuses on PPL/cross-LM signals (the source of cross_lm_best 0.284 leaderboard).
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
    p.add_argument("--out-prefix", type=str, default="submission_v2")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory to write CSVs (default: SUBMISSIONS_DIR)")
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


def _oof_logreg(X: np.ndarray, y: np.ndarray, C: float, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = _make_logreg(C)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    final = _make_logreg(C)
    final.fit(X, y)
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
    print(f"Loading features from: {cache}")

    # LABEL-FREE feature sources only:
    feat_a       = _load_pkl(cache, "a")              # GPT-2 small lp + GLTR + ngram diversity
    feat_bino    = _load_pkl(cache, "bino")           # GPT-2 + GPT-2 medium PPL ratio
    feat_bino_s  = _load_pkl(cache, "bino_strong")    # Pythia-1.4b + 2.8b
    feat_bino_xl = _load_pkl(cache, "bino_xl")        # Pythia-2.8b + 6.9b
    feat_olmo7b  = _load_pkl(cache, "olmo_7b")        # OLMo-2-7B-Instruct PPL
    feat_olmo13b = _load_pkl(cache, "olmo_13b")
    feat_olmo1b  = _load_pkl(cache, "multi_lm")       # multi_lm includes OLMo-1B
    feat_judge   = _load_pkl(cache, "lm_judge")
    feat_judge7b = _load_pkl(cache, "judge_olmo7b")
    feat_better_liu = _load_pkl(cache, "better_liu")
    feat_fdgpt   = _load_pkl(cache, "fdgpt")
    feat_d       = _load_pkl(cache, "d")               # sentence-transformer adj cosine

    have = {n: f is not None for n, f in [
        ("a", feat_a), ("bino", feat_bino), ("bino_s", feat_bino_s),
        ("bino_xl", feat_bino_xl), ("olmo7b", feat_olmo7b), ("olmo13b", feat_olmo13b),
        ("olmo1b", feat_olmo1b), ("judge", feat_judge), ("judge7b", feat_judge7b),
        ("better_liu", feat_better_liu), ("fdgpt", feat_fdgpt), ("d", feat_d),
    ]}
    print("  available:", {k: v for k, v in have.items() if v})

    # Concatenate all label-free features
    parts = [f for f in [feat_a, feat_bino, feat_bino_s, feat_bino_xl,
                          feat_olmo7b, feat_olmo13b, feat_olmo1b, feat_judge,
                          feat_judge7b, feat_better_liu, feat_fdgpt, feat_d] if f is not None]
    full = pd.concat(parts, axis=1).fillna(0.0)
    print(f"  raw feature count: {full.shape[1]}")

    # Build the WINNING cross-LM v1 derived features (these are the leaderboard 0.284 signal)
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
    # NEW v3 derivations: more pairs + interactions (still no labels used)
    if "olmo13b_lp_mean" in full and "olmo7b_lp_mean" in full:
        derived["cross_olmo13b_vs_7b_lp"] = full["olmo13b_lp_mean"] - full["olmo7b_lp_mean"]
    if "olmo13b_lp_mean" in full and "lp_per" in full:
        derived["cross_olmo13b_vs_gpt2med_lp"] = full["olmo13b_lp_mean"] - full["lp_per"]
    if "olmo7b_lp_mean" in full and "bino_strong_lp_obs" in full:
        derived["cross_olmo7b_vs_pythia14b_lp"] = full["olmo7b_lp_mean"] - full["bino_strong_lp_obs"]
    if "olmo7b_lp_mean" in full:
        derived["olmo7b_lp_mean_sq"] = full["olmo7b_lp_mean"] ** 2
    if "olmo7b_lp_mean" in full and "lp_mean" in full:
        derived["cross_olmo7b_vs_gpt2_lp"] = full["olmo7b_lp_mean"] - full["lp_mean"]
    full = pd.concat([full, pd.DataFrame(derived)], axis=1).reset_index(drop=True)
    print(f"  added derived: {len(derived)} (total cols: {full.shape[1]})")

    X_full = full.fillna(0.0).values.astype(np.float32)
    X_lab = X_full[:n_lab]
    X_test = X_full[n_lab:]
    name_to_idx = {n: i for i, n in enumerate(full.columns)}

    # ── VIEWS (each gives a different inductive bias)
    views: dict[str, list[str]] = {}

    # V1: cross-LM derived ONLY (the 11 derived features — pure cross-LM signal)
    views["cross_only"] = list(derived.keys())

    # V2: cross-LM derived + olmo7b raw features (winning signal stack)
    if feat_olmo7b is not None:
        views["cross_plus_olmo7b"] = list(derived.keys()) + list(feat_olmo7b.columns)

    # V3: all binoculars + olmo PPL + cross-LM (PPL-only stack)
    ppl_cols = []
    for fb in [feat_bino, feat_bino_s, feat_bino_xl, feat_olmo7b, feat_olmo13b]:
        if fb is not None: ppl_cols += list(fb.columns)
    views["ppl_full"] = ppl_cols + list(derived.keys())

    # V4: branch_a alone (universal LM feats, no labels)
    if feat_a is not None:
        views["a_only"] = list(feat_a.columns)

    # V5: a + cross-LM (universal + winning)
    if feat_a is not None:
        views["a_plus_cross"] = list(feat_a.columns) + list(derived.keys())

    # V6: aux signals only (judge + better_liu + fdgpt + d) — orthogonal heuristics
    aux_cols = []
    for fb in [feat_judge, feat_judge7b, feat_better_liu, feat_fdgpt, feat_d]:
        if fb is not None: aux_cols += list(fb.columns)
    if aux_cols:
        views["aux_only"] = aux_cols

    # V7: ALL label-free features
    views["full_no_leak"] = list(full.columns)

    print(f"\nViews: {[(k, len(v)) for k, v in views.items()]}")

    # ── OOF + test for each view × C
    oof_stack = []
    test_stack = []
    view_names = []

    for name, cols in views.items():
        idx = [name_to_idx[c] for c in cols if c in name_to_idx]
        Xv_lab = X_lab[:, idx]
        Xv_test = X_test[:, idx]

        for C in [0.01, 0.05, 0.5]:
            oof, model = _oof_logreg(Xv_lab, y, C, args.n_splits, args.seed)
            tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  [{name}_C{C}] OOF TPR@1%FPR: {tpr:.4f}  (cols={len(cols)})")
            test_pred = model.predict_proba(Xv_test)[:, 1]
            oof_stack.append(oof)
            test_stack.append(test_pred)
            view_names.append(f"{name}_C{C}")

    OOF = np.column_stack(oof_stack)
    TEST = np.column_stack(test_stack)
    print(f"\nStack shape: OOF={OOF.shape} TEST={TEST.shape}")

    # ── Save best single base model as a candidate
    base_tprs = np.array([tpr_at_fpr(OOF[:, i].tolist(), y.tolist(), 0.01) for i in range(OOF.shape[1])])
    best_i = int(np.argmax(base_tprs))
    print(f"\nBest single base: {view_names[best_i]} OOF={base_tprs[best_i]:.4f}")

    out1 = out_dir / f"{args.out_prefix}_best_base.csv"
    pd.DataFrame({"id": test_df["id"].tolist(),
                  "score": np.clip(TEST[:, best_i], 0.001, 0.999)}).to_csv(out1, index=False)
    print(f"Saved: {out1}")

    # ── Meta LogReg (sweep C, pick best)
    best_meta_C = None; best_meta_tpr = -1
    for meta_C in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + 7)
        meta_oof = np.zeros(len(y))
        for tr, va in skf.split(OOF, y):
            m = _make_logreg(meta_C)
            m.fit(OOF[tr], y[tr])
            meta_oof[va] = m.predict_proba(OOF[va])[:, 1]
        t = tpr_at_fpr(meta_oof.tolist(), y.tolist(), 0.01)
        print(f"[META C={meta_C}] OOF TPR@1%FPR: {t:.4f}")
        if t > best_meta_tpr:
            best_meta_tpr = t; best_meta_C = meta_C

    print(f"[META BEST] C={best_meta_C} OOF={best_meta_tpr:.4f}")
    meta = _make_logreg(best_meta_C)
    meta.fit(OOF, y)
    test_meta = meta.predict_proba(TEST)[:, 1]
    out2 = out_dir / f"{args.out_prefix}_meta.csv"
    pd.DataFrame({"id": test_df["id"].tolist(),
                  "score": np.clip(test_meta, 0.001, 0.999)}).to_csv(out2, index=False)
    print(f"Saved: {out2}")

    # ── Top-K rank-average (K=3 picks best diversity-quality tradeoff)
    top_idx = np.argsort(-base_tprs)[:5]
    test_ranks = np.column_stack([rankdata(TEST[:, i], method="average") / len(TEST) for i in top_idx])
    weights = base_tprs[top_idx] ** 2
    weights = weights / weights.sum()
    weighted = (test_ranks * weights).sum(axis=1)
    weighted = (weighted - weighted.min()) / (weighted.max() - weighted.min() + 1e-9)
    out3 = out_dir / f"{args.out_prefix}_top5_weighted.csv"
    pd.DataFrame({"id": test_df["id"].tolist(),
                  "score": np.clip(weighted, 0.001, 0.999)}).to_csv(out3, index=False)
    print(f"Saved: {out3}  (top5: {[view_names[i] for i in top_idx]})")

    # ── PURE cross-LM only model (mimic cross_lm v1 leaderboard winner exactly)
    # cross_only view, C=0.5 was OOF 0.1148 (low) but test signal != OOF signal
    cross_view = [name_to_idx[c] for c in views["cross_only"]]
    Xc_lab = X_lab[:, cross_view]
    Xc_test = X_test[:, cross_view]
    pure_model = _make_logreg(0.05)
    pure_model.fit(Xc_lab, y)
    pure_pred = pure_model.predict_proba(Xc_test)[:, 1]
    out4 = out_dir / f"{args.out_prefix}_pure_cross.csv"
    pd.DataFrame({"id": test_df["id"].tolist(),
                  "score": np.clip(pure_pred, 0.001, 0.999)}).to_csv(out4, index=False)
    print(f"Saved: {out4}  (pure 11-cross-LM features, C=0.05)")


if __name__ == "__main__":
    main()
