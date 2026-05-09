#!/usr/bin/env python3
"""Task 3: LLM Watermark Detection — main pipeline.

Usage examples:
  # Phase 1 baseline (no Binoculars, no Branch D):
  python main.py --phase 1

  # Phase 2 full (all branches, GPU recommended for binoculars):
  python main.py --phase 2

  # Eval only (metrics on val, no submission file):
  python main.py --phase 2 --eval-only

  # 2400-row fallback if API rejects 2250:
  python main.py --phase 2 --n-rows 2400
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr

TASK_DIR = Path(__file__).parent
CACHE_DIR = TASK_DIR / "cache"
SUBMISSIONS_DIR = ROOT / "submissions"


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 watermark detector")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Local dir with train.csv/validation.csv/test.csv; HF if omitted")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "task3_submission.csv")
    p.add_argument("--phase", type=int, default=2, choices=[1, 2],
                   help="1=baseline (no binoculars/branch-d), 2=full")
    p.add_argument("--skip-binoculars", action="store_true")
    p.add_argument("--skip-branch-d", action="store_true")
    p.add_argument("--n-rows", type=int, default=2250, choices=[2250, 2400],
                   help="Submission row count (2400 fallback if API rejects 2250)")
    p.add_argument("--eval-only", action="store_true",
                   help="Compute val metrics only; skip test inference and CSV")
    p.add_argument("--force-extract", action="store_true",
                   help="Re-extract features even if cache exists")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> pd.DataFrame:
    import json
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_splits(data_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_dir and data_dir.exists():
        # 5-file JSONL layout from Dataset.zip
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
        available = list(ds.keys())
        print(f"  HF splits: {available}")

        # 5-split layout: train_clean / train_wm / valid_clean / valid_wm / test
        if "train_clean" in available:
            train_clean = ds["train_clean"].to_pandas(); train_clean["label"] = 0
            train_wm = ds["train_wm"].to_pandas();       train_wm["label"] = 1
            valid_clean = ds["valid_clean"].to_pandas(); valid_clean["label"] = 0
            valid_wm = ds["valid_wm"].to_pandas();       valid_wm["label"] = 1
            train = pd.concat([train_clean, train_wm], ignore_index=True)
            val = pd.concat([valid_clean, valid_wm], ignore_index=True)
            test = ds["test"].to_pandas()
        elif "train" in available and "validation" in available:
            train = ds["train"].to_pandas()
            val = ds["validation"].to_pandas()
            test = ds["test"].to_pandas()
        else:
            raise ValueError(f"Unrecognised HF split layout: {available}")

    # Normalize text column name
    for df in (train, val, test):
        if "text" not in df.columns:
            candidates = [c for c in df.columns if df[c].dtype == object and c != "label"]
            if candidates:
                df.rename(columns={candidates[0]: "text"}, inplace=True)

    # Add sequential id to test if missing
    if "id" not in test.columns:
        test["id"] = range(1, len(test) + 1)

    return train, val, test


# ── Feature extraction with caching ──────────────────────────────────────────

def extract_cached(
    name: str,
    texts: list[str],
    extract_fn,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = cache_dir / f"features_{name}.pkl"
    if not force and cache_path.exists():
        print(f"  [cache] Loading {name} features ({cache_path.name})")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  [extract] {name} ({len(texts)} texts)...")
    rows = []
    for t in tqdm(texts, desc=name, ncols=80):
        try:
            rows.append(extract_fn(str(t)))
        except Exception as e:
            print(f"  [warn] {name} failed on text: {e}")
            rows.append({})

    df = pd.DataFrame(rows).fillna(0.0)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"  [cache] Saved {name} features → {cache_path.name} ({df.shape})")
    return df


# ── LightGBM training ─────────────────────────────────────────────────────────

def train_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "n_jobs": -1,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def train_lgbm_fixed(X: np.ndarray, y: np.ndarray, n_rounds: int):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=n_rounds)


# ── OOF + calibration ─────────────────────────────────────────────────────────

def run_oof(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, int]:
    from cv_utils import run_oof as _run_oof
    return _run_oof(X, y, train_lgbm, n_splits=n_splits, seed=seed)


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
    from cv_utils import bootstrap_tpr_ci
    return bootstrap_tpr_ci(scores, labels)


# ── Submission ─────────────────────────────────────────────────────────────────

def validate_submission(df: pd.DataFrame, n_rows: int) -> None:
    assert len(df) == n_rows, f"Expected {n_rows} rows, got {len(df)}"
    assert list(df.columns) == ["id", "score"], f"Wrong columns: {df.columns.tolist()}"
    assert df["id"].nunique() == len(df), "Duplicate IDs detected"
    assert df["score"].between(0.0, 1.0).all(), f"Scores outside [0,1]: {df['score'].describe()}"
    assert df["score"].notna().all(), "NaN scores found"


def build_submission(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    n_rows: int,
) -> pd.DataFrame:
    # Determine IDs — use existing column or generate 1-indexed
    if "id" in test_df.columns:
        ids = test_df["id"].tolist()
    else:
        ids = list(range(1, len(test_df) + 1))

    if len(ids) < n_rows:
        # 2400 fallback: pad with 0.5 (neutral uninformative)
        extra = n_rows - len(ids)
        last_id = ids[-1] if ids else 0
        ids += list(range(int(last_id) + 1, int(last_id) + 1 + extra))
        scores = np.concatenate([scores, np.full(extra, 0.5)])
    else:
        ids = ids[:n_rows]
        scores = scores[:n_rows]

    return pd.DataFrame({"id": ids, "score": np.clip(scores, 0.0, 1.0)})


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data loading
    print("Loading dataset...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    all_labeled = pd.concat([train_df, val_df], ignore_index=True)
    all_texts_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_texts = all_texts_df["text"].tolist()
    n_labeled = len(all_labeled)

    y_labeled = all_labeled["label"].astype(int).values
    print(f"  labeled: {n_labeled}  positive rate: {y_labeled.mean():.3f}")

    # ── 2. Feature extraction
    from features import branch_a, branch_bc, branch_d, binoculars

    print("\nExtracting features...")

    fa = extract_cached("a", all_texts, branch_a.extract, args.cache_dir, args.force_extract)

    # Fit Unigram green list on TRAIN labels only (avoid val leakage)
    gl_cache = args.cache_dir / "green_list.pkl"
    if not args.force_extract and gl_cache.exists():
        from transformers import AutoTokenizer
        gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
        with open(gl_cache, "rb") as f:
            gl = pickle.load(f)
        print("  [cache] Loaded green_list.pkl")
    else:
        from transformers import AutoTokenizer
        gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
        gl = branch_bc.UnigramGreenList()
        gl.fit(train_df["text"].tolist(), train_df["label"].tolist(), gpt2_tok)
        with open(gl_cache, "wb") as f:
            pickle.dump(gl, f)
        print("  [extract] Fitted UnigramGreenList on training split")

    fb = extract_cached(
        "bc", all_texts, lambda t: branch_bc.extract(t, gl, gpt2_tok), args.cache_dir, args.force_extract
    )

    use_bino = args.phase >= 2 and not args.skip_binoculars
    fb_bino = (
        extract_cached("bino", all_texts, binoculars.extract, args.cache_dir, args.force_extract)
        if use_bino
        else None
    )

    use_d = args.phase >= 2 and not args.skip_branch_d
    fd = (
        extract_cached("d", all_texts, branch_d.extract, args.cache_dir, args.force_extract)
        if use_d
        else None
    )

    # ── 3. Build feature matrix
    parts = [fa.reset_index(drop=True), fb.reset_index(drop=True)]
    if fb_bino is not None:
        parts.append(fb_bino.reset_index(drop=True))
    if fd is not None:
        parts.append(fd.reset_index(drop=True))

    X_full = pd.concat(parts, axis=1).fillna(0.0).values.astype(np.float32)
    X_labeled = X_full[:n_labeled]
    X_test = X_full[n_labeled:]

    print(f"\nFeature matrix: {X_labeled.shape} labeled + {X_test.shape} test")
    print(f"Features: {pd.concat(parts, axis=1).columns.tolist()}")

    # ── 4. 5-fold OOF
    print(f"\nRunning {args.n_splits}-fold OOF...")
    oof, mean_best_iter = run_oof(X_labeled, y_labeled, args.n_splits, args.seed)

    oof_tpr = tpr_at_fpr(oof.tolist(), y_labeled.tolist(), 0.01)
    ci = bootstrap_ci(oof, y_labeled)
    print(f"OOF TPR@1%FPR: {oof_tpr:.4f}  CI(5/95): [{ci[0]:.4f}, {ci[2]:.4f}]")
    print(f"Mean best iteration from OOF folds: {mean_best_iter}")

    # Per-subtype breakdown if column exists
    if "watermark_type" in all_labeled.columns:
        for wt in all_labeled["watermark_type"].dropna().unique():
            mask = all_labeled["watermark_type"] == wt
            if mask.sum() > 5:
                t = tpr_at_fpr(oof[mask].tolist(), y_labeled[mask].tolist(), 0.01)
                print(f"  TPR@1%FPR [{wt}]: {t:.4f}")

    # ── 5. Isotonic calibration
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y_labeled)
    cal_oof = calibrator.transform(oof)
    cal_tpr = tpr_at_fpr(cal_oof.tolist(), y_labeled.tolist(), 0.01)
    print(f"Calibrated OOF TPR@1%FPR: {cal_tpr:.4f}")

    if args.eval_only:
        print("\n--eval-only: done.")
        return

    # ── 6. Final model on all labeled data (fixed rounds from OOF)
    print(f"\nTraining final model (n_rounds={mean_best_iter}) on all {n_labeled} labeled samples...")
    final_model = train_lgbm_fixed(X_labeled, y_labeled, mean_best_iter)

    # ── 7. Test inference
    raw = final_model.predict(X_test)
    cal = np.clip(calibrator.transform(raw), 0.0, 1.0)

    print(f"Test score distribution: min={cal.min():.4f} mean={cal.mean():.4f} max={cal.max():.4f}")

    # ── 8. Build and validate submission
    sub = build_submission(test_df, cal, args.n_rows)
    validate_submission(sub, args.n_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
