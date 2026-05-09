from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cv_utils import bootstrap_tpr_ci, make_stratified_folds, tpr_at_fpr
from features import (
    BinocularsConfig,
    BinocularsExtractor,
    BranchAConfig,
    BranchAExtractor,
    BranchBCConfig,
    BranchBCExtractor,
    BranchBigramConfig,
    BranchBigramExtractor,
    BranchDConfig,
    BranchDExtractor,
)


@dataclass
class Task3Artifacts:
    feature_columns: list[str]
    lgb_model: lgb.Booster
    calibrator: IsotonicRegression | LogisticRegression | None
    bc_extractor: BranchBCExtractor
    bigram_extractor: BranchBigramExtractor | None = None


def _read_split(path: Path, require_label: bool) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in {path}")
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    if require_label and "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {path}")
    return df


def _read_hf_split(dataset_name: str, split: str, require_label: bool) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in HF split {split}")
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    if require_label and "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in HF split {split}")
    return df


def _read_zip_split(zip_path: Path, split: str, require_label: bool) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
    split = split.lower()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        candidates = [n for n in names if split in Path(n).stem.lower()]
        if not candidates:
            raise ValueError(f"Could not find split '{split}' in {zip_path}")
        pref_ext = [".csv", ".jsonl", ".parquet"]
        chosen = None
        for ext in pref_ext:
            for c in candidates:
                if c.lower().endswith(ext):
                    chosen = c
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = candidates[0]
        raw = zf.read(chosen)
    lower = chosen.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    elif lower.endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        raise ValueError(f"Unsupported split format in zip: {chosen}")
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in zip split file: {chosen}")
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    if require_label and "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in zip split file: {chosen}")
    return df


def _read_zip_member_df(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    raw = zf.read(member_name)
    lower = member_name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if lower.endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if lower.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    raise ValueError(f"Unsupported split format in zip: {member_name}")


def _ensure_text_id(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column in dataset")
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(1, len(df) + 1)
    return df


def _read_zip_train_val_from_parts(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        by_stem = {Path(n).stem.lower(): n for n in names}
        required = ["train_clean", "train_wm", "valid_clean", "valid_wm"]
        if not all(key in by_stem for key in required):
            raise ValueError("ZIP does not contain split-part files train_clean/train_wm/valid_clean/valid_wm")

        train_clean = _ensure_text_id(_read_zip_member_df(zf, by_stem["train_clean"]))
        train_wm = _ensure_text_id(_read_zip_member_df(zf, by_stem["train_wm"]))
        valid_clean = _ensure_text_id(_read_zip_member_df(zf, by_stem["valid_clean"]))
        valid_wm = _ensure_text_id(_read_zip_member_df(zf, by_stem["valid_wm"]))

    train_clean = train_clean.copy()
    train_wm = train_wm.copy()
    valid_clean = valid_clean.copy()
    valid_wm = valid_wm.copy()
    train_clean["label"] = 0
    train_wm["label"] = 1
    valid_clean["label"] = 0
    valid_wm["label"] = 1

    train_df = pd.concat([train_clean, train_wm], ignore_index=True)
    val_df = pd.concat([valid_clean, valid_wm], ignore_index=True)
    if "id" not in train_df.columns or train_df["id"].duplicated().any():
        train_df["id"] = np.arange(1, len(train_df) + 1)
    if "id" not in val_df.columns or val_df["id"].duplicated().any():
        val_df["id"] = np.arange(1, len(val_df) + 1)
    return train_df, val_df


def _load_train_val(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.data_source == "zip":
        try:
            train_df = _read_zip_split(Path(args.zip_path), args.zip_train_split, require_label=True)
            val_df = _read_zip_split(Path(args.zip_path), args.zip_val_split, require_label=True)
            return train_df, val_df
        except ValueError:
            return _read_zip_train_val_from_parts(Path(args.zip_path))
    if args.data_source == "hf":
        train_df = _read_hf_split(args.hf_dataset, args.hf_train_split, require_label=True)
        val_df = _read_hf_split(args.hf_dataset, args.hf_val_split, require_label=True)
        return train_df, val_df
    train_df = _read_split(Path(args.train_path), require_label=True)
    val_df = _read_split(Path(args.val_path), require_label=True)
    return train_df, val_df


def _load_test(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "zip":
        return _ensure_text_id(_read_zip_split(Path(args.zip_path), args.zip_test_split, require_label=False))
    if args.data_source == "hf":
        return _read_hf_split(args.hf_dataset, args.hf_test_split, require_label=False)
    return _read_split(Path(args.test_path), require_label=False)


def _build_extractors(args: argparse.Namespace):
    branch_a = BranchAExtractor(
        BranchAConfig(model_name=args.a_model, max_length=args.max_length, device=args.device)
    )
    extra_tokenizers = [t.strip() for t in args.bc_extra_tokenizers.split(",") if t.strip()]
    branch_bc = BranchBCExtractor(
        BranchBCConfig(
            tokenizer_name=args.bc_tokenizer,
            extra_tokenizer_names=extra_tokenizers,
            max_length=args.max_length,
        )
    )
    branch_bigram = (
        BranchBigramExtractor(
            BranchBigramConfig(
                tokenizer_name=args.bc_tokenizer,
                max_length=args.max_length,
            )
        )
        if args.use_bigram
        else None
    )
    branch_d = (
        BranchDExtractor(BranchDConfig(model_name=args.d_model, device=args.device))
        if args.use_branch_d
        else None
    )
    binoculars = (
        BinocularsExtractor(
            BinocularsConfig(
                observer_model=args.binoculars_observer,
                performer_model=args.binoculars_performer,
                max_length=args.max_length,
                device=args.device,
            )
        )
        if args.use_binoculars
        else None
    )
    return branch_a, branch_bc, branch_bigram, branch_d, binoculars


def _extract_features(
    df: pd.DataFrame,
    branch_a: BranchAExtractor,
    branch_bc: BranchBCExtractor,
    branch_bigram: BranchBigramExtractor | None,
    branch_d: BranchDExtractor | None,
    binoculars: BinocularsExtractor | None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for text in df["text"].tolist():
        feats = {}
        feats.update(branch_a.featurize(text))
        feats.update(branch_bc.featurize(text))
        if branch_bigram is not None:
            feats.update(branch_bigram.featurize(text))
        if branch_d is not None:
            feats.update(branch_d.featurize(text))
        if binoculars is not None:
            feats.update(binoculars.featurize(text))
        rows.append(feats)
    return pd.DataFrame(rows).fillna(0.0)


def _validate_submission(df: pd.DataFrame, expected_rows: int = 2250) -> None:
    if list(df.columns) != ["id", "score"]:
        raise ValueError("Submission columns must be exactly: id,score")
    if len(df) != expected_rows:
        raise ValueError(f"Submission rows={len(df)}, expected={expected_rows}")
    if df["id"].duplicated().any():
        raise ValueError("Submission contains duplicated ids")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise ValueError("Submission contains non-finite scores")
    if not ((df["score"] >= 0.0) & (df["score"] <= 1.0)).all():
        raise ValueError("Submission score must be in [0,1]")


def train(args: argparse.Namespace) -> None:
    train_df, val_df = _load_train_val(args)
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    y = all_df["label"].astype(int).to_numpy()

    branch_a, branch_bc, branch_bigram, branch_d, binoculars = _build_extractors(args)
    branch_bc.fit(all_df["text"].tolist(), y.tolist())
    if branch_bigram is not None:
        branch_bigram.fit(all_df["text"].tolist(), y.tolist())
    x_all = _extract_features(all_df, branch_a, branch_bc, branch_bigram, branch_d, binoculars)
    feature_cols = x_all.columns.tolist()

    folds = make_stratified_folds(y, n_splits=args.n_splits, seed=args.seed)
    oof = np.zeros(len(all_df), dtype=float)
    fold_scores: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(x_all.iloc[tr_idx], y[tr_idx])
        dva = lgb.Dataset(x_all.iloc[va_idx], y[va_idx], reference=dtr)
        model = lgb.train(
            {
                "objective": "binary",
                "learning_rate": args.lr,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "lambda_l2": args.lambda_l2,
                "verbosity": -1,
                "seed": args.seed + fold_idx,
            },
            dtr,
            num_boost_round=args.num_boost_round,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        pred = model.predict(x_all.iloc[va_idx])
        oof[va_idx] = pred
        fold_tpr = tpr_at_fpr(pred, y[va_idx], target_fpr=0.01)
        fold_scores.append(fold_tpr)

    # Platt scaling: LR on 1D raw score → smooth P(watermarked) in (0,1).
    # IsotonicRegression produces "step" functions that collapse many predictions
    # to exact 0.0/1.0, destroying the ranking signal needed for TPR@1%FPR.
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(oof.reshape(-1, 1), y)
    oof_cal = calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]

    mean_tpr = float(np.mean(fold_scores))
    std_tpr = float(np.std(fold_scores))
    oof_tpr = tpr_at_fpr(oof_cal, y, target_fpr=0.01)
    oof_uncal_tpr = tpr_at_fpr(oof, y, target_fpr=0.01)
    ci5, ci50, ci95 = bootstrap_tpr_ci(oof_cal, y, target_fpr=0.01, n_boot=args.n_boot, seed=args.seed)

    print(f"Fold TPR@1%FPR: {[round(s, 4) for s in fold_scores]}")
    print(f"CV mean/std: {mean_tpr:.4f}/{std_tpr:.4f}")
    print(f"OOF raw TPR@1%FPR: {oof_uncal_tpr:.4f}")
    print(f"OOF Platt-calibrated TPR@1%FPR: {oof_tpr:.4f}")
    print(f"Bootstrap TPR@1%FPR [5/50/95]: {ci5:.4f}/{ci50:.4f}/{ci95:.4f}")

    final_model = lgb.train(
        {
            "objective": "binary",
            "learning_rate": args.lr,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "lambda_l2": args.lambda_l2,
            "verbosity": -1,
            "seed": args.seed,
        },
        lgb.Dataset(x_all, y),
        num_boost_round=max(100, int(args.num_boost_round * 0.6)),
    )

    artifacts = Task3Artifacts(
        feature_columns=feature_cols,
        lgb_model=final_model,
        calibrator=calibrator,
        bc_extractor=branch_bc,
        bigram_extractor=branch_bigram,
    )
    out = Path(args.artifacts_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved artifacts: {out}")


def infer(args: argparse.Namespace) -> None:
    test_df = _load_test(args)
    with Path(args.artifacts_path).open("rb") as f:
        artifacts: Task3Artifacts = pickle.load(f)

    branch_a, branch_bc_fresh, branch_bigram_fresh, branch_d, binoculars = _build_extractors(args)
    branch_bc = artifacts.bc_extractor if artifacts.bc_extractor.soft_green_weights else branch_bc_fresh
    branch_bigram = (
        artifacts.bigram_extractor
        if (artifacts.bigram_extractor is not None and artifacts.bigram_extractor.soft_bigram_weights)
        else branch_bigram_fresh
    )
    x_test = _extract_features(test_df, branch_a, branch_bc, branch_bigram, branch_d, binoculars)
    for col in artifacts.feature_columns:
        if col not in x_test.columns:
            x_test[col] = 0.0
    x_test = x_test[artifacts.feature_columns]

    raw = artifacts.lgb_model.predict(x_test)
    if artifacts.calibrator is None:
        cal = raw.astype(float)
    elif hasattr(artifacts.calibrator, "predict_proba"):
        # Platt scaling (LogisticRegression)
        cal = artifacts.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        # Legacy IsotonicRegression artifacts
        cal = np.clip(artifacts.calibrator.transform(raw), 0.0, 1.0)
    sub = pd.DataFrame({"id": test_df["id"], "score": cal.astype(float)})
    _validate_submission(sub, expected_rows=args.expected_rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Saved submission: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3 watermark detector")
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--data-source", choices=["zip", "hf", "csv"], default="zip")
    p.add_argument("--zip-path", default="Dataset.zip")
    p.add_argument("--zip-train-split", default="train")
    p.add_argument("--zip-val-split", default="valid")
    p.add_argument("--zip-test-split", default="test")
    p.add_argument("--hf-dataset", default="SprintML/llm-watermark-detection")
    p.add_argument("--hf-train-split", default="train")
    p.add_argument("--hf-val-split", default="validation")
    p.add_argument("--hf-test-split", default="test")
    p.add_argument("--train-path", default="data/task3/train.csv")
    p.add_argument("--val-path", default="data/task3/val.csv")
    p.add_argument("--test-path", default="data/task3/test.csv")
    p.add_argument("--artifacts-path", default="code/attacks/task3/cache/model.pkl")
    p.add_argument("--out-csv", default="submissions/task3_watermark_detection.csv")
    p.add_argument("--expected-rows", type=int, default=2250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--num-boost-round", type=int, default=400)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-data-in-leaf", type=int, default=15)
    p.add_argument("--feature-fraction", type=float, default=0.7)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--lambda-l2", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--a-model", default="gpt2")
    p.add_argument("--bc-tokenizer", default="gpt2")
    p.add_argument(
        "--bc-extra-tokenizers",
        default="",
        help="Comma-separated extra tokenizer names for BranchBC multi-tokenizer mode. "
        "E.g. 'facebook/opt-1.3b' (Kirchenbauer original model tokenizer).",
    )
    p.add_argument("--use-bigram", action="store_true", help="Enable BranchBigram (KGW-targeted)")
    p.add_argument("--d-model", default="all-MiniLM-L6-v2")
    p.add_argument("--use-branch-d", action="store_true")
    p.add_argument("--use-binoculars", action="store_true")
    p.add_argument("--binoculars-observer", default="gpt2")
    p.add_argument("--binoculars-performer", default="gpt2-medium")
    return p.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        train(cli_args)
    else:
        infer(cli_args)
