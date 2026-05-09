from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import main as task3_main
from cv_utils import bootstrap_tpr_ci, make_stratified_folds, tpr_at_fpr


@dataclass
class Variant:
    name: str
    use_branch_d: bool
    use_binoculars: bool
    use_bigram: bool = False
    bc_extra_tokenizers: str = ""


def _mutate_for_truncation(df: pd.DataFrame, max_words: int | None) -> pd.DataFrame:
    if max_words is None:
        return df
    out = df.copy()
    out["text"] = out["text"].astype(str).map(lambda x: " ".join(x.split()[:max_words]))
    return out


def _length_slice_report(scores: np.ndarray, labels: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    q = np.quantile(lengths, [0.25, 0.5, 0.75])
    bins = {
        "q1": lengths <= q[0],
        "q2": (lengths > q[0]) & (lengths <= q[1]),
        "q3": (lengths > q[1]) & (lengths <= q[2]),
        "q4": lengths > q[2],
    }
    out: dict[str, float] = {}
    for name, mask in bins.items():
        if int(mask.sum()) < 8:
            out[f"tpr_{name}"] = 0.0
            continue
        out[f"tpr_{name}"] = tpr_at_fpr(scores[mask], labels[mask], target_fpr=0.01)
    return out


def _run_variant(base_args: argparse.Namespace, variant: Variant, trunc_words: int | None) -> dict:
    args = argparse.Namespace(**vars(base_args))
    args.use_branch_d = variant.use_branch_d
    args.use_binoculars = variant.use_binoculars
    args.use_bigram = variant.use_bigram
    args.bc_extra_tokenizers = variant.bc_extra_tokenizers

    train_df, val_df = task3_main._load_train_val(args)
    train_df = _mutate_for_truncation(train_df, trunc_words)
    val_df = _mutate_for_truncation(val_df, trunc_words)
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    y = all_df["label"].astype(int).to_numpy()
    lengths = all_df["text"].astype(str).map(lambda x: len(x.split())).to_numpy()

    branch_a, branch_bc, branch_bigram, branch_d, binoculars = task3_main._build_extractors(args)
    branch_bc.fit(all_df["text"].tolist(), y.tolist())
    if branch_bigram is not None:
        branch_bigram.fit(all_df["text"].tolist(), y.tolist())
    x_all = task3_main._extract_features(all_df, branch_a, branch_bc, branch_bigram, branch_d, binoculars)

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
        fold_scores.append(tpr_at_fpr(pred, y[va_idx], target_fpr=0.01))

    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(oof.reshape(-1, 1), y)
    oof_cal = cal.predict_proba(oof.reshape(-1, 1))[:, 1]
    q5, q50, q95 = bootstrap_tpr_ci(oof_cal, y, target_fpr=0.01, n_boot=args.n_boot, seed=args.seed)

    report = {
        "variant": variant.name,
        "trunc_words": trunc_words,
        "use_branch_d": variant.use_branch_d,
        "use_binoculars": variant.use_binoculars,
        "fold_tpr": [float(v) for v in fold_scores],
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_tpr_cal": float(tpr_at_fpr(oof_cal, y, target_fpr=0.01)),
        "bootstrap_tpr_q5": float(q5),
        "bootstrap_tpr_q50": float(q50),
        "bootstrap_tpr_q95": float(q95),
    }
    report.update(_length_slice_report(oof_cal, y, lengths))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3 experiments and ablations")
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
    p.add_argument("--results-path", default="code/attacks/task3/results/ablation_report.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=300)
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
    p.add_argument("--bc-extra-tokenizers", default="")
    p.add_argument("--use-bigram", action="store_true")
    p.add_argument("--d-model", default="all-MiniLM-L6-v2")
    p.add_argument("--binoculars-observer", default="gpt2")
    p.add_argument("--binoculars-performer", default="gpt2-medium")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = [
        Variant("A+BC+Bigram", use_branch_d=False, use_binoculars=False, use_bigram=True),
        Variant("A+BC+Bigram+OPT", use_branch_d=False, use_binoculars=False,
                use_bigram=True, bc_extra_tokenizers="facebook/opt-1.3b"),
        Variant("A+BC+Bigram+D", use_branch_d=True, use_binoculars=False, use_bigram=True),
        Variant("FULL+Bigram+OPT", use_branch_d=True, use_binoculars=True,
                use_bigram=True, bc_extra_tokenizers="facebook/opt-1.3b"),
    ]
    trunc_set = [None]

    all_reports: list[dict] = []
    for trunc in trunc_set:
        for v in variants:
            print(f"[run] variant={v.name} trunc_words={trunc}")
            all_reports.append(_run_variant(args, v, trunc_words=trunc))

    out = Path(args.results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
