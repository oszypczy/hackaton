#!/usr/bin/env python3
"""Task 3: LLM Watermark Detection — main pipeline.

Usage examples:
  # Phase 1 baseline (branch_a only, logreg):
  python main.py --phase 1 --classifier logreg

  # Phase 2 full without branch_bc (no bimodal collapse):
  python main.py --phase 2 --skip-branch-bc --classifier logreg

  # Phase 2 full, all branches, logreg:
  python main.py --phase 2 --classifier logreg

  # Old LightGBM mode:
  python main.py --phase 2 --classifier lgbm

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
    p.add_argument("--skip-branch-bc", action="store_true",
                   help="Skip branch_bc (green-list features) to avoid bimodal collapse")
    p.add_argument("--use-bigram", action="store_true",
                   help="Add bigram greenlist features (KGW/Kirchenbauer-specific)")
    p.add_argument("--use-strong-bino", action="store_true",
                   help="Add stronger Binoculars features (Pythia-1.4b/2.8b)")
    p.add_argument("--use-xl-bino", action="store_true",
                   help="Add XL Binoculars features (Pythia-2.8b/6.9b, ~20GB GPU)")
    p.add_argument("--use-fdgpt", action="store_true",
                   help="Add Fast-DetectGPT analytical curvature features (Pythia-2.8b)")
    p.add_argument("--use-strong-a", action="store_true",
                   help="Add stronger branch_a features (Pythia-2.8b)")
    p.add_argument("--use-multi-lm", action="store_true",
                   help="Add multi-LM PPL features (OPT-1.3b)")
    p.add_argument("--use-multi-lm-v2", action="store_true",
                   help="Add extended multi-LM PPL (Phi-2/Qwen2/Llama-chat/Mistral-instruct)")
    p.add_argument("--use-lm-judge", action="store_true",
                   help="Add LM-as-judge zero-shot detection features (OLMo-instruct)")
    p.add_argument("--use-olmo7b", action="store_true",
                   help="Add OLMo-2-7B-Instruct PPL features (amplifies OLMo-1B breakthrough)")
    p.add_argument("--use-judge-phi2", action="store_true",
                   help="Add LM-judge under Phi-2 (Microsoft 2.7B instruct)")
    p.add_argument("--use-judge-mistral", action="store_true",
                   help="Add LM-judge under Mistral-7B-Instruct")
    p.add_argument("--use-judge-olmo7b", action="store_true",
                   help="Add LM-judge under OLMo-2-7B-Instruct (best size + best signal combo)")
    p.add_argument("--use-olmo13b", action="store_true",
                   help="Add OLMo-2-13B-Instruct PPL features (next size up from 7B)")
    p.add_argument("--use-judge-olmo13b", action="store_true",
                   help="Add LM-judge under OLMo-2-13B-Instruct (largest OLMo judge)")
    p.add_argument("--use-judge-chat", action="store_true",
                   help="OLMo-7B judge with PROPER chat template + 5 prompts (vs plain text 3)")
    p.add_argument("--use-olmo7b-chunks", action="store_true",
                   help="OLMo-7B PPL per chunk (5 chunks) - local PPL variation features")
    p.add_argument("--use-stylometric", action="store_true",
                   help="Add stylometric features (CPU only, fast)")
    p.add_argument("--use-better-liu", action="store_true",
                   help="Add extended semantic features (Liu/Semantic detector)")
    p.add_argument("--use-roberta", action="store_true",
                   help="Add RoBERTa-base mean-pooled embedding (768 features, requires C<=0.001)")
    p.add_argument("--use-kgw-llama", action="store_true",
                   help="KGW direct detection with Llama-2/Llama-3/Mistral tokenizers (gated)")
    p.add_argument("--roberta-pca-dim", type=int, default=32,
                   help="PCA dim reduction for RoBERTa features (0 = no PCA, raw 768)")
    p.add_argument("--use-kgw", action="store_true",
                   help="Add direct KGW reference detection features (multi-tokenizer)")
    p.add_argument("--use-kgw-v2", action="store_true",
                   help="Add KGW v2 features (extra hash_keys + h=2 multigram, gpt2 only)")
    p.add_argument("--classifier", choices=["lgbm", "logreg"], default="logreg",
                   help="Classifier: logreg (continuous output) or lgbm (default was lgbm)")
    p.add_argument("--logreg-C", type=float, default=0.05,
                   help="LogReg regularization strength (smaller = more regularized)")
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


# ── Classifier helpers ────────────────────────────────────────────────────────

class _LogRegModel:
    """Wraps sklearn Pipeline so .predict(X) returns probabilities."""
    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]


def _make_logreg(C: float = 0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def train_logreg(C: float = 0.05):
    """Returns a train_fn(X_tr, y_tr, X_va, y_va) -> model compatible with run_oof."""
    def _train(X_tr, y_tr, X_va, y_va):
        pipe = _make_logreg(C)
        pipe.fit(X_tr, y_tr)
        return _LogRegModel(pipe)
    return _train


def train_logreg_final(X: np.ndarray, y: np.ndarray, C: float = 0.05) -> _LogRegModel:
    pipe = _make_logreg(C)
    pipe.fit(X, y)
    return _LogRegModel(pipe)


def train_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "n_jobs": -1,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    return lgb.train(
        params,
        dtr,
        num_boost_round=1000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def train_lgbm_fixed(X: np.ndarray, y: np.ndarray, n_rounds: int):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=n_rounds)


# ── OOF ───────────────────────────────────────────────────────────────────────

def run_oof(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int, train_fn) -> tuple[np.ndarray, int]:
    from cv_utils import run_oof as _run_oof
    return _run_oof(X, y, train_fn, n_splits=n_splits, seed=seed)


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
    print(f"  classifier={args.classifier}  skip_branch_bc={args.skip_branch_bc}")

    fa = extract_cached("a", all_texts, branch_a.extract, args.cache_dir, args.force_extract)

    parts = [fa.reset_index(drop=True)]

    if not args.skip_branch_bc:
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
        parts.append(fb.reset_index(drop=True))
    else:
        print("  [skip] branch_bc (--skip-branch-bc set)")

    if args.use_bigram:
        bg_cache = args.cache_dir / "bigram_greenlist.pkl"
        if not args.force_extract and bg_cache.exists():
            from transformers import AutoTokenizer
            gpt2_tok2 = AutoTokenizer.from_pretrained("gpt2")
            with open(bg_cache, "rb") as f:
                bg_gl = pickle.load(f)
            print("  [cache] Loaded bigram_greenlist.pkl")
        else:
            from transformers import AutoTokenizer
            gpt2_tok2 = AutoTokenizer.from_pretrained("gpt2")
            bg_gl = branch_bc.BigramGreenList()
            bg_gl.fit(train_df["text"].tolist(), train_df["label"].tolist(), gpt2_tok2)
            with open(bg_cache, "wb") as f:
                pickle.dump(bg_gl, f)
            print("  [extract] Fitted BigramGreenList on training split")

        fb_bg = extract_cached(
            "bigram", all_texts,
            lambda t: branch_bc.extract_bigram(t, bg_gl, gpt2_tok2),
            args.cache_dir, args.force_extract,
        )
        parts.append(fb_bg.reset_index(drop=True))

    use_bino = args.phase >= 2 and not args.skip_binoculars
    if use_bino:
        fb_bino = extract_cached("bino", all_texts, binoculars.extract, args.cache_dir, args.force_extract)
        parts.append(fb_bino.reset_index(drop=True))

    use_d = args.phase >= 2 and not args.skip_branch_d
    if use_d:
        fd = extract_cached("d", all_texts, branch_d.extract, args.cache_dir, args.force_extract)
        parts.append(fd.reset_index(drop=True))

    if args.use_strong_bino:
        from features import binoculars_strong
        fb_strong = extract_cached("bino_strong", all_texts, binoculars_strong.extract,
                                   args.cache_dir, args.force_extract)
        parts.append(fb_strong.reset_index(drop=True))

    if args.use_xl_bino:
        from features import binoculars_xl
        fb_xl = extract_cached("bino_xl", all_texts, binoculars_xl.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_xl.reset_index(drop=True))

    if args.use_fdgpt:
        from features import fast_detectgpt
        fb_fd = extract_cached("fdgpt", all_texts, fast_detectgpt.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_fd.reset_index(drop=True))

    if args.use_strong_a:
        from features import branch_a_strong
        fb_as = extract_cached("a_strong", all_texts, branch_a_strong.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_as.reset_index(drop=True))

    if args.use_multi_lm:
        from features import multi_lm_ppl
        fb_mlm = extract_cached("multi_lm", all_texts, multi_lm_ppl.extract,
                                args.cache_dir, args.force_extract)
        parts.append(fb_mlm.reset_index(drop=True))

    if args.use_multi_lm_v2:
        from features import multi_lm_v2
        fb_mlm2 = extract_cached("multi_lm_v2", all_texts, multi_lm_v2.extract,
                                 args.cache_dir, args.force_extract)
        parts.append(fb_mlm2.reset_index(drop=True))

    if args.use_lm_judge:
        from features import lm_judge
        fb_judge = extract_cached("lm_judge", all_texts, lm_judge.extract,
                                  args.cache_dir, args.force_extract)
        parts.append(fb_judge.reset_index(drop=True))

    if args.use_olmo7b:
        from features import olmo_7b
        fb_olmo7b = extract_cached("olmo_7b", all_texts, olmo_7b.extract,
                                   args.cache_dir, args.force_extract)
        parts.append(fb_olmo7b.reset_index(drop=True))

    if args.use_judge_phi2:
        from features import lm_judge_phi2
        fb_jp = extract_cached("judge_phi2", all_texts, lm_judge_phi2.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_jp.reset_index(drop=True))

    if args.use_judge_mistral:
        from features import lm_judge_mistral
        fb_jm = extract_cached("judge_mistral", all_texts, lm_judge_mistral.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_jm.reset_index(drop=True))

    if args.use_judge_olmo7b:
        from features import lm_judge_olmo7b
        fb_jo7 = extract_cached("judge_olmo7b", all_texts, lm_judge_olmo7b.extract,
                                args.cache_dir, args.force_extract)
        parts.append(fb_jo7.reset_index(drop=True))

    if args.use_olmo13b:
        from features import olmo_13b
        fb_o13 = extract_cached("olmo_13b", all_texts, olmo_13b.extract,
                                args.cache_dir, args.force_extract)
        parts.append(fb_o13.reset_index(drop=True))

    if args.use_judge_olmo13b:
        from features import lm_judge_olmo13b
        fb_jo13 = extract_cached("judge_olmo13b", all_texts, lm_judge_olmo13b.extract,
                                 args.cache_dir, args.force_extract)
        parts.append(fb_jo13.reset_index(drop=True))

    if args.use_judge_chat:
        from features import judge_olmo7b_chat
        fb_chat = extract_cached("judge_chat", all_texts, judge_olmo7b_chat.extract,
                                 args.cache_dir, args.force_extract)
        parts.append(fb_chat.reset_index(drop=True))

    if args.use_olmo7b_chunks:
        from features import olmo_7b_chunks
        fb_chunks = extract_cached("olmo7b_chunks", all_texts, olmo_7b_chunks.extract,
                                   args.cache_dir, args.force_extract)
        parts.append(fb_chunks.reset_index(drop=True))

    if args.use_stylometric:
        from features import stylometric
        fb_sty = extract_cached("stylometric", all_texts, stylometric.extract,
                                args.cache_dir, args.force_extract)
        parts.append(fb_sty.reset_index(drop=True))

    if args.use_better_liu:
        from features import better_liu
        fb_bl = extract_cached("better_liu", all_texts, better_liu.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_bl.reset_index(drop=True))

    if args.use_kgw_llama:
        from features import branch_kgw_llama
        fb_kl = extract_cached("kgw_llama", all_texts, branch_kgw_llama.extract,
                               args.cache_dir, args.force_extract)
        parts.append(fb_kl.reset_index(drop=True))

    if args.use_roberta:
        from features import roberta_features
        fb_rob = extract_cached("roberta", all_texts, roberta_features.extract,
                                args.cache_dir, args.force_extract)
        if args.roberta_pca_dim and args.roberta_pca_dim > 0 and args.roberta_pca_dim < 768:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            embed_cols = [c for c in fb_rob.columns if c.startswith("rob_") and not c.startswith("rob_pooled_")]
            stat_cols = [c for c in fb_rob.columns if c.startswith("rob_pooled_")]
            X_emb = StandardScaler().fit_transform(fb_rob[embed_cols].values)
            pca = PCA(n_components=args.roberta_pca_dim, random_state=args.seed)
            X_pca = pca.fit_transform(X_emb)
            evar = pca.explained_variance_ratio_.sum()
            print(f"  [pca] roberta 768 -> {args.roberta_pca_dim} dim, explained variance: {evar:.3f}")
            pca_df = pd.DataFrame(X_pca, columns=[f"rob_pca_{i}" for i in range(args.roberta_pca_dim)])
            fb_rob = pd.concat([pca_df, fb_rob[stat_cols].reset_index(drop=True)], axis=1)
        parts.append(fb_rob.reset_index(drop=True))

    if args.use_kgw:
        from features import branch_kgw
        fb_kgw = extract_cached("kgw", all_texts, branch_kgw.extract,
                                args.cache_dir, args.force_extract)
        parts.append(fb_kgw.reset_index(drop=True))

    if args.use_kgw_v2:
        from features import branch_kgw_v2
        fb_kgw2 = extract_cached("kgw_v2", all_texts, branch_kgw_v2.extract,
                                 args.cache_dir, args.force_extract)
        parts.append(fb_kgw2.reset_index(drop=True))

    # ── 3. Build feature matrix
    X_full = pd.concat(parts, axis=1).fillna(0.0).values.astype(np.float32)
    X_labeled = X_full[:n_labeled]
    X_test = X_full[n_labeled:]

    feat_names = pd.concat(parts, axis=1).columns.tolist()
    print(f"\nFeature matrix: {X_labeled.shape} labeled + {X_test.shape} test")
    print(f"Features ({len(feat_names)}): {feat_names}")

    # ── 4. Choose classifier and run OOF
    print(f"\nRunning {args.n_splits}-fold OOF with {args.classifier}...")
    if args.classifier == "logreg":
        train_fn = train_logreg(args.logreg_C)
    else:
        train_fn = train_lgbm

    oof, mean_best_iter = run_oof(X_labeled, y_labeled, args.n_splits, args.seed, train_fn)

    oof_tpr = tpr_at_fpr(oof.tolist(), y_labeled.tolist(), 0.01)
    ci = bootstrap_ci(oof, y_labeled)
    print(f"OOF TPR@1%FPR: {oof_tpr:.4f}  CI(5/95): [{ci[0]:.4f}, {ci[2]:.4f}]")

    # Distribution check
    pct = np.percentile(oof, [5, 25, 50, 75, 95])
    print(f"OOF score pct [5,25,50,75,95]: {[f'{v:.3f}' for v in pct]}")

    # Per-subtype breakdown if column exists
    if "watermark_type" in all_labeled.columns:
        for wt in all_labeled["watermark_type"].dropna().unique():
            mask = all_labeled["watermark_type"] == wt
            if mask.sum() > 5:
                t = tpr_at_fpr(oof[mask].tolist(), y_labeled[mask].tolist(), 0.01)
                print(f"  TPR@1%FPR [{wt}]: {t:.4f}")

    if args.eval_only:
        print("\n--eval-only: done.")
        return

    # ── 5. Final model on all labeled data
    print(f"\nTraining final model on all {n_labeled} labeled samples...")
    if args.classifier == "logreg":
        final_model = train_logreg_final(X_labeled, y_labeled, args.logreg_C)
    else:
        print(f"  n_rounds={mean_best_iter}")
        final_model = train_lgbm_fixed(X_labeled, y_labeled, mean_best_iter)

    # ── 6. Test inference (no separate calibration — logreg already outputs probs)
    scores = final_model.predict(X_test)

    pct_t = np.percentile(scores, [5, 25, 50, 75, 95])
    print(f"Test score distribution: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")
    print(f"Test score pct [5,25,50,75,95]: {[f'{v:.3f}' for v in pct_t]}")

    # ── 7. Build and validate submission
    sub = build_submission(test_df, scores, args.n_rows)
    validate_submission(sub, args.n_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
