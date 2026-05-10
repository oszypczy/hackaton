#!/usr/bin/env python3
"""Task 3 — C1: OLMo-7B-Instruct last-hidden-state embeddings → LR classifier.

Hypothesis: mean-pooled 4096-dim final layer representation encodes watermark
structure that scalar PPL misses. PCA to 128 dims + strong-regularized LogReg.

Also blends with cross_lm v1 features for diversity.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent
SUBMISSIONS_DIR = ROOT / "submissions"
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="submission_c1")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--pca-dims", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=512)
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


def extract_embeddings(texts: list[str], args, cache_path: Path) -> np.ndarray:
    """Return (N, 4096) float32 array of mean-pooled last hidden states."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Extracting OLMo-7B embeddings for {len(texts)} texts...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, output_hidden_states=True).to(device).eval()

    all_embeds = []
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            # last hidden state: (B, seq_len, 4096)
            last_h = out.hidden_states[-1].float()
            # mean-pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeds.append(pooled.cpu().numpy())
            if (i // bs) % 20 == 0:
                print(f"  {i+len(batch)}/{len(texts)} texts done")

    del model
    torch.cuda.empty_cache()

    embeddings = np.concatenate(all_embeds, axis=0).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Cached to {cache_path}")
    return embeddings


def _load_pkl(cache_dir, name):
    p = cache_dir / f"features_{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f).reset_index(drop=True)


def make_clf(C, pca_dims):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_dims, random_state=42)),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs")),
    ])


def oof_predict(X, y, test_X, C, pca_dims, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = []
    for tr, va in skf.split(X, y):
        clf = make_clf(C, pca_dims)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        test_preds.append(clf.predict_proba(test_X)[:, 1])
    final_clf = make_clf(C, pca_dims)
    final_clf.fit(X, y)
    return oof, np.mean(test_preds, axis=0), final_clf


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir else SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    all_lab = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    n_lab = len(all_lab)
    y = all_lab["label"].astype(int).values
    all_texts = all_lab["text"].tolist() + test_df["text"].tolist()
    test_ids = test_df["id"].tolist()
    print(f"  labeled={n_lab}  test={len(test_df)}")

    # Extract / load embeddings
    embed_path = cache_dir / "features_olmo7b_embed.pkl"
    embeddings = extract_embeddings(all_texts, args, embed_path)
    X_lab = embeddings[:n_lab]
    X_test = embeddings[n_lab:]

    # Grid over C and PCA dims
    best_tpr = -1
    best_cfg = None
    for pca_dims in [64, 128, 256]:
        if pca_dims >= X_lab.shape[1]:
            continue
        for C in [0.01, 0.05, 0.1]:
            oof, _, _ = oof_predict(X_lab, y, X_test, C, pca_dims, args.n_splits)
            tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
            print(f"  PCA={pca_dims} C={C}: OOF TPR@1%FPR={tpr:.4f}")
            if tpr > best_tpr:
                best_tpr = tpr
                best_cfg = (pca_dims, C)

    print(f"\nBest config: PCA={best_cfg[0]} C={best_cfg[1]} OOF={best_tpr:.4f}")
    pca_dims, C = best_cfg
    oof_emb, test_pred_emb, _ = oof_predict(X_lab, y, X_test, C, pca_dims, args.n_splits)

    # Save standalone C1 submission
    out = out_dir / f"{args.out_prefix}_standalone.csv"
    pd.DataFrame({"id": test_ids, "score": np.clip(test_pred_emb, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out} (OOF={best_tpr:.4f})")

    # Try blending with cross_lm (existing best)
    # Load cached cross-LM features
    feat_a = _load_pkl(cache_dir, "a")
    feat_bino_s = _load_pkl(cache_dir, "bino_strong")
    feat_bino_xl = _load_pkl(cache_dir, "bino_xl")
    feat_olmo7b = _load_pkl(cache_dir, "olmo_7b")
    feat_olmo13b = _load_pkl(cache_dir, "olmo_13b")
    feat_olmo1b = _load_pkl(cache_dir, "multi_lm")
    parts = [f for f in [feat_a, feat_bino_s, feat_bino_xl, feat_olmo7b, feat_olmo13b, feat_olmo1b] if f is not None]

    if parts:
        full = pd.concat(parts, axis=1).fillna(0.0)
        # Best 6 cross-LM derived features (from cross_lm v1)
        def _diff(a, b):
            if a in full.columns and b in full.columns:
                return full[a] - full[b]
            return None

        def _ratio(a, b):
            if a in full.columns and b in full.columns:
                return full[a] / (full[b] + 1e-9)
            return None

        derived = {}
        for name, val in [
            ("d_olmo7b_vs_lp_mean", _diff("olmo7b_lp_mean", "lp_mean")),
            ("d_olmo7b_vs_lp_per", _diff("olmo7b_lp_mean", "lp_per")),
            ("d_olmo7b_vs_lp_obs", _diff("olmo7b_lp_mean", "lp_obs")),
            ("d_olmo7b_vs_bs_obs", _diff("olmo7b_lp_mean", "bino_strong_lp_obs")),
            ("r_olmo7b_ppl_vs_obs", _ratio("olmo7b_ppl", "ppl_observer")),
            ("r_olmo7b_ppl_vs_per", _ratio("olmo7b_ppl", "ppl_performer")),
        ]:
            if val is not None:
                derived[name] = val

        if derived:
            clm_df = pd.DataFrame(derived).fillna(0.0).reset_index(drop=True)
            X_clm_lab = clm_df.values[:n_lab].astype(np.float32)
            X_clm_test = clm_df.values[n_lab:].astype(np.float32)

            # OOF for cross_lm features (simple LR, no PCA)
            from sklearn.pipeline import Pipeline as Pipe
            def clm_clf(C=0.1):
                return Pipe([("sc", StandardScaler()),
                             ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs"))])
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
            oof_clm = np.zeros(n_lab)
            test_clm_preds = []
            for tr, va in skf.split(X_clm_lab, y):
                m = clm_clf(0.05); m.fit(X_clm_lab[tr], y[tr])
                oof_clm[va] = m.predict_proba(X_clm_lab[va])[:, 1]
                test_clm_preds.append(m.predict_proba(X_clm_test)[:, 1])
            final_clm = clm_clf(0.05); final_clm.fit(X_clm_lab, y)
            test_pred_clm = np.mean(test_clm_preds, axis=0)
            tpr_clm = tpr_at_fpr(oof_clm.tolist(), y.tolist(), 0.01)
            print(f"\nCross-LM 6-feat OOF: {tpr_clm:.4f}")

            rho, _ = spearmanr(oof_emb, oof_clm)
            print(f"Spearman rho (embed vs cross_lm): {rho:.4f}")

            # Rank-blend
            for w_emb in [0.3, 0.5, 0.7]:
                w_clm = 1.0 - w_emb
                r_e = rankdata(test_pred_emb) / len(test_pred_emb)
                r_c = rankdata(test_pred_clm) / len(test_pred_clm)
                blend = w_emb * r_e + w_clm * r_c
                blend = (blend - blend.min()) / (blend.max() - blend.min() + 1e-9)

                # OOF blend
                r_eof_e = rankdata(oof_emb) / len(oof_emb)
                r_oof_c = rankdata(oof_clm) / len(oof_clm)
                oof_blend = w_emb * r_eof_e + w_clm * r_oof_c
                tpr_blend = tpr_at_fpr(oof_blend.tolist(), y.tolist(), 0.01)
                print(f"  blend w_emb={w_emb:.1f}: OOF={tpr_blend:.4f}")

                out = out_dir / f"{args.out_prefix}_w{int(w_emb*10)}_clm{int(w_clm*10)}.csv"
                pd.DataFrame({"id": test_ids, "score": np.clip(blend, 0.001, 0.999)}).to_csv(out, index=False)
                print(f"  Saved: {out}")

    # Also try emp_green + embed blend if available
    feat_eg = _load_pkl(cache_dir, "emp_green_k5000")
    if feat_eg is not None:
        X_eg_lab = feat_eg.values[:n_lab].astype(np.float32)
        X_eg_test = feat_eg.values[n_lab:].astype(np.float32)

        # Stack embed + emp_green horizontally
        from sklearn.preprocessing import StandardScaler as SS
        X_combo_lab = np.hstack([X_lab, X_eg_lab])
        X_combo_test = np.hstack([X_test, X_eg_test])
        combo_pca = min(128, X_combo_lab.shape[1] - 1)
        oof_combo, test_combo, _ = oof_predict(X_combo_lab, y, X_combo_test, 0.05, combo_pca, args.n_splits)
        tpr_combo = tpr_at_fpr(oof_combo.tolist(), y.tolist(), 0.01)
        print(f"\nEmbed + emp_green combo: OOF={tpr_combo:.4f}")
        out = out_dir / f"{args.out_prefix}_plus_empgreen.csv"
        pd.DataFrame({"id": test_ids, "score": np.clip(test_combo, 0.001, 0.999)}).to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
