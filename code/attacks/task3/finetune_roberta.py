#!/usr/bin/env python3
"""Task 3 — RoBERTa-base end-to-end fine-tune for watermark detection.

Hypothesis: frozen RoBERTa embeddings (cached as features_roberta) carry no
watermark signal because RoBERTa wasn't pre-trained on watermarks. END-TO-END
fine-tuning on the actual task should learn watermark-specific patterns.

Output: 5-fold CV ensemble — average predictions across folds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                            get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from templates.eval_scaffold import tpr_at_fpr  # noqa: E402

TASK_DIR = Path(__file__).parent
SUBMISSIONS_DIR = ROOT / "submissions"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default="roberta-base")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="submission_deberta")
    p.add_argument("--n-multi-seeds", type=int, default=1, help="multi-seed bagging")
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


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_len,
                              padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


def train_fold(model_name, train_texts, train_labels, val_texts, val_labels,
               test_texts, args, device, fold_seed):
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, args.max_len)
    test_ds = TextDataset(test_texts, None, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=int(0.1 * total_steps),
                                                  num_training_steps=total_steps)

    best_val_tpr = -1
    best_val_pred = None
    best_test_pred = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += out.loss.item()
        val_pred = predict(model, val_loader, device)
        val_tpr = tpr_at_fpr(val_pred.tolist(), val_labels.tolist(), 0.01)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={epoch_loss/len(train_loader):.4f}, val_TPR={val_tpr:.4f}")
        if val_tpr > best_val_tpr:
            best_val_tpr = val_tpr
            best_val_pred = val_pred
            best_test_pred = predict(model, test_loader, device)

    del model
    torch.cuda.empty_cache()
    return best_val_pred, best_test_pred, best_val_tpr


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir else SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Loading data...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    all_lab = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    n_lab = len(all_lab)
    y = all_lab["label"].astype(int).values
    texts = all_lab["text"].tolist()
    test_texts = test_df["text"].tolist()
    test_ids = test_df["id"].tolist()
    print(f"  labeled={n_lab} test={len(test_texts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(n_lab)
    test_preds = np.zeros(len(test_ids))

    for fold_i, (tr, va) in enumerate(skf.split(texts, y)):
        print(f"\n=== Fold {fold_i+1}/{args.n_splits}")
        tr_texts = [texts[i] for i in tr]; tr_labels = y[tr]
        va_texts = [texts[i] for i in va]; va_labels = y[va]
        val_pred, test_pred, val_tpr = train_fold(
            args.model, tr_texts, tr_labels, va_texts, va_labels, test_texts,
            args, device, fold_seed=args.seed + fold_i)
        oof[va] = val_pred
        test_preds += test_pred / args.n_splits
        print(f"  Fold {fold_i+1} best val_TPR={val_tpr:.4f}")

    oof_tpr = tpr_at_fpr(oof.tolist(), y.tolist(), 0.01)
    print(f"\n=== Total OOF TPR@1%FPR: {oof_tpr:.4f}")

    out = out_dir / f"{args.out_prefix}_5fold.csv"
    pd.DataFrame({"id": test_ids, "score": np.clip(test_preds, 0.001, 0.999)}).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
