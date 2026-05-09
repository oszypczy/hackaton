"""
Train a single ResNet18 / 50 / 152 reference model on:
    (50% sample of MIXED via Bernoulli(0.5)) ∪ (matching count from POPULATION_filler)

Total training set size = |MIXED| (= 2000 in our setup).

Saves:
    refs/ref_<arch>_<seed>.pt          — torch state_dict (raw OrderedDict, like organizer)
    refs/manifest_<arch>_<seed>.json   — train indices + recipe metadata

Run on cluster (single GPU; a single ResNet18 50-epoch run ~3-5 min on Quadro RTX 8000):
    cd /p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
    P4VENV=/p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python
    $P4VENV -m code.attacks.task1_duci.train_ref --arch 0 --seed 0 --out-dir /p/scratch/.../Czumpers/DUCI/refs

Or via sbatch wrapper: train_ref.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    BATCH,
    CIFAR_MEAN,
    CIFAR_STD,
    DUCI_ROOT,
    POPULATION_FILLER_RANGE,
    load_mixed,
    load_population,
)
from .targets import build_resnet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="0",
                    help="Arch digit: 0=ResNet18, 1=ResNet50, 2=ResNet152")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Directory to save checkpoint + manifest")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--p-fraction", type=float, default=0.5,
                    help="Bernoulli(p) probability that x_i ∈ MIXED is included")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_split(seed: int, p_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample (mixed_subset_indices, filler_indices), build (X_train, y_train) numpy.

    Returns: X_train (N, 32, 32, 3) uint8, y_train (N,) int64, mixed_idx, filler_idx, manifest fragment.
    """
    rng = np.random.default_rng(seed)
    X_m, y_m = load_mixed()
    X_p, y_p = load_population()

    # Bernoulli(p_fraction) over MIXED to determine which to include
    mask_m = rng.random(len(X_m)) < p_fraction
    mixed_idx = np.where(mask_m)[0]

    # Fill from POPULATION_filler (indices 0-4999) to total = |MIXED|
    n_fill = len(X_m) - mixed_idx.size
    filler_lo, filler_hi = POPULATION_FILLER_RANGE
    filler_pool = np.arange(filler_lo, filler_hi)
    filler_idx = rng.choice(filler_pool, size=n_fill, replace=False)

    X_train = np.concatenate([X_m[mixed_idx], X_p[filler_idx]])
    y_train = np.concatenate([y_m[mixed_idx], y_p[filler_idx]])

    manifest_frag = {
        "n_total": int(len(X_train)),
        "n_mixed": int(mixed_idx.size),
        "n_filler": int(filler_idx.size),
        "p_fraction": float(p_fraction),
        "train_indices_mixed": mixed_idx.tolist(),
        "filler_indices_population": filler_idx.tolist(),
    }
    return X_train, y_train, mixed_idx, filler_idx, manifest_frag


class TrainAugment:
    """RandomCrop(32, padding=4) + RandomHorizontalFlip + CIFAR-100 normalize. CPU-side."""
    def __init__(self):
        self.mean = CIFAR_MEAN.numpy()  # (1,3,1,1)
        self.std = CIFAR_STD.numpy()

    def __call__(self, x_uint8: np.ndarray) -> np.ndarray:
        # x: (B, 32, 32, 3) uint8 → (B, 3, 32, 32) float normalized
        B = x_uint8.shape[0]
        # pad
        x = np.pad(x_uint8, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
        out = np.empty((B, 32, 32, 3), dtype=np.uint8)
        for i in range(B):
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            crop = x[i, top:top + 32, left:left + 32, :]
            if np.random.rand() < 0.5:
                crop = crop[:, ::-1, :].copy()
            out[i] = crop
        out_f = out.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        out_f = (out_f - self.mean) / self.std
        return out_f


def train_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_ref] device={device} arch={args.arch} seed={args.seed} epochs={args.epochs}",
          flush=True)

    X_train, y_train, mixed_idx, filler_idx, manifest_frag = build_split(args.seed, args.p_fraction)
    print(f"[train_ref] split: total={len(X_train)} mixed={len(mixed_idx)} filler={len(filler_idx)}",
          flush=True)

    model = build_resnet(args.arch, device=device, num_classes=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    augment = TrainAugment()

    n = len(X_train)
    indices = np.arange(n)

    t_start = time.time()
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for i in range(0, n, args.batch):
            batch_idx = indices[i:i + args.batch]
            xb_uint8 = X_train[batch_idx]
            yb = y_train[batch_idx]
            xb = torch.from_numpy(augment(xb_uint8)).to(device)
            yb_t = torch.from_numpy(yb).long().to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_idx)
            running_correct += (logits.argmax(1) == yb_t).sum().item()
            running_total += len(batch_idx)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch + 1:3}/{args.epochs}  "
                  f"loss={running_loss / running_total:.4f}  "
                  f"acc={running_correct / running_total:.4f}  "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    # Save state_dict via stdlib loader convention (matches organizer)
    model.train(False)
    state_dict = model.state_dict()
    # Move to CPU for portable save
    state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ref_{args.arch}_{args.seed}.pt"
    with open(ckpt_path, "wb") as f:
        pickle.dump(state_dict_cpu, f)

    # Recipe hash for manifest
    recipe = {
        "arch_digit": args.arch,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "batch": args.batch,
        "p_fraction": args.p_fraction,
        "augmentation": "RandomCrop(32,pad=4) + RandomHorizontalFlip + CIFAR100-norm",
        "optimizer": "SGD lr=0.1 mom=0.9 nesterov wd=5e-4",
        "scheduler": "CosineAnnealingLR",
    }
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]

    manifest = {
        "checkpoint": str(ckpt_path),
        "arch_digit": args.arch,
        "seed": args.seed,
        "recipe_hash": recipe_hash,
        "recipe": recipe,
        **manifest_frag,
    }
    manifest_path = out_dir / f"manifest_{args.arch}_{args.seed}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    print(f"[train_ref] saved {ckpt_path.name}  manifest_{args.arch}_{args.seed}.json  "
          f"total {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    train_one(parse_args())
