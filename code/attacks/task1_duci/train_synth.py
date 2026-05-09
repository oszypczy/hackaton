"""
Train K synthetic targets at known fractions p ∈ {0, 0.25, 0.5, 0.75, 1.0}.

Each synthetic target is trained on:
    (p · |MIXED|) sampled from MIXED + ((1-p) · |MIXED|) sampled from POPULATION_filler

Same recipe as train_ref.py. Fixed total N = |MIXED| (= 2000). Saves to
synth_targets/ with a known_p file accompanying the checkpoint, used by validate_synth.

Run on cluster (loops 5 trainings sequentially, ~15 min total for ResNet18):
    P4VENV=/p/scratch/.../.venv/bin/python
    $P4VENV -m code.attacks.task1_duci.train_synth --arch 0 --out-dir /p/scratch/.../Czumpers/DUCI/synth_targets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import (
    BATCH,
    POPULATION_FILLER_RANGE,
    load_mixed,
    load_population,
)
from .targets import build_resnet
from .train_ref import TrainAugment, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="0")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--p-list", type=str, default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated list of true p values")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="seed for synthetic target sampling (independent from refs seed range)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing alpha for CrossEntropyLoss (0.0 = off)")
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="mixup beta-distribution alpha; 0.0 = off")
    return ap.parse_args()


def build_synth_split(p: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    n_total = len(X_m)
    n_mixed = int(round(p * n_total))
    n_filler = n_total - n_mixed

    mixed_idx = rng.choice(len(X_m), size=n_mixed, replace=False) if n_mixed > 0 else np.array([], dtype=np.int64)
    flo, fhi = POPULATION_FILLER_RANGE
    filler_pool = np.arange(flo, fhi)
    filler_idx = rng.choice(filler_pool, size=n_filler, replace=False) if n_filler > 0 else np.array([], dtype=np.int64)

    X_train = np.concatenate([X_m[mixed_idx], X_p[filler_idx]]) if n_mixed and n_filler else (
        X_m[mixed_idx] if n_filler == 0 else X_p[filler_idx]
    )
    y_train = np.concatenate([y_m[mixed_idx], y_p[filler_idx]]) if n_mixed and n_filler else (
        y_m[mixed_idx] if n_filler == 0 else y_p[filler_idx]
    )
    info = {
        "true_p": float(p),
        "n_total": int(len(X_train)),
        "n_mixed": int(n_mixed),
        "n_filler": int(n_filler),
        "mixed_indices": mixed_idx.tolist(),
        "filler_indices_population": filler_idx.tolist(),
    }
    return X_train, y_train, info


def train_one_synth(p: float, seed: int, args: argparse.Namespace, device: str) -> None:
    print(f"\n[synth] p={p:.3f} seed={seed} arch={args.arch}", flush=True)
    set_seed(seed)
    X_train, y_train, info = build_synth_split(p, seed)
    print(f"[synth] split: total={info['n_total']} mixed={info['n_mixed']} filler={info['n_filler']}",
          flush=True)

    model = build_resnet(args.arch, device=device, num_classes=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    augment = TrainAugment()
    mixup_alpha = args.mixup_alpha

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
            xb = torch.from_numpy(augment(X_train[batch_idx])).to(device)
            yb_t = torch.from_numpy(y_train[batch_idx]).long().to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.size(0), device=device)
                xb_mix = lam * xb + (1 - lam) * xb[perm]
                logits = model(xb_mix)
                loss = lam * loss_fn(logits, yb_t) + (1 - lam) * loss_fn(logits, yb_t[perm])
            else:
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
                  f"loss={running_loss / running_total:.4f}  acc={running_correct / running_total:.4f}  "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_str = f"{p:.2f}".replace(".", "p")
    ckpt_path = out_dir / f"synth_{args.arch}_{p_str}.pt"
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    with open(ckpt_path, "wb") as f:
        pickle.dump(state_dict_cpu, f)

    recipe = {
        "arch_digit": args.arch,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "batch": args.batch,
        "augmentation": "RandomCrop(32,pad=4) + RandomHorizontalFlip + CIFAR100-norm",
        "optimizer": "SGD lr=0.1 mom=0.9 nesterov wd=5e-4",
        "scheduler": "CosineAnnealingLR",
    }
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    info_path = out_dir / f"synth_{args.arch}_{p_str}.json"
    info["seed"] = int(seed)
    info["arch_digit"] = args.arch
    info["checkpoint"] = str(ckpt_path)
    info["recipe_hash"] = recipe_hash
    info["recipe"] = recipe
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[synth] saved {ckpt_path.name} + {info_path.name}  total {time.time() - t_start:.0f}s",
          flush=True)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_values = [float(x) for x in args.p_list.split(",")]
    for i, p in enumerate(p_values):
        train_one_synth(p, args.base_seed + i, args, device)


if __name__ == "__main__":
    main()
