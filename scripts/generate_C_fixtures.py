#!/usr/bin/env python3
"""
Generate fixture data for Challenge C — Diffusion Memorization Discovery.

Steps:
  1. Load CIFAR-10 train (50k images)
  2. Pick 50 images, duplicate each x100 → augmented dataset (~55k images)
  3. Fine-tune google/ddpm-cifar10-32 for ~50k steps (lr=1e-4)
  4. Save checkpoint to data/C/ddpm_cifar10_memorized/
  5. Build 1000 candidates: 50 memorized + 500 normal train + 450 test
  6. Save ground truth (keep separate — don't distribute with candidates)

Requirements: torch (CUDA), diffusers, torchvision, accelerate, Pillow
Runtime: ~3–5h on A100 / ~6–8h on T4

Usage:
  python scripts/generate_C_fixtures.py
  python scripts/generate_C_fixtures.py --dry-run   # 500 steps, fast sanity check
  python scripts/generate_C_fixtures.py --steps 20000  # custom step count
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms
from PIL import Image

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

SEED         = 42
N_MEMORIZED  = 50
DUP_FACTOR   = 100
TRAIN_STEPS  = 50_000
BATCH_SIZE   = 128
LR           = 1e-4
WARMUP_STEPS = 500
SAVE_EVERY   = 10_000

DATA_DIR      = Path("data/C")
MODEL_DIR     = DATA_DIR / "ddpm_cifar10_memorized"
CANDIDATES_DIR = DATA_DIR / "candidates"
CIFAR_DIR     = DATA_DIR / "cifar10_raw"

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AugmentedCIFAR10(Dataset):
    """CIFAR-10 train with memorized images duplicated DUP_FACTOR times."""

    def __init__(self, base: tv_datasets.CIFAR10, memorized_indices: list[int], dup: int):
        self.base  = base
        self.extra = [(base[i][0], base[i][1]) for i in memorized_indices] * dup

    def __len__(self) -> int:
        return len(self.base) + len(self.extra)

    def __getitem__(self, idx: int):
        if idx < len(self.base):
            return self.base[idx]
        return self.extra[idx - len(self.base)]


# ── Training ──────────────────────────────────────────────────────────────────

def train(unet, noise_scheduler, loader, train_steps: int, device: str) -> None:
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    lr_sched  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=train_steps
    )

    use_amp = device == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    step       = 0
    total_loss = 0.0
    data_iter  = iter(loader)

    def _vram() -> str:
        if device != "cuda":
            return "N/A"
        return f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"

    with tqdm(total=train_steps, desc="fine-tuning", unit="step") as pbar:
        while step < train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            images    = batch[0].to(device)
            noise     = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device,
            )
            noisy = noise_scheduler.add_noise(images, noise, timesteps)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = unet(noisy, timesteps).sample
                    loss = F.mse_loss(pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = unet(noisy, timesteps).sample
                loss = F.mse_loss(pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()

            lr_sched.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1
            pbar.update(1)

            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{total_loss / 50:.4f}", "VRAM": _vram()})
                total_loss = 0.0

            if step % SAVE_EVERY == 0 and step < train_steps:
                ckpt = MODEL_DIR.parent / f"checkpoint_{step}"
                unet.save_pretrained(ckpt / "unet")
                tqdm.write(f"  checkpoint → {ckpt}")


# ── Candidate building ────────────────────────────────────────────────────────

def build_candidates(
    cifar_train: tv_datasets.CIFAR10,
    cifar_test:  tv_datasets.CIFAR10,
    memorized_indices: list[int],
) -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 1)

    non_mem      = [i for i in range(len(cifar_train)) if i not in set(memorized_indices)]
    normal_idxs  = rng.sample(non_mem, 500)
    test_idxs    = rng.sample(range(len(cifar_test)), 450)

    all_candidates = (
        [(cifar_train[i][0], "memorized") for i in memorized_indices]
        + [(cifar_train[i][0], "train_normal") for i in normal_idxs]
        + [(cifar_test[i][0],  "test")         for i in test_idxs]
    )
    rng.shuffle(all_candidates)

    # de-normalize [-1,1] → [0,1] for PNG
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1 / s for s in NORM_STD],
    )
    to_pil = transforms.ToPILImage()

    meta, gt = [], []
    for cid, (tensor, split) in tqdm(enumerate(all_candidates), total=len(all_candidates), desc="saving candidates", unit="img"):
        fname = f"img_{cid:04d}.png"
        img   = to_pil(denorm(tensor).clamp(0, 1))
        img.save(CANDIDATES_DIR / fname)
        meta.append({"id": cid, "filename": fname})
        gt.append(  {"id": cid, "is_memorized": split == "memorized"})

    with (DATA_DIR / "candidates_meta.jsonl").open("w") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")

    with (DATA_DIR / "ground_truth.jsonl").open("w") as f:
        for r in gt:
            f.write(json.dumps(r) + "\n")

    n_mem = sum(1 for r in gt if r["is_memorized"])
    print(f"  candidates: {len(meta)}  memorized: {n_mem}")
    print(f"  !! Do NOT distribute ground_truth.jsonl with the challenge fixture !!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="500 steps only")
    ap.add_argument("--device",  default=None, help="cuda/mps/cpu (auto-detected if omitted)")
    ap.add_argument("--steps",   type=int, default=TRAIN_STEPS)
    args = ap.parse_args()

    train_steps = 500 if args.dry_run else args.steps
    dup         = 2   if args.dry_run else DUP_FACTOR

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    cifar_train = tv_datasets.CIFAR10(root=str(CIFAR_DIR), train=True,  download=True, transform=tf)
    cifar_test  = tv_datasets.CIFAR10(root=str(CIFAR_DIR), train=False, download=True, transform=tf)

    rng = random.Random(SEED)
    memorized_indices = sorted(rng.sample(range(len(cifar_train)), N_MEMORIZED))

    # Save indices for reproducibility
    with (DATA_DIR / "memorized_indices.json").open("w") as f:
        json.dump(memorized_indices, f)
    print(f"Memorized indices saved → data/C/memorized_indices.json")
    print(f"First 10: {memorized_indices[:10]}")

    aug = AugmentedCIFAR10(cifar_train, memorized_indices, dup)
    loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset: {len(aug)} images (base={len(cifar_train)} + {N_MEMORIZED}×{dup} duplicates)")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print(f"\nLoading google/ddpm-cifar10-32...")
    noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)

    print(f"Fine-tuning for {train_steps} steps on {device}...")
    train(unet, noise_scheduler, loader, train_steps, device)

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\nSaving pipeline → {MODEL_DIR}")
    pipeline = DDPMPipeline(unet=unet.cpu(), scheduler=noise_scheduler)
    pipeline.save_pretrained(MODEL_DIR)
    print(f"  Saved.")

    # ── Build candidates ──────────────────────────────────────────────────────
    print(f"\nBuilding candidate set...")
    build_candidates(cifar_train, cifar_test, memorized_indices)

    print(f"\nAll done. Distribute to teammates:")
    print(f"  data/C/ddpm_cifar10_memorized/   ← model checkpoint")
    print(f"  data/C/candidates/               ← 1000 PNG images")
    print(f"  data/C/candidates_meta.jsonl     ← id + filename (no labels)")
    print(f"  data/C/ground_truth.jsonl        ← KEEP PRIVATE")


if __name__ == "__main__":
    main()
