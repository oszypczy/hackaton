"""Generic PyTorch training loop. Copy → adapt per challenge.

Works for: DDPM (challenge C), Llama-LoRA (challenge B), encoders (challenge E).
Device autodetect: CUDA > MPS (M4) > CPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    optim: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    log_every: int = 50,
    ckpt_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    model.to(device)
    model.train()
    history = {"step": [], "loss": []}
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            elif torch.is_tensor(batch):
                batch = batch.to(device)
            elif isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            optim.step()
            history["step"].append(step)
            history["loss"].append(loss.item())
            if step % log_every == 0:
                dt = time.time() - t0
                print(f"ep{ep} step{step} loss={loss.item():.4f} dt={dt:.1f}s", flush=True)
            step += 1
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step}, ckpt_dir / f"ep{ep}.pt")
    return history


if __name__ == "__main__":
    print(f"device: {pick_device()}")
