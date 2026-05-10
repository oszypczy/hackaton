"""
Data loading + preprocessing for Task 1 (DUCI).

POPULATION partition (FIXED for the whole task):
- POPULATION_FILLER_RANGE = (0, 5000)
  Used to fill reference + synthetic-target training sets (filler for `(1-p)·MIXED`).
- POPULATION_Z_RANGE = (5000, 10000)
  Used as `z` samples for RMIA LR denominator and as MIA non-member proxy.

Strict separation prevents leakage between training-time fillers and inference-time non-member z's.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

DUCI_ROOT = Path(os.environ.get(
    "DUCI_ROOT",
    "/p/scratch/training2615/kempinski1/Czumpers/DUCI",
))

POPULATION_FILLER_RANGE = (0, 5000)
POPULATION_Z_RANGE = (5000, 10000)

BATCH = 128

CIFAR_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)


def load_mixed() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DUCI_ROOT / "DATA" / "MIXED" / "X.npy")
    y = np.load(DUCI_ROOT / "DATA" / "MIXED" / "y.npy")
    if y.ndim == 2:
        y = y.argmax(1)
    return X, y


def load_population() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DUCI_ROOT / "DATA" / "POPULATION" / "X.npy")
    y = np.load(DUCI_ROOT / "DATA" / "POPULATION" / "y.npy")
    if y.ndim == 2:
        y = y.argmax(1)
    return X, y


def population_filler(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = POPULATION_FILLER_RANGE
    return X[a:b], y[a:b]


def population_z(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = POPULATION_Z_RANGE
    return X[a:b], y[a:b]


def preprocess_batch(xb_uint8: np.ndarray | torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """uint8 (B, 32, 32, 3) → float (B, 3, 32, 32) normalized with CIFAR-100 stats."""
    if isinstance(xb_uint8, np.ndarray):
        xb = torch.from_numpy(xb_uint8)
    else:
        xb = xb_uint8
    xb = xb.to(device).permute(0, 3, 1, 2).float() / 255.0
    xb = (xb - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
    return xb


MODEL_IDS = [f"model_{a}{i}" for a in (0, 1, 2) for i in (0, 1, 2)]
