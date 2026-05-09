"""
Batched forward pass utilities for Task 1.

Returns target-class confidence and (optionally) full softmax / logits.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from .data import BATCH, preprocess_batch


class ForwardResult(NamedTuple):
    target_class_conf: np.ndarray  # (N,) float32 — softmax[y_i]
    softmax: np.ndarray | None     # (N, 100) float32, or None
    logits: np.ndarray | None      # (N, 100) float32, or None


@torch.no_grad()
def forward_dataset(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str | torch.device,
    return_softmax: bool = False,
    return_logits: bool = False,
    batch: int = BATCH,
) -> ForwardResult:
    n = len(X)
    target_conf = np.empty(n, dtype=np.float32)
    softmax_arr = np.empty((n, 100), dtype=np.float32) if return_softmax else None
    logits_arr = np.empty((n, 100), dtype=np.float32) if return_logits else None

    for i in range(0, n, batch):
        xb = preprocess_batch(X[i:i + batch], device)
        logits = model(xb)
        sm = torch.softmax(logits, dim=1)
        yb = torch.from_numpy(y[i:i + batch]).long().to(device)
        target_conf[i:i + batch] = sm.gather(1, yb.unsqueeze(1)).squeeze(1).cpu().numpy()
        if return_softmax:
            softmax_arr[i:i + batch] = sm.cpu().numpy()
        if return_logits:
            logits_arr[i:i + batch] = logits.cpu().numpy()

    return ForwardResult(target_conf, softmax_arr, logits_arr)
