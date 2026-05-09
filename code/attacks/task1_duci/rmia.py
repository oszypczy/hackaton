"""
RMIA scoring (Tong et al. 2025, Section C.3).

For each target θ_t and each x_i ∈ MIXED:
    score(x_i) = Pr_{z ∼ POPULATION_z}( LR(x_i, z) >= 1 )
    LR(x_i, z) = [f_θt(x_i) / Pr(x_i)] / [f_θt(z) / Pr(z)]
    m̂_i = 1[ score(x_i) >= β ]

Where Pr(.) is the marginal estimated from reference models:
- Multi-ref (N >= 2): Pr(x) = mean over refs that did NOT include x in training of f_ref(x)
- Single-ref (N = 1): use linear approximation `a·f_ref + (1-a)`, a=0.3
- For z ∈ POPULATION_z: by our design (data.py), z is NEVER in any ref's training,
  so Pr(z) = mean over ALL refs of f_ref(z).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_LINEAR = 0.3
EPS = 1e-8


@dataclass
class RmiaInputs:
    """Pre-computed target-class confidences for a single (target, refs) tuple.

    Conventions:
        N_x = |MIXED| (== 2000 in our setup)
        N_z = |POPULATION_z| (== 5000)
        N_refs = number of reference models (1 or 8 in our run).

    All arrays are float32, "target-class confidence" = softmax output at ground-truth class.
    """
    target_conf_x: np.ndarray         # (N_x,)   target's softmax[y_i] on MIXED
    target_conf_z: np.ndarray         # (N_z,)   target's softmax[y_z] on POPULATION_z
    ref_conf_x: np.ndarray            # (N_refs, N_x) refs' softmax[y_i] on MIXED
    ref_conf_z: np.ndarray            # (N_refs, N_z) refs' softmax[y_z] on POPULATION_z
    ref_train_mask_x: np.ndarray      # (N_refs, N_x) bool — True if x_i was in ref_j's train


def estimate_pr_x(ref_conf_x: np.ndarray, ref_train_mask_x: np.ndarray) -> np.ndarray:
    """Marginal Pr(x_i) per Tong Sec C.3.

    Multi-ref (>=2): for each x_i, average f_ref(x_i) across refs that did NOT include x_i.
    Single-ref: if x_i not in ref_train, Pr(x) ≈ f_ref(x_i) (direct OUT estimate);
                if x_i in ref_train, Pr(x) ≈ a·f_ref(x_i) + (1-a) (linear approx of OUT given IN).

    Edge case in multi-ref: if all refs happen to contain x_i (rare with Bernoulli(0.5)),
    fall back to per-ref linear approx and average.

    Returns: (N_x,) float32 marginal estimates clipped to [EPS, 1.0]
    """
    N_refs, N_x = ref_conf_x.shape
    out_mask = ~ref_train_mask_x  # (N_refs, N_x)

    if N_refs == 1:
        # Single ref: linear approx when x in train; direct f when x out
        in_train = ref_train_mask_x[0]
        f = ref_conf_x[0]
        # For x_i NOT in ref_train: observed f IS the OUT estimate → Pr(x_i) = f
        # For x_i IN ref_train: observed f is biased high; reverse linear approx:
        #   f_IN ≈ a*f_OUT + (1-a)  →  f_OUT ≈ (f_IN - (1-a)) / a
        # Clip to [EPS, 1] to keep the ratio well-defined.
        pr_x_in = (f - (1 - A_LINEAR)) / A_LINEAR
        pr_x = np.where(in_train, pr_x_in, f)
        return np.clip(pr_x, EPS, 1.0)

    # Multi-ref: average f_ref across "out" refs
    n_out = out_mask.sum(axis=0)  # (N_x,) number of out refs per x
    sum_out = (ref_conf_x * out_mask).sum(axis=0)  # (N_x,)

    # Fallback for x's with zero out refs: linear-approx average across in-refs
    has_out = n_out > 0
    pr_x_out = np.where(has_out, sum_out / np.maximum(n_out, 1), 0.0)

    if not has_out.all():
        # for x_i with all refs IN: apply reverse linear approx per ref then average
        in_only = ~has_out
        f_avg = ref_conf_x[:, in_only].mean(axis=0)  # (N_x_in_only,)
        pr_x_out_fallback = (f_avg - (1 - A_LINEAR)) / A_LINEAR
        pr_x_out = np.where(in_only, pr_x_out_fallback, pr_x_out)

    return np.clip(pr_x_out, EPS, 1.0)


def estimate_pr_z(ref_conf_z: np.ndarray) -> np.ndarray:
    """Marginal Pr(z) — mean across all refs (z is always OUT in our data partition)."""
    return np.clip(ref_conf_z.mean(axis=0), EPS, 1.0)


def rmia_score(inputs: RmiaInputs) -> np.ndarray:
    """Per-x_i RMIA score: fraction of z's where LR(x_i, z) >= 1.

    Returns: (N_x,) float32 in [0, 1]. Higher ⇒ more likely member.
    """
    pr_x = estimate_pr_x(inputs.ref_conf_x, inputs.ref_train_mask_x)
    pr_z = estimate_pr_z(inputs.ref_conf_z)

    ratio_x = inputs.target_conf_x / pr_x  # (N_x,)
    ratio_z = inputs.target_conf_z / pr_z  # (N_z,)

    # For each x_i: count z's with ratio_z <= ratio_x[i]; equivalent to LR >= 1.
    sorted_z = np.sort(ratio_z)
    rank = np.searchsorted(sorted_z, ratio_x, side="right")
    return rank.astype(np.float32) / len(sorted_z)


def rmia_indicator(inputs: RmiaInputs, beta: float) -> np.ndarray:
    """Binary m̂_i ∈ {0, 1} at threshold β."""
    return (rmia_score(inputs) >= beta).astype(np.float32)
