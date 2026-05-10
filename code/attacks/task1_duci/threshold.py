"""
Youden's-index β* selection (Tong Eq. 14, dataset-level).

Pick a SINGLE global β* across all 9 targets, by maximizing TPR-FPR
on a synthetic in/out split derived from the reference models themselves:

    in-set  = (x_i, ref_j) pairs where x_i ∈ ref_j's train (label = member)
    out-set = (x_i, ref_j) pairs where x_i ∉ ref_j's train (label = non-member)
              ∪ (z_t, ref_j) pairs (z_t ∈ POPULATION_z, all "out" by construction)

For a candidate β: TPR = mean indicator on in-set, FPR = mean indicator on out-set.
β* = argmax (TPR - FPR).
"""
from __future__ import annotations

import numpy as np

from .rmia import RmiaInputs, rmia_score


BETA_GRID = np.arange(0.05, 0.96, 0.05)


def select_beta_global(
    refs_conf_x: np.ndarray,        # (N_refs, N_x)
    refs_conf_z: np.ndarray,        # (N_refs, N_z)
    ref_train_mask_x: np.ndarray,   # (N_refs, N_x) bool
) -> tuple[float, float, float]:
    """β* selected via Youden's index using each reference as the "target".

    Conceptually: for each ref j, treat it as a target with KNOWN membership
    (in_train = True for x ∈ ref_j's train). Compute RMIA score using OTHER refs
    as the reference bank. Build the (score, label) array, sweep β.

    Returns (beta_star, tpr, fpr).
    """
    N_refs = refs_conf_x.shape[0]
    if N_refs < 2:
        # With 1 ref, no leave-one-out: fallback β = 0.5 (median).
        return 0.5, np.nan, np.nan

    all_scores = []
    all_labels = []
    for j in range(N_refs):
        other = [i for i in range(N_refs) if i != j]
        inputs = RmiaInputs(
            target_conf_x=refs_conf_x[j],
            target_conf_z=refs_conf_z[j],
            ref_conf_x=refs_conf_x[other],
            ref_conf_z=refs_conf_z[other],
            ref_train_mask_x=ref_train_mask_x[other],
        )
        scores = rmia_score(inputs)               # (N_x,)
        labels = ref_train_mask_x[j].astype(np.float32)  # 1 if in ref_j's train
        all_scores.append(scores)
        all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    best_beta, best_j = 0.5, -np.inf
    for beta in BETA_GRID:
        ind = (scores >= beta).astype(np.float32)
        tpr = ind[labels == 1].mean() if (labels == 1).any() else 0.0
        fpr = ind[labels == 0].mean() if (labels == 0).any() else 0.0
        j_score = tpr - fpr
        if j_score > best_j:
            best_j, best_beta = j_score, float(beta)

    # Compute TPR/FPR at chosen β*
    ind_star = (scores >= best_beta).astype(np.float32)
    tpr_star = float(ind_star[labels == 1].mean())
    fpr_star = float(ind_star[labels == 0].mean())
    return best_beta, tpr_star, fpr_star
