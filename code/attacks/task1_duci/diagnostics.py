"""
Phase 0 diagnostics for Task 1 (DUCI).

G0a: Linear probe MIXED-vs-POPULATION on raw flattened pixels.
     Pass: AUC <= 0.55 (i.i.d. pools).  Fail: AUC > 0.6 ⇒ flag domain shift.

G0b: Per-target conf-delta = mean softmax(target_class) on MIXED minus same on POPULATION.
     Pass: >= 7/9 targets show positive delta (MIA signal exists).
     Fail: <= 6/9 ⇒ models too undertrained / pools mismatched, pivot to MLE.

Run on JURECA login node:
    cd /p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
    /p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python -m code.attacks.task1_duci.diagnostics
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .data import MODEL_IDS, load_mixed, load_population, population_z
from .forward import forward_dataset
from .targets import load_target


def linear_probe_g0a(X_mixed: np.ndarray, X_pop: np.ndarray) -> float:
    """G0a: 5-fold CV AUC of LogReg on flattened raw pixels distinguishing MIXED vs POPULATION."""
    n_pop_used = min(len(X_pop), 4 * len(X_mixed))
    rng = np.random.default_rng(42)
    pop_idx = rng.choice(len(X_pop), size=n_pop_used, replace=False)
    X = np.concatenate([
        X_mixed.reshape(len(X_mixed), -1).astype(np.float32) / 255.0,
        X_pop[pop_idx].reshape(n_pop_used, -1).astype(np.float32) / 255.0,
    ])
    y = np.concatenate([np.ones(len(X_mixed)), np.zeros(n_pop_used)]).astype(np.int64)
    clf = LogisticRegression(max_iter=200, C=0.1, solver="liblinear")
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    return float(aucs.mean())


def conf_delta_g0b(device: str = "cuda") -> dict[str, dict[str, float]]:
    """G0b: per-target mean target-class softmax on MIXED vs POPULATION_z."""
    X_m, y_m = load_mixed()
    X_p_full, y_p_full = load_population()
    X_pz, y_pz = population_z(X_p_full, y_p_full)

    results: dict[str, dict[str, float]] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        model = load_target(mid, device=device)
        m_res = forward_dataset(model, X_m, y_m, device=device)
        p_res = forward_dataset(model, X_pz, y_pz, device=device)
        m_conf = float(m_res.target_class_conf.mean())
        p_conf = float(p_res.target_class_conf.mean())
        results[mid] = {
            "mean_conf_MIXED": m_conf,
            "mean_conf_POPULATION_z": p_conf,
            "delta": m_conf - p_conf,
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mid}  MIXED={m_conf:.4f}  POP_z={p_conf:.4f}  delta={m_conf - p_conf:+.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
    return results


def class_balance_check(y_m: np.ndarray, y_p: np.ndarray, n_classes: int = 100) -> tuple[float, bool]:
    """Chi-square distance between class histograms; True if 'balanced enough'."""
    h_m = np.bincount(y_m, minlength=n_classes) / len(y_m)
    h_p = np.bincount(y_p, minlength=n_classes) / len(y_p)
    chi2 = float(np.sum((h_m - h_p) ** 2 / np.maximum(h_p, 1e-6)))
    return chi2, chi2 < 0.5


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    print(f"MIXED:      X={X_m.shape} {X_m.dtype}, y={y_m.shape} (classes={int(y_m.max()) + 1})", flush=True)
    print(f"POPULATION: X={X_p.shape} {X_p.dtype}, y={y_p.shape} (classes={int(y_p.max()) + 1})", flush=True)

    print("\n--- Class balance check ---", flush=True)
    chi2, balanced = class_balance_check(y_m, y_p)
    print(f"  chi-square distance MIXED vs POPULATION class hist: {chi2:.4f}  "
          f"({'OK' if balanced else 'WARN: imbalance'})", flush=True)

    print("\n--- G0a: linear probe MIXED-vs-POPULATION on raw pixels ---", flush=True)
    t0 = time.time()
    auc_g0a = linear_probe_g0a(X_m, X_p)
    print(f"  AUC = {auc_g0a:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    if auc_g0a <= 0.55:
        print("  G0a PASS: pools look i.i.d.", flush=True)
    elif auc_g0a <= 0.60:
        print("  G0a MARGINAL: weak distribution shift, monitor", flush=True)
    else:
        print(f"  G0a FAIL: AUC > 0.60 — domain shift detected, flag in NOTES", flush=True)

    print("\n--- G0b: per-target conf-delta (MIXED minus POPULATION_z) ---", flush=True)
    deltas = conf_delta_g0b(device=device)

    n_pos = sum(1 for v in deltas.values() if v["delta"] > 0)
    print(f"\n  Positive deltas: {n_pos}/9", flush=True)
    by_arch = {"resnet18": [], "resnet50": [], "resnet152": []}
    arch_label = {"0": "resnet18", "1": "resnet50", "2": "resnet152"}
    for mid, d in deltas.items():
        arch = arch_label[mid.removeprefix("model_")[0]]
        by_arch[arch].append(d["delta"])
    for arch, arr in by_arch.items():
        if arr:
            print(f"  {arch:10}: mean delta = {np.mean(arr):+.4f}  "
                  f"min/max = {np.min(arr):+.4f} / {np.max(arr):+.4f}", flush=True)
    if n_pos >= 7:
        print("  G0b PASS: MIA signal exists across most targets", flush=True)
    else:
        print(f"  G0b FAIL: only {n_pos}/9 positive deltas — pivot to Avg-Logit MLE backup", flush=True)


if __name__ == "__main__":
    main()
