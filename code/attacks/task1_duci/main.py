"""
Task 1 (DUCI) orchestrator.

CLI:
    --refs-dir <dir>    directory containing ref_<arch>_<seed>.pt + manifest_<arch>_<seed>.json
    --out <path>        output submission CSV path (must be named submission.csv per spec)
    [--snap-grid]       snap p̂ to nearest 0.05
    [--clamp-lo F]      clamp lower bound (default 0.025)
    [--clamp-hi F]      clamp upper bound (default 0.975)
    [--n-z 4096]        number of POPULATION_z samples to use (default: full 5000)

Pipeline:
    1. Load 9 organizer targets (.pkl).
    2. Load all references in --refs-dir.
    3. Forward-pass MIXED + POPULATION_z through all (1 + 9 + N_refs) models, get target-class softmax.
    4. β* via Youden's index across reference leave-one-out (threshold.py).
    5. Per target: rmia_score → m̂ at β* → debias to p̂ → clamp.
    6. Write submission CSV.

Run on cluster (login node ok for inference):
    P4VENV=/p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python
    $P4VENV -m code.attacks.task1_duci.main --refs-dir /p/scratch/.../Czumpers/DUCI/refs \
        --out /p/scratch/.../Czumpers/DUCI/submission.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .data import (
    DUCI_ROOT,
    MODEL_IDS,
    load_mixed,
    load_population,
    population_z,
)
from .debias import P_MAX, P_MIN, clamp, debias, snap_5pct, write_submission_csv
from .forward import forward_dataset
from .rmia import RmiaInputs, rmia_score
from .targets import build_resnet, load_target
from .threshold import select_beta_global


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--snap-grid", action="store_true")
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    ap.add_argument("--n-z", type=int, default=0,
                    help="0 = use all POPULATION_z; otherwise random subsample")
    ap.add_argument("--rmia-seed", type=int, default=0,
                    help="seed for POPULATION_z subsample if --n-z > 0")
    return ap.parse_args()


def discover_refs(refs_dir: Path) -> list[dict]:
    """Discover reference checkpoints + manifests in refs_dir.

    Returns list of {checkpoint, manifest} dicts sorted by (arch_digit, seed).
    """
    out = []
    for mp in sorted(refs_dir.glob("manifest_*.json")):
        with open(mp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            print(f"  [warn] manifest {mp.name} but checkpoint missing: {ckpt}", flush=True)
            continue
        out.append({"manifest": m, "manifest_path": mp, "checkpoint_path": ckpt})
    return out


def load_ref_state(checkpoint_path: Path):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def forward_target_class(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                         device: str) -> np.ndarray:
    return forward_dataset(model, X, y, device=device).target_class_conf


def run() -> None:
    args = parse_args()
    refs_dir = Path(args.refs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}", flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_m, y_m = load_mixed()
    X_p_full, y_p_full = load_population()
    X_z, y_z = population_z(X_p_full, y_p_full)

    if args.n_z > 0 and args.n_z < len(X_z):
        rng = np.random.default_rng(args.rmia_seed)
        idx = rng.choice(len(X_z), size=args.n_z, replace=False)
        X_z, y_z = X_z[idx], y_z[idx]
    print(f"[main] MIXED: N={len(X_m)}  POPULATION_z used: N={len(X_z)}", flush=True)

    # ── Discover references ───────────────────────────────────────────────────
    refs = discover_refs(refs_dir)
    if not refs:
        raise RuntimeError(f"no references found in {refs_dir}")
    print(f"[main] found {len(refs)} references", flush=True)

    # ── Forward-pass refs ─────────────────────────────────────────────────────
    refs_conf_x = []  # (N_refs, N_x)
    refs_conf_z = []  # (N_refs, N_z)
    ref_train_mask_x = []  # (N_refs, N_x) bool

    for r in refs:
        m = r["manifest"]
        arch = m["arch_digit"]
        ckpt_path = r["checkpoint_path"]
        t0 = time.time()
        model = build_resnet(arch, device=device, num_classes=100)
        state = load_ref_state(ckpt_path)
        model.load_state_dict(state)
        model.train(False)

        conf_x = forward_target_class(model, X_m, y_m, device=device)
        conf_z = forward_target_class(model, X_z, y_z, device=device)
        refs_conf_x.append(conf_x)
        refs_conf_z.append(conf_z)

        train_idx_set = set(m["train_indices_mixed"])
        mask = np.array([i in train_idx_set for i in range(len(X_m))], dtype=bool)
        ref_train_mask_x.append(mask)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  ref arch={arch} seed={m['seed']}  "
              f"in_count={int(mask.sum())}/{len(mask)}  ({time.time() - t0:.1f}s)", flush=True)

    refs_conf_x = np.stack(refs_conf_x)
    refs_conf_z = np.stack(refs_conf_z)
    ref_train_mask_x = np.stack(ref_train_mask_x)

    # ── β* selection ─────────────────────────────────────────────────────────
    print("\n[main] selecting β* via Youden's index (leave-one-ref-out)", flush=True)
    beta_star, tpr_hat, fpr_hat = select_beta_global(refs_conf_x, refs_conf_z, ref_train_mask_x)
    print(f"  β* = {beta_star:.3f}  TPR̂ = {tpr_hat:.3f}  FPR̂ = {fpr_hat:.3f}  "
          f"Youden gap = {tpr_hat - fpr_hat:+.3f}", flush=True)

    if not np.isfinite(tpr_hat):
        # single-ref fallback: estimate TPR/FPR from the one ref directly
        in_mask = ref_train_mask_x[0]
        # build a synthetic "target" RMIA score using the ref as both target and ref
        single_ref_inputs = RmiaInputs(
            target_conf_x=refs_conf_x[0],
            target_conf_z=refs_conf_z[0],
            ref_conf_x=refs_conf_x,
            ref_conf_z=refs_conf_z,
            ref_train_mask_x=ref_train_mask_x,
        )
        scores = rmia_score(single_ref_inputs)
        ind = (scores >= beta_star).astype(np.float32)
        tpr_hat = float(ind[in_mask].mean()) if in_mask.any() else 0.6
        fpr_hat = float(ind[~in_mask].mean()) if (~in_mask).any() else 0.4
        print(f"  [single-ref fallback] TPR̂ = {tpr_hat:.3f}  FPR̂ = {fpr_hat:.3f}", flush=True)

    # ── Forward-pass 9 targets ────────────────────────────────────────────────
    print("\n[main] scoring 9 targets", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        model = load_target(mid, device=device)
        target_conf_x = forward_target_class(model, X_m, y_m, device=device)
        target_conf_z = forward_target_class(model, X_z, y_z, device=device)

        inputs = RmiaInputs(
            target_conf_x=target_conf_x,
            target_conf_z=target_conf_z,
            ref_conf_x=refs_conf_x,
            ref_conf_z=refs_conf_z,
            ref_train_mask_x=ref_train_mask_x,
        )
        scores = rmia_score(inputs)
        m_hat = (scores >= beta_star).astype(np.float32)

        p_raw = debias(m_hat, tpr_hat, fpr_hat)
        p_clamped = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        if args.snap_grid:
            p_clamped = snap_5pct(p_clamped)
        # CSV expects model_id as '00' '01' etc., matching debias.write_submission_csv conventions
        key = mid.removeprefix("model_")
        predictions[key] = p_clamped
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mid}  m̂_mean={m_hat.mean():.3f}  p̂_raw={p_raw:+.4f}  "
              f"p̂_final={p_clamped:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    # ── Write submission CSV ──────────────────────────────────────────────────
    write_submission_csv(predictions, str(out_path))
    print(f"\n[main] wrote {out_path}", flush=True)
    print(f"[main] preds: " + "  ".join(f"{k}={predictions[k]:.3f}" for k in sorted(predictions)),
          flush=True)


if __name__ == "__main__":
    run()
