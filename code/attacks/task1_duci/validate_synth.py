"""
Validate full RMIA + Tong-debias pipeline on synthetic targets with KNOWN p.

Reads synth_targets/synth_<arch>_<p>.{pt,json} written by train_synth.py.
Reuses references in --refs-dir (same as main.py).
Reports MAE vs known p, plus per-target |error|.

Decision gate G2C (per plan):
  MAE <= 0.06 ⇒ pipeline works, proceed
  0.06 < MAE <= 0.12 ⇒ marginal; submit but queue MLE backup
  MAE > 0.12 ⇒ pipeline broken, pivot to MLE
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_mixed, load_population, population_z
from .debias import P_MAX, P_MIN, clamp, debias, snap_5pct
from .forward import forward_dataset
from .main import discover_refs, load_ref_state
from .rmia import RmiaInputs, rmia_score
from .targets import build_resnet
from .threshold import select_beta_global


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", type=str, required=True)
    ap.add_argument("--synth-dir", type=str, required=True)
    ap.add_argument("--snap-grid", action="store_true")
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    return ap.parse_args()


def discover_synth(synth_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(synth_dir.glob("synth_*.json")):
        with open(jp) as f:
            info = json.load(f)
        ckpt = Path(info["checkpoint"])
        if not ckpt.exists():
            print(f"  [warn] manifest {jp.name} but checkpoint missing", flush=True)
            continue
        out.append({"info": info, "checkpoint_path": ckpt})
    return out


def main() -> None:
    args = parse_args()
    refs_dir = Path(args.refs_dir)
    synth_dir = Path(args.synth_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_m, y_m = load_mixed()
    X_p_full, y_p_full = load_population()
    X_z, y_z = population_z(X_p_full, y_p_full)
    print(f"[validate_synth] device={device}  MIXED N={len(X_m)}  POPULATION_z N={len(X_z)}",
          flush=True)

    # Load refs (same as main.py)
    refs = discover_refs(refs_dir)
    if not refs:
        raise RuntimeError(f"no references in {refs_dir}")
    print(f"[validate_synth] loaded {len(refs)} references", flush=True)

    refs_conf_x_list = []
    refs_conf_z_list = []
    ref_train_mask_x = []
    for r in refs:
        m = r["manifest"]
        model = build_resnet(m["arch_digit"], device=device, num_classes=100)
        model.load_state_dict(load_ref_state(r["checkpoint_path"]))
        model.train(False)
        cx = forward_dataset(model, X_m, y_m, device=device).target_class_conf
        cz = forward_dataset(model, X_z, y_z, device=device).target_class_conf
        refs_conf_x_list.append(cx)
        refs_conf_z_list.append(cz)
        s = set(m["train_indices_mixed"])
        ref_train_mask_x.append(np.array([i in s for i in range(len(X_m))], dtype=bool))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    refs_conf_x = np.stack(refs_conf_x_list)
    refs_conf_z = np.stack(refs_conf_z_list)
    ref_train_mask_x = np.stack(ref_train_mask_x)

    beta_star, tpr, fpr = select_beta_global(refs_conf_x, refs_conf_z, ref_train_mask_x)
    print(f"[validate_synth] β*={beta_star:.3f}  TPR̂={tpr:.3f}  FPR̂={fpr:.3f}", flush=True)

    # Score synthetic targets
    synth = discover_synth(synth_dir)
    if not synth:
        raise RuntimeError(f"no synthetic targets in {synth_dir}")
    print(f"[validate_synth] {len(synth)} synthetic targets", flush=True)

    abs_errors = []
    print(f"\n  {'true_p':>8}  {'p̂_raw':>10}  {'p̂_clamped':>10}  {'|err|':>8}", flush=True)
    print(f"  " + "-" * 44, flush=True)
    for s in synth:
        info = s["info"]
        true_p = info["true_p"]
        arch = info["arch_digit"]
        t0 = time.time()
        model = build_resnet(arch, device=device, num_classes=100)
        with open(s["checkpoint_path"], "rb") as f:
            model.load_state_dict(pickle.load(f))
        model.train(False)
        target_conf_x = forward_dataset(model, X_m, y_m, device=device).target_class_conf
        target_conf_z = forward_dataset(model, X_z, y_z, device=device).target_class_conf
        inputs = RmiaInputs(
            target_conf_x=target_conf_x,
            target_conf_z=target_conf_z,
            ref_conf_x=refs_conf_x,
            ref_conf_z=refs_conf_z,
            ref_train_mask_x=ref_train_mask_x,
        )
        scores = rmia_score(inputs)
        m_hat = (scores >= beta_star).astype(np.float32)
        p_raw = debias(m_hat, tpr, fpr)
        p_final = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        if args.snap_grid:
            p_final = snap_5pct(p_final)
        err = abs(p_final - true_p)
        abs_errors.append(err)
        print(f"  {true_p:>8.3f}  {p_raw:>+10.4f}  {p_final:>10.4f}  {err:>8.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    mae = float(np.mean(abs_errors))
    print(f"\n[validate_synth] MAE = {mae:.4f}  (N={len(abs_errors)})", flush=True)
    if mae <= 0.06:
        verdict = "G2C PASS — pipeline works, proceed to submission"
    elif mae <= 0.12:
        verdict = "G2C MARGINAL — submit but queue Avg-Logit MLE backup"
    else:
        verdict = "G2C FAIL — pivot to Avg-Logit MLE backup"
    print(f"[validate_synth] {verdict}", flush=True)

    # Spearman-ish ordering check
    true_ps = [s["info"]["true_p"] for s in synth]
    pred_ps = []  # need re-pass — store as we go above
    # (already printed; recompute via second pass not needed; trust per-target |err|)


if __name__ == "__main__":
    main()
