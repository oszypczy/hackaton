"""
Combine multiple synth dirs into one virtual synth set, then run MLE.

Reads --synth-dirs (comma-separated), aggregates all manifests + ckpts.
For multi-seed averaging: pass two dirs of same arch — both will contribute
synth points at potentially same p (averaged in fit by linear regression).

For dense-calibration: pass synth_targets_20ep + synth_targets_20ep_extra
(different p values) — combined gives 13 points.
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl
import time
from pathlib import Path

import numpy as np
import torch

from .data import MODEL_IDS, load_mixed, load_population, population_z
from .debias import P_MAX, P_MIN, clamp, write_submission_csv
from .forward import forward_dataset
from .mle import SIGNALS, compute_signals, fit_predict_poly, loo_mae_poly
from .targets import build_resnet, load_target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dirs", type=str, required=True,
                    help="comma-separated list of synth dirs to combine for arch=0")
    ap.add_argument("--synth-dirs-r50", type=str, default="")
    ap.add_argument("--synth-dirs-r152", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--use-signal", type=str, default="mean_loss_mixed")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    return ap.parse_args()


def discover_combined(dirs_csv: str, want_arch: str | None = None) -> list[dict]:
    out = []
    for d in dirs_csv.split(","):
        d = d.strip()
        if not d:
            continue
        for jp in sorted(Path(d).glob("synth_*.json")):
            with open(jp) as f:
                m = json.load(f)
            ckpt = Path(m["checkpoint"])
            if not ckpt.exists():
                continue
            if want_arch is not None and str(m["arch_digit"]) != want_arch:
                continue
            out.append({"manifest": m, "ckpt": ckpt, "true_p": float(m["true_p"]),
                        "arch_digit": str(m["arch_digit"])})
    out.sort(key=lambda x: (x["true_p"], x["ckpt"].name))
    return out


def evaluate(synths, X_m, y_m, X_z, y_z, device: str):
    sigs_per: dict[str, list[float]] = {k: [] for k in SIGNALS}
    ps: list[float] = []
    for s in synths:
        model = build_resnet(s["arch_digit"], device=device, num_classes=100)
        with open(s["ckpt"], "rb") as f:
            state = _pkl.load(f)
        model.load_state_dict(state)
        model.train(False)
        sigs = compute_signals(model, X_m, y_m, X_z, y_z, device)
        for k, v in sigs.items():
            sigs_per[k].append(v)
        ps.append(s["true_p"])
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return sigs_per, ps


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[mlec] device={device}  MIXED N={len(X_m)}  POP_z N={len(X_z)}", flush=True)

    archs = {
        "0": args.synth_dirs,
        "1": args.synth_dirs_r50 if args.synth_dirs_r50 else args.synth_dirs,
        "2": args.synth_dirs_r152 if args.synth_dirs_r152 else args.synth_dirs,
    }

    arch_calib: dict[str, dict] = {}
    for arch in ("0", "1", "2"):
        synths = discover_combined(archs[arch], arch)
        if len(synths) < 3:
            # Fallback: use ANY arch from same dirs (cross-arch calibration)
            synths_any = discover_combined(archs[arch], None)
            if len(synths_any) >= 3:
                print(f"  [warn] no arch={arch} synths in {archs[arch]}; "
                      f"falling back to {len(synths_any)} cross-arch synths", flush=True)
                synths = synths_any
            else:
                raise RuntimeError(f"need >=3 synths for arch={arch}, got {len(synths)} in {archs[arch]}")
        print(f"\n[mlec] arch={arch}: {len(synths)} synth points (combined)", flush=True)
        for s in synths:
            print(f"    {s['ckpt'].parent.name}/{s['ckpt'].name}  p={s['true_p']:.2f}")
        sigs_per, ps = evaluate(synths, X_m, y_m, X_z, y_z, device)
        sig = args.use_signal
        loo = loo_mae_poly(sigs_per[sig], ps, args.degree)
        zipped = sorted(zip(ps, sigs_per[sig]))
        print(f"  signal={sig} deg={args.degree} LOO-MAE={loo:.4f}  "
              f"data: " + " ".join(f"({p:.2f},{v:+.2f})" for p, v in zipped))
        arch_calib[arch] = {"sigs_per": sigs_per, "ps": ps}

    print(f"\n[mlec] scoring 9 real targets:", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        arch = mid.removeprefix("model_")[0]
        cal = arch_calib[arch]
        ps_synth = cal["ps"]

        model = load_target(mid, device=device)
        sigs_t = compute_signals(model, X_m, y_m, X_z, y_z, device)
        if args.ensemble:
            preds = []
            for k in SIGNALS:
                p_k = fit_predict_poly(cal["sigs_per"][k], ps_synth, sigs_t[k], args.degree)
                preds.append(p_k)
            p_raw = float(np.mean(preds))
        else:
            sig = args.use_signal
            p_raw = fit_predict_poly(cal["sigs_per"][sig], ps_synth, sigs_t[sig], args.degree)

        p_clamped = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        key = mid.removeprefix("model_")
        predictions[key] = p_clamped
        print(f"  {mid}  arch={arch}  p_raw={p_raw:+.4f}  p_final={p_clamped:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_submission_csv(predictions, str(out_path))
    print(f"\n[mlec] wrote {out_path}", flush=True)
    print(f"[mlec] preds: " + "  ".join(f"{k}={predictions[k]:.3f}" for k in sorted(predictions)),
          flush=True)


if __name__ == "__main__":
    run()
