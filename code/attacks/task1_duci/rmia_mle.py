"""
RMIA-MLE: use mean(rmia_score) on MIXED as signal in MLE-style linear calibration.

Background (Phase A, 2026-05-09):
    Targets POP_z acc ~= 0.27 (all 3 archs); our refs/synth saturate at 0.17-0.19
    even at 200ep -- recipe gap (organizer's held-out filler != our POPULATION).
    Tong Eq.4 strict debias requires regime match -> fails (SUB-1: 0.46).

Pivot:
    1. Compute RMIA score per MIXED sample (uses target + refs Pr(x), Pr(z))
    2. Take mean across MIXED -> scalar signal per target
    3. Calibrate linear signal-vs-p on synth bank at known p (MLE-style)
    4. Predict p for real targets via linear inverse
    Bypasses Eq.4 TPR/FPR calibration -> robust to regime/arch mismatch.

Per-arch:
    arch=0 (R18): refs=R18 50ep, synth=R18 80ep
    arch=1 (R50): refs=R18 cross-arch, synth=R50
    arch=2 (R152): refs=R18 cross-arch, synth=R152

Trust boundary note: ref/synth checkpoints are stdlib-pickle dumps written by
our own train_ref.py / train_synth.py -- internal trust, same pattern as
extract_signals.py / mle.py / validate_synth.py. Organizer .pkl files use
targets.load_target.

Run on cluster:
    P4VENV=/p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python
    $P4VENV -m code.attacks.task1_duci.rmia_mle \\
        --refs-dir /p/scratch/.../DUCI/refs \\
        --synth-dir-r18 /p/scratch/.../DUCI/synth_targets_80ep_r18 \\
        --synth-dir-r50 /p/scratch/.../DUCI/synth_targets_80ep_r50 \\
        --synth-dir-r152 /p/scratch/.../DUCI/synth_targets_80ep_r152 \\
        --out submissions/task1_duci_rmia_mle.csv
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal trust boundary: own train_ref/train_synth outputs
import time
from pathlib import Path

import numpy as np
import torch

from .data import MODEL_IDS, load_mixed, load_population, population_z
from .debias import P_MAX, P_MIN, clamp, write_submission_csv
from .forward import forward_dataset
from .rmia import RmiaInputs, rmia_score
from .targets import build_resnet, load_target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", type=str, required=True,
                    help="dir with ref_<arch>_<seed>.pt + manifest_*.json (used cross-arch)")
    ap.add_argument("--synth-dir-r18", type=str, required=True,
                    help="synth dir for arch=0 R18 calibration")
    ap.add_argument("--synth-dir-r50", type=str, default="",
                    help="synth dir for arch=1 R50 calibration; empty = use r18 dir")
    ap.add_argument("--synth-dir-r152", type=str, default="",
                    help="synth dir for arch=2 R152 calibration; empty = use r18 dir")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    ap.add_argument("--no-clamp", action="store_true",
                    help="disable clamping (organizer says p is continuous + can be 0/1)")
    ap.add_argument("--r152-cal-from", type=str, default="2",
                    help="arch digit (0/1/2) to use for R152 targets calibration; "
                         "default 2 = own; '0' = use R18 cal cross-arch for R152")
    return ap.parse_args()


def discover_refs(refs_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(refs_dir.glob("manifest_*.json")):
        with open(jp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            continue
        out.append({"manifest": m, "ckpt": ckpt})
    return out


def discover_synth(synth_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(synth_dir.glob("synth_*.json")):
        with open(jp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            continue
        out.append({"info": m, "ckpt": ckpt, "true_p": float(m["true_p"]),
                    "arch_digit": str(m["arch_digit"])})
    return out


def load_state(path: Path):
    """Internal: load own train_ref/train_synth pickle dumps."""
    with open(path, "rb") as f:
        return _pkl.load(f)


def load_refs_signals(refs: list[dict], X_m: np.ndarray, y_m: np.ndarray,
                      X_z: np.ndarray, y_z: np.ndarray, device: str):
    cx_list, cz_list, mask_list = [], [], []
    for r in refs:
        m = r["manifest"]
        arch = m["arch_digit"]
        ckpt = r["ckpt"]
        t0 = time.time()
        model = build_resnet(arch, device=device, num_classes=100)
        model.load_state_dict(load_state(ckpt))
        model.train(False)
        cx = forward_dataset(model, X_m, y_m, device=device).target_class_conf
        cz = forward_dataset(model, X_z, y_z, device=device).target_class_conf
        cx_list.append(cx)
        cz_list.append(cz)
        train_set = set(m["train_indices_mixed"])
        mask = np.array([i in train_set for i in range(len(X_m))], dtype=bool)
        mask_list.append(mask)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  ref arch={arch} seed={m['seed']:2d} in_count={int(mask.sum()):4d}/{len(mask)} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return (np.stack(cx_list), np.stack(cz_list), np.stack(mask_list))


def compute_signal(target_conf_x: np.ndarray, target_conf_z: np.ndarray,
                   refs_conf_x: np.ndarray, refs_conf_z: np.ndarray,
                   ref_train_mask_x: np.ndarray) -> tuple[float, np.ndarray]:
    inputs = RmiaInputs(
        target_conf_x=target_conf_x,
        target_conf_z=target_conf_z,
        ref_conf_x=refs_conf_x,
        ref_conf_z=refs_conf_z,
        ref_train_mask_x=ref_train_mask_x,
    )
    scores = rmia_score(inputs)
    return float(scores.mean()), scores


def fit_linear(signals: list[float], ps: list[float]) -> tuple[float, float]:
    s = np.array(signals, dtype=np.float64)
    p = np.array(ps, dtype=np.float64)
    coeffs = np.polyfit(p, s, 1)
    return float(coeffs[0]), float(coeffs[1])


def predict_p(signal: float, a: float, b: float) -> float:
    if abs(a) < 1e-12:
        return 0.5
    return (signal - b) / a


def loo_mae(signals: list[float], ps: list[float]) -> tuple[float, list[float]]:
    n = len(signals)
    errs = []
    for i in range(n):
        s_train = [s for j, s in enumerate(signals) if j != i]
        p_train = [p for j, p in enumerate(ps) if j != i]
        a, b = fit_linear(s_train, p_train)
        p_pred = predict_p(signals[i], a, b)
        errs.append(abs(p_pred - ps[i]))
    return float(np.mean(errs)), errs


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[rmia_mle] device={device}  MIXED N={len(X_m)}  POP_z N={len(X_z)}", flush=True)

    refs_dir = Path(args.refs_dir)
    refs = discover_refs(refs_dir)
    if not refs:
        raise RuntimeError(f"no refs in {refs_dir}")
    print(f"[rmia_mle] {len(refs)} refs in {refs_dir.name}", flush=True)
    refs_conf_x, refs_conf_z, ref_train_mask_x = load_refs_signals(
        refs, X_m, y_m, X_z, y_z, device)

    synth_dirs = {
        "0": Path(args.synth_dir_r18),
        "1": Path(args.synth_dir_r50) if args.synth_dir_r50 else Path(args.synth_dir_r18),
        "2": Path(args.synth_dir_r152) if args.synth_dir_r152 else Path(args.synth_dir_r18),
    }
    print(f"[rmia_mle] synth dirs: 0={synth_dirs['0'].name}  1={synth_dirs['1'].name}  "
          f"2={synth_dirs['2'].name}", flush=True)

    arch_calib: dict[str, dict] = {}
    skip_archs: list[str] = []
    if args.r152_cal_from != "2":
        skip_archs.append("2")
        print(f"[rmia_mle] R152 cal will use arch={args.r152_cal_from} (cross-arch)", flush=True)
    for arch in ("0", "1", "2"):
        if arch in skip_archs:
            continue
        d = synth_dirs[arch]
        synths = [s for s in discover_synth(d) if s["arch_digit"] == arch]
        if len(synths) < 3:
            raise RuntimeError(f"need >=3 synths for arch={arch}, got {len(synths)} in {d}")
        synths.sort(key=lambda s: s["true_p"])

        signals_synth = []
        ps_synth = []
        for s in synths:
            t0 = time.time()
            model = build_resnet(arch, device=device, num_classes=100)
            model.load_state_dict(load_state(s["ckpt"]))
            model.train(False)
            tc_x = forward_dataset(model, X_m, y_m, device=device).target_class_conf
            tc_z = forward_dataset(model, X_z, y_z, device=device).target_class_conf
            sig, _ = compute_signal(tc_x, tc_z, refs_conf_x, refs_conf_z, ref_train_mask_x)
            signals_synth.append(sig)
            ps_synth.append(s["true_p"])
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  synth arch={arch} p={s['true_p']:.2f}  rmia_signal={sig:.4f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)

        a, b = fit_linear(signals_synth, ps_synth)
        loo, per_loo = loo_mae(signals_synth, ps_synth)
        verdict = "PASS" if loo < 0.040 else ("MARGINAL" if loo < 0.080 else "FAIL")
        print(f"  arch={arch}  fit: signal = {a:+.4f}*p + {b:+.4f}  LOO-MAE={loo:.4f} [{verdict}]",
              flush=True)
        for s_v, p_v, e in zip(signals_synth, ps_synth, per_loo):
            print(f"    p={p_v:.2f}  signal={s_v:.4f}  LOO|err|={e:.4f}", flush=True)
        arch_calib[arch] = {
            "a": a, "b": b, "loo": loo, "per_loo": per_loo,
            "signals": signals_synth, "ps": ps_synth,
        }

    # Cross-arch fallback for skipped archs
    for skip in skip_archs:
        src = args.r152_cal_from if skip == "2" else "0"
        if src not in arch_calib:
            raise RuntimeError(f"cannot fallback arch={skip} to arch={src}: src not calibrated")
        print(f"[rmia_mle] arch={skip} cal := arch={src} (cross-arch fallback)", flush=True)
        arch_calib[skip] = {**arch_calib[src], "from_arch": src}

    print("\n[rmia_mle] scoring 9 real targets:", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        arch = mid.removeprefix("model_")[0]
        cal = arch_calib[arch]
        model = load_target(mid, device=device)
        tc_x = forward_dataset(model, X_m, y_m, device=device).target_class_conf
        tc_z = forward_dataset(model, X_z, y_z, device=device).target_class_conf
        sig, _ = compute_signal(tc_x, tc_z, refs_conf_x, refs_conf_z, ref_train_mask_x)
        p_raw = predict_p(sig, cal["a"], cal["b"])
        p_final = p_raw if args.no_clamp else clamp(p_raw, args.clamp_lo, args.clamp_hi)
        p_final = float(np.clip(p_final, 0.0, 1.0))
        key = mid.removeprefix("model_")
        predictions[key] = p_final
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mid}  arch={arch}  signal={sig:.4f}  p_raw={p_raw:+.4f}  "
              f"p_final={p_final:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    write_submission_csv(predictions, str(out_path))
    print(f"\n[rmia_mle] wrote {out_path}", flush=True)
    print(f"[rmia_mle] preds: " + "  ".join(f"{k}={predictions[k]:.4f}"
                                             for k in sorted(predictions)), flush=True)
    print(f"[rmia_mle] LOO-MAE per arch: 0={arch_calib['0']['loo']:.4f}  "
          f"1={arch_calib['1']['loo']:.4f}  2={arch_calib['2']['loo']:.4f}", flush=True)


if __name__ == "__main__":
    run()
