"""
Phase 4 backup: Avg-Logit MLE (with polynomial fit, signal ensemble, per-arch synth).

Calibrate p_hat via synthetic targets trained at known p in {0, 0.25, 0.5, 0.75, 1.0}+.

For each candidate signal s and each polynomial degree d:
    1. compute s(synth_p) for all synth targets
    2. fit polynomial regression of degree d: s = poly(p)
    3. invert numerically (binary search over [0,1]) for each real target
    4. clamp to [0.025, 0.975]

Pick best (signal, degree) by leave-one-out MAE on synth.

Per-arch: if --synth-dir-r50 / --synth-dir-r152 given, use those for arch=1/2 targets.
Otherwise use --synth-dir for everything.

Signals:
    - mean_logit_mixed     : mean target-class logit on MIXED
    - mean_conf_mixed      : mean target-class softmax on MIXED
    - mean_loss_mixed      : mean -log(target-class softmax) on MIXED
    - delta_conf           : (mean_conf MIXED) - (mean_conf POP_z)
    - delta_loss           : (mean_loss POP_z) - (mean_loss MIXED)
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
from .targets import build_resnet, load_target

SIGNALS = ["mean_logit_mixed", "mean_conf_mixed", "mean_loss_mixed",
           "delta_conf", "delta_loss"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=str, required=True,
                    help="default synth dir (used for arch=0 and as fallback)")
    ap.add_argument("--synth-dir-r50", type=str, default="",
                    help="optional arch-matched synth dir for R50 targets")
    ap.add_argument("--synth-dir-r152", type=str, default="",
                    help="optional arch-matched synth dir for R152 targets")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--use-signal", type=str, default="")
    ap.add_argument("--degree", type=int, default=0,
                    help="0 = auto-pick by LOO; otherwise force polynomial degree (1, 2, 3)")
    ap.add_argument("--ensemble", action="store_true",
                    help="average predictions across all signals at chosen degree")
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    ap.add_argument("--dump-signals", type=str, default="",
                    help="optional path to JSON dump of synth calibration + per-target signals "
                         "(for downstream plotting / presentation; default empty = no dump)")
    return ap.parse_args()


def discover_synth(synth_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(synth_dir.glob("synth_*.json")):
        with open(jp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            print(f"  [warn] manifest {jp.name} but ckpt missing: {ckpt}", flush=True)
            continue
        out.append({"manifest": m, "ckpt": ckpt, "true_p": float(m["true_p"]),
                    "arch_digit": m["arch_digit"]})
    return out


def compute_signals(model, X_m, y_m, X_z, y_z, device: str) -> dict:
    res_m = forward_dataset(model, X_m, y_m, device=device,
                            return_softmax=True, return_logits=True)
    res_z = forward_dataset(model, X_z, y_z, device=device,
                            return_softmax=True, return_logits=False)
    target_logits_m = np.take_along_axis(res_m.logits, y_m[:, None].astype(np.int64), axis=1).squeeze(1)
    target_conf_m = res_m.target_class_conf
    target_conf_z = res_z.target_class_conf
    target_loss_m = -np.log(np.clip(target_conf_m, 1e-12, 1.0))
    target_loss_z = -np.log(np.clip(target_conf_z, 1e-12, 1.0))
    return {
        "mean_logit_mixed": float(target_logits_m.mean()),
        "mean_conf_mixed": float(target_conf_m.mean()),
        "mean_loss_mixed": float(target_loss_m.mean()),
        "delta_conf": float(target_conf_m.mean() - target_conf_z.mean()),
        "delta_loss": float(target_loss_z.mean() - target_loss_m.mean()),
    }


def fit_predict_poly(s_synth, p_synth, s_target, degree: int) -> float:
    s_arr = np.array(s_synth, dtype=np.float64)
    p_arr = np.array(p_synth, dtype=np.float64)
    if len(p_arr) <= degree:
        degree = max(1, len(p_arr) - 1)
    coeffs = np.polyfit(p_arr, s_arr, degree)

    if degree == 1:
        # Analytical inverse: s = a*p + b => p = (s - b) / a
        a, b = coeffs
        if abs(a) < 1e-12:
            return 0.5
        return float((s_target - b) / a)

    # Higher-degree: solve poly(p) - s_target = 0 via numpy roots, pick real root in [0,1]
    poly_eq = np.poly1d(coeffs - np.array([0] * (len(coeffs) - 1) + [s_target]))
    roots = np.roots(poly_eq)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    if real_roots:
        in_range = [r for r in real_roots if -0.5 <= r <= 1.5]
        if in_range:
            return float(min(in_range, key=lambda r: abs(r - 0.5)))
        return float(min(real_roots, key=lambda r: min(abs(r), abs(r - 1))))
    # Fallback: dense grid
    p_grid = np.linspace(0.0, 1.0, 100001)
    s_grid = np.poly1d(coeffs)(p_grid)
    if not (s_grid.max() - s_grid.min()) > 1e-9:
        return 0.5
    return float(p_grid[int(np.argmin(np.abs(s_grid - s_target)))])


def loo_mae_poly(s_synth, p_synth, degree: int) -> float:
    n = len(s_synth)
    errs = []
    for i in range(n):
        s_train = [s for j, s in enumerate(s_synth) if j != i]
        p_train = [p for j, p in enumerate(p_synth) if j != i]
        p_pred = fit_predict_poly(s_train, p_train, s_synth[i], degree)
        errs.append(abs(p_pred - p_synth[i]))
    return float(np.mean(errs))


def evaluate_synth(synths, X_m, y_m, X_z, y_z, device: str):
    """Return dict signal -> list of (signal_value, true_p) sorted by p."""
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


def pick_best_signal_degree(sigs_per, ps, degrees=(1, 2)):
    best = None
    table = []
    for k in SIGNALS:
        for d in degrees:
            try:
                m = loo_mae_poly(sigs_per[k], ps, d)
            except Exception:
                m = float("inf")
            table.append((m, k, d))
            if best is None or m < best[0]:
                best = (m, k, d)
    return best, table


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[mle] device={device}  MIXED N={len(X_m)}  POP_z N={len(X_z)}", flush=True)

    arch_dirs = {"0": Path(args.synth_dir)}
    if args.synth_dir_r50:
        arch_dirs["1"] = Path(args.synth_dir_r50)
    else:
        arch_dirs["1"] = Path(args.synth_dir)
    if args.synth_dir_r152:
        arch_dirs["2"] = Path(args.synth_dir_r152)
    else:
        arch_dirs["2"] = Path(args.synth_dir)
    print(f"[mle] arch dirs: 0={arch_dirs['0']}  1={arch_dirs['1']}  2={arch_dirs['2']}", flush=True)

    arch_calib: dict[str, dict] = {}
    for arch in ("0", "1", "2"):
        d = arch_dirs[arch]
        synths = discover_synth(d)
        synths_arch = [s for s in synths if s["arch_digit"] == arch]
        if not synths_arch:
            print(f"  [warn] no arch={arch} synths in {d}; using all available", flush=True)
            synths_arch = synths
        if len(synths_arch) < 3:
            raise RuntimeError(f"need >=3 synths for arch={arch}, got {len(synths_arch)}")
        print(f"\n[mle] arch={arch}: {len(synths_arch)} synth targets from {d.name}", flush=True)
        sigs_per, ps = evaluate_synth(synths_arch, X_m, y_m, X_z, y_z, device)
        for k in SIGNALS:
            zipped = sorted(zip(ps, sigs_per[k]))
            print(f"  {k:20s}  " + "  ".join(f"p={p:.2f}:{v:+.3f}" for p, v in zipped), flush=True)
        best, table = pick_best_signal_degree(sigs_per, ps)
        print(f"  best: {best[1]} deg={best[2]} LOO-MAE={best[0]:.4f}", flush=True)
        arch_calib[arch] = {"sigs_per": sigs_per, "ps": ps,
                            "best_signal": best[1], "best_degree": best[2],
                            "best_loo": best[0], "table": table}

    if args.use_signal:
        for a in arch_calib:
            arch_calib[a]["best_signal"] = args.use_signal
    if args.degree > 0:
        for a in arch_calib:
            arch_calib[a]["best_degree"] = args.degree

    print(f"\n[mle] scoring 9 real targets:", flush=True)
    predictions: dict[str, float] = {}
    target_signals: dict[str, dict[str, float]] = {}
    target_p_raw: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        arch = mid.removeprefix("model_")[0]
        cal = arch_calib[arch]
        signal = cal["best_signal"]
        degree = cal["best_degree"]
        sigs_synth = cal["sigs_per"][signal]
        ps_synth = cal["ps"]

        model = load_target(mid, device=device)
        sigs_t = compute_signals(model, X_m, y_m, X_z, y_z, device)
        s_t = sigs_t[signal]

        if args.ensemble:
            preds = []
            for k in SIGNALS:
                p_k = fit_predict_poly(cal["sigs_per"][k], ps_synth, sigs_t[k], degree)
                preds.append(p_k)
            p_raw = float(np.mean(preds))
            note = f"ensemble({signal}deg={degree})"
        else:
            p_raw = fit_predict_poly(sigs_synth, ps_synth, s_t, degree)
            note = f"{signal}deg={degree}"

        p_clamped = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        key = mid.removeprefix("model_")
        predictions[key] = p_clamped
        target_signals[key] = {k: float(v) for k, v in sigs_t.items()}
        target_p_raw[key] = float(p_raw)
        print(f"  {mid}  arch={arch}  s={s_t:+.4f}  p_raw={p_raw:+.4f}  "
              f"p_final={p_clamped:.4f}  ({note}, {time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_submission_csv(predictions, str(out_path))
    print(f"\n[mle] wrote {out_path}", flush=True)
    print(f"[mle] preds: " + "  ".join(f"{k}={predictions[k]:.3f}" for k in sorted(predictions)),
          flush=True)

    if args.dump_signals:
        dump_path = Path(args.dump_signals)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        arch_names = {"0": "ResNet18", "1": "ResNet50", "2": "ResNet152"}
        synth_block = {}
        for arch_key in ("0", "1", "2"):
            cal = arch_calib[arch_key]
            best_sig = cal["best_signal"]
            best_deg = cal["best_degree"]
            sigs = cal["sigs_per"][best_sig]
            ps = cal["ps"]
            zipped = sorted(zip(ps, sigs))
            ps_sorted = [float(p) for p, _ in zipped]
            sigs_sorted = [float(s) for _, s in zipped]
            poly_coeffs = list(map(float, np.polyfit(ps_sorted, sigs_sorted, best_deg)))
            loo_actual = loo_mae_poly(sigs_sorted, ps_sorted, best_deg)
            synth_block[arch_key] = {
                "arch_name": arch_names[arch_key],
                "best_signal": best_sig,
                "best_degree": int(best_deg),
                "ps": ps_sorted,
                "signals": sigs_sorted,
                "poly_coeffs": poly_coeffs,
                "loo_mae": float(loo_actual),
                "all_signals": {
                    k: [float(v) for _, v in sorted(zip(ps, cal["sigs_per"][k]))]
                    for k in SIGNALS
                },
            }
        targets_block = []
        for key in sorted(predictions):
            arch_key = key[0]
            best_sig = arch_calib[arch_key]["best_signal"]
            targets_block.append({
                "model_id": key,
                "arch": arch_key,
                "arch_name": arch_names[arch_key],
                "best_signal": best_sig,
                "signal": float(target_signals[key][best_sig]),
                "all_signals": target_signals[key],
                "p_raw": target_p_raw[key],
                "p_hat": float(predictions[key]),
            })
        payload = {
            "method": "Avg-loss MLE (synth-calibrated polynomial inversion)",
            "synth": synth_block,
            "targets": targets_block,
        }
        with open(dump_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[mle] dumped signals to {dump_path}", flush=True)


if __name__ == "__main__":
    run()
