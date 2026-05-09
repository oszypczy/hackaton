"""
Phase 4 backup: Avg-Logit MLE.

Calibrate p̂ via synthetic targets trained at known p in {0, 0.25, 0.5, 0.75, 1.0}.
For each candidate signal s:
    1. compute s(synth_p) for all 5 synth targets
    2. fit linear regression: s = a*p + b
    3. for each real target t, compute s(t) and predict p_t = (s(t) - b) / a
    4. clamp to [0.025, 0.975]

Signals tried:
    - mean_logit_mixed     : mean target-class logit on MIXED
    - mean_conf_mixed      : mean target-class softmax on MIXED
    - mean_loss_mixed      : mean -log(target-class softmax) on MIXED
    - delta_conf           : (mean_conf MIXED) - (mean_conf POP_z) - self-normalized

Reports cross-val MAE per signal (leave-one-synth-out) - picks best.

CLI:
    --synth-dir <dir>      directory of synth_<arch>_<pNN>.{pt,json}
    --out <csv>            output CSV path
    [--use-signal name]    force a specific signal (skips selection)
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--use-signal", type=str, default="",
                    help="force signal: mean_logit_mixed | mean_conf_mixed | mean_loss_mixed | delta_conf")
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
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
    """Return all 4 signals for a single model."""
    res_m = forward_dataset(model, X_m, y_m, device=device,
                            return_softmax=True, return_logits=True)
    res_z = forward_dataset(model, X_z, y_z, device=device,
                            return_softmax=True, return_logits=False)

    target_logits_m = np.take_along_axis(res_m.logits, y_m[:, None].astype(np.int64), axis=1).squeeze(1)
    target_conf_m = res_m.target_class_conf
    target_conf_z = res_z.target_class_conf
    target_loss_m = -np.log(np.clip(target_conf_m, 1e-12, 1.0))

    return {
        "mean_logit_mixed": float(target_logits_m.mean()),
        "mean_conf_mixed": float(target_conf_m.mean()),
        "mean_loss_mixed": float(target_loss_m.mean()),
        "delta_conf": float(target_conf_m.mean() - target_conf_z.mean()),
    }


def fit_predict(s_synth: list[float], p_synth: list[float], s_target: float) -> float:
    """Linear regression s = a*p + b, return inverse for s_target."""
    s_arr = np.array(s_synth, dtype=np.float64)
    p_arr = np.array(p_synth, dtype=np.float64)
    a, b = np.polyfit(p_arr, s_arr, 1)
    if abs(a) < 1e-9:
        return 0.5
    return float((s_target - b) / a)


def loo_mae(s_synth: list[float], p_synth: list[float]) -> float:
    n = len(s_synth)
    errs = []
    for i in range(n):
        s_train = [s for j, s in enumerate(s_synth) if j != i]
        p_train = [p for j, p in enumerate(p_synth) if j != i]
        p_pred = fit_predict(s_train, p_train, s_synth[i])
        p_pred = np.clip(p_pred, 0.0, 1.0)
        errs.append(abs(p_pred - p_synth[i]))
    return float(np.mean(errs))


def run() -> None:
    args = parse_args()
    synth_dir = Path(args.synth_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mle] device={device}  synth-dir={synth_dir}", flush=True)

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[mle] MIXED N={len(X_m)}  POP_z N={len(X_z)}", flush=True)

    synths = discover_synth(synth_dir)
    if len(synths) < 3:
        raise RuntimeError(f"need >=3 synth targets in {synth_dir}, found {len(synths)}")
    print(f"[mle] {len(synths)} synth targets:")
    synth_signals: dict[str, list[float]] = {k: [] for k in
        ["mean_logit_mixed", "mean_conf_mixed", "mean_loss_mixed", "delta_conf"]}
    synth_p: list[float] = []
    for s in synths:
        t0 = time.time()
        model = build_resnet(s["arch_digit"], device=device, num_classes=100)
        with open(s["ckpt"], "rb") as f:
            state = _pkl.load(f)
        model.load_state_dict(state)
        model.train(False)

        sigs = compute_signals(model, X_m, y_m, X_z, y_z, device)
        for k, v in sigs.items():
            synth_signals[k].append(v)
        synth_p.append(s["true_p"])
        print(f"  synth p={s['true_p']:.2f}  "
              f"logit={sigs['mean_logit_mixed']:+.3f}  "
              f"conf={sigs['mean_conf_mixed']:.3f}  "
              f"loss={sigs['mean_loss_mixed']:.3f}  "
              f"delta={sigs['delta_conf']:+.3f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[mle] LOO-MAE per signal:")
    loo: dict[str, float] = {}
    for k, vs in synth_signals.items():
        mm = loo_mae(vs, synth_p)
        loo[k] = mm
        print(f"  {k:22s}  LOO-MAE = {mm:.4f}", flush=True)

    if args.use_signal:
        signal = args.use_signal
        print(f"[mle] using forced signal: {signal}", flush=True)
    else:
        signal = min(loo, key=loo.get)
        print(f"[mle] best signal: {signal}  (LOO-MAE = {loo[signal]:.4f})", flush=True)

    s_synth = synth_signals[signal]

    print(f"\n[mle] scoring 9 real targets via signal '{signal}':", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        model = load_target(mid, device=device)
        sigs = compute_signals(model, X_m, y_m, X_z, y_z, device)
        s_t = sigs[signal]
        p_raw = fit_predict(s_synth, synth_p, s_t)
        p_clamped = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        key = mid.removeprefix("model_")
        predictions[key] = p_clamped
        print(f"  {mid}  s={s_t:+.4f}  p_raw={p_raw:+.4f}  p_final={p_clamped:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_submission_csv(predictions, str(out_path))
    print(f"\n[mle] wrote {out_path}", flush=True)
    print(f"[mle] preds: " + "  ".join(f"{k}={predictions[k]:.3f}" for k in sorted(predictions)),
          flush=True)


if __name__ == "__main__":
    run()
