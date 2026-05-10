"""
Extract a rich signal vector for every (organizer_target, synth_target) and dump to npz.

Saved arrays (one row per model):
    model_id : "model_00".."model_22" or "synth_<arch>_<p>"
    arch     : "0"|"1"|"2"
    true_p   : float (NaN for organizer targets, exact p for synth)
    is_synth : bool
    mean_loss_mixed         : mean -log p(y|x) on MIXED
    mean_loss_pop           : mean -log p(y|x) on POP_Z
    delta_loss              : mean_loss_pop - mean_loss_mixed
    loss_ratio              : mean_loss_mixed / mean_loss_pop
    p25_loss_mixed          : 25th percentile of per-sample loss on MIXED
    p10_loss_mixed          : 10th percentile (top 10% easiest) on MIXED
    p75_loss_mixed          : 75th percentile (top 25% hardest)
    aug_invariance_mixed    : mean across MIXED of (1 - var of target_conf across 4 augs)
    aug_invariance_pop      : same on POP_Z (baseline / normaliser)
    aug_inv_diff            : aug_invariance_mixed - aug_invariance_pop
    mean_logit_mixed        : mean target-class raw logit on MIXED
    mean_conf_mixed         : mean target-class softmax on MIXED
    mean_conf_pop           : mean target-class softmax on POP_Z
    delta_conf              : mean_conf_mixed - mean_conf_pop

NOTE: synth checkpoint state dicts are stdlib-pickle dumps written by our own
train_synth.py (not user content); same pattern as mle.py. Internal trust boundary.

Synth targets are loaded from any number of dirs passed via --synth-dirs.
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal: loads our own train_synth.py outputs
import time
from pathlib import Path

import numpy as np
import torch

from .data import (
    BATCH,
    CIFAR_MEAN,
    CIFAR_STD,
    MODEL_IDS,
    load_mixed,
    load_population,
    population_z,
)
from .targets import build_resnet, load_target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dirs", type=str, required=True,
                    help="comma-separated list of synth_targets_* dirs")
    ap.add_argument("--out", type=str, required=True, help="output .npz path")
    ap.add_argument("--n-aug", type=int, default=4,
                    help="number of augmented forward passes per sample")
    ap.add_argument("--include-targets", action="store_true", default=True)
    return ap.parse_args()


def discover_synths(dirs_csv: str) -> list[dict]:
    out = []
    for d_str in dirs_csv.split(","):
        d = Path(d_str.strip())
        if not d.exists():
            print(f"[warn] dir missing: {d}", flush=True)
            continue
        for jp in sorted(d.glob("synth_*.json")):
            with open(jp) as f:
                m = json.load(f)
            ckpt = Path(m["checkpoint"])
            if not ckpt.exists():
                continue
            out.append({
                "model_id": jp.stem,
                "arch": str(m["arch_digit"]),
                "true_p": float(m["true_p"]),
                "ckpt": ckpt,
                "is_synth": True,
                "manifest_dir": d.name,
            })
    return out


@torch.no_grad()
def forward_softmax(model, X: np.ndarray, y: np.ndarray, device: str):
    n = len(X)
    tc = np.empty(n, dtype=np.float32)
    tl = np.empty(n, dtype=np.float32)
    for i in range(0, n, BATCH):
        xb = torch.from_numpy(X[i:i + BATCH]).to(device).permute(0, 3, 1, 2).float() / 255.0
        xb = (xb - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
        logits = model(xb)
        sm = torch.softmax(logits, dim=1)
        yb = torch.from_numpy(y[i:i + BATCH]).long().to(device)
        tc[i:i + BATCH] = sm.gather(1, yb.unsqueeze(1)).squeeze(1).cpu().numpy()
        tl[i:i + BATCH] = logits.gather(1, yb.unsqueeze(1)).squeeze(1).cpu().numpy()
    return tc, tl


@torch.no_grad()
def forward_aug_target_conf(model, X: np.ndarray, y: np.ndarray, device: str,
                            n_aug: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    out = np.empty((n, n_aug), dtype=np.float32)
    pad = 4
    for k in range(n_aug):
        flips = rng.random(n) < 0.5
        crop_x = rng.integers(0, 2 * pad + 1, size=n)
        crop_y = rng.integers(0, 2 * pad + 1, size=n)
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            xb_np = X[i:j]
            xb = torch.from_numpy(xb_np).to(device).permute(0, 3, 1, 2).float() / 255.0
            xb = torch.nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
            B = xb.shape[0]
            cropped = torch.empty(B, 3, 32, 32, device=device, dtype=xb.dtype)
            for b in range(B):
                cy, cx = int(crop_y[i + b]), int(crop_x[i + b])
                patch = xb[b:b + 1, :, cy:cy + 32, cx:cx + 32]
                if flips[i + b]:
                    patch = torch.flip(patch, dims=[3])
                cropped[b] = patch[0]
            cropped = (cropped - CIFAR_MEAN.to(device)) / CIFAR_STD.to(device)
            logits = model(cropped)
            sm = torch.softmax(logits, dim=1)
            yb = torch.from_numpy(y[i:j]).long().to(device)
            tc = sm.gather(1, yb.unsqueeze(1)).squeeze(1).cpu().numpy()
            out[i:j, k] = tc
    return out


def compute_model_signals(model, X_m, y_m, X_z, y_z, device, n_aug):
    tc_m, tl_m = forward_softmax(model, X_m, y_m, device)
    tc_z, _ = forward_softmax(model, X_z, y_z, device)
    aug_m = forward_aug_target_conf(model, X_m, y_m, device, n_aug, seed=42)
    aug_z = forward_aug_target_conf(model, X_z, y_z, device, n_aug, seed=42)

    loss_m = -np.log(np.clip(tc_m, 1e-12, 1.0))
    loss_z = -np.log(np.clip(tc_z, 1e-12, 1.0))
    inv_m = 1.0 - aug_m.var(axis=1)
    inv_z = 1.0 - aug_z.var(axis=1)

    return {
        "mean_loss_mixed": float(loss_m.mean()),
        "mean_loss_pop": float(loss_z.mean()),
        "delta_loss": float(loss_z.mean() - loss_m.mean()),
        "loss_ratio": float(loss_m.mean() / max(loss_z.mean(), 1e-9)),
        "p25_loss_mixed": float(np.percentile(loss_m, 25)),
        "p10_loss_mixed": float(np.percentile(loss_m, 10)),
        "p75_loss_mixed": float(np.percentile(loss_m, 75)),
        "aug_invariance_mixed": float(inv_m.mean()),
        "aug_invariance_pop": float(inv_z.mean()),
        "aug_inv_diff": float(inv_m.mean() - inv_z.mean()),
        "mean_logit_mixed": float(tl_m.mean()),
        "mean_conf_mixed": float(tc_m.mean()),
        "mean_conf_pop": float(tc_z.mean()),
        "delta_conf": float(tc_m.mean() - tc_z.mean()),
    }


SIGNAL_KEYS = [
    "mean_loss_mixed", "mean_loss_pop", "delta_loss", "loss_ratio",
    "p25_loss_mixed", "p10_loss_mixed", "p75_loss_mixed",
    "aug_invariance_mixed", "aug_invariance_pop", "aug_inv_diff",
    "mean_logit_mixed", "mean_conf_mixed", "mean_conf_pop", "delta_conf",
]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[extract] device={device}  MIXED={len(X_m)}  POP_z={len(X_z)}", flush=True)

    items: list[dict] = []
    if args.include_targets:
        for mid in MODEL_IDS:
            arch = mid.removeprefix("model_")[0]
            items.append({"model_id": mid, "arch": arch, "true_p": float("nan"),
                          "is_synth": False, "ckpt": None, "manifest_dir": ""})
    items.extend(discover_synths(args.synth_dirs))
    print(f"[extract] {len(items)} models total", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for idx, it in enumerate(items):
        ts = time.time()
        if it["is_synth"]:
            model = build_resnet(it["arch"], device=device, num_classes=100)
            with open(it["ckpt"], "rb") as f:
                state = _pkl.load(f)
            model.load_state_dict(state)
            model.train(False)
        else:
            model = load_target(it["model_id"], device=device)
        sigs = compute_model_signals(model, X_m, y_m, X_z, y_z, device, args.n_aug)
        rows.append({**{k: it[k] for k in ("model_id", "arch", "true_p", "is_synth", "manifest_dir")},
                     **sigs})
        print(f"  [{idx + 1}/{len(items)}] {it['model_id']:20s} arch={it['arch']} "
              f"p={it['true_p']:.2f}  loss={sigs['mean_loss_mixed']:.3f}  "
              f"deltaL={sigs['delta_loss']:+.3f}  invDiff={sigs['aug_inv_diff']:+.4f}  "
              f"({time.time() - ts:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = {
        "model_id": np.array([r["model_id"] for r in rows]),
        "arch": np.array([r["arch"] for r in rows]),
        "true_p": np.array([r["true_p"] for r in rows], dtype=np.float64),
        "is_synth": np.array([r["is_synth"] for r in rows], dtype=bool),
        "manifest_dir": np.array([r["manifest_dir"] for r in rows]),
    }
    for k in SIGNAL_KEYS:
        arr[k] = np.array([r[k] for r in rows], dtype=np.float64)
    np.savez(out_path, **arr)
    print(f"\n[extract] wrote {out_path}  total={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
