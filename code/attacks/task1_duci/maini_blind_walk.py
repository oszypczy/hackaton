"""
Maini DI 2021 — Blind Walk feature extraction (paper §5.1).

For each sample x with true label y, sample D random unit directions from
{Uniform, Gaussian, Laplace} and walk k·step until model's prediction changes
from y. Distance traveled = k·step (δ unit-norm). Aggregate per-sample 30-d
embedding into model-level scalar signals to MLE-calibrate proportion p̂ via
synth bank with known p (see maini_mle.py).

Key adaptation vs paper: paper uses 30-d → confidence regressor → Welch t-test
(binary membership). We aggregate to **scalar signals** and invert MLE curve
on synth — gives continuous p̂ ∈ [0, 1] needed for Task 1 DUCI.

Trust boundary note: synth/ref checkpoints are stdlib-pickle dumps written by
our own train_synth.py / train_ref.py (internal trust, same convention as
mle.py / main.py / extract_signals.py / probe_regime.py). Organizer .pkl
loaded via targets.load_target.
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal trust: own train_synth/train_ref outputs
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import (
    MODEL_IDS,
    load_mixed,
    load_population,
    population_z,
    preprocess_batch,
)
from .targets import build_resnet, load_target


# ---------------------------------------------------------------------------
# Blind Walk core
# ---------------------------------------------------------------------------

DIST_TYPES = ("uniform", "gaussian", "laplace")


@dataclass
class BlindWalkConfig:
    n_dirs_per_dist: int = 10
    distributions: tuple[str, ...] = DIST_TYPES
    step: float = 0.05  # normalized-image-space step (≈0.013 pixel for CIFAR std)
    max_steps: int = 80
    sample_batch: int = 16  # process this many MIXED/POP samples at once
    seed: int = 0

    @property
    def n_dirs_total(self) -> int:
        return self.n_dirs_per_dist * len(self.distributions)


def _sample_unit_direction(d_type: str, shape: tuple[int, ...],
                           generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Sample direction tensor of given shape, ℓ2-normalized per-direction.

    shape = (D, C, H, W) — sample D independent directions in one go.
    """
    if d_type == "uniform":
        delta = torch.empty(shape, device=device).uniform_(-1.0, 1.0, generator=generator)
    elif d_type == "gaussian":
        delta = torch.randn(shape, device=device, generator=generator)
    elif d_type == "laplace":
        # Laplace(0, 1) via inverse CDF: -sign(u)·log(1 - 2|u|), u ~ U(-0.5, 0.5)
        u = torch.empty(shape, device=device).uniform_(-0.5, 0.5, generator=generator)
        delta = -torch.sign(u) * torch.log1p(-2.0 * u.abs() + 1e-12)
    else:
        raise ValueError(f"unknown distribution {d_type}")

    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat = flat / norms
    return flat.view_as(delta)


@torch.no_grad()
def blind_walk_batch(
    model: torch.nn.Module,
    X_uint8: np.ndarray,
    y: np.ndarray,
    config: BlindWalkConfig,
    device: torch.device,
) -> np.ndarray:
    """Run Blind Walk for a batch of samples; return (N, D) distance matrix.

    D = config.n_dirs_total (default 30 = 10 × 3 distributions).
    Distance = step × #steps_until_label_flip. Capped at max_steps × step.
    Misclassified-from-start samples → dist = 0 (paper §5.1 convention).
    """
    n = X_uint8.shape[0]
    D = config.n_dirs_total
    step = config.step
    max_steps = config.max_steps

    distances = np.zeros((n, D), dtype=np.float32)
    gen = torch.Generator(device=device).manual_seed(config.seed)

    sb = config.sample_batch
    for i in range(0, n, sb):
        xb_uint8 = X_uint8[i:i + sb]
        yb = torch.from_numpy(y[i:i + sb]).long().to(device)
        b = xb_uint8.shape[0]

        x_norm = preprocess_batch(xb_uint8, device)  # (b, 3, 32, 32)

        init_logits = model(x_norm)
        init_pred = init_logits.argmax(1)
        init_correct = (init_pred == yb)  # (b,)

        for d_idx, d_type in enumerate(config.distributions):
            ndp = config.n_dirs_per_dist
            shape = (b * ndp, 3, 32, 32)
            deltas = _sample_unit_direction(d_type, shape, gen, device).view(b, ndp, 3, 32, 32)

            active = init_correct[:, None].expand(b, ndp).clone()  # (b, ndp)
            steps_taken = torch.zeros(b, ndp, dtype=torch.int32, device=device)
            cur = x_norm[:, None].expand(b, ndp, 3, 32, 32).contiguous()

            for _k in range(1, max_steps + 1):
                if not active.any():
                    break
                active_idx = active.nonzero(as_tuple=False)  # (M, 2)
                if active_idx.numel() == 0:
                    break

                cur_active = cur[active_idx[:, 0], active_idx[:, 1]] + \
                             step * deltas[active_idx[:, 0], active_idx[:, 1]]
                logits = model(cur_active)
                pred_active = logits.argmax(1)
                yb_active = yb[active_idx[:, 0]]

                steps_taken[active_idx[:, 0], active_idx[:, 1]] += 1
                cur[active_idx[:, 0], active_idx[:, 1]] = cur_active

                still_active = (pred_active == yb_active)
                flipped = ~still_active
                if flipped.any():
                    flip_idx = active_idx[flipped]
                    active[flip_idx[:, 0], flip_idx[:, 1]] = False

            slice_lo = d_idx * ndp
            slice_hi = (d_idx + 1) * ndp
            d_dist = (steps_taken.float() * step).cpu().numpy()
            d_dist[(~init_correct).cpu().numpy()] = 0.0
            distances[i:i + b, slice_lo:slice_hi] = d_dist

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return distances


# ---------------------------------------------------------------------------
# Multi-signal extraction
# ---------------------------------------------------------------------------

def aggregate_signals(dist_mixed: np.ndarray, dist_z: np.ndarray,
                      config: BlindWalkConfig) -> dict[str, float]:
    """Reduce per-sample (N, 30) distance matrices to scalar signals.

    Signals (we'll pick best per arch via LOO-MAE on synth):
      - mean_{mixed,z}_all, delta_all, ratio_all, log_ratio_all
      - per-distribution: mean_{mixed,z}_<dist>, delta_<dist>
      - median variants (heavy-tail-robust)
    """
    out: dict[str, float] = {}
    out["mean_mixed_all"] = float(np.mean(dist_mixed))
    out["mean_z_all"] = float(np.mean(dist_z))
    out["delta_all"] = out["mean_mixed_all"] - out["mean_z_all"]
    eps = 1e-6
    out["ratio_all"] = out["mean_mixed_all"] / (out["mean_z_all"] + eps)
    out["log_ratio_all"] = float(np.log(out["mean_mixed_all"] + eps) -
                                  np.log(out["mean_z_all"] + eps))
    out["median_mixed_all"] = float(np.median(dist_mixed))
    out["median_z_all"] = float(np.median(dist_z))
    out["delta_median_all"] = out["median_mixed_all"] - out["median_z_all"]

    n_per = config.n_dirs_per_dist
    for i, d_type in enumerate(config.distributions):
        sl = slice(i * n_per, (i + 1) * n_per)
        out[f"mean_mixed_{d_type}"] = float(np.mean(dist_mixed[:, sl]))
        out[f"mean_z_{d_type}"] = float(np.mean(dist_z[:, sl]))
        out[f"delta_{d_type}"] = out[f"mean_mixed_{d_type}"] - out[f"mean_z_{d_type}"]

    return out


def extract_model_signals(
    model: torch.nn.Module,
    X_m: np.ndarray, y_m: np.ndarray,
    X_z: np.ndarray, y_z: np.ndarray,
    config: BlindWalkConfig,
    device: torch.device,
) -> dict:
    t0 = time.time()
    print(f"  [bw] mixed N={len(X_m)}  pop_z N={len(X_z)}  D={config.n_dirs_total}  "
          f"step={config.step}  max_steps={config.max_steps}", flush=True)
    dist_mixed = blind_walk_batch(model, X_m, y_m, config, device)
    t1 = time.time()
    print(f"  [bw] mixed done in {t1 - t0:.1f}s  mean={dist_mixed.mean():.4f}  "
          f"flip_frac={(dist_mixed > 0).mean():.3f}", flush=True)
    dist_z = blind_walk_batch(model, X_z, y_z, config, device)
    t2 = time.time()
    print(f"  [bw] pop_z done in {t2 - t1:.1f}s  mean={dist_z.mean():.4f}  "
          f"flip_frac={(dist_z > 0).mean():.3f}", flush=True)
    sigs = aggregate_signals(dist_mixed, dist_z, config)
    return {
        "signals": sigs,
        "dist_mixed_summary": {
            "mean": float(dist_mixed.mean()),
            "median": float(np.median(dist_mixed)),
            "std": float(dist_mixed.std()),
            "frac_flipped": float((dist_mixed > 0).mean()),
            "frac_max_steps": float((dist_mixed >= config.max_steps * config.step - 1e-6).mean()),
        },
        "dist_z_summary": {
            "mean": float(dist_z.mean()),
            "median": float(np.median(dist_z)),
            "std": float(dist_z.std()),
            "frac_flipped": float((dist_z > 0).mean()),
            "frac_max_steps": float((dist_z >= config.max_steps * config.step - 1e-6).mean()),
        },
        "config": {
            "n_dirs_per_dist": config.n_dirs_per_dist,
            "distributions": list(config.distributions),
            "step": config.step,
            "max_steps": config.max_steps,
            "seed": config.seed,
        },
        "elapsed_s": float(time.time() - t0),
    }


def _load_internal_pickle(ckpt_path: Path):
    # internal trust: synth/ref checkpoints from our own train_*.py
    with open(ckpt_path, "rb") as f:
        return _pkl.load(f)


def load_model_by_kind(kind: str, model_id_or_path: str, arch_digit: str | None,
                        device: torch.device) -> torch.nn.Module:
    """kind: 'target' | 'synth' | 'ref'"""
    if kind == "target":
        return load_target(model_id_or_path, device=device)

    if arch_digit is None:
        raise ValueError(f"arch_digit required for kind={kind}")
    model = build_resnet(arch_digit, device=device, num_classes=100)
    state = _load_internal_pickle(Path(model_id_or_path))
    model.load_state_dict(state)
    model.train(False)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("target", help="extract signals for one organizer target")
    p_t.add_argument("--model-id", type=str, required=True)
    p_t.add_argument("--out", type=str, required=True)
    _add_bw_flags(p_t)

    p_all = sub.add_parser("targets", help="extract signals for all 9 organizer targets")
    p_all.add_argument("--out-dir", type=str, required=True)
    _add_bw_flags(p_all)

    p_s = sub.add_parser("synth", help="extract signals for all synth in a dir")
    p_s.add_argument("--synth-dir", type=str, required=True)
    p_s.add_argument("--out-dir", type=str, required=True)
    _add_bw_flags(p_s)

    p_f = sub.add_parser("feasibility", help="quick feasibility check on 3 models")
    p_f.add_argument("--target-id", type=str, default="model_00")
    p_f.add_argument("--synth-dir", type=str, required=True)
    p_f.add_argument("--n-samples", type=int, default=100)
    _add_bw_flags(p_f)

    return ap.parse_args()


def _add_bw_flags(p):
    p.add_argument("--n-dirs", type=int, default=10)
    p.add_argument("--distributions", type=str, default="uniform,gaussian,laplace")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--sample-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)


def _config_from_args(args) -> BlindWalkConfig:
    return BlindWalkConfig(
        n_dirs_per_dist=args.n_dirs,
        distributions=tuple(args.distributions.split(",")),
        step=args.step,
        max_steps=args.max_steps,
        sample_batch=args.sample_batch,
        seed=args.seed,
    )


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def cmd_target(args, device, X_m, y_m, X_z, y_z) -> None:
    config = _config_from_args(args)
    model = load_target(args.model_id, device=device)
    payload = extract_model_signals(model, X_m, y_m, X_z, y_z, config, device)
    payload["model_id"] = args.model_id
    payload["kind"] = "target"
    payload["arch_digit"] = args.model_id.removeprefix("model_")[0]
    _save_json(Path(args.out), payload)
    print(f"[maini_bw] saved {args.out}", flush=True)


def cmd_targets(args, device, X_m, y_m, X_z, y_z) -> None:
    config = _config_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid in MODEL_IDS:
        out_path = out_dir / f"target_{mid}.json"
        if out_path.exists():
            print(f"[maini_bw] skip {mid} — exists", flush=True)
            continue
        print(f"\n[maini_bw] === {mid} ===", flush=True)
        model = load_target(mid, device=device)
        payload = extract_model_signals(model, X_m, y_m, X_z, y_z, config, device)
        payload["model_id"] = mid
        payload["kind"] = "target"
        payload["arch_digit"] = mid.removeprefix("model_")[0]
        _save_json(out_path, payload)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _discover_synth_manifests(synth_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(synth_dir.glob("synth_*.json")):
        with open(jp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            print(f"  [warn] manifest {jp.name} but ckpt missing: {ckpt}")
            continue
        out.append({"manifest": m, "ckpt": ckpt, "stem": jp.stem})
    return out


def cmd_synth(args, device, X_m, y_m, X_z, y_z) -> None:
    config = _config_from_args(args)
    synth_dir = Path(args.synth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = _discover_synth_manifests(synth_dir)
    print(f"[maini_bw] found {len(manifests)} synth in {synth_dir}", flush=True)
    for entry in manifests:
        m = entry["manifest"]
        out_path = out_dir / f"{entry['stem']}.json"
        if out_path.exists():
            print(f"[maini_bw] skip {entry['stem']} — exists", flush=True)
            continue
        print(f"\n[maini_bw] === {entry['stem']}  arch={m['arch_digit']}  p={m['true_p']} ===",
              flush=True)
        model = load_model_by_kind("synth", str(entry["ckpt"]),
                                    arch_digit=m["arch_digit"], device=device)
        payload = extract_model_signals(model, X_m, y_m, X_z, y_z, config, device)
        payload["kind"] = "synth"
        payload["arch_digit"] = m["arch_digit"]
        payload["true_p"] = float(m["true_p"])
        payload["source_manifest"] = str(synth_dir / f"{entry['stem']}.json")
        _save_json(out_path, payload)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cmd_feasibility(args, device, X_m, y_m, X_z, y_z) -> None:
    config = _config_from_args(args)
    rng = np.random.default_rng(0)
    idx_m = rng.choice(len(X_m), size=min(args.n_samples, len(X_m)), replace=False)
    idx_z = rng.choice(len(X_z), size=min(args.n_samples, len(X_z)), replace=False)
    X_m_s, y_m_s = X_m[idx_m], y_m[idx_m]
    X_z_s, y_z_s = X_z[idx_z], y_z[idx_z]
    print(f"[feasibility] using {len(X_m_s)} MIXED + {len(X_z_s)} POP_z samples", flush=True)
    print(f"[feasibility] dirs={config.n_dirs_total} step={config.step} max_steps={config.max_steps}",
          flush=True)

    results = {}
    print(f"\n--- target {args.target_id} ---")
    model = load_target(args.target_id, device=device)
    out = extract_model_signals(model, X_m_s, y_m_s, X_z_s, y_z_s, config, device)
    results[f"target_{args.target_id}"] = out["signals"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    manifests = _discover_synth_manifests(Path(args.synth_dir))
    arch_target = args.target_id.removeprefix("model_")[0]
    pick = {0.0: None, 1.0: None}
    for entry in manifests:
        m = entry["manifest"]
        if m["arch_digit"] != arch_target:
            continue
        for tp in (0.0, 1.0):
            if abs(m["true_p"] - tp) < 1e-3 and pick[tp] is None:
                pick[tp] = entry
    for tp, entry in pick.items():
        if entry is None:
            print(f"[feasibility] no synth at p={tp} arch={arch_target}", flush=True)
            continue
        print(f"\n--- synth p={tp} ({entry['stem']}) ---")
        model = load_model_by_kind("synth", str(entry["ckpt"]),
                                    arch_digit=arch_target, device=device)
        out = extract_model_signals(model, X_m_s, y_m_s, X_z_s, y_z_s, config, device)
        results[f"synth_p{tp:.2f}"] = out["signals"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== feasibility summary ===")
    keys = ["mean_mixed_all", "mean_z_all", "delta_all", "ratio_all", "log_ratio_all"]
    print(f"{'model':30s} " + "  ".join(f"{k:18s}" for k in keys))
    for name, sigs in results.items():
        print(f"{name:30s} " + "  ".join(f"{sigs[k]:+.5f}" for k in keys))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[maini_bw] device={device}  cmd={args.cmd}", flush=True)

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)

    if args.cmd == "target":
        cmd_target(args, device, X_m, y_m, X_z, y_z)
    elif args.cmd == "targets":
        cmd_targets(args, device, X_m, y_m, X_z, y_z)
    elif args.cmd == "synth":
        cmd_synth(args, device, X_m, y_m, X_z, y_z)
    elif args.cmd == "feasibility":
        cmd_feasibility(args, device, X_m, y_m, X_z, y_z)
    else:
        raise ValueError(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
