"""Tong RMIA + Eq.4 with ARCH-MATCHED ref banks per target.

Target arch=0 (R18) -> R18 refs
Target arch=1 (R50) -> R50 refs
Target arch=2 (R152) -> R152 refs

Each ref bank can have its own Youden's beta_star + TPR/FPR.

Trust boundary note: ref checkpoints are stdlib-pickle dumps written by
our own train_ref.py (internal trust, same pattern as main.py / mle.py /
extract_signals.py / probe_regime.py). Organizer .pkl files use targets.load_target.

Run on cluster:
    P4VENV=/p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python
    $P4VENV -m code.attacks.task1_duci.main_arch_matched \\
        --refs-dir-r18 /p/scratch/.../DUCI/refs_n7000_100ep \\
        --refs-dir-r50 /p/scratch/.../DUCI/refs_n7000_100ep_r50 \\
        --refs-dir-r152 /p/scratch/.../DUCI/refs_n7000_100ep_r152 \\
        --out submissions/task1_duci_tong_arch_matched.csv
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal trust: own train_ref outputs
import time
from pathlib import Path

import numpy as np
import torch

from .data import MODEL_IDS, load_mixed, load_population, population_z
from .debias import P_MAX, P_MIN, clamp, debias, write_submission_csv
from .forward import forward_dataset
from .rmia import RmiaInputs, rmia_score
from .targets import build_resnet, load_target
from .threshold import select_beta_global


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir-r18", type=str, required=True)
    ap.add_argument("--refs-dir-r50", type=str, default="",
                    help="empty -> fall back to r18 cross-arch for R50 targets")
    ap.add_argument("--refs-dir-r152", type=str, default="",
                    help="empty -> fall back to r18 cross-arch for R152 targets")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--clamp-lo", type=float, default=P_MIN)
    ap.add_argument("--clamp-hi", type=float, default=P_MAX)
    ap.add_argument("--no-clamp", action="store_true")
    return ap.parse_args()


def discover_refs(d: Path) -> list[dict]:
    out = []
    for jp in sorted(d.glob("manifest_*.json")):
        with open(jp) as f:
            m = json.load(f)
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            continue
        out.append({"manifest": m, "ckpt": ckpt})
    return out


def load_state_internal(p: Path):
    with open(p, "rb") as f:
        return _pkl.load(f)


def precompute_ref_bank(refs: list[dict], X_m, y_m, X_z, y_z, device: str):
    cx_list, cz_list, mask_list = [], [], []
    for r in refs:
        m = r["manifest"]
        arch = m["arch_digit"]
        t0 = time.time()
        model = build_resnet(arch, device=device, num_classes=100)
        model.load_state_dict(load_state_internal(r["ckpt"]))
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
        print(f"  ref arch={arch} seed={m['seed']:3d} in_count={int(mask.sum()):4d} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return np.stack(cx_list), np.stack(cz_list), np.stack(mask_list)


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_m, y_m = load_mixed()
    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[arch_matched] device={device} MIXED={len(X_m)} POP_z={len(X_z)}", flush=True)

    arch_refs_dirs = {
        "0": Path(args.refs_dir_r18),
        "1": Path(args.refs_dir_r50) if args.refs_dir_r50 else Path(args.refs_dir_r18),
        "2": Path(args.refs_dir_r152) if args.refs_dir_r152 else Path(args.refs_dir_r18),
    }
    print(f"[arch_matched] refs: 0={arch_refs_dirs['0'].name}  "
          f"1={arch_refs_dirs['1'].name}  2={arch_refs_dirs['2'].name}", flush=True)

    arch_bank: dict[str, dict] = {}
    for arch in ("0", "1", "2"):
        d = arch_refs_dirs[arch]
        refs = [r for r in discover_refs(d) if r["manifest"]["arch_digit"] == arch]
        if not refs:
            refs = [r for r in discover_refs(d) if r["manifest"]["arch_digit"] == "0"]
            if refs:
                print(f"[arch_matched] arch={arch}: no arch-matched refs in {d.name}; "
                      f"using arch=0 refs cross-arch", flush=True)
        if not refs:
            raise RuntimeError(f"no refs for arch={arch} in {d}")
        print(f"[arch_matched] arch={arch}: {len(refs)} refs from {d.name}", flush=True)
        cx, cz, mask = precompute_ref_bank(refs, X_m, y_m, X_z, y_z, device)
        beta, tpr, fpr = select_beta_global(cx, cz, mask)
        print(f"  beta*={beta:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}  gap={tpr-fpr:+.3f}", flush=True)
        arch_bank[arch] = {"cx": cx, "cz": cz, "mask": mask, "beta": beta, "tpr": tpr, "fpr": fpr}

    print("\n[arch_matched] scoring 9 targets:", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        t0 = time.time()
        arch = mid.removeprefix("model_")[0]
        b = arch_bank[arch]
        model = load_target(mid, device=device)
        tc_x = forward_dataset(model, X_m, y_m, device=device).target_class_conf
        tc_z = forward_dataset(model, X_z, y_z, device=device).target_class_conf
        inputs = RmiaInputs(target_conf_x=tc_x, target_conf_z=tc_z,
                            ref_conf_x=b["cx"], ref_conf_z=b["cz"],
                            ref_train_mask_x=b["mask"])
        scores = rmia_score(inputs)
        m_hat = (scores >= b["beta"]).astype(np.float32)
        p_raw = debias(m_hat, b["tpr"], b["fpr"])
        if args.no_clamp:
            p_final = p_raw
        else:
            p_final = clamp(p_raw, args.clamp_lo, args.clamp_hi)
        p_final = float(np.clip(p_final, 0.0, 1.0))
        key = mid.removeprefix("model_")
        predictions[key] = p_final
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mid} arch={arch} m_hat_mean={m_hat.mean():.3f} "
              f"p_raw={p_raw:+.4f} p_final={p_final:.4f} ({time.time() - t0:.1f}s)",
              flush=True)

    write_submission_csv(predictions, str(out_path))
    print(f"\n[arch_matched] wrote {out_path}", flush=True)
    print(f"[arch_matched] preds: " + "  ".join(f"{k}={predictions[k]:.4f}"
                                                  for k in sorted(predictions)), flush=True)


if __name__ == "__main__":
    run()
