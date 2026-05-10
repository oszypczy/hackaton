"""
Phase A diagnostic: POP_z accuracy per model.

For each (target | ref | synth) checkpoint, compute:
- top-1 accuracy on POPULATION_z
- mean cross-entropy loss on POPULATION_z
- mean target-class softmax on POPULATION_z

Goal: find epoch count per arch where ref bank's POP_z acc matches the 9 target
models (organizer hint says targets are at ~0.27 POP_z acc — undertrained).

NOTE on trust boundary: ref + synth checkpoints are stdlib-pickle dumps written
by our own train_ref.py / train_synth.py — internal trust, same pattern as
extract_signals.py. Organizer .pkl files are loaded via targets.load_target.

Usage:
    python -m code.attacks.task1_duci.probe_regime \\
        --refs-dirs /p/scratch/.../DUCI/refs_10ep,/p/scratch/.../DUCI/refs_20ep,... \\
        --synth-dirs /p/scratch/.../DUCI/synth_targets_20ep_r50,... \\
        --out /tmp/probe_regime.json
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal: loads our own train_ref.py / train_synth.py outputs
import time
from pathlib import Path

import numpy as np
import torch

from .data import BATCH, MODEL_IDS, load_population, population_z, preprocess_batch
from .targets import build_resnet, load_target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dirs", type=str, default="",
                    help="comma-separated dirs containing ref_<arch>_<seed>.pt + manifest_*.json")
    ap.add_argument("--synth-dirs", type=str, default="",
                    help="comma-separated dirs with synth_<arch>_<p>.pt + synth_*.json")
    ap.add_argument("--include-targets", action="store_true", default=True)
    ap.add_argument("--out", type=str, required=True, help="output JSON path")
    return ap.parse_args()


@torch.no_grad()
def compute_popz_stats(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: str) -> dict:
    n = len(X)
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_conf = 0.0
    for i in range(0, n, BATCH):
        xb = preprocess_batch(X[i:i + BATCH], device)
        yb = torch.from_numpy(y[i:i + BATCH]).long().to(device)
        logits = model(xb)
        sm = torch.softmax(logits, dim=1)
        tc = sm.gather(1, yb.unsqueeze(1)).squeeze(1)
        sum_conf += tc.sum().item()
        sum_loss += -torch.log(torch.clamp(tc, min=1e-12)).sum().item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.shape[0]
    return {
        "popz_acc": correct / total,
        "popz_loss": sum_loss / total,
        "popz_conf": sum_conf / total,
        "n": total,
    }


def discover_refs(dirs_csv: str) -> list[dict]:
    out = []
    for d_str in dirs_csv.split(","):
        d_str = d_str.strip()
        if not d_str:
            continue
        d = Path(d_str)
        if not d.exists():
            print(f"[warn] dir missing: {d}", flush=True)
            continue
        for jp in sorted(d.glob("manifest_*.json")):
            with open(jp) as f:
                m = json.load(f)
            ckpt = Path(m["checkpoint"])
            if not ckpt.exists():
                continue
            out.append({
                "model_id": jp.stem,
                "arch": str(m["arch_digit"]),
                "ckpt": ckpt,
                "kind": "ref",
                "manifest_dir": d.name,
                "epochs": int(m.get("recipe", {}).get("epochs", -1)),
                "true_p": float("nan"),
            })
    return out


def discover_synths(dirs_csv: str) -> list[dict]:
    out = []
    for d_str in dirs_csv.split(","):
        d_str = d_str.strip()
        if not d_str:
            continue
        d = Path(d_str)
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
                "ckpt": ckpt,
                "kind": "synth",
                "manifest_dir": d.name,
                "epochs": int(m.get("recipe", {}).get("epochs", -1)),
                "true_p": float(m["true_p"]),
            })
    return out


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    X_p, y_p = load_population()
    X_z, y_z = population_z(X_p, y_p)
    print(f"[probe] device={device}  POP_z={len(X_z)}", flush=True)

    items: list[dict] = []
    if args.include_targets:
        for mid in MODEL_IDS:
            arch = mid.removeprefix("model_")[0]
            items.append({
                "model_id": mid, "arch": arch, "ckpt": None,
                "kind": "target", "manifest_dir": "", "epochs": -1,
                "true_p": float("nan"),
            })
    items.extend(discover_refs(args.refs_dirs))
    items.extend(discover_synths(args.synth_dirs))
    print(f"[probe] {len(items)} models total", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for idx, it in enumerate(items):
        ts = time.time()
        if it["kind"] == "target":
            model = load_target(it["model_id"], device=device)
        else:
            model = build_resnet(it["arch"], device=device, num_classes=100)
            with open(it["ckpt"], "rb") as f:
                state = _pkl.load(f)
            model.load_state_dict(state)
            model.train(False)

        stats = compute_popz_stats(model, X_z, y_z, device)
        row = {**{k: it[k] for k in ("model_id", "arch", "kind", "manifest_dir", "epochs", "true_p")},
               **stats}
        rows.append(row)
        print(f"  [{idx + 1:3}/{len(items)}] {it['kind']:6s} {it['model_id']:24s} arch={it['arch']} "
              f"ep={it['epochs']:3d} p={it['true_p']:.2f}  POP_z acc={stats['popz_acc']:.4f}  "
              f"loss={stats['popz_loss']:.3f}  conf={stats['popz_conf']:.4f}  ({time.time() - ts:.1f}s)",
              flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[probe] wrote {out_path}  total={time.time() - t0:.0f}s", flush=True)

    print("\n=== Summary by (kind, arch, epochs) ===")
    by_group: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["kind"], r["arch"], r["epochs"])
        by_group.setdefault(key, []).append(r["popz_acc"])
    for key in sorted(by_group):
        accs = np.array(by_group[key])
        print(f"  {key[0]:6s} arch={key[1]} ep={key[2]:3d}  n={len(accs):2d}  "
              f"POP_z acc mean={accs.mean():.4f} std={accs.std():.4f} "
              f"[min={accs.min():.4f} max={accs.max():.4f}]")


if __name__ == "__main__":
    main()
