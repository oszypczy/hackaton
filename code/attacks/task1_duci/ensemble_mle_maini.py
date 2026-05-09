"""Ensemble MLE 80ep R18 (SUB-5 continuous) with Maini DI per-arch predictions.

Inputs:
- MLE continuous CSV (e.g. submission_mle_80ep_r18_precise.csv) — full SUB-5 predictions
- Maini predictions CSV from maini_mle.py — per-arch grid-search winner
- Optional weights: --w-mle, --w-maini (default 0.5 each)

Outputs:
- Continuous ensemble CSV
- snap_10 CSV (rounded to 0.1)
- snap_05 CSV (rounded to 0.05)
- Diff vs SUB-9 (snap_10 of MLE alone) for each model

Usage:
    python -m code.attacks.task1_duci.ensemble_mle_maini \\
        --mle submissions/task1_duci_sub5_continuous.csv \\
        --maini submissions/task1_duci_maini.csv \\
        --out-continuous submissions/task1_ens_mle_maini_continuous.csv \\
        --out-snap10 submissions/task1_ens_mle_maini_snap10.csv \\
        --w-mle 0.5 --w-maini 0.5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_IDS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]


def load_csv(path: Path) -> dict[str, float]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["model_id"]] = float(row["proportion"])
    return out


def write_csv(preds: dict[str, float], path: Path, snap: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("model_id,proportion\n")
        for mid in MODEL_IDS:
            v = preds[mid]
            if snap > 0:
                v = round(v / snap) * snap
            v = max(0.0, min(1.0, v))
            f.write(f"{mid},{v:.6f}\n" if snap == 0 else f"{mid},{v:.1f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mle", type=Path, required=True, help="MLE continuous CSV")
    ap.add_argument("--maini", type=Path, required=True, help="Maini predictions CSV")
    ap.add_argument("--w-mle", type=float, default=0.5)
    ap.add_argument("--w-maini", type=float, default=0.5)
    ap.add_argument("--out-continuous", type=Path, required=True)
    ap.add_argument("--out-snap10", type=Path, default=None)
    ap.add_argument("--out-snap05", type=Path, default=None)
    ap.add_argument("--snap10-offset", type=float, default=0.0,
                    help="Apply offset before rounding to 0.1 (e.g. -0.025)")
    args = ap.parse_args()

    mle = load_csv(args.mle)
    maini = load_csv(args.maini)
    assert sum([args.w_mle, args.w_maini]) > 0

    w_total = args.w_mle + args.w_maini
    w_mle = args.w_mle / w_total
    w_maini = args.w_maini / w_total

    ens = {}
    for mid in MODEL_IDS:
        if mid not in mle:
            print(f"[ensemble] missing MLE pred for {mid} — skipping ensemble"); return
        if mid not in maini:
            print(f"[ensemble] missing Maini pred for {mid} — using MLE only")
            ens[mid] = mle[mid]
        else:
            ens[mid] = w_mle * mle[mid] + w_maini * maini[mid]

    print(f"\n[ensemble] weights: MLE={w_mle:.3f} Maini={w_maini:.3f}")
    print("[ensemble] per-target predictions:")
    print(f"  {'mid':>4s}  {'MLE':>7s}  {'Maini':>7s}  {'ens':>7s}  {'snap10':>6s}")
    for mid in MODEL_IDS:
        v = ens[mid]
        v_snap = round((v + args.snap10_offset) / 0.1) * 0.1
        print(f"  {mid:>4s}  {mle[mid]:.4f}  {maini.get(mid, float('nan')):.4f}"
              f"  {v:.4f}  {v_snap:.1f}")

    # Write outputs
    write_csv(ens, args.out_continuous, snap=0)
    print(f"\n[ensemble] wrote continuous → {args.out_continuous}")

    if args.out_snap10:
        ens_snap10 = {k: round((v + args.snap10_offset) / 0.1) * 0.1
                      for k, v in ens.items()}
        write_csv(ens_snap10, args.out_snap10, snap=0.1)
        print(f"[ensemble] wrote snap_10 → {args.out_snap10}")
    if args.out_snap05:
        ens_snap05 = {k: round((v + args.snap10_offset) / 0.05) * 0.05
                      for k, v in ens.items()}
        write_csv(ens_snap05, args.out_snap05, snap=0.05)
        print(f"[ensemble] wrote snap_05 → {args.out_snap05}")


if __name__ == "__main__":
    main()
