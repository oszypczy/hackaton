"""Calibration plot — single reference bank, all 9 targets dropped onto the fit.

Reads data/signals.json (output of mle.py --dump-signals).
The pipeline uses one ResNet18 attacker bank for all three target architectures,
so a single calibration line tells the whole story.

Produces 02_calibration.png at 300 DPI, 16:9.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "signals.json"
OUT = ROOT / "figures" / "02_calibration.png"

ARCH_COLORS = {"0": "#4C78A8", "1": "#54A24B", "2": "#F58518"}
ARCH_NAMES = {"0": "ResNet18", "1": "ResNet50", "2": "ResNet152"}


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} — run dump_signals.sh on cluster first")

    with open(DATA) as f:
        payload = json.load(f)

    cal = payload["synth"]["0"]
    ps = np.array(cal["ps"], dtype=float)
    sigs = np.array(cal["signals"], dtype=float)
    coeffs = cal["poly_coeffs"]
    a, b = float(coeffs[0]), float(coeffs[1])
    loo = float(cal["loo_mae"])
    sig_name = cal["best_signal"]

    targets = payload["targets"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Linear fit line spanning [0, 1]
    x_line = np.linspace(0.0, 1.0, 200)
    y_line = a * x_line + b
    ax.plot(x_line, y_line, color="#444", linewidth=2.6, alpha=0.85, zorder=2,
            label=f"linear fit:  signal = {a:+.2f}·p {b:+.2f}")

    # Reference bank
    ax.scatter(ps, sigs, s=220, color="#444", edgecolor="white", linewidth=2.0,
               zorder=4, marker="o",
               label=f"reference bank (5 ResNet18, known $p$)  —  LOO-MAE = {loo:.3f}")

    # Targets — color by architecture
    for arch_key in ("0", "1", "2"):
        arch_targets = [t for t in targets if t["arch"] == arch_key]
        xs = [t["p_hat"] for t in arch_targets]
        ys = [t["signal"] for t in arch_targets]
        ax.scatter(xs, ys, s=170, color=ARCH_COLORS[arch_key], edgecolor="#222",
                   linewidth=1.5, marker="D", zorder=5,
                   label=f"target — {ARCH_NAMES[arch_key]} ($\\hat{{p}}$ recovered)")
        # Add tiny model_id labels
        for t in arch_targets:
            ax.annotate(t["model_id"], (t["p_hat"], t["signal"]),
                        xytext=(7, -14), textcoords="offset points",
                        fontsize=10, color=ARCH_COLORS[arch_key], weight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("training proportion  $p$  (known for synth bank, recovered for targets)",
                  fontsize=13)
    ax.set_ylabel("MIA signal  =  mean loss on candidate dataset", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=11.5, framealpha=0.95)

    fig.suptitle(
        "One attacker reference bank calibrates all nine targets",
        fontsize=18, weight="bold", color="#222", y=0.995,
    )
    fig.text(
        0.5, 0.945,
        "Forward pass on the candidate dataset → average loss → invert through the linear fit",
        ha="center", va="top", fontsize=12.5, style="italic", color="#555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"wrote {OUT}  (signal={sig_name}, a={a:.3f}, b={b:.3f}, LOO={loo:.3f})")


if __name__ == "__main__":
    main()
