"""Data-flow diagram for Task 1 (DUCI) presentation slide.

Pure-matplotlib boxes-and-arrows; no input data needed. Output:
figures/01_pipeline.png at 300 DPI, 16:9 ratio.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "01_pipeline.png"

# Deuteranopia-safe palette
C_INPUT = "#4C78A8"   # blue — data sources
C_REF = "#F58518"     # orange — attacker reference
C_PROC = "#54A24B"    # green — processing
C_OUT = "#B279A2"     # purple — output
EDGE = "#2A2A2A"
BG = "#FFFFFF"

FONT_TITLE = 17
FONT_BODY = 13
FONT_SMALL = 11


def box(ax, x, y, w, h, text, color, fontsize=FONT_BODY, weight="normal"):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, edgecolor=EDGE, facecolor=color, alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            color="white" if color != "#FFFFFF" else "#222")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.8, mutation_scale=18),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label,
                ha="center", va="bottom", fontsize=FONT_SMALL, style="italic", color="#444")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(8, 8.4, "Pipeline: continuous dataset-usage estimation",
            ha="center", va="center", fontsize=FONT_TITLE, weight="bold", color="#222")
    ax.text(8, 7.95,
            "Single attacker reference bank  →  calibrated MIA signal  →  per-target inversion",
            ha="center", va="center", fontsize=FONT_BODY, style="italic", color="#555")

    # Row 1: inputs
    box(ax, 0.5, 5.4, 4.2, 1.4,
        "9 black-box targets\n(ResNet18 / 50 / 152)\nunknown $p \\in [0,1]$",
        C_INPUT, fontsize=FONT_BODY, weight="bold")

    box(ax, 5.9, 5.4, 4.2, 1.4,
        "Single attacker reference bank\n5 ResNet18 models\nknown $p \\in \\{0, ¼, ½, ¾, 1\\}$",
        C_REF, fontsize=FONT_BODY, weight="bold")

    box(ax, 11.3, 5.4, 4.2, 1.4,
        "MIXED dataset\n(N = 2000, CIFAR-100)\n+ population auxiliary",
        C_INPUT, fontsize=FONT_BODY, weight="bold")

    # Row 2: forward pass + signal
    box(ax, 2.0, 3.3, 12.0, 1.3,
        "Forward pass on candidate dataset  →  mean loss  →  one scalar signal per model",
        C_PROC, fontsize=FONT_BODY, weight="bold")

    arrow(ax, 2.6, 5.4, 4.5, 4.6)
    arrow(ax, 8.0, 5.4, 8.0, 4.6)
    arrow(ax, 13.4, 5.4, 11.5, 4.6)

    # Row 3: calibration + inversion
    box(ax, 0.8, 1.0, 6.6, 1.6,
        "Linear calibration on synth bank\n$\\mathrm{loss} = a \\cdot p + b$\nsingle fit, all 3 architectures",
        C_PROC, fontsize=FONT_BODY)

    box(ax, 8.6, 1.0, 6.6, 1.6,
        "Invert for each target\n$\\hat{p} = (\\mathrm{signal} - b) / a$\n→ continuous $\\hat{p} \\in [0,1]$",
        C_OUT, fontsize=FONT_BODY, weight="bold")

    arrow(ax, 5.5, 3.3, 4.1, 2.6, label="synth bank only")
    arrow(ax, 10.5, 3.3, 11.9, 2.6, label="9 targets")
    arrow(ax, 7.4, 1.8, 8.6, 1.8, label="$a, b$")

    # Footer note
    ax.text(8, 0.35,
            "Method follows Tong et al. (DUCI, ICLR 2025) — single-reference setting reproduces published MAE.",
            ha="center", va="center", fontsize=FONT_SMALL, style="italic", color="#666")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
