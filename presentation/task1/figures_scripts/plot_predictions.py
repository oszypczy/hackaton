"""Per-target prediction bar chart for the 9 hackathon ResNet models.

Reads data/signals.json. Produces 03_predictions.png at 300 DPI, 16:9.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "signals.json"
OUT = ROOT / "figures" / "03_predictions.png"

ARCH_COLORS = {"0": "#4C78A8", "1": "#54A24B", "2": "#F58518"}
ARCH_NAMES = {"0": "ResNet18", "1": "ResNet50", "2": "ResNet152"}


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} — run dump_signals.sh on cluster first")

    with open(DATA) as f:
        payload = json.load(f)

    targets = sorted(payload["targets"], key=lambda t: t["model_id"])
    ids = [t["model_id"] for t in targets]
    p_hats = [t["p_hat"] for t in targets]
    archs = [t["arch"] for t in targets]
    colors = [ARCH_COLORS[a] for a in archs]

    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(ids))
    bars = ax.bar(x, p_hats, color=colors, edgecolor="#222", linewidth=1.2, width=0.72)

    # Value labels on bars
    for rect, p in zip(bars, p_hats):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                f"{p:.2f}", ha="center", va="bottom", fontsize=11, weight="bold", color="#222")

    # Reference lines
    ax.axhline(0.5, linestyle="--", linewidth=1.0, color="#888", alpha=0.7, zorder=0)
    ax.text(len(ids) - 0.4, 0.51, "$p = 0.5$", fontsize=10, color="#888", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=12)
    ax.set_xlabel("target model id  (first digit = architecture)", fontsize=13)
    ax.set_ylabel("estimated training proportion  $\\hat{p}$", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Legend by architecture
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=ARCH_COLORS[a],
                      edgecolor="#222", linewidth=1.0, label=ARCH_NAMES[a])
        for a in ("0", "1", "2")
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=11, framealpha=0.95,
              title="architecture", title_fontsize=11)

    fig.suptitle(
        "Continuous dataset-usage estimate $\\hat{p}$ for the 9 hackathon ResNet targets",
        fontsize=17, weight="bold", color="#222", y=0.995,
    )
    fig.text(
        0.5, 0.945,
        "Public benchmark MAE = 0.053 — matches Tong et al. ICLR 2025 single-reference setting",
        ha="center", va="top", fontsize=12.5, style="italic", color="#555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
