"""Per-PII routing diagram for Task 2 ensemble (slot 4 of 6).

Layout:
  Top: title + 1-line problem statement.
  Middle 70%: two side-by-side blocks
              LEFT  = EMAIL / PHONE fallback chain (single best + 2 fallbacks)
              RIGHT = CREDIT plurality voting (4-column comparison table with
                      placeholder column rejected, agreement highlighted)
  Bottom 12%: lift-over-single-best-prompt score box.

Output: figures/04_ensemble_routing.png at 300 DPI, 16:9.

Visual continuity with figures/01_split_screen.png (slot 3 / Artur):
- same monospace family for chat-like and digit-like content
- same C_V2 orange for "winner / chosen candidate"
- same C_V1 gray for fallback / not-active
- same C_DELTA green for positive delta
- same C_BAD dark red for placeholder / rejected
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "04_ensemble_routing.png"

# Palette — matches plot_split_screen.py for slide-pair continuity
C_V1 = "#9CA3AF"            # muted gray — fallback / not active
C_V2 = "#D97706"            # warm orange — chosen / winning candidate
C_HIGHLIGHT_V2 = "#FED7AA"  # light orange — agreement highlight
C_HIGHLIGHT_GROUP = "#FFEDD5"  # very pale orange — voting agreement region
C_DELTA = "#15803D"         # green — positive delta
C_BG = "#FFFFFF"
C_BLOCK_BG = "#F9FAFB"      # very pale gray — block background
C_PANEL_BG = "#FFFBEB"      # very pale yellow — bottom-strip / score panel
C_EDGE = "#374151"
C_TEXT = "#111827"
C_DIM = "#6B7280"
C_BAD = "#B91C1C"           # dark red — placeholder / rejected

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"

FONT_TITLE = 18
FONT_SUBTITLE = 11.5
FONT_HEADER = 14
FONT_HEADER_SMALL = 10.5
FONT_BODY = 11
FONT_MONO = 10.5
FONT_MONO_BIG = 12
FONT_CAPTION = 9.5
FONT_SCORE_LABEL = 11.5
FONT_SCORE_DELTA = 16


def section_box(ax, x, y, w, h, accent_color, fill=C_BLOCK_BG, lw=1.5):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=lw, edgecolor=accent_color, facecolor=fill,
    )
    ax.add_patch(patch)


def fallback_box(ax, x, y, w, h, label, value, *, active: bool, check: bool = False):
    """One row of the fallback chain. Active = orange; non-active = gray."""
    accent = C_V2 if active else C_V1
    fill = C_HIGHLIGHT_V2 if active else "#FFFFFF"
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.6 if active else 1.2,
        edgecolor=accent, facecolor=fill,
    )
    ax.add_patch(patch)

    label_x = x + 0.012
    value_x = x + 0.18
    label_color = accent if active else C_DIM
    weight = "bold" if active else "normal"

    label_text = label
    if check and active:
        label_text = f"{label}  ✓"

    ax.text(label_x, y + h * 0.55, label_text,
            ha="left", va="center",
            fontsize=FONT_BODY, family=SANS, color=label_color, weight=weight)
    ax.text(value_x, y + h * 0.55, value,
            ha="left", va="center",
            fontsize=FONT_MONO, family=MONO,
            color=C_TEXT if active else C_DIM,
            weight="bold" if active else "normal")


def down_arrow(ax, x, y_top, y_bot, label):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.4, mutation_scale=14))
    ax.text(x + 0.012, (y_top + y_bot) / 2, label,
            ha="left", va="center",
            fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")


def credit_column(ax, x, y, w, h, header, value, *, agreeing: bool, placeholder: bool):
    """One column of the CREDIT voting table. agreeing = orange tint background."""
    if agreeing:
        bg = C_HIGHLIGHT_V2
        edge = C_V2
    elif placeholder:
        bg = "#FEE2E2"  # very pale red
        edge = C_BAD
    else:
        bg = "white"
        edge = C_V1

    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.4, edgecolor=edge, facecolor=bg,
    )
    ax.add_patch(patch)

    # Column header (P1/P2/...)
    ax.text(x + w / 2, y + h - 0.022, header,
            ha="center", va="center",
            fontsize=FONT_HEADER_SMALL, family=SANS,
            color=edge, weight="bold")

    # Digit string (broken into 4 lines: 4-4-4-4 stacked)
    blocks = value.split()
    line_h = (h - 0.05) / max(len(blocks), 1)
    cur_y = y + h - 0.05 - line_h * 0.5
    for blk in blocks:
        text_color = C_TEXT
        if placeholder:
            text_color = C_BAD
        ax.text(x + w / 2, cur_y, blk,
                ha="center", va="center",
                fontsize=FONT_MONO_BIG, family=MONO,
                color=text_color,
                weight="bold" if agreeing else "normal")
        cur_y -= line_h

    if placeholder:
        # Strikethrough across each block (rendered as a thick red line per block)
        cur_y = y + h - 0.05 - line_h * 0.5
        for blk in blocks:
            ax.plot([x + w * 0.18, x + w * 0.82], [cur_y, cur_y],
                    color=C_BAD, linewidth=2.2, solid_capstyle="round")
            cur_y -= line_h


def score_pill(ax, x, y, w, h, label_text, delta_text):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        linewidth=1.6, edgecolor=C_DELTA, facecolor=C_PANEL_BG,
    )
    ax.add_patch(patch)
    ax.text(x + w * 0.05, y + h / 2, label_text,
            ha="left", va="center",
            fontsize=FONT_SCORE_LABEL, family=SANS, color=C_TEXT)
    ax.text(x + w * 0.95, y + h / 2, delta_text,
            ha="right", va="center",
            fontsize=FONT_SCORE_DELTA, family=SANS,
            color=C_DELTA, weight="bold")


def main():
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor=C_BG)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # === Title ===
    fig.text(0.5, 0.955,
             "Task 2 — PII ensemble: per-PII routing across prompt CSVs",
             ha="center", va="center",
             fontsize=FONT_TITLE, family=SANS, weight="bold", color=C_TEXT)
    fig.text(0.5, 0.918,
             "Different prompts succeed on different rows — combine them with per-feature rules.",
             ha="center", va="center",
             fontsize=FONT_SUBTITLE, family=SANS, color=C_DIM, style="italic")

    # === Column headers ===
    fig.text(0.255, 0.872, "EMAIL  &  PHONE",
             ha="center", va="center",
             fontsize=FONT_HEADER, family=SANS, weight="bold", color=C_V2)
    fig.text(0.255, 0.842, "single best prompt with fallback chain",
             ha="center", va="center",
             fontsize=FONT_HEADER_SMALL, family=SANS, color=C_DIM, style="italic")

    fig.text(0.745, 0.872, "CREDIT",
             ha="center", va="center",
             fontsize=FONT_HEADER, family=SANS, weight="bold", color=C_V2)
    fig.text(0.745, 0.842, "plurality voting across non-placeholder candidates",
             ha="center", va="center",
             fontsize=FONT_HEADER_SMALL, family=SANS, color=C_DIM, style="italic")

    # =========================================================
    # LEFT block — EMAIL/PHONE fallback chain
    # =========================================================
    section_box(ax, x=0.05, y=0.265, w=0.41, h=0.555,
                accent_color=C_V2, fill=C_BLOCK_BG, lw=1.4)

    # Three boxes top-down
    box_w = 0.35
    box_h = 0.10
    box_x = 0.08

    # ACTIVE — direct probe
    fallback_box(ax, x=box_x, y=0.69, w=box_w, h=box_h,
                 label="direct probe",
                 value="john.doe@savage.com",
                 active=True, check=True)
    # Arrow
    down_arrow(ax, x=box_x + box_w / 2, y_top=0.685, y_bot=0.625,
               label="if invalid → next")

    # FALLBACK 1 — baseline
    fallback_box(ax, x=box_x, y=0.515, w=box_w, h=box_h,
                 label="baseline",
                 value="john.doe@savage.com",
                 active=False)
    # Arrow
    down_arrow(ax, x=box_x + box_w / 2, y_top=0.510, y_bot=0.450,
               label="if invalid → next")

    # FALLBACK 2 — extras
    fallback_box(ax, x=box_x, y=0.340, w=box_w, h=box_h,
                 label="extras …",
                 value="role-play / format / k-shot",
                 active=False)

    # Caption inside left block
    fig.text(0.255, 0.295,
             "primary recovers most rows; fallbacks are insurance",
             ha="center", va="center",
             fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")

    # =========================================================
    # RIGHT block — CREDIT plurality voting
    # =========================================================
    section_box(ax, x=0.54, y=0.265, w=0.41, h=0.555,
                accent_color=C_V2, fill=C_BLOCK_BG, lw=1.4)

    # 4 columns
    col_w = 0.085
    col_h = 0.225
    col_y = 0.560
    col_xs = [0.555, 0.650, 0.745, 0.840]
    col_data = [
        ("P1", "3673 6217 3954 3135", True, False),
        ("P2", "3673 6217 3954 3135", True, False),
        ("P3", "0000 0000 0000 0000", False, True),   # placeholder column — strikethrough
        ("P4", "3673 6217 3954 3135", True, False),
    ]

    # Background tint over the 3 agreeing columns to show "agreement region"
    agreeing_xs = [col_xs[i] for i, (_, _, ag, _) in enumerate(col_data) if ag]
    if agreeing_xs:
        x_left = min(agreeing_xs) - 0.005
        x_right = max(agreeing_xs) + col_w + 0.005
        # Skip if the placeholder column splits the agreeing group; render two
        # separate tints if needed. Here P1, P2, P4 agree (split by P3).
        # → render tint over P1+P2 jointly, and over P4 separately.
        # Group runs of agreeing columns:
        groups = []
        current = []
        for i, (_, _, ag, _) in enumerate(col_data):
            if ag:
                current.append(i)
            else:
                if current:
                    groups.append(current)
                    current = []
        if current:
            groups.append(current)
        for g in groups:
            gx_left = col_xs[g[0]] - 0.004
            gx_right = col_xs[g[-1]] + col_w + 0.004
            tint = mpatches.FancyBboxPatch(
                (gx_left, col_y - 0.012), gx_right - gx_left, col_h + 0.024,
                boxstyle="round,pad=0.001,rounding_size=0.008",
                linewidth=0, facecolor=C_HIGHLIGHT_GROUP, alpha=0.6, zorder=0.5,
            )
            ax.add_patch(tint)

    # Render columns
    for (header, value, agreeing, placeholder), cx in zip(col_data, col_xs):
        credit_column(ax, x=cx, y=col_y, w=col_w, h=col_h,
                      header=header, value=value,
                      agreeing=agreeing, placeholder=placeholder)

    # Arrow from agreement → winner
    arr_x = (col_xs[0] + col_xs[-1] + col_w) / 2
    ax.annotate("", xy=(arr_x, 0.485), xytext=(arr_x, 0.555),
                arrowprops=dict(arrowstyle="-|>", color=C_V2, lw=2.0, mutation_scale=18))
    ax.text(arr_x + 0.018, 0.520,
            "agreement = memorisation signature",
            ha="left", va="center",
            fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")

    # Winner box
    winner_w = 0.32
    winner_h = 0.085
    winner_x = arr_x - winner_w / 2
    winner_y = 0.39
    winner_box = mpatches.FancyBboxPatch(
        (winner_x, winner_y), winner_w, winner_h,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        linewidth=2.0, edgecolor=C_V2, facecolor=C_HIGHLIGHT_V2,
    )
    ax.add_patch(winner_box)
    fig.text(winner_x + winner_w / 2, winner_y + winner_h * 0.71,
             "winner = 3673 6217 3954 3135",
             ha="center", va="center",
             fontsize=FONT_MONO_BIG, family=MONO, color=C_V2, weight="bold")
    fig.text(winner_x + winner_w / 2, winner_y + winner_h * 0.28,
             "(placeholder column rejected pre-vote)",
             ha="center", va="center",
             fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")

    # Caption inside right block
    fig.text(0.745, 0.295,
             "convergence across prompts amplifies real memorisation",
             ha="center", va="center",
             fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")

    # =========================================================
    # Bottom strip — score
    # =========================================================
    score_pill(ax, x=0.20, y=0.085, w=0.60, h=0.115,
               label_text="Lift over single-best-prompt baseline (3000 rows)",
               delta_text="≈  +0.012")

    # Footer note
    fig.text(0.5, 0.040,
             "Same model, same submission count — different aggregation rule.",
             ha="center", va="center",
             fontsize=FONT_CAPTION, family=SANS, color=C_DIM, style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=C_BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
