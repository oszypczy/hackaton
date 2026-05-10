# Presentation — Task 3 (watermark submission)

## Folder `task3/` — 2 min, English, **solution-only** (no task intro)

- PNGs: `slide1_solution.png`, `slide2_heads.png`, `slide3_tradeoffs_en.png`
- Script + regen prompt: [task3/slides_content.md](task3/slides_content.md)

## Files (`presentation/` root)

| File | Description |
|------|-------------|
| `task3/slides.md` | Marp: three full-bleed PNGs → export PDF. |
| `task3/data_flow_task3.png` | High-level **solution** flow (English): shared layer → two heads → rank fusion → CSV. |
| `task3/evolution-chart.html` | Bar chart: milestone scores + illustrative **final blend** bar (edit `MILESTONES` in HTML if needed). |
| `task3/data_flow_explained_PL.md` | Polish walk-through of `data_flow_task3.png` (reference). |

## Notes on `task3/evolution-chart.html`

There is no single canonical API score stored in-repo for the rank-blend CSV. The chart uses an **illustrative** last bar between `cross_lm (~0.284)` and a reference top snapshot (~0.40). Edit `MILESTONES` in the HTML to match your records.
