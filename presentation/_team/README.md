# Team presentation — convention & shared style

> 6 minutes total team presentation, 3 tasks, ~2 minutes per task.
> Each task owner builds their slot independently in `presentation/task<N>/`,
> then the designer combines the three prompts + assets into one coherent deck.

## Folder convention (mirror per task)

```
presentation/
├── _team/
│   └── README.md                 ← this file
├── task1/                        ← Oliwier (DUCI)
│   ├── data/signals.json
│   ├── figures/01_pipeline.png
│   ├── figures/02_calibration.png
│   ├── figures/03_predictions.png
│   ├── figures_scripts/plot_*.py
│   ├── narrative.md              ← 2-min spoken script + cheatsheet
│   └── claude_design_prompt.md   ← brief for Claude Design
├── task2/                        ← (PII Extraction)
│   └── ...
└── task3/                        ← (Watermark Detection)
    └── ...
```

## Shared style guide (suggestion — designer may unify)

| Property | Value |
|---|---|
| Aspect ratio | 16:9 |
| Figure DPI | 300 (presentation-grade) |
| Background | white / very pale gray |
| Body font | sans-serif (Inter / Helvetica), ≥ 18 pt on slide |
| Headline color | `#222` |
| Per-architecture (task1) palette | `#4C78A8` (blue), `#54A24B` (green), `#F58518` (orange) |
| Math | one inline equation max per slide |

If task2 / task3 already have an established palette, keep yours.
Designer is free to harmonize across the deck.

## Hard time budget

- Task 1: **2 minutes** (slides 1–2 of 6)
- Task 2: **2 minutes** (slides 3–4 of 6)
- Task 3: **2 minutes** (slides 5–6 of 6)

Inter-task transition slide is optional; if used, it counts against
whichever task is taking the cleaner cut.

## Process — combining the three prompts

1. Each owner finalizes their `claude_design_prompt.md` with all assets
   in `figures/` ready.
2. Concatenate the three prompts into a single Claude Design session, in
   order task1 → task2 → task3.
3. Add a short opening: *"This is a 6-slide team presentation. Maintain a
   consistent design language across all slides; the per-task prompts
   describe each slot."*
4. Iterate.

## Don't

- ❌ Don't overwrite another task's `claude_design_prompt.md`.
- ❌ Don't move shared figures into `_team/` — keep figures with their task.
- ❌ Don't commit large model artifacts or raw signal CSVs > 5 MB; keep
  them on the cluster and only pull JSON summaries to `data/`.
