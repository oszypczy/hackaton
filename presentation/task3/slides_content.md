# Task 3 — 2-minute deck (English, **solution only**)

No task wording on slides—only **our method**. Voice-over can assume the audience already knows the setting.

## Slide images (16:9)

| File | Content |
|------|---------|
| `slide1_solution.png` | Pipeline: **shared features** → **Head A / Head B** → **mean of ranks** → **CSV** |
| `slide2_heads.png` | What differs between heads: **OLMo-7B-Instruct** full-distribution stats vs **green-list z-tests** |
| `slide3_tradeoffs_en.png` | Strengths vs limitations of **this** design |

Extras (optional): `data_flow_task3.png`, screenshot from `evolution-chart.html`.

---

## Voice-over (short)

**Slide 1 (~40 s):** One wide feature stack for every text; we train **two** logistic regressions that only differ by a small specialist block. On the test set we **average ranks** of the two score vectors, then rescale—**not** a raw average of probabilities.

**Slide 2 (~45 s):** Head A adds **per-step softmax shape** under **OLMo-7B-Instruct**—entropy, ranks, bursts. Head B adds cheap **multi-configuration green-list statistics** without another large LM forward. Same labels, same baseline columns, **six cross-model deltas** built from existing scores.

**Slide 3 (~35 s):** **Pros:** complementary signals, rank merge aligned with **threshold-at-fixed-FPR** evaluation, modular heads. **Cons:** fixed **equal** weights in rank fusion (not learned); many overlapping features so we rely on **strong regularization**; list-style head is sensitive to **tokenizer / scheme mismatch** vs the true generator.

---

## Regeneration prompt (Claude Design / similar)

```
Three 16:9 English slides. Do NOT explain the competition task or metrics from scratch.
Only describe OUR ensemble: shared features, two heads (OLMo-7B-Instruct distribution vs green-list z-tests), LogReg each, rank fusion, CSV.
Style: flat infographic, navy #1a365d, teal accent, minimal text (icons + 3–5 words per slide body).
Slide 3: strengths vs limitations of this design only—do NOT mention leaderboard, public/private splits, or GPU/compute cost.
No footer label "BEST3". Optional tiny footer: "CISPA SprintML" only.
```

---

Deeper logic (not for slides): [../../docs/tasks/task3_solution_logic_linear_EN.md](../../docs/tasks/task3_solution_logic_linear_EN.md)
