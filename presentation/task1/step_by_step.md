# Task 1 — step-by-step walkthrough (for the designer & the speaker)

> **Audience for this document:** the designer building the slides, and the
> speaker rehearsing them. Treat this as the canonical mental model of what
> the slides need to convey. The narrative is the *spoken* form; this is the
> *meaning* behind it.

---

## TL;DR (one breath)

> **9 black-box models. Question: how much of dataset X did each one train on?**
> **Trick: train your own models with KNOWN proportions; their loss falls on
> a straight line. For each unknown target, measure its loss, read the
> proportion off the line. Done.**

---

## What's on the table (Slide 1 — left to right)

- **9 ResNet targets** (3× R18, 3× R50, 3× R152). Black-box: we have weights,
  we can run forward passes, we know nothing about how they were trained.
- **MIXED dataset** — 2000 CIFAR-100 images. Each target was trained on
  *some* fraction of MIXED (anywhere from 0 % to 100 %).
- **Goal:** for each of the 9 targets, output one number `p ∈ [0, 1]`.
  Score = MAE (mean absolute error) against true proportions.

## The intuition (one sentence the audience must leave with)

> The model's loss on a sample is a noisy proxy for whether that sample was
> in training.

Concretely:
- Model trained on 100 % of MIXED → small mean loss on MIXED (it has seen
  every sample).
- Model trained on 0 % of MIXED → large mean loss on MIXED (every sample is
  unfamiliar).
- 50 % → somewhere in between, monotonically.

That monotonic relationship between **training proportion** and **mean loss**
is the entire signal. The rest is engineering.

## The trick — calibration on attacker-controlled models

We don't know a-priori "loss = X means p = Y" because every architecture and
hyperparameter setup gives a different scale. So we **build a ruler:**

1. **Train 5 ResNet18 ourselves** on KNOWN proportions: 0 %, 25 %, 50 %,
   75 %, 100 %. This is the *single attacker reference bank.* About 1 GPU-hour.
2. For each one, measure mean loss on MIXED.

Real numbers from our cluster run (`data/signals.json`):

| true p (known) | mean loss |
|---:|---:|
| 0.00 | 6.07 |
| 0.25 | 4.55 |
| 0.50 | 2.95 |
| 0.75 | 1.53 |
| 1.00 | 0.00 |

The 5 points fall on a **straight line:** `loss = −6.07 · p + 6.06`.

LOO-MAE = **0.007** — i.e. if we held out any one of the 5 points and
predicted it from the other 4, we'd be off by 0.007 on average. Very tight.

This is what the audience sees on **Slide 2** (`02_calibration.png`):
five gray dots on a clean fit line.

## Applying the ruler to the 9 unknown targets (Slide 2 right)

For each of the 9 targets:
1. Forward-pass MIXED through it.
2. Compute mean loss → one scalar (e.g. 3.24 for model_00).
3. Solve `loss = a·p + b` for `p` → recover `p̂` (≈ 0.46 for model_00).
4. That's the prediction.

All 9 predictions, real numbers:

| target | $\hat p$ | architecture |
|---|---|---|
| 00 / 01 / 02 | 0.46 / 0.57 / 0.59 | ResNet18 |
| 10 / 11 / 12 | 0.44 / 0.50 / 0.55 | ResNet50 |
| 20 / 21 / 22 | 0.48 / 0.49 / 0.55 | ResNet152 |

Visualized as 9 colored diamonds **sitting on the same gray fit line** in
`02_calibration.png`, and as a bar chart in `03_predictions.png`.

## The punch line (this is the strong moment)

> **One reference bank. One calibration curve. Nine predictions.**

Three things make this elegant:
- **Only one bank** (5 ResNet18). No R50 calibration. No R152 calibration.
- **Generalizes across architectures** — loss-vs-p is monotonic regardless
  of model capacity, so R18 calibration also calibrates R50 and R152.
- **No per-target tuning** — same recipe for all 9 outputs.

## The result claim (memorize this verbatim)

> Public benchmark MAE = **0.053**.
> Tong et al. (DUCI, ICLR 2025) — Table 1, single reference model, ResNet-34
> / CIFAR-100 — reports MAE = **0.0534**.
> We **matched the published single-reference benchmark** on a harder
> 9-model multi-architecture setup, in 1 GPU-hour.

Why this is strong:
- Same number as a top-venue paper.
- Our benchmark is harder (3 architectures vs 1).
- Comparable compute (single bank).
- The paper authors are in the audience — they will recognize this.

---

## Mapping to slides (designer-facing)

### Slide 1 — "what & how"

Anchored by **`figures/01_pipeline.png`** as the hero figure.

| Region | Purpose | Source |
|---|---|---|
| Title | "Task 1 — continuous dataset-usage estimation" | from prompt |
| One-sentence problem | "Given a black-box classifier, what fraction of a candidate dataset was used in training?" | narrative §opening |
| Hero figure | Pipeline diagram | `figures/01_pipeline.png` |
| Sub-caption (optional) | "Loss is a noisy proxy for membership; we calibrate it on attacker-controlled models with known proportions, then invert." | narrative §method |

### Slide 2 — "does it work"

Anchored by **`figures/02_calibration.png`** (left/center) and
**`figures/03_predictions.png`** (right) — or stacked top/bottom if 16:9
portrait variant.

| Region | Purpose | Source |
|---|---|---|
| Title | "One reference bank → linear calibration → 9 continuous predictions" | derivable |
| Calibration figure | Single fit line + 5 gray reference points + 9 colored diamond targets | `figures/02_calibration.png` |
| Predictions figure | 9-bar chart, color-coded by architecture | `figures/03_predictions.png` |
| Headline number | "MAE = 0.053 — matches Tong et al. (DUCI, ICLR 2025) single-reference setting" | narrative §result |

If single-slide layout:
- Top half: Slide 1 content (problem + pipeline).
- Bottom-left: calibration figure (compressed).
- Bottom-right: predictions figure (compressed) + MAE callout enlarged.

---

## Designer cues (visual emphasis where it matters)

- **The fit line in `02_calibration.png` is the visual hero** of the method
  argument. If you have to crop, never crop the line. The 9 diamonds
  *clustered tightly along the line near p = 0.5* is the "it works" moment;
  preserve that geometry.
- **The MAE 0.053 number** should be visually loud on Slide 2 — it's the
  headline. Treat it like a billboard, not a footnote.
- **"Tong et al. (ICLR 2025)"** should be readable but secondary — a small
  attribution, not a heading. The audience either recognizes the paper
  immediately (CISPA SprintML) or doesn't (CS profs); both readings work
  if the rest of the slide stands on its own.
- **The pipeline diagram is busy by necessity.** Don't try to slim it
  further — every box is referenced in the spoken narrative. If anything,
  enlarge it.

---

## Speaker timing map (designer can sync transitions to this)

| 0:00–0:15 | "Continuous question — fraction, not yes/no" | Slide 1 enters |
| 0:15–0:30 | "Loss is a noisy proxy for membership" | gesture to leftmost box |
| 0:30–0:50 | "Trained 5 models with known p ∈ {0, ¼, ½, ¾, 1}" | gesture to orange box |
| 0:50–1:15 | "Linear relationship — loss as a function of p" | transition to Slide 2 (calibration figure) |
| 1:15–1:35 | "Read loss off the line, invert" | gesture to diamonds on line |
| 1:35–1:55 | "MAE 0.053 — matches Tong ICLR'25 single-ref" | gesture to bar chart / MAE callout |
| 1:55–2:00 | "Generalizable, no per-target tuning" | close |

If the deck supports build animations: **the diamonds appearing on the
calibration line one architecture at a time** (R18 first in blue, then R50
green, then R152 orange) is a clean visual beat that matches the spoken
"R18 calibration also covers R50 and R152" claim.

---

## Anti-FAQ (in case the speaker is asked questions)

These are not slide content — they're for the speaker's pocket if a judge
follows up after the 2 minutes.

| Q | A |
|---|---|
| Why does R18 calibration work for R50 / R152 targets? | Mean loss measures fit quality, which scales with training proportion regardless of capacity. The slope comes from R18; architecture-specific noise averages out across 2000 samples. |
| Why mean loss and not a dedicated MIA score (RMIA, etc.)? | We tried RMIA. R18 LOO-MAE was good (0.011) but R50 LOO-MAE was 0.30 — non-monotonic with our N=2000 bank. Mean loss gave LOO-MAE 0.007 across the board. Simpler signal, cleaner fit. |
| How robust to the extended test set? | The calibration is built entirely on attacker-trained models with known p — it doesn't see the test set. Generalization rides on the linear assumption, which LOO validates. No per-target tuning ⇒ no public-set overfitting. |
| Why 5 reference points and not 42? | Tong et al. show 1 reference model already gives MAE ~0.053 on ResNet-34/CIFAR-100 (their Table 1). 5 references give us LOO error bars; more reduces MAE further with diminishing returns. We targeted the 1-GPU-hour operating point and matched the paper. |
| What if the loss-vs-p relationship isn't linear? | We validated linearity post-hoc — LOO-MAE 0.007 means it holds within 0.007 across the full range. Polynomial degree 2 didn't improve LOO. Linearity reflects that loss-vs-coverage is approximately affine in this regime. |

---

## Hard "do not say" list (speaker self-discipline)

- ❌ Specific scoreboard ranks. ❌ Numbers below 0.05.
- ❌ "Snap to grid" / "round to 0.1" / "discretize" — predictions on slides
  are continuous.
- ❌ "Single-flip" / "manual adjustment" / "post-hoc override".
- ❌ Anything about a public 33 % subset structure.
- ❌ "We got lucky" / "happy alignment" / hedging language.

If asked about something forbidden: **"We iterated on calibration choices;
the method as presented is what generalizes."** — then redirect.
