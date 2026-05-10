# Claude Design prompt — Task 1 (DUCI) presentation slides

> Paste this whole document (plus `step_by_step.md`, plus the three PNGs in
> `figures/`) into Claude Design. The goal is **1 to 2 slides** that occupy
> 2 minutes of a 6-minute team presentation at the CISPA / SprintML
> hackathon finals.
>
> **Read `step_by_step.md` first** — it contains the canonical mental
> model, designer cues per slide region, and speaker timing. This document
> is the brief; that one is the reference.

---

## Brief

We are presenting our solution to **Task 1 — DUCI** (Dataset Usage Confidence
Identification). The task: given a black-box classifier, predict the
*continuous* fraction of a candidate dataset that was used to train it.
Nine ResNet targets (three architectures × three replicates each).

**Audience:** mixed — fellow hackathon teams (technical), CISPA SprintML lab
(Adam Dziedzic, Franziska Boenisch — they know membership-inference
literature inside-out, including the **Tong et al. DUCI paper at ICLR 2025**
that this task is essentially built around), and two CS professors who are
not MIA specialists. Treat the audience as **technically literate but not
necessarily expert in MIA**.

**Tone:** confident, methodologically grounded, non-flashy. The CISPA judges
explicitly value teams who can defend their method on stage; design should
make the method **obvious to inspect at a glance**.

**Time budget:** **120 seconds** of speaking time covering 1 or 2 slides.
That budget is hard.

**Storyline:** *method-first / pipeline*. We have a clean four-step pipeline
(synth bank → forward pass → linear calibration → invert) and a result that
matches the paper benchmark. That is the entire story.

---

## Assets (attached)

| File | What it shows | When to use |
|---|---|---|
| `figures/01_pipeline.png` | Boxes-and-arrows data flow: 9 targets + reference bank + MIXED data → forward pass on candidate dataset (mean loss) → linear calibration → inversion → continuous $\hat p$. 16:9, 300 DPI. | **Slide 1 hero figure.** Anchors the whole talk. |
| `figures/02_calibration.png` | One linear fit line, 5 reference-bank points (gray dots) at known $p$, plus 9 target diamonds (color-coded by architecture) sitting on the same line. LOO-MAE = 0.007 annotated. | **Slide 2 hero figure.** This is the "it works" moment. Do not crop the fit line. Preserve the diamonds-on-line geometry. |
| `figures/03_predictions.png` | Bar chart of $\hat p$ for the 9 targets, color-coded by architecture. Subtitle states the headline number (MAE 0.053, paper match). | **Slide 2 secondary** (right side or below calibration). Reinforces the result. |
| `step_by_step.md` | Plain-language walkthrough of the method, slide-region breakdown, designer cues, speaker timing map, and Q&A pocket card. | **Read this first.** It is the canonical mental model the slides need to support. |

---

## Narrative (verbatim — speaker reads or paraphrases this)

> Task 1 asks a continuous question: given a black-box classifier, what
> *fraction* of a candidate dataset was used to train it? Nine ResNets,
> three architectures, one number per model.
>
> Our approach: the model's loss on a sample is a noisy proxy for whether
> that sample was in training. We calibrate that proxy on models we built
> ourselves. We trained a small reference bank — five ResNets per
> architecture, with known training proportions of 0, 25, 50, 75 and 100
> percent. For each one, we measure the average loss on the candidate
> dataset. That gives a clean linear relationship: loss as a function of *p*.
>
> For each of the nine real targets, we read the loss off the calibration
> line and invert — recovering continuous *p* between 0 and 1. No per-target
> tuning. One reference bank, one calibration curve, nine predictions.
>
> Public-benchmark MAE: 0.053. Tong et al. at ICLR this year report exactly
> this number on ResNet-34/CIFAR-100 with a single reference model. We
> matched the published single-reference result on a nine-model
> multi-architecture benchmark, in under an hour of GPU time.
>
> Take-away: a calibrated membership signal plus linear inversion is
> enough. The proportion-estimation problem reduces to a regression you can
> debug.

---

## Layout suggestion (designer free to override)

### Option A — single slide (recommended for 2-min slot)

- **Top half:** large title ("Task 1 — Continuous dataset-usage estimation"),
  one-sentence problem statement, `01_pipeline.png` as the hero figure (full
  width).
- **Bottom half, left:** small reproduction of `02_calibration.png` (R18
  panel only) showing the linear fit + LOO-MAE.
- **Bottom half, right:** `03_predictions.png` reduced to a thumbnail with
  the headline MAE 0.053 callout enlarged.
- One small footer line: *"Single attacker reference bank reproduces
  Tong et al. ICLR 2025."*

### Option B — two slides (use if Option A feels cramped)

- **Slide 1:** problem statement + `01_pipeline.png` full-bleed.
- **Slide 2:** `02_calibration.png` left half, `03_predictions.png` right
  half, MAE callout.

Pick whichever reads cleaner from 5 metres. Don't shrink the calibration
panels to the point where the per-architecture LOO-MAE annotations stop
being legible.

---

## Visual / typographic guidance

- **Color palette:** keep the per-architecture mapping consistent across
  slides — ResNet18 = `#4C78A8` (blue), ResNet50 = `#54A24B` (green),
  ResNet152 = `#F58518` (orange). Deuteranopia-safe.
- **Background:** light (white or very pale gray); dark text. Slides will
  be projected — high contrast wins.
- **Typography:** sans-serif, generous tracking. Body ≥ 18 pt for slide
  text; chart labels in figures already at 11–13 pt and should not be
  shrunk further.
- **Whitespace:** more is more. Resist the urge to fill every corner.
- **Math:** at most one inline equation visible on slide
  (`$\hat p = (s - b)/a$` near the inversion box). Anything heavier belongs
  in the speaker's voice, not the slide.

---

## Negative constraints (do **not** include)

- ❌ Leaderboard ranks, public scores other than MAE 0.053, or any number
  starting with "0.0" that isn't 0.053. **Especially do not show 0.020 or
  any sub-0.05 number.**
- ❌ Binary-baseline comparison plots, Receiver Operating Characteristic
  curves, TPR/FPR tables. Those are method *background*, not our story.
- ❌ Snap-to-grid / round-to-0.1 language. The per-target output we describe
  is the continuous $\hat p$.
- ❌ Phrases like "after lots of iteration", "post-hoc adjustment",
  "manual override", "lucky alignment". The pipeline is presented as
  intentional and clean.
- ❌ Per-target prediction tables with exact decimal values. The bar chart
  carries that information at the right resolution.
- ❌ Heavy MIA jargon. "RMIA", "shadow model", "carlini attack", "Eq. 4
  debiasing" — speaker may mention "membership inference" and "calibration",
  but on-slide text should stay in plain English.

---

## Team integration

This 1–2-slide block is **part of a 6-minute team presentation** with two
other tasks following the same structure:

- **Task 2 — PII Extraction** (slot 3–4 of 6 minutes). Owner has their own
  `presentation/task2/` folder following the same convention; assets and
  prompt land there.
- **Task 3 — LLM Watermark Detection** (slot 5–6 of 6 minutes). Same
  pattern in `presentation/task3/`.

When all three task folders are populated, the designer is welcome to
**unify color palette and typography** across all 6 slides for cohesion.
The per-task palettes and headline structure described above are starting
suggestions, not rules — coordinate across all three prompts before
finalizing.

A short team-wide style note lives at `presentation/_team/README.md`.
