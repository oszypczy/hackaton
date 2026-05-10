# Claude Design prompt — Task 2 (PII Extraction) ENSEMBLE slide

> Paste this whole document (plus `pawel_step_by_step.md`) into Claude
> Design. The goal is **1 slide** that occupies 1 minute of a 6-minute
> team presentation at the CISPA / SprintML hackathon finals. This is
> **slot 4 of 6** — the second of two task-2 slides.
>
> **Read `pawel_step_by_step.md` first** — it contains the canonical
> mental model, designer cues per slide region, and speaker timing.
> This document is the brief; that one is the reference.

---

## Brief

This is the **second** Task 2 slide. The first one (teammate's) covers
the prompt-attack methodology — how a single prompt is constructed and
which one of five strategies won on validation. **This slide picks up
where that one ends**, on the fallback panel, and generalises the
fallback idea into a per-PII ensemble across multiple prompt CSVs.

**Audience:** mixed — fellow hackathon teams, CISPA SprintML lab
(Adam Dziedzic, Franziska Boenisch — they know PII extraction
literature; **PII-Scope arXiv:2410.06704** is a relevant reference
for ICL-style aggregation, but **do not put citations on the slide**),
and two CS professors who are not memorization-attack specialists.

**Tone:** confident, methodologically grounded, non-flashy. The story
is *every prompt is partial; per-feature aggregation rules close the
gaps*. Design should make the per-PII split obvious and the
agreement-as-signal idea legible.

**Time budget:** **60 seconds** of speaking time covering 1 slide.

**Storyline:** *5–6 prompt CSVs from the previous slide → EMAIL/PHONE
take fallback chain → CREDIT takes plurality vote on non-placeholders
→ +1 point of additional lift over single-best-prompt.*

**Important — what NOT to say:** the speaker will not mention "OCR",
"we discovered validation has visible PII", "lucky", "we failed",
specific job IDs, internal CSV filenames, the 213-non-redacted public
LB bonus, or the existence of a public/private split. See the hard
do-not-say list in `pawel_step_by_step.md`.

---

## Assets to be produced

| File | What it shows | When to use |
|---|---|---|
| `figures/04_ensemble_routing.png` (TBD) | Two-block split: LEFT = EMAIL/PHONE fallback chain (single best prompt → fallback waterfall). RIGHT = CREDIT plurality voting (3-4 source columns, agreement highlight on majority digit string). Bottom: small score box with delta vs single-best-prompt. 16:9, 300 DPI. | **The entire slide.** |
| `pawel_step_by_step.md` | Plain-language walkthrough, slide-region breakdown, designer cues, speaker timing, Q&A pocket card. | **Read this first.** |

---

## Narrative (verbatim — speaker reads or paraphrases this)

> Different prompts succeed on different rows. Across the campaigns
> my teammate just walked through, some prompts return clean memorised
> PII, some return placeholders, some return the wrong type. We
> collected every CSV and stitched them together as a structured
> fallback — that's our ensemble.
>
> We use a different aggregation rule per PII type, because each fails
> differently. EMAIL and PHONE — the model genuinely memorised the
> patterns — so we pick the single best prompt with a fallback chain
> through the others. CREDIT — the model has no visual anchor and
> defaults to placeholders — so we vote across non-placeholder
> candidates from every source.
>
> Plurality voting on CREDIT exploits source agreement: when three
> different prompts converge on the same digit string, that's a strong
> signal of memorisation. Disagreement signals halucynation, which we
> down-weight.
>
> Compared to using the single best prompt alone, per-PII routing with
> non-placeholder voting lifted us by another point — small absolute,
> meaningful at the scale of 3000 rows.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ Title — "Task 2 — PII ensemble: per-PII routing across prompt CSVs"  │
│ Subtitle — one-sentence problem statement                            │
├────────────────────────────────────┬─────────────────────────────────┤
│ EMAIL & PHONE — fallback chain     │  CREDIT — plurality voting      │
│ (single best + insurance)          │  (across prompt CSVs)           │
│                                    │                                 │
│  ┌──────────────────────┐          │   ┌─────┬────┬────┬────┐        │
│  │ direct probe ✓       │          │   │ P1  │ P2 │ P3 │ P4 │        │
│  │ john.doe@savage.com  │          │   ├─────┼────┼────┼────┤        │
│  └──────────────────────┘          │   │3673 │3673│0000│1419│        │
│       ↓ (if invalid)               │   │6217 │6217│0000│3392│        │
│  ┌──────────────────────┐          │   │3954 │3954│0000│6594│        │
│  │ baseline (fallback)  │          │   │3135 │3135│0000│4993│        │
│  └──────────────────────┘          │   └──┬──┴─┬──┴─×──┴────┘        │
│       ↓ (if invalid)               │      └────┴───── majority       │
│  ┌──────────────────────┐          │      = 3673 6217 3954 3135      │
│  │ extras …             │          │       (placeholder rejected ×)  │
│  └──────────────────────┘          │                                 │
├────────────────────────────────────┴─────────────────────────────────┤
│            Lift over single-best-prompt: +1 point on 3000 rows       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Visual / typographic guidance

- **Match teammate's slide aesthetic.** Same color family, same
  monospace font for chat-like blocks (Fira Code / JetBrains Mono /
  Menlo). Sans-serif (Inter / Helvetica) for headers and labels.
- **Color logic** (mirroring teammate's V1/V2 contrast):
  - Selected / winning candidate = warm orange `#D97706` (same as
    teammate's V2 accent).
  - Fallback / not-active = muted gray `#9CA3AF`.
  - Rejected placeholder (`0000…`) = dark red `#B91C1C` with strikethrough.
  - Positive delta = green `#15803D`, bold, slightly larger.
- **The agreement highlight on the CREDIT block** is the visual climax
  — make it loud. The 2-3 columns whose digit strings agree should be
  visually grouped (subtle tinted background or a connecting bracket).
- **The placeholder column** should be clearly de-emphasised — gray,
  strikethrough digits — so the audience reads "this one was rejected"
  without the speaker having to say it.
- **Body ≥ 18 pt.** Background white or very pale gray.
- **One score number** at the bottom centre — the ensemble lift over
  single-best-prompt. Do not show absolute leaderboard numbers — keep
  deltas relative.
- **No on-slide math.** No regex symbols on slide. Aggregation rules
  belong in speaker voice, not on the slide.

---

## Negative constraints (do **not** include)

- ❌ "OCR" / "discovery" / "validation set wasn't scrubbed" anywhere on
  slide. Same rule as teammate's slide.
- ❌ Specific leaderboard numbers (0.40, 0.5328, etc.). Slide shows
  *delta over single-best-prompt only*.
- ❌ Internal CSV filenames: `task2_smart_ensemble_v2.csv`,
  `submission_v0_blank_question_repeat_*.csv`, etc. Slide says
  "5–6 prompt CSVs" or "the prompt-attack outputs".
- ❌ Slurm job IDs, commit hashes, file paths.
- ❌ Internal team labels: "Path A", "Path B", "smart_v2",
  "shadow attack". Use "prompt-attack pipeline" and "ensemble".
- ❌ The 213 non-redacted bonus / public-LB jump. Not part of final
  scoring; mentioning it weakens the slot.
- ❌ "Hallucination" without context. If used, "model fills in
  placeholder values when recall fails".
- ❌ "We got lucky" / "after a lot of iteration" / "we failed first".
  Method is presented as systematic.
- ❌ References to Membership Decoding, Bayesian calibration,
  contrastive decoding — those are research-prompt extensions
  *we did not ship*. Save for Q&A.
- ❌ Citations on slide. Speaker may name PII-Scope in Q&A.
- ❌ Per-row prediction tables.

---

## Team integration

This 1-slide block is **slot 4 of a 6-slide team presentation**:

- **Task 1 — DUCI** (slots 1–2 of 6 minutes). See `presentation/task1/`.
  Palette: ResNet18 = `#4C78A8` (blue), ResNet50 = `#54A24B` (green),
  ResNet152 = `#F58518` (orange).
- **Task 2 — prompt-attack pipeline** (slot 3 of 6 minutes). See
  `presentation/task2/narrative.md` + `step_by_step.md` +
  `claude_design_prompt.md`. Established palette: V1 grey, V2 orange,
  delta green. **Match this aesthetic.**
- **Task 2 — ensemble** (slot 4 of 6 minutes — THIS SLOT). Picks up
  on teammate's fallback panel and generalises to per-PII aggregation.
  Score numbers on this slide are deltas relative to single-best-prompt
  (teammate's V2), not raw leaderboard scores.
- **Task 3 — Watermark Detection** (slots 5–6 of 6 minutes). See
  `presentation/task3/`.

When all three task folders are populated, the designer is welcome to
unify color palette and typography across all 6 slides for cohesion.
For task 2 specifically, the two slides (3 + 4) should share a single
visual identity — same fonts, same color logic, same monospace
treatment for code-like blocks. The audience should perceive them as
a continuous two-step story.
