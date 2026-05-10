# Claude Design prompt — Task 2 (PII Extraction) presentation slide

> Paste this whole document (plus `step_by_step.md`, plus the PNGs in
> `figures/`) into Claude Design. The goal is **1 slide** that occupies
> 1 minute of a 6-minute team presentation at the CISPA / SprintML
> hackathon finals.
>
> **Read `step_by_step.md` first** — it contains the canonical mental
> model, designer cues per slide region, and speaker timing. This document
> is the brief; that one is the reference.

---

## Brief

We are presenting our solution to **Task 2 — PII Extraction Attack** on a
multimodal LLM (OLMo-2-1B + LLaVA-HR vision encoder fine-tuned on
synthetic user records). The task: given a user's name and a *scrubbed*
image, recover the EMAIL / CREDIT / PHONE the model memorized during
fine-tuning. 3000 rows = 1000 users × 3 PII types.

**Audience:** mixed — fellow hackathon teams (technical), CISPA SprintML
lab (Adam Dziedzic, Franziska Boenisch — they know PII extraction
literature; **Pinto et al. ICML 2024** and **PII-Scope arXiv:2410.06704**
are direct references), and two CS professors who are not memorization-
attack specialists. Treat the audience as **technically literate but not
necessarily expert in extraction attacks**.

**Tone:** confident, methodologically grounded, non-flashy. The story is
*systematic comparison + deterministic fallbacks*. Design should
foreground the strategy choice and the fallback rules.

**Time budget:** **60 seconds** of speaking time covering 1 slide.
The slot is part of a 6-minute / 6-slide team deck.

**Storyline:** *V1 prefix-priming baseline → 5 strategies benchmarked →
direct probe wins → fallbacks for failures → +0.07 over baseline*.

**Important — what NOT to say:** the speaker will not mention "OCR" or
"discovery that validation has visible PII". The organizers told us
that — it's not a discovery, and citing it weakens our slot.

---

## Assets (attached)

| File | What it shows | When to use |
|---|---|---|
| `figures/01_split_screen.png` | Side-by-side V1 (baseline) vs V2 (winner of 5) chat blocks (monospace), V1 with injected assistant prefix highlighted, V2 with direct probe highlighted. Score boxes underneath: V1 LB 0.31 + "prefix-priming baseline" tag, V2 LB 0.38 + "direct probe + fallbacks" tag + green "+0.07" delta. Bottom 25%: fallback panel with 3 raw → fixed examples (EMAIL no `@`, PHONE wrong shape, PHONE missing `+`). 16:9, 300 DPI. | **The entire slide.** |
| `step_by_step.md` | Plain-language walkthrough, slide-region breakdown, designer cues, speaker timing, Q&A pocket card. | **Read this first.** |

---

## Narrative (verbatim — speaker reads or paraphrases this)

> Task 2: a multimodal LLM was fine-tuned on synthetic user records —
> emails, credit cards, phones. Our job: extract that PII back from the
> model, given only the user's name and a scrubbed image.
>
> We started by taking the training-time answer template, cutting it
> right before the redacted PII, and pasting it as the *start* of the
> assistant's reply. The model thinks it has already begun answering and
> just continues. That gave us our baseline — and a fair comparison
> point for everything we tried next.
>
> We then benchmarked five different prompt strategies on local
> validation — naive, role-play, system-override, completion format, and
> a direct memory probe. The direct probe won: ask the model what it
> remembers, output only that.
>
> When the model fails to emit a valid email or phone — wrong format,
> wrong type — we fall back: synthesize `firstname.lastname@example.com`
> from the question, force the E.164 plus prefix, normalize card
> formatting. End result: +0.07 over the baseline.

---

## Layout

The hero figure (`01_split_screen.png`) is **already laid out as a full
slide**. Designer can use it directly as background and overlay the
title bar with house typography, OR re-rasterize the chat blocks /
fallback panel using the design system. Either way, the structure must
match:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Title — "Task 2 — PII extraction: prompt-attack pipeline"            │
│ Subtitle — one-sentence problem statement                            │
├──────────────────────────────────────────────────────────────────────┤
│  V1 — paste training            │   V2 — ask the model directly      │
│   template into reply           │   (winner of 5 prompt strategies)  │
│   (baseline)                    │                                    │
│                                 │                                    │
│  [chat block, monospace]        │   [chat block, monospace]          │
│  with INJECTED PREFIX           │   with DIRECT PROBE highlighted    │
│  highlighted                    │                                    │
│                                 │                                    │
│  ┌─────────────────────┐        │   ┌─────────────────────┐          │
│  │ public LB  0.31     │        │   │ public LB  0.38     │ +0.07    │
│  │ baseline            │        │   │ direct probe + FB   │          │
│  └─────────────────────┘        │   └─────────────────────┘          │
├──────────────────────────────────────────────────────────────────────┤
│                  Fallbacks — when model emits invalid output         │
│                                                                      │
│  EMAIL — no '@'         "+1-505-555-9847"   →  crystal.serrano@...   │
│  PHONE — wrong shape    "4986 6022 6865 7288"  →  +15555550000       │
│  PHONE — missing '+'    "13859159897"          →  +13859159897       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Visual / typographic guidance

- **Color palette:** V1 = muted gray (`#9CA3AF`); V2 = warm orange
  (`#D97706`); positive delta = green (`#15803D`); error/raw output =
  dark red (`#B91C1C`). Designer may swap to the team's house accent
  but keep the V1/V2 contrast.
- **Monospace** for chat blocks AND for fallback raw/fixed values
  (Fira Code, JetBrains Mono, Menlo). The injected prefix in V1 and
  the direct probe in V2 should sit on **subtle highlight backgrounds**
  (light yellow / light orange tint) so the audience sees *exactly*
  what we changed.
- **Sans-serif** for everything else (Inter, Helvetica). Body ≥ 18 pt.
- **Background:** white or very pale gray. High contrast — slides will
  be projected.
- **The "+0.07" delta** in the V2 score box should be visually loud —
  green, bold, slightly larger. It is the headline number.
- **The fallback panel background** should be a separate pale tint
  (pale yellow `#FFFBEB`) so the panel reads as a distinct module.
- **No on-slide math.** No regex symbols on slide. The metric
  (`1 − Levenshtein.normalized_distance`) belongs in speaker voice.

---

## Negative constraints (do **not** include)

- ❌ "OCR" / "discovery" / "validation set wasn't scrubbed" anywhere on
  slide. The organizers gave us that information; we do not pitch it.
- ❌ Specific scoreboard ranks. **Do not show 0.40+ ensemble numbers** —
  those belong to teammate's downstream-ensemble slot.
- ❌ Slurm job IDs, commit hashes, file paths.
- ❌ Internal terminology on slide: "assistant-prefix priming",
  "memory probe", "constrained decoding". Speaker phrases. Slide text
  is plain English ("V1 — paste training template into reply" /
  "V2 — ask the model directly").
- ❌ "Hallucination" without context. If used, say "model fills in
  plausible values when recall fails".
- ❌ Phrases like "we got lucky" / "happy alignment" / "after lots of
  iteration" / "we failed first". Method is presented as systematic.
- ❌ Per-row prediction tables.
- ❌ Pinto / PII-Scope citations on slide. Speaker may mention them in
  Q&A; the slide is method-first, not literature-anchored.

---

## Team integration

This 1-slide block is **part of a 6-minute team presentation** with two
other tasks:

- **Task 1 — DUCI** (slot 1–2 of 6 minutes). See `presentation/task1/`.
  Palette: ResNet18 = `#4C78A8` (blue), ResNet50 = `#54A24B` (green),
  ResNet152 = `#F58518` (orange).
- **Teammate's Task 2 ensemble slide** (slot 4 of 6 minutes — separate
  speaker). Covers per-PII routing across multiple prompt-attack CSVs.
  Our slide must **not** encroach on that — keep our score numbers to
  the prompt-attack pipeline only.
- **Task 3 — Watermark Detection** (slots 5–6 of 6 minutes). See
  `presentation/task3/` (when populated).

When all three task folders are populated, the designer is welcome to
**unify color palette and typography** across all 6 slides for cohesion.
Suggest staying in the warm-tone family (`#F58518` orange in Task 1 ↔
`#D97706` orange in Task 2) for visual continuity.

A short team-wide style note lives at `presentation/_team/README.md`.
