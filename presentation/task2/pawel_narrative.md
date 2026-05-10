# Task 2 — PII extraction (slot 2/2: ensemble) — 1-min narrative

> Method-first framing. Audience: hackathon participants + CISPA SprintML +
> 2 CS professors. Target language: English. Total spoken time ≈ 60 s
> (≈ 130 words at 130 wpm). Speaker = task 2 ensemble owner (murdzek2).
>
> **Picks up where teammate's prompt-attack slot ends.** Teammate finishes on
> the *fallback panel* — speaker steps in immediately, framing fallbacks as
> a connecting tissue: per-prompt success is partial; ensemble is the
> systematic generalisation of that idea.

---

## 0:00 – 0:15  Pickup from teammate (≈ 35 words)

> Different prompts succeed on different rows. Across the campaigns my
> teammate just walked through, some prompts return clean memorised PII,
> some return placeholders, some return the wrong type. We collected every
> CSV and stitched them together as a structured fallback — that's our
> ensemble.

[on screen: Slide enters — multi-source routing diagram (`figures/04_ensemble_routing.png`)]

## 0:15 – 0:35  Per-PII routing (≈ 50 words)

> We use a different aggregation rule per PII type, because each fails
> differently. EMAIL and PHONE — the model genuinely memorised the
> patterns — so we pick the single best prompt with a fallback chain
> through the others. CREDIT — the model has no visual anchor and
> defaults to placeholders — so we vote across non-placeholder candidates
> from every source.

[gesture to LEFT block — EMAIL/PHONE chain. Then RIGHT block — CREDIT vote]

## 0:35 – 0:50  Why this works (≈ 30 words)

> Plurality voting on CREDIT exploits source agreement: when three
> different prompts converge on the same digit string, that's a strong
> signal of memorisation. Disagreement signals halucynation, which we
> down-weight.

[gesture to bottom panel — agreement indicators across 3-4 source CSVs]

## 0:50 – 1:00  Result (≈ 25 words)

> Compared to using the single best prompt alone, per-PII routing with
> non-placeholder voting lifted us by another point — small absolute,
> meaningful at the scale of 3000 rows.

[gesture to score delta — final 0.40 vs single-prompt-best 0.39]

---

## Speaker cheat-sheet (≤ 5 bullets to memorize)

1. **Pickup:** different prompts succeed on different rows → ensemble is
   the structured fallback.
2. **Per-PII routing:** EMAIL/PHONE → fallback chain across prompts
   (single best + fallbacks). CREDIT → plurality vote across non-dummy
   candidates.
3. **Why CREDIT is different:** no visual anchor → model defaults to
   placeholder. Voting on non-placeholders amplifies real memorisation.
4. **Why EMAIL/PHONE is different:** real memorisation signal in single
   best prompt. Voting would dilute.
5. **Closer:** ensemble lifts another point on top of the prompt-attack
   pipeline — same dataset, different aggregation rule.

## Self-check timing

- 130 spoken words ≈ 60 s at 130 wpm. Buffer ≈ 5 s.
- If running long: cut "disagreement signals halucynation" sentence —
  the diagram already conveys it.
- If running short: add — *"the dummy-detection rules are tuned on
  validation: zeros, arithmetic sequences, throwaway domains all flagged
  before voting."*

## Hard "do-not-say" list

- ❌ "OCR" / "we discovered the validation set wasn't scrubbed" — same
  rule as teammate slot. Organizers gave us this; not a discovery.
- ❌ Specific scoreboard ranks or comparisons to other teams.
- ❌ "Hallucination" without context — say "the model fills in
  placeholder values when it lacks a visual anchor".
- ❌ "We failed first" / "we got lucky" / "after a lot of iteration".
  Frame the progression as systematic.
- ❌ Internal job IDs, slurm specifics, commit hashes, file paths,
  CSV filenames.
- ❌ Per-prompt CSV identifiers ("question_repeat", "K4_directprobe",
  "shadow_hybrid"…) — too internal. Slide says "5–6 source CSVs".
- ❌ "Path A / Path B" team-internal labels — say "the prompt-attack
  pipeline" and "the ensemble".
- ❌ The 213 non-redacted public-LB bonus. It's known to organizers and
  not part of final scoring. Mentioning it weakens the slot.

## Bridge phrases (for the live handoff from teammate)

If teammate ends crisply on "+0.07 over baseline":
> "Different prompts succeed on different rows. Across the campaigns
> my teammate just walked through…"

If teammate ends loose:
> "And that fallback panel — that's the seed for what I'll show next."
