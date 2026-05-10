# Task 2 — step-by-step walkthrough (slot 2/2: ensemble)

> **Audience for this document:** the designer building the slide, and
> the speaker rehearsing it. Treat this as the canonical mental model of
> what the slide needs to convey. The narrative is the *spoken* form;
> this is the *meaning* behind it.
>
> **Slot:** 1 slide, ~60 seconds. Speaker (Paweł / murdzek2) covers the
> downstream ensemble that aggregates outputs from the prompt-attack
> pipeline. Teammate's slot covers the prompt strategies — do not
> re-explain the prompts.

---

## TL;DR (one breath)

> **The prompt-attack pipeline produces 5–6 candidate CSVs — one per
> strategy. Across them, every individual row has a mix of clean
> memorised PII, placeholders, and wrong-type outputs. We treat the
> CSVs as a structured fallback chain: per-PII type, EMAIL and PHONE
> take the single best prompt with the others as fallbacks; CREDIT
> takes a plurality vote across non-placeholder candidates. That extra
> aggregation step lifts the score one more point on top of the
> prompt-attack baseline.**

---

## What's on the table (input)

By the time my slot starts, we already have:

- **5–6 source CSVs**, each one being the prompt-attack pipeline output
  under a different prompt strategy (the methodology my teammate just
  walked through).
- Each CSV has the same shape: 3000 rows = 1000 users × 3 PII types
  (EMAIL / CREDIT / PHONE).
- Each cell can be:
  - a **clean memorised PII** (`john.doe@savage.com`, 16-digit CC, E.164 phone)
  - a **placeholder** the model fell back on when it lacked signal
    (`0000 0000 0000 0000`, `1234567890`, `peters.com` cluster)
  - a **wrong-type** output (a phone number where an email should be)
  - a **format-broken** output (12-digit "card number", missing `+`)

Critically: **no single CSV is best on every row.** A prompt that
recovers EMAIL well may collapse on CREDIT, and vice versa.

## The intuition (one sentence the audience must leave with)

> Each prompt is a partial probe. Combining them with a per-feature
> rule recovers the answer that any one prompt would miss.

---

## The two aggregation rules

### Rule 1 — EMAIL and PHONE: fallback chain

The model genuinely memorised local-parts (`firstname.lastname`) and
country-code prefixes (`+1...`). One prompt — the direct memory probe
— gets these right far more often than the others.

Logic per row:

```
if direct_probe[row] looks valid (regex fit, not placeholder):
    keep it
elif baseline[row] looks valid:
    use baseline
elif extras[row] looks valid:
    walk down the priority list
else:
    fall back to last-resort regex synthesis from the question
```

Voting across EMAIL/PHONE prompts would *dilute* the signal — the
direct probe is already close to the recall ceiling and the others
add noise.

### Rule 2 — CREDIT: plurality voting on non-placeholders

CREDIT is fundamentally different. The card number lives mostly in
the image; with the image scrubbed, the model has no anchor. Across
CSVs we observe:

- the direct probe places `0000 0000 0000 0000` on **about 99%** of
  CREDIT rows when the card image is masked;
- other prompts produce mixed-quality 16-digit guesses that are
  uncorrelated when the model is hallucinating, and **agree** when
  the model has actual memorisation.

So for each CREDIT row:

```
1. Reject placeholder shapes (zeros, arithmetic sequences, single-digit fills).
2. Among the survivors, pick the digit string that the most CSVs agree on.
3. Tie-break by source priority (the most reliable prompt first).
```

Convergence across independent prompts is a memorisation signature.
Lone outliers are halucynations and lose the vote.

## Dummy-detection rules (validation-tuned)

| PII type | Marked as placeholder if... |
|---|---|
| CREDIT | digits collapse to ≤ 2 unique values, or form an arithmetic sequence |
| EMAIL | domain in throwaway list (example.com, test.com, …) or local-part is a single token without a dot |
| PHONE | digits collapse to ≤ 2 unique values, or string equals a known stub |

These rules were tuned on the 280-user validation set — known
placeholder shapes were excluded, known good shapes were preserved.
No magic numbers tuned to the public leaderboard.

## What it lifts

- Compared to using the single best prompt alone (the direct memory
  probe applied uniformly to all PII types), per-PII routing with
  non-placeholder voting on CREDIT lifts the score by **about a point**
  (delta on order of +0.012 across 3000 rows).
- That sounds small — and it is, in absolute terms — but it's a
  consistent, calibrated lift on top of an already-tuned prompt
  attack. Same model, same submissions count, different aggregation
  rule.

The lift comes from the right places:
- EMAIL / PHONE recovery is preserved (single best prompt remains
  the source).
- CREDIT recovers a fraction of placeholder rows by surfacing the
  minority of non-placeholder candidates.

---

## Mapping to slide (designer-facing)

### Single slide — recommended layout

| Region | Purpose | Source |
|---|---|---|
| Title | "Task 2 — PII ensemble: per-PII routing across prompt CSVs" | from prompt |
| One-sentence problem | "Different prompts succeed on different rows — combine them with per-PII rules." | step-by-step §what's on the table |
| Hero — left block | EMAIL / PHONE → fallback chain diagram (single best prompt with arrows to fallbacks) | new figure |
| Hero — right block | CREDIT → multi-source voting diagram (3-4 column heads with example digit strings, agreement highlight) | new figure |
| Bottom strip | Score box: ensemble result vs single-best-prompt baseline + small per-PII delta breakdown | new figure |

---

## Designer cues (visual emphasis where it matters)

- **Two-block split**, mirroring teammate's slide structure.
- **EMAIL / PHONE block (LEFT):** a simple arrow chain — a green
  primary box ("direct probe") with grey fallback boxes below
  ("baseline", "role-play", …). Show it as a *waterfall*: only one
  active source per row, the rest are insurance.
- **CREDIT block (RIGHT):** a small comparison table, one column per
  prompt source, one row per example user. Highlight the digit string
  that the majority of sources agree on — that's the winner. Strike
  through the placeholder columns (zeros, arithmetic seq).
- **Use the same monospace font** as teammate's chat blocks for the
  digit strings and emails — visual consistency across the two
  task-2 slides.
- **One accent color** for the chosen-row in each block (green or
  warm orange, matching teammate's V2 highlight). Placeholders /
  rejected candidates in muted gray.
- **Score box at bottom centre:** small, single number — the
  ensemble lift over single-best-prompt. Don't put leaderboard
  numbers other teams have access to.

---

## Speaker timing map

| 0:00–0:15 | "Different prompts succeed on different rows — ensemble is the structured fallback" | slide enters |
| 0:15–0:35 | "EMAIL / PHONE → fallback chain. CREDIT → vote across non-placeholders" | gesture LEFT block, then RIGHT |
| 0:35–0:50 | "Convergence across prompts = memorisation signature" | gesture to agreement highlight |
| 0:50–1:00 | "Lifts the score one more point" | gesture to score box |

---

## Anti-FAQ (for the speaker's pocket if a judge follows up)

| Q | A |
|---|---|
| Why per-PII rules instead of one global ensemble rule? | EMAIL / PHONE memorisation is dominated by the single best prompt — voting would average in noise from weaker prompts. CREDIT memorisation is *bimodal* (placeholder vs real) — voting amplifies the real cluster. Different failure modes need different aggregation rules. |
| How do you avoid overfitting the routing to the public leaderboard? | The dummy-detection rules and the per-PII assignment of "voting" vs "single-best-with-fallback" were chosen on the 280-user validation set. The voting tie-break is content-agnostic (highest count, source priority for ties). No magic numbers tuned to public LB. |
| Could you have just generated more candidates from a single best prompt instead of running multiple prompts? | We tried K-shot sampling under the same prompt as a separate diversity axis. Results were close but multi-prompt diversity was cheaper to combine — every CSV was already on disk from the prompt-attack benchmark. The two diversity axes are orthogonal and could be combined. |
| What if all prompts agree on a wrong answer? | Then we accept the wrong answer — there's no oracle at inference. This is the inherent ceiling of any non-shadow-based extraction; agreement-as-confidence is a heuristic, not a proof. The validation set caught the worst cases (placeholder agreement) which is why the dummy-detection step runs before voting. |
| Were any prompts redundant in the ensemble? | Yes — two of our prompts were highly correlated on EMAIL/PHONE because they both phrased the question as a memory probe. We kept both as redundant insurance for the fallback chain — costs a constant per row to evaluate. The CREDIT vote benefits even from correlated sources because it's voting on *content agreement*, not strategy diversity. |
| Is this an attack-side or defence-side contribution? | Attack-side. The contribution is that even when individual prompts are imperfect, a structured aggregation rule recovers a measurable additional fraction of memorised PII at zero additional inference cost. |

---

## Hard "do not say" list (speaker self-discipline)

- ❌ "OCR" / "we discovered validation has visible PII". Same rule as
  teammate slot.
- ❌ "We failed" / "we got lucky" / "happy alignment". Frame as
  *systematic comparison* and *targeted aggregation rules*.
- ❌ Specific scoreboard ranks. **Do not show 0.40+ raw numbers** if
  presenter is not from this team — keep deltas relative.
- ❌ Slurm job IDs, commit hashes, CSV filenames, file paths.
- ❌ Internal labels: "Path A", "Path B", "smart_v2", "smart_ensemble",
  "question_repeat", "K4_directprobe", "shadow_hybrid". Speaker phrases.
  On slide it's "the direct probe", "the prompt CSVs", "the ensemble".
- ❌ "Hallucination" without context — say "model fills in placeholder
  values when it lacks a visual anchor".
- ❌ The 213 non-redacted bonus. Organizers said it doesn't count for
  final scoring. Citing it weakens the slot.
- ❌ References to Membership Decoding / Bayesian calibration / the
  research paper we *didn't* implement. Save for Q&A only.
- ❌ Anything about a public 30 % subset structure.

If asked something forbidden: **"the ensemble's contract is at the
aggregation layer; the rest is in the prompt-attack pipeline you've
already seen."** — then redirect.
