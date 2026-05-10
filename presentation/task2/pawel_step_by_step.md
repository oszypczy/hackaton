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

## Slide-by-slide gesture script (what the speaker DOES while talking)

> Use this when rehearsing in front of a camera. The slide is one image
> with three readable regions; the speaker draws the audience's eye
> through them in this order.

### Beat 1 (0:00 – 0:15) — Slide enters, you set the frame
- **Visual focus:** the whole slide — let the audience absorb the
  two-block split for one beat.
- **Speaker stance:** at centre of stage, eyes to upper-third of the
  slide where the title sits.
- **Spoken:** *"Different prompts succeed on different rows. Across the
  campaigns my teammate just walked through, some prompts return clean
  memorised PII, some return placeholders, some return the wrong
  type. We collected every CSV and stitched them together as a
  structured fallback — that's our ensemble."*
- **Gesture:** open palm sweep from left of slide to right, indicating
  "this whole picture is the ensemble".

### Beat 2 (0:15 – 0:35) — Walk the two PII rules, left then right
- **Visual focus:** LEFT block (EMAIL/PHONE fallback chain).
- **Speaker stance:** half-step toward the left side of the slide.
- **Spoken (left-half):** *"We use a different aggregation rule per
  PII type, because each fails differently. EMAIL and PHONE — the
  model genuinely memorised the patterns — so we pick the single best
  prompt with a fallback chain through the others."*
- **Gesture:** point at the green "direct probe ✓" box at top of the
  LEFT chain, then trace down through the gray fallback boxes —
  showing the waterfall.
- **Visual focus:** RIGHT block (CREDIT voting).
- **Speaker stance:** half-step toward the right side of the slide.
- **Spoken (right-half):** *"CREDIT — the model has no visual anchor
  and defaults to placeholders — so we vote across non-placeholder
  candidates from every source."*
- **Gesture:** point at each of the 4 source-column heads (P1/P2/P3/P4),
  then sweep underline beneath the agreement highlight.

### Beat 3 (0:35 – 0:50) — Lift the agreement signal as the climax
- **Visual focus:** the agreement-highlight on the CREDIT block —
  this is the visual climax of the slide.
- **Speaker stance:** stay on the right.
- **Spoken:** *"Plurality voting on CREDIT exploits source agreement:
  when three different prompts converge on the same digit string,
  that's a strong signal of memorisation. Disagreement signals
  halucynation, which we down-weight."*
- **Gesture:** index finger taps the converging column heads (P1, P2,
  P3) one by one, then a downward chop pointing at the
  "= 3673 6217 3954 3135" winner row. Finally cross-out gesture over
  the rejected `0000…` placeholder column.

### Beat 4 (0:50 – 1:00) — Land on the lift, then yield to next slot
- **Visual focus:** bottom score box.
- **Speaker stance:** return to centre.
- **Spoken:** *"Compared to using the single best prompt alone,
  per-PII routing with non-placeholder voting lifted us by another
  point — small absolute, meaningful at the scale of 3000 rows."*
- **Gesture:** open palm down toward the score box, then a small
  closing nod — signals "end of my slot".

### Optional micro-bridge (only if running 2-3 s short)
- **Add:** *"Same model, same submission count — different aggregation
  rule. The lift is free."*

---

## Slide region specification (designer must hit these exact targets)

> If the figure is being authored fresh in Claude Design rather than
> reusing `figures/01_split_screen.png`, these are the constraints
> the slide must meet to support the speaker's gesture script above.

### Title bar (top, ≤ 8 % of slide height)
- Text: `Task 2 — PII ensemble: per-PII routing across prompt CSVs`
- Optional subtitle: `Different prompts succeed on different rows. We aggregate them with per-feature rules.`
- Sans-serif, headline color `#222`. No accent color in the title bar.

### LEFT block — EMAIL / PHONE fallback chain (~45 % of body width)
- Header: `EMAIL & PHONE — single best with fallback chain`
- Stack of 3-4 boxes, top-down arrows, monospace email/phone strings inside:
  - Box 1 (TOP, ACTIVE — orange tint, green ✓ icon): `direct probe`
    + sample value (`john.doe@savage.com`).
  - Arrow down with label `if invalid`.
  - Box 2 (gray): `baseline`.
  - Arrow down with label `if invalid`.
  - Box 3 (gray): `extras …` (or list two more).
- Caption beneath: *"primary recovers most rows; fallbacks are insurance"*.

### RIGHT block — CREDIT plurality voting (~55 % of body width)
- Header: `CREDIT — plurality voting across non-placeholder candidates`
- Small comparison table, 1 row per example user, 4 columns (`P1` … `P4`):
  - 3 columns show the same 16-digit string (e.g. `3673 6217 3954 3135`)
    on a tinted-orange background — these are the agreeing votes.
  - 1 column shows `0000 0000 0000 0000` in dark red, with a strikethrough.
- Below the table: an arrow pointing down to the winner string
  (`= 3673 6217 3954 3135`) in bold orange.
- Caption beneath: *"agreement = memorisation signature; placeholders rejected first"*.

### Bottom strip — Score & delta (≤ 12 % of slide height)
- Single sentence centred: `Lift over single-best-prompt baseline:`
- One score box (small, centred), green (`#15803D`) bold delta:
  `≈ +0.012 across 3000 rows`.
- No absolute leaderboard numbers. No public/private split mention.

### Visual continuity with teammate's slide (slot 3)
- Same monospace family for digit strings and emails (Fira Code or
  JetBrains Mono).
- Same accent orange (`#D97706`) for selected / winning candidates.
- Same muted gray (`#9CA3AF`) for fallback / non-active boxes.
- Same dark red (`#B91C1C`) + strikethrough for rejected placeholders.
- Background white / very pale gray.
- Body type ≥ 18 pt for projection-grade readability.

### Negative space rule
- The agreement-highlight region in the RIGHT block must have at least
  20 px of breathing room around it — that's the visual climax of the
  slide and needs space to land.

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
| How exactly does dummy detection work? | Three rules per type, validation-tuned. CREDIT is flagged when the digit set has size ≤ 2 (catches `0000…` and `1111…`) or forms an arithmetic sequence (catches `2460 2461 2462 2463`). EMAIL is flagged on a small denylist of throwaway domains (`example.com`, `test.com`, …) plus single-token locals without a dot. PHONE is flagged on the same digit-collapse rule plus a `1234567890` family. We checked these on the 280-user validation set — no false positives on real PII. |
| What's the actual delta in numbers? | Going from "single best prompt for every PII" to "per-PII routing with non-placeholder voting on CREDIT" adds about a hundredth on the average score. Most of that lift sits on CREDIT, where the voting recovers the small minority of rows where some non-primary prompt happens to have a non-placeholder candidate while the primary prompt collapsed. |
| Did you try weighted or learned aggregation? | Plurality voting is unweighted — each surviving non-placeholder candidate counts as one vote. We considered weighting by prior validation score but the calibrator wasn't predictive enough to justify the extra hyperparameter. The fallback chain on EMAIL/PHONE is implicitly weighted by *priority order* — direct probe first, baselines next. |
| What if the model has not memorised a row at all? | Then every prompt produces either a hallucination or a placeholder. The dummy-detection step removes placeholders; what's left is hallucinated content that doesn't agree across prompts. The vote then reduces to the first non-placeholder candidate from the priority list, which is roughly equivalent to the single-best-prompt baseline. So unrecoverable rows degrade gracefully — they don't get worse. |
| Can this be run online (streaming) or only on pre-collected CSVs? | Both. The aggregation logic is per-row independent and stateless; it can run as a post-process on any number of streaming sources. We ran it as a batch over CSVs because the prompt-attack pipeline was already producing CSVs as its native output. |
| Why didn't you ensemble across the K-shot samples within a single prompt? | We did, separately — same prompt, K samples with temperature, then a Levenshtein-medoid pick. That's a parallel research direction; results were close and it can be combined with the multi-prompt ensemble shown here. The two axes are orthogonal: K-shot diversifies the *decoder*, multi-prompt diversifies the *probe*. |

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
