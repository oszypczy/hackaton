# Synthesis — 3-agent findings post-v2 (leaderboard 0.381)

> Generated 2026-05-09 after spawning 3 parallel agents on direct_probe results.
> Sources:
> - `findings/method_depth.md` (agent #1, prompt-paradigm depth)
> - `findings/literature_review.md` (agent #2, papers + task spec)
> - `findings/hybrid_alternative.md` (agent #3, hybrid/multi-stage)

## Convergence map

|                                         | Agent #1 | Agent #2 | Agent #3 |
|-----------------------------------------|----------|----------|----------|
| Per-PII routing (DP→EMAIL/PHONE, BL→CC) | ✓        |          | ✓        |
| K-shot candidate + logprob ranking      |          | ✓        | ✓        |
| EMAIL candidate-domain re-ranking       | ✓        |          |          |
| Verbatim-prefix BYPASS chat template    |          | ✓        |          |
|   (contrarian — argues direct_probe is anti-pattern) |          |          |          |

## Tension to resolve

- **Empirical (agents #1, #3):** direct_probe gave +0.08 lift over baseline on val_pii blank-mode (n=840). Build on it.
- **Literature (agent #2):** Carlini'21 §6.5 + Nasr'23 §5 argue overfitted models want **literal training prefix without chat template wrapper**. Calls direct_probe an anti-pattern.

Both can be right: direct_probe > chat-template-wrapped baseline, AND raw verbatim prefix > direct_probe. Resolved empirically by P2 ablation.

## Action plan (priority-ranked)

### P1 — Per-PII routing (LOW cost, HIGH confidence)
- direct_probe → EMAIL + PHONE
- baseline (with assistant prefix) → CREDIT (restores 4-4-4-4 training format)
- ~15 LoC: per-pii_type strategy dispatch in main.py
- 1 full eval val_pii 840 blank-mode (~14 min)
- Expected lift on task/: +0.01–0.03 OVERALL (CREDIT format restoration dominant)
- Risk: minimal — baseline CREDIT pipeline already in production

### P2 — Verbatim-prefix bypass (LOW cost, contrarian)
- New strategy `verbatim_prefix`: skip apply_chat_template, build raw `[SYSTEM]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER_PREFIX]` matching training format exactly
- ~30 LoC + new strategy entry in strategies.py
- 1 full eval val_pii 840 blank-mode (~14 min)
- Expected: unknown — could win, could regress
- Value: resolves literature vs empirical disagreement, is a single ablation

### P3 — K-shot candidate + own-logprob ranking (MEDIUM cost)
- Sample K=8 candidates per sample with temperature decay (1.0 → 0.0), pick by mean per-token NLL or Levenshtein-medoid
- ~80 LoC + new entrypoint
- Eval on subset 840 with K=4: ~50 min; full predict K=4 = ~3.5h
- Expected lift: +0.02–0.04 (likely CREDIT since current is floor)
- Risk: 8× cost, lift uncertain — must validate on val_pii blank first

### P4 — EMAIL candidate-domain re-ranking (DEFER)
- Enumerate domain candidates (gmail/yahoo/savage/elliott/...) per record, pick by logprob over the email tail
- Requires logprob extraction per candidate — non-trivial
- Expected lift: medium for EMAIL only

## Recommended execution sequence

1. **NOW: P1 + P2 in parallel** — two separate sbatch eval jobs on val_pii blank-mode (~14 min wall clock, two GPUs)
2. Pick better of {P1, P2} → predict task/ with that strategy (52 min) → submit anchor v3
3. **If time:** P3 K-shot ensemble. Subset eval first to confirm lift, then full predict (~3.5h)
4. **Probably out of scope:** P4

## Heads up — paper provenance check (RESOLVED)

Agent #2 originally flagged that the corpus did NOT contain:
- "Pinto et al. ICML'24 — Extracting Training Data from DocVQA" (arXiv:2407.08707)
- "PII-Scope" (arXiv:2410.06704 v2)

**Now downloaded** to `references/papers/{pinto_2407.08707,pii_scope_2410.06704}.pdf` + extracted txt. A 4th agent verified both papers and found **prior NOTES.md descriptions were partially WRONG**.

## Paper analysis update (Pinto + PII-Scope verified, agent #4)

Source: `findings/paper_analysis_pinto_piiscope.md`

### Corrections to prior NOTES.md / STRATEGY.md
| Our prior claim | Paper actually says |
|---|---|
| "PII-Scope: 5 templates T1-T5" | **4 templates A/B/C/D** (lines 315-319) |
| "5× boost from template diversity" | **5.4× = 4 templates × 64 top-k samples = 256 queries aggregated**. Boost vs single-query best (2.6% → 14.0%), not vs naive baseline. |
| "Pinto: blackout PII region + arbitrary probe" | `(I⁻ᵃ, Q_original)`: image with answer-OCR-bbox whited out + the EXACT training question. NOT an arbitrary probe. |
| "Question paraphrase doesn't matter much" | **OPPOSITE.** Pinto §5.2: paraphrasing Q drops extraction MORE than perturbing the image. Q-phrasing is the dominant lever. |

### Pinto §5 key findings (lines 680-890)
- §5.2: Q-paraphrasing via PaLM2 → biggest extractability drop. Image-only (random Q) extracts ~0; Q-only (random image) still extracts 4-21 samples → **question carries more information than image** for triggering memorization.
- §5.3: Brightness ×0.5–×2 retains most extraction; rotations ±5/±10° hurt more — spatial layout matters more than pixel values.
- §4: Lower training resolution → MORE memorization. Our target was overfitted intentionally → expect strong memorization signal.

### PII-Scope §6-7 key findings
- §6.2 lines 779-783: **Template D wins** because it appears as a literal substring in Enron training data ("---Original Message---\nFrom: {Name} [mailto:"). Multi-line, email-header-like.
- §6.4: PII-Compass — prepend FULL true prefix of ANOTHER subject before query. +3× over plain template. L=100 typically optimal.
- §6.3: ICL demo SELECTION matters more than count: 2 well-chosen ≈ 32 random. Order matters.
- §7.2: Continual/adaptive extraction — feed extracted PII back as new ICL demos, re-attack. ~2× lift over 10 rounds, saturating ~round 5.
- Per-PII boost varies a lot: true-prefix +2.5×, Template +5.4×, ICL +2.9×, PII-Compass +3.0×. **Phone harder than email.**

### NEW recommendations (not in prior agents' lists)

**P5 — Email-header Template D phrasing for EMAIL** [PII-Scope §6.2]
- Strategy variant: `"---Original Message---\nFrom: {Name} [mailto:"` for EMAIL
- Reasoning: matches multi-line email-header substring that appeared verbatim in training corpora; +1.7× over single-line "X's email is".
- Cost: 1 new strategy entry, single eval.
- Expected: **+0.05-0.10 on EMAIL** (only) IF training format matches.

**P6 — Validation_pii as one-shot in-context demos** [PII-Scope §6.4 + §6.3]
- For each task/ row, prepend ONE (Name → PII) pair from validation_pii as a demo:
  - `"{ValName}'s email is {ValEmail}\n{TaskName}'s email is "`
- Demo selection: pick val_pii subject with most surface similarity (last-name length, first letter, area code if PHONE) — quality >> quantity.
- Cost: O(1) per row, no training. ~30 LoC.
- Expected: **+0.05+ across all PIIs**, potentially largest on CREDIT (which is currently floor — verbatim format demo may trigger format mimicry even w/o memorization).
- Verifiable on val_pii blank.

**P7 — Original-question reverse-engineering** [Pinto §5.2]
- Manually inspect 5-10 val_pii images for the EXACT training-time question/caption (visible OCR-able text in the image template).
- Use that verbatim as the probe — replaces our paraphrased `direct_probe`.
- Cost: 30 min manual inspection + new strategy. 0 GPU-min for the inspection.
- Expected: **+0.10-0.15 globally** if we recover the training prompt. Largest single lift available per Pinto's results.

### Updated execution priority

Now we have 7 candidate paths (P1-P7). Recommended ordering after P1+P2 evals return:

1. **P1 + P2 eval results** (running) — confirms per-pii routing + chat-template-bypass
2. **P7 first** (manual ~30 min, big upside) — recover training-time question by inspecting val_pii images
3. **P5** (single new strategy, easy ablation) — Template-D email-header phrasing
4. **P6** (one-shot demo) — most impact for CREDIT floor
5. **P3** K-shot ensemble — last because expensive

P4 (EMAIL domain re-rank) absorbed into P6 (demos give domain prior implicitly).
