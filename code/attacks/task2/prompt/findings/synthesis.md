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

## Heads up — paper provenance check

Agent #2 reported that paper corpus does NOT contain:
- "Pinto et al. ICML'24 — Extracting Training Data from DocVQA" (arXiv:2407.08707)
- "PII-Scope" (arXiv:2410.06704 v2)

These were both cited as anchor papers in `NOTES.md` and `STRATEGY.md` for Path A. If they aren't actually in the corpus, our descriptions of their methods (e.g., PII-Scope's "5 templates × 5x boost", Pinto's "blackout PII region recipe") may be reconstructions from secondary sources rather than verified. Worth a quick check: `ls references/papers/` and grep for arXiv IDs.

Doesn't change current plan — agent #2 reconstructed the multi-template idea from Carlini'21 §5.1 (3 strategies × 6 metrics) which IS in corpus. But for the final presentation, we should not cite Pinto/PII-Scope unless we can produce them.
