# v3 plan — synthesized from 3 agents (2026-05-09 evening)

> Po P1/P2 fail (LB 0.381 v2 direct_probe stays top). Sources: `v3_strategy_ranking.md`,
> `error_pattern_analysis.md`, `research_prompt.md`. Synthesis below ranks ALL three.

## Convergence (3 agents same call)

- **EMAIL domain = biggest unmeasured lever.** Error P3: 39% local-correct, 0% domain-correct.
  V3-rank V7 (domain-freq prior). Research Q3 (edit-distance-aware aggregation).
- **CREDIT floor needs FORMAT, not content.** Error P1: 46% wrong digit-count.
  V3-rank V1 demo injects literal 4-4-4-4 token stream direct_probe cannot.
- **Strategy ensemble oracle ceiling +0.054** (error P8) ↔ V3-rank V3/V5 K-shot medoid ↔
  Research Q3. All three converge on aggregation as next-tier ceiling.

## Tension

- V3-rank trusts blank calibrator (delta +0.031 ± 0.005). Error: oracle is blank-only;
  task/ is original-image; CREDIT collapse `0000 0000…` ×110/280 on blank → 1/1000 task/.
  V1 lift estimates may be inflated. **A3 (rerun multi-eval ORIGINAL) before V1 predict.**
- V3-rank treats V7 (domain rerank) as "free side-effect". Error P3 says it IS the
  biggest unmeasured lever. → wire V7 INTO V1 post-process, not as standalone.

## Phased plan (16h runway, 5-min cooldown, val_pii calibrator-gated)

### Phase 0 — free post-process replay [NOW, zero GPU, ~30 min]

Apply on existing v2 CSV + v1 fallback for narrowly-defined failure cases:
- **P1** CREDIT length+format normalize: extract digits → if 13≤n<16 pad last digit, n>16 keep first 16 → reformat `\d{4} \d{4} \d{4} \d{4}`
- **P4** EMAIL template-echo blacklist: `^(card|tel)@(dateofbirth|card)\.(com|net)$` and `*@example.com` → fallback to v1 prediction at that row
- **P5** PHONE `+1?555\d{7}$` placeholder → fallback to v1 prediction
- **P7** PHONE country-code reformat: extract digits → drop leading 1 → take last 10 → prefix `+1`

Submit v2.1 → expected LB **0.40–0.41**. Zero risk, deterministic.

### Phase 1 — calibrator parity check [30 min GPU]

A3: rerun 3-strategy multi-eval (direct_probe + verbatim_prefix + role_play_dba) on
val_pii **image_mode=original** (currently only blank). Resolves: does V1 demo lift
hold under original-image conditions, or does image already kill the CREDIT collapse
that demo would fix?

### Phase 2 — V1 oneshot_demo (+ V2 Template-D email side-test) [28 min multi_eval, 52 min predict]

- Per-PII routing on top of one-shot val_pii ICL demo. Demo selection by surface
  similarity (last-name length / area-code / 4-4-4-4 format anchor).
- V2 piggyback: same eval, EMAIL slot replaced with Template-D `"---Original Message---\nFrom: {Name} [mailto:"`
- Wire P3 (rare-domain rerank) into post-process step at V1's output stage.
- Submit v3 → expected LB **0.42–0.46** (if A3 confirms lift).

### Phase 3 — Claude Research async [parallel with Phase 2]

User pasted `research_prompt.md` to Claude Research. Output reviewed when V1 lands.
Use only if V1 underperforms (informs v4 K-shot family). Q3 of prompt is the
load-bearing question — Levenshtein-aware aggregation has highest expected return
absent additional ground truth.

### Phase 4 (optional, ≥4 GPU-h runway) — V5 K-shot × demo combo

K=4 sampling on V1 demo prompt, EMAIL+PHONE only, medoid pick. +0.005–0.02 over V1.
Skip if v3 already lands ≥0.42.

### Phase 5 (parallel, 30 min user-side) — V4 manual training-Q recovery

Inspect 10 val_pii images for literal training-time Q phrasing. Replace direct_probe's
paraphrase. Free, but confidence reduced post-P2 fail (Carlini/Nasr verbatim hypothesis
already falsified locally).

## Rejected (from this synthesis)

- V3-rank V3 standalone (K-shot without demo) — V1+post-process reaches +0.054 ceiling cheaper.
- V3-rank V6 PII-Compass long-prefix — superset V1 cost, uncertain marginal return.
- More chat-template tweaks (P1/P2 family dead).
- Image perturbations (paper_analysis:23 — layout ≪ Q-phrasing).
- system_override / completion_format (NOTES.md:411–417 negative).

## Submission budget

5 valuable submits remaining. Ordered: v2.1 (Phase 0) → v3 (Phase 2 V1) → v4 (Phase 4
K-shot if applicable) → v5 (Phase 5 V4 manual if hits) → 1 reserve. Each gated on
val_pii blank ≥+0.02 over current best (calibrator delta-reliable per synthesis.md:155).

## Status checkpoints

- [x] Phase 0 post-processor written + v2.1 CSV produced
- [x] Phase 0 submitted, LB recorded — **NO LB MOVE** (format-only fixes below noise)
- [~] Phase 1 A3 — SKIPPED (Phase 0 confirmed format wins are insufficient)
- [x] Phase 2 V1 strategy implemented + eval submitted
- [x] Phase 2 V1 eval done — **REJECTED** (-0.049 vs DP, demo-leak pollution)
- [x] Research v2 reviewed (`new_researchclaude.md`)
- [ ] **Phase 4 (REVISED)** — K-shot + Levenshtein medoid on direct_probe (this session)
- [ ] **Phase 5 (PARALLEL)** — Contrastive Decoding with shadow_lmm (separate session, branch task2-prompt-cd)

## Revised plan post-V1 fail (2026-05-09 22:50)

Pivot: drop V1 family. Run two strategies in parallel:

### Phase 4 (this session, branch `task2-prompt`) — K-shot + Lev medoid

K=8 sampling on direct_probe (typical-τ=0.4, top_p=0.95) + Levenshtein medoid aggregator.
Per research §3.1 + §3.2 + §3.4: medoid = Bayes-opt for 1-Lev_norm metric.
Implementation: ~50 LoC (`aggregator.py` + `attack.py` sampling mode).
Eval cost: ~112 min (8× direct_probe). Predict cost: K=4 → 3.5h, K=8 → 7h.

### Phase 5 (parallel session, branch `task2-prompt-cd`) — Contrastive Decoding

α·logits_target − β·logits_amateur via custom decoder. Amateur = `shadow_lmm/`
(clean OLMo-2-1B, multimodal, no PII finetune). Memorized PII tokens have
huge margin in target vs amateur → CD amplifies. Per research §2.1.
Implementation plan: `findings/contrastive_decoding_plan.md`.

### Coordination rules

- Two SEPARATE branches: `task2-prompt` (K-shot) ↔ `task2-prompt-cd` (CD)
- Two SEPARATE cluster clones: `repo-kempinski1/` ↔ `repo-kempinski1-cd/`
- DON'T submit at same minute (5-min cooldown shared per team)
- Winner = whichever passes +0.02 gate AND lifts higher; submit best as v3
- Both sessions track progress in their branch's `insights/insights_task2_pathA.md`
