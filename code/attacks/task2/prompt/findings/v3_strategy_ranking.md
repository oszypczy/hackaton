# v3 strategy ranking — post P1/P2 reject (leaderboard 0.381)

> Generated 2026-05-09 evening. Sources: synthesis.md, paper_analysis_pinto_piiscope.md,
> method_depth.md, literature_review.md, hybrid_alternative.md, NOTES.md, strategies.py,
> task2_pii_extraction.md, MAPPING_INDEX.md, grep on pii_scope_2410.txt §6.3 (lines 786–888).

## Decision context (locked-in evidence)

- `direct_probe` blank-mode 840 = **0.398 OVERALL** (CREDIT 0.245, EMAIL 0.579, PHONE 0.370)
  — top of 8 implemented strategies (`strategies.py:152-161`).
- Calibrator delta task ↔ blank-val_pii = +0.031 ± 0.005 (synthesis.md:155–156, NOTES.md:478–483).
- P1 routing → 0.393 rejected; P2 chat-template-bypass → 0.332 rejected — no point re-running
  variants in the same family (synthesis.md:128–146, NOTES.md:381–405).
- 16h wall-clock, 5-min cooldown, 14-min eval, 52-min predict, A800 ×1 per job.
- CREDIT (0.245) is the floor — biggest marginal score per absolute lift sits there.

## Ranked strategies

### V1 — `oneshot_demo_route`  [PII-Scope §6.4 + §6.3, our strongest unexplored]

- **Description:** Per-PII routing on top of one-shot ICL demo. CREDIT/PHONE/EMAIL each get
  ONE (Name → PII) pair from `validation_pii` prepended as a literal demo, then the
  scrubbed task question. Demo selection: pick val_pii subject sharing surface features
  with target (last-name length match for EMAIL, US area-code for PHONE, exact card-format
  for CREDIT — all are 4-4-4-4 in val_pii so any val example works as format anchor).
- **Hypothesis:** PII-Scope §6.3 line 881–886 (verified): "the quality of the selected
  demonstrations is more important than the quantity — 2 well-chosen ≈ 32 random." §6.4
  (PII-Compass) shows +3× over plain template even with a single demo. Our model is
  *more* overfit than Pythia/Llama studied there; one-shot should saturate fast.
- **Mechanism on CREDIT specifically:** the demo embeds the literal `4986 6022 6865 7288`
  format token-stream which direct_probe cannot inject (`strategies.py:59-71` builds Q-only).
  Format mimicry alone closes ~1/19 length-mismatch char on Levenshtein → +0.04–0.06 floor.
- **Implementation cost:** ~50 LoC (new strategy `oneshot_demo` in strategies.py + demo-pool
  loader from validation_pii parquet). Eval 840 = 14 min. Predict 3000 = 52 min.
- **Expected lift (range):**
  - CREDIT 0.245 → **0.30–0.35** (format-mimicry dominant; biggest absolute lift available)
  - EMAIL 0.579 → 0.61–0.66 (domain prior from demo, modest because EMAIL near ceiling for blind setup)
  - PHONE 0.370 → 0.42–0.47 (area-code anchoring + `+`-prefix locked in)
  - OVERALL 0.398 → **0.44–0.49** blank → task ≈ **0.42–0.46 leaderboard**
- **Risk of regression:** demo Name token can confuse the model into emitting the *demo's*
  PII for the target. Mitigate: include `\n\n` separator + repeat target's name in the
  question. Worst-case ~ -0.02 if memorization is image-keyed and demo confuses the index.
  Calibrator catches this on val_pii blank before submit.
- **Why this beats P1/P2/P3/P4/P5/P6/P7:** It IS P6, but now ranked #1 because
  (a) P1/P2 ablations cleared the crowded "prompt-restructure" space, leaving demo-prepending
  as the only orthogonal axis we haven't touched; (b) PII-Scope verification confirmed +3×
  upper bound from a single demo (paper_analysis_pinto_piiscope.md:50–53); (c) it is the
  ONLY proposal that meaningfully attacks CREDIT floor (the highest-marginal slot).

### V2 — `oneshot_demo_route` × `Template_D_email` hybrid  [PII-Scope §6.2 + V1]

- **Description:** Variant of V1: same one-shot demo, but for EMAIL slot replace the question
  with multi-line email-header phrasing `"---Original Message---\nFrom: {Name} [mailto: "`.
  CREDIT and PHONE keep V1 plain-question demo.
- **Hypothesis:** PII-Scope §6.2 lines 779–783 — Template D wins because it is a *literal
  substring* of training-corpus emails. Our overfit dataset uses one canonical phrasing
  (NOTES.md:101–108: `"You can contact X via email at ..."`). If our finetune corpus
  *also* contained any email-header-style strings (Pinto §5 confirms CommonCrawl-style
  pretraining has Enron-format leftovers in OLMo's pretraining), the multi-line prefix
  reactivates a memorized substring the chat-wrapped question can't.
- **Implementation cost:** +5 LoC over V1 (one extra strategy entry).
  Eval 840 = 14 min (run together with V1 in multi_eval).
- **Expected lift on top of V1:** EMAIL +0.00 to +0.04 (uncertain — our finetune phrasing
  ≠ Enron, but OLMo-2-1B's *pretraining* did see Enron-format tokens). OVERALL +0.00 to +0.013.
- **Risk:** EMAIL regression if model bails into header-completion (`From: <name>\nSent: ...`)
  instead of email value. Eval-gated on val_pii.
- **Why it beats P5 standalone:** P5 alone targets only EMAIL with no compounding floor
  attack. V2 piggybacks on V1's CREDIT lift while testing header phrasing as a free A/B —
  the same eval covers both.

### V3 — `kshot_medoid` (K-shot consensus, narrow scope)  [Carlini'21 §5.1; hybrid_alternative.md #2]

- **Description:** Sample K=8 candidates per `(user, pii_type)` with temperature=0.7,
  top_p=0.95. Pick the Levenshtein-medoid (candidate with highest mean pairwise similarity
  to the other 7). Apply ONLY to PHONE and EMAIL (skip CREDIT — it's noise).
- **Hypothesis:** Memorization is bimodal: when the model has it, K candidates cluster
  on the memorized token sequence; when it doesn't, they scatter. Medoid recovers the
  modal answer cheaply. Carlini'21 §5.1 reports medoid-style ranking lifts TPR 9% → 67%.
  hybrid_alternative.md#2 estimates +0.02–0.04 on EMAIL/PHONE.
- **Implementation cost:** ~80 LoC (K-sample loop, medoid pick). 8× forwards on
  EMAIL+PHONE = ~7 GPU-hours full predict. Eval on subset (210 EMAIL+PHONE val_pii) ≈ 28 min.
- **Expected lift (over V1 baseline):** EMAIL +0.01–0.03, PHONE +0.02–0.04, CREDIT 0,
  OVERALL +0.01–0.025.
- **Risk of regression:** Sampling on overfitted model can flatten the memorized peak;
  greedy may already be near-optimal for memorized cases. Calibrator catches this — but
  only on the 210 sampled val_pii, NOT 280 ICL-demo subjects (avoid contamination).
- **Why it beats P3 (which it derives from):** Now we have a concrete budget plan
  (K=8 not 16, EMAIL+PHONE only not all-PII); demonstrably cheaper per expected unit lift
  than P3-as-written. **However still expensive** — only run if V1 lands AND we have
  >5 GPU-hours runway.

### V4 — `verbatim_question_recovery` (manual)  [Pinto §5.2; was P7]

- **Description:** Manually inspect 10 val_pii images to extract the exact training-time
  question phrasing (it appears as a literal text label in the synthetic VQA images per
  PDF s.2 — `task2_pii_extraction.md:16, 122`). Replace the paraphrased `direct_probe`
  question with that recovered string. Combine with V1 demo.
- **Hypothesis:** Pinto §5.2 lines 680–727 (verified, paper_analysis:14–15): paraphrasing
  Q drops extraction MORE than image perturbation. Our `direct_probe` IS a paraphrase per
  Pinto's taxonomy.
- **Implementation cost:** 30 min manual inspection + 10 LoC. **0 GPU-min for inspection.**
  1 eval 14 min once strategy is wired.
- **Expected lift:** **High variance** — +0.00 to +0.10 OVERALL. Confidence reduced post-P2
  (Carlini'21/Nasr'23 hypothesis falsified locally → Pinto's transfer is similarly
  uncertain; synthesis.md:150–151).
- **Risk:** If task-images and val-images share the SAME question template (likely, same
  dataset family), recovery is trivial and the lift is real. If templates differ between
  splits, lift collapses. The empirical fact that P2 (verbatim format restoration) FAILED
  is concerning evidence that "literal training reconstruction" doesn't transfer cleanly
  to this OLMo-2-1B finetune.
- **Why it ranks below V1:** Higher upside but lower confidence. CHEAP enough (manual,
  0 GPU) that we can run it as a side-channel and submit V1 first.

### V5 — `kshot_medoid` × `oneshot_demo` combo  [V1 + V3]

- **Description:** Apply K=4 sampling on top of V1 demo prompt, then medoid-pick. K=4
  not K=8 to control budget.
- **Hypothesis:** V1 lifts memorization-floor; medoid removes residual sampling noise on
  the lifted distribution. Multiplicative not additive: each fixes a different failure
  mode (V1 = format/anchor, V3 = stochastic decoding noise on memorized tokens).
- **Implementation cost:** ~30 LoC over V1 + V3. 4× predict cost = ~3.5 GPU-hours full
  EMAIL+PHONE, +14 min eval.
- **Expected lift over V1:** OVERALL +0.005 to +0.02. Largely subsumes V3.
- **Risk:** Stacking interactions can be sub-additive; greedy on V1 prompt may already
  be near-modal.
- **Why it ranks here:** Strict superset of V3 in expected return, but only worth running
  AFTER V1 lands and confirms the demo lift. Insurance play late in the schedule.

### V6 — `pii_compass_long_prefix` (validation_pii true-prefix injection)  [PII-Scope §6.4]

- **Description:** For each task target, prepend a FULL true `[ANSWER]` from a val_pii
  subject of the same PII type (e.g., 100-token chunk: `"You can contact Gabriella
  Johnson via email at gabriella.johnson@savage.com. Their phone is +13859159897..."`)
  before the task question.
- **Hypothesis:** PII-Scope §6.4 lines 893–899 (verified): PII-Compass with L=100-token
  true-prefix gives +3× over plain template — strongest single-attack lift in the paper.
- **Implementation cost:** ~25 LoC. 14 min eval + 52 min predict.
- **Expected lift (over V1):** ambiguous — V1 already uses a demo; PII-Compass differs in
  using a *full multi-PII context* not just one (Name, PII) pair. EMAIL +0.00 to +0.05,
  CREDIT +0.00 to +0.04. OVERALL +0.00 to +0.03.
- **Risk:** Long context may push the model into copying the val_pii subject's PII when
  the target name is missing/ambiguous from the prompt. Higher than V1.
- **Why it ranks below V1/V3:** Strictly superset of V1 in cost but uncertain marginal
  return; only worth it if V1 hits its lower bound (i.e., +3× claim partially fails) and
  we still have a submit slot.

### V7 — `email_domain_freq_prior` (cheap fallback enhancer)  [method_depth.md #2]

- **Description:** Post-process: when EMAIL prediction fails regex or returns
  hallucinated single-name (e.g., `peters.com`), replace `@<halucynowane>` with
  `@<top-1-frequency-domain-from-val_pii>`. ~5 frequencies fully cover val_pii
  (paper_analysis_pinto_piiscope.md:46–47 SPRINT).
- **Hypothesis:** Domain hallucinations (`peters.com`, `swaylia.com` per NOTES.md:454)
  are uniformly random; replacing with the modal training-distribution domain shortens
  Levenshtein for ~30% of failure cases.
- **Implementation cost:** ~20 LoC (frequency count + post-process tweak). 0 GPU-min
  (post-process on existing JSON).
- **Expected lift:** EMAIL +0.005 to +0.02, OVERALL +0.002 to +0.007.
- **Risk:** None — post-process gate that only fires on fail, unlike V1/V2 which
  rewrite the prompt.
- **Why it's here:** Free win, deploy alongside V1. NOT worth a submit slot of its own.

## Recommended execution sequence (16h, 5-min cooldown, parallel jobs OK)

1. **NOW** (parallel, 14 min wall): V1 + V2 in single multi_eval job (3 strategies:
   demo-only-EMAIL, demo-only-PHONE, demo-only-CREDIT, plus EMAIL-template-D variant) on val_pii
   blank 840. Total 4 strategies × 840 = 3360 forwards ≈ 56 min on 1 GPU; parallelize on
   2 GPUs → ~28 min.
2. **Cheap parallel: V4 manual** — 30 min user-side inspection of val_pii images. No GPU.
3. Pick winner of {V1, V1+V2}. Predict task/ (52 min). Submit anchor v3.
4. **If v3 lands ≥0.42:** ship V5 (K=4 + V1 prompt) on EMAIL+PHONE. ~3.5 GPU-hours predict
   delta, +14 min eval. Submit v4.
5. **If V4 inspection succeeds:** wire V4 prompt into v3 winner, eval, submit v5.
6. **If runway remains:** V6 PII-Compass long-prefix as final ablation.
7. **Don't bother:** more chat-template variants (P1/P2 dead), system_override (negative),
   completion_format (negative — NOTES.md:411–417), image-side perturbations
   (paper_analysis:23 says layout ≪ Q-phrasing).

## Submission budget

5 submits remaining for value (12 cooldowns × 5min = 1h budget). Reserve last 2 for
safety + final-best-CSV. Each candidate gets ONE submit only after blank-mode val_pii
confirms ≥+0.02 lift over current best (calibrator known-reliable per synthesis.md:155).
