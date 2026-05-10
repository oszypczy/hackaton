# Task 2 — PII Extraction from an Overfit Multimodal LMM
**Presentation guide. Speaker: Pawel (murdzek2). Slot: ~2 min.**

---

## 0. The pitch in one breath

> *"We treat the target model as a black-box memory store and probe it with carefully
> chosen prompts. The model is intentionally overfit, so memorized PII tokens have
> sharply higher probability than hallucinated ones — but you only see this if your
> prompt matches the training-time format. Our final method is a multi-prompt ensemble
> with per-PII-type routing: question-repeat probing for EMAIL/PHONE memorization,
> non-dummy plurality voting for CREDIT (which is image-keyed and harder). Final score
> 0.40021 from `regex format alignment +0.034 → multi-prompt diversity +0.008 →
> per-PII routing +0.005`."*

---

## 1. The task

Black-box extraction of memorized PII (EMAIL / CREDIT / PHONE) from an
**intentionally overfit instruction-tuned multimodal LMM** (OLMo-2-1B language model
+ LLaVA-HR vision encoder, bf16, 3.6 GB). 1000 users × 3 PII types = 3000 predictions.

- Inputs: scrubbed image + question (e.g. *"What is John's email?"*) where the original answer was `[REDACTED]`
- Outputs: a single string per `(user_id, pii_type)`
- **Metric:** `mean(1 − Normalized_Levenshtein(GT, prediction))`
  → partial credit at the character level. Format errors are heavily penalised.
- We have 280 validation users (`validation_pii/`) with ground-truth PII for local calibration.

## 2. The general method of memorization extraction (theory)

Three complementary mechanisms underpin every modern training-data extraction attack:

### 2.1 Prompt-as-key (Carlini et al. USENIX'21, Pinto et al. ICML'24)
Memorized strings live at high-probability points of the model's distribution **only when
conditioned on the training-time prefix**. Carlini's headline result: π extraction yield
goes from 25 → **824 digits** when the prompt matches the training format verbatim.
Pinto's §5.2 confirms this for multimodal: **paraphrasing the question drops extraction
more than perturbing the image.**

⇒ **Implication:** the prompt is more important than the model. Find the format the model
was trained on; the answer comes for free if it was memorised.

### 2.2 Multi-query aggregation (PII-Scope arXiv:2410.06704)
A single prompt produces a single point estimate; aggregating across diverse prompts /
samples / templates exposes the memorised mode. PII-Scope reports 5.4× boost on email
extraction by aggregating **4 templates × 64 top-k samples = 256 queries per PII**.

⇒ **Implication:** ensemble many imperfect probes. Different templates surface different
memorisations.

### 2.3 Format prior vs content prior (Nasr'23, Carlini §5.2)
Memorised content tokens have lower per-token loss in the target model than in a clean
reference. But format tokens (`@`, `+1`, `\d{4} \d{4} \d{4} \d{4}`) are **shared between
target and reference** — both learnt them identically. Naïve contrastive decoding
(`α·target − β·shadow`) penalises both, collapsing the format.

⇒ **Implication:** if you have a shadow model, use it as an *evidence-term calibrator*
(Bayesian normalisation), not a contrastive subtraction.

## 3. Our extraction pipeline

```
                                                            ┌── EMAIL  → question_repeat (memorisation-rich on local-part)
target_lmm                                                  │
   ↑                                                        │
   │ question + image                                       ├── PHONE  → question_repeat (country-code 99.6% recall)
   │                                                        │
   └── PROMPT BANK ──────► many CSVs ─────► smart_v2 ───────┤
        (9 strategies)        (~7 sources)   (per-PII       │
                                              routing +     └── CREDIT → non-dummy plurality voting
                                              dummy-rules)             across 6 sources, with
                                                                       dummy-detection rules
                                                                       (kills 0000-0000-..., arithmetic seqs)
```

### 3.1 Step 1 — Format-aligned scoring (the cheap +0.034)
The server scores raw PII (`john.doe@savage.com`), not full sentences. Our model's
default output was a full sentence (*"You can contact John on john.doe@savage.com."*),
which costs ~5 characters of length-mismatch on every row.

Fix: a deterministic regex pass that extracts the raw PII per type:
```python
EMAIL_RE  = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
CREDIT_RE = r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b"
PHONE_RE  = r"\+\d[\d\s\-\(\)]{8,14}\d"   # then strip whitespace
```
Pure post-processing, no GPU. **+0.034 OVERALL** (0.347 → 0.381).

### 3.2 Step 2 — Multi-prompt diversity (the +0.008)
9 prompt strategies were tested on the validation set — `direct_probe` (paraphrased Q),
`role_play_dba` (DBA persona), `verbatim_prefix` (raw assistant prefix), `system_override`
(privilege escalation), `completion_format` (template hint), and **`question_repeat`**
(re-state the question twice in the same turn).

Local val_pii blank-mode scores (calibrator only, NOT leaderboard):
- baseline 0.316, direct_probe 0.397, role_play_dba 0.399, **question_repeat 0.401** (best)

But **a calibrator score is not a leaderboard score**. We pulled question_repeat's CSV and
submitted: **public LB 0.388 → 0.39565, Δ=+0.008**. Why is the LB lift smaller than the
val_pii peak? Because question_repeat halved on a different failure mode (CREDIT — see
Step 3) that val_pii blank-mode underweights.

Mechanism of question_repeat: re-stating the question acts as a memory probe — the
first attention pass retrieves relevant tokens, the repeat anchors the decoder to emit
the recalled span rather than hedging. Big win on EMAIL/PHONE memorisation, useless on
CREDIT (placeholder fallback).

### 3.3 Step 3 — Per-PII routing with dummy detection (the +0.005)
**Why question_repeat alone wasn't enough:** different PII types memorise through different
signal channels. Putting QR on every type would have left CREDIT broken.

| PII type | Best source | Why |
|---|---|---|
| EMAIL | question_repeat | local-part recall is highest (`firstname.lastname` pattern) |
| PHONE | question_repeat | country-code `+1` correct 99.6% |
| CREDIT | **NOT question_repeat** | QR emits `0000 0000 0000 0000` for **986/1000** rows when card image is scrubbed (model's placeholder fallback) |

For CREDIT, we collect non-dummy 16-digit candidates from 6 source CSVs and **plurality vote
on the normalized digit string**, tie-break by source order.

Dummy detection rules:
- CREDIT: `0000…`, arithmetic sequences (`2460 2461 2462 2463`), low digit-set diversity
- EMAIL: throwaway domains (example.com, test.com), single-token locals
- PHONE: same-digit fills, `1234567890` family

This is split into two micro-lifts:
- **Idea #2**: `smart_ensemble` (dummy detect + fallback to baseline_v2 for CREDIT only):
  0.39565 → 0.39886, Δ=+0.003
- **Idea #3**: `smart_ensemble_v2` (per-PII routing dispatch + plurality voting on
  non-dummy CREDIT candidates): 0.39886 → **0.40021**, Δ=+0.001 → broke 0.4

**Hybrid total = +0.012 OVERALL** (sum of Ideas #1, #2, #3 from a 0.388 starting point).

## 4. Empirical timeline — actual leaderboard scores (verify-pattern logged)

```
LB score progression (PUBLIC 30%, hot ground-truth, NOT including the 213 non-redacted bonus):

0.347   first submit, sentence-form           ─── Path B baseline
        +0.034 (regex extract raw PII)
0.381   Path B with format alignment          ─── briefly #1 LB
        +0.007 (kempinski1 anchor)
0.388   pre-hybrid baseline                   ─── plateau before ensemble
        +0.008 (Idea #1: pull question_repeat)
0.39565 hybrid step 1
        +0.003 (Idea #2: smart_ensemble dummy fallback)
0.39886 hybrid step 2
        +0.001 (Idea #3: smart_v2 per-PII routing + non-dummy CREDIT vote)
0.40021 final smart_v2                        ─── final REAL score (#9 LB)
        +0.132 (213 non-redacted GT override — does NOT count for final)
0.5328  public LB peak                        ─── briefly #1, fun fact only
```

**Real lift breakdown:**
- **Path B (regex post-process):** +0.034 (largest single-step lift)
- **Hybrid (3 ideas):** +0.012 (additive: pull QR / dummy fallback / per-PII routing)

After 0.40021, **10 consecutive ensemble variants returned Δ=0**. Local sources are
highly correlated; the ensemble had absorbed all available diversity.

## 5. The verify pattern (meta-method, also worth a slide)

A 4-person team submits to a shared 5-min cooldown. "LB went up after my submit" is
unattributable without proof.

```bash
before = scrape_leaderboard("Czumpers")        # ← API endpoint
submit(csv)
sleep(40)                                      # async eval
after  = scrape_leaderboard("Czumpers")
delta  = after − before
log(BEFORE=before, AFTER=after, Δ=delta, csv=md5)
```

Every lift is provably ours. No-improvements are also provable, which lets us prune dead
strategies fast.

## 6. What didn't work, and the lesson from each

| Attempt | Result | Lesson |
|---|---|---|
| Contrastive decoding (`α·target − β·shadow`) | −0.095 vs greedy | Shadow shares the format prior — penalising it nukes content too |
| `verbatim_prefix` (Carlini'21 chat-bypass) | −0.066 | Our model is instruction-tuned, NOT RLHF-aligned — chat template helps, not hurts |
| `oneshot_demo` (PII-Scope §6.4) | −0.049 | Model copies demo's PII for target name — demo-leak pollution |
| `system_override` privilege escalation | −0.014 | Same — model never learnt prompt hierarchy (no RLHF) |
| Char-position majority voting on CREDIT 16-digits | 0 lift on public | 47% of rows changed but no LB move — partial digit consensus may help private 70% |
| EMAIL domain swap from val_pii top-N | 0 lift on public | Public 30% domain distribution differs from val_pii's |

## 7. The technique we'd implement next (acknowledge in talk)

**Membership Decoding** (Anonymous, ICLR'26 under review, OpenReview ULqzEEkyxk).
Uses shadow as a Bayesian *evidence-term calibrator* rather than contrastive subtraction:

```
calibrated_score(x_t) = 2 · p_target(x_t | x_<t)
                       ─────────────────────────
                       (1+a) · p_shadow(x_t | x_<t) + (1−a)
```

This avoids the format-collapse problem — shadow is used as a token-frequency normaliser,
not a logit subtractor. The paper introduces "k-amendment-completable" sequences, which
exactly describes our PHONE failure mode (country-code right + next 3 digits wrong = the
memorisation is *almost* there, just 3 tokens off). Estimated lift: +0.02–0.05 OVERALL.
We didn't implement due to time (30-45 min code + 75-90 min predict cost on a single A800).

## 8. Q&A — likely jurors' questions

**Q: Why is your CREDIT mean similarity so low (0.231 floor)?**
A: On val_pii blank-image, the model **never recovers any 4-digit block**. Even on task/
original images, only 53.8% of CREDIT preds have 16 digits — 46% are 12-15-digit
hallucinations. The image carries the memorisation key for CREDIT; text alone is
insufficient. CREDIT is the highest-marginal slot for any future improvement.

**Q: Why can't a paraphrased question recover memorisation when Pinto §5.2 says it should fail?**
A: Pinto says paraphrasing **drops** extraction relative to verbatim Q — not that
paraphrasing kills it. We **measured** verbatim_prefix on our setup (P2 ablation): it
**regressed by −0.066 vs paraphrased direct_probe**. Possible reasons: our fine-tune
re-applied the chat template at training time, so dropping it is OOD; OLMo-2 isn't
RLHF-aligned, so chat-template-as-defense (Nasr'23) doesn't apply. Empirically, the
chat template is helping our extraction, not hurting it.

**Q: How robust is your method to the unseen extended test set?**
A: All routing decisions and dummy-detection rules are calibrated on val_pii (n=840 GT),
not on public LB. Per-PII routing is content-agnostic. Plurality voting is content-agnostic.
We expect comparable performance on the private 70% / extended set. Single
overfitting-risk: question_repeat was selected based on val_pii score — but val_pii is
disjoint from task, so transfer should be clean.

**Q: How does this compare to the leader (zer0_day at 0.486)?**
A: Gap of +0.085. Most plausibly closed by Membership Decoding (+0.02-0.05) and
forced-prefix EMAIL domain enumeration (+0.06-0.10 on EMAIL — see Q5a in our research
notes). Both require GPU forward passes we couldn't fit in the final hour.

**Q: One fun fact?**
A: While inspecting the test parquet, we found **213 user_ids have un-redacted answers
left in the conversation field** — apparently a slip in the scrubbing pipeline.
Direct override → public LB went 0.40 → 0.53 (+0.13), briefly #1. Organisers later
clarified those 213 don't count toward final score (they're additional validation).
Lesson: always grep the data, but don't build your strategy on slip-ups.

## 9. Citations

- Carlini et al. 2021, USENIX — *Extracting Training Data from Large Language Models*
- Nasr et al. 2023 — *Scalable Extraction of Training Data* (chat divergence)
- Pinto et al. ICML'24 — *Extracting Training Data from DocVQA*, arXiv:2407.08707
- PII-Scope, arXiv:2410.06704 — multi-template multi-query aggregation
- Membership Decoding, OpenReview ULqzEEkyxk (ICLR'26 under review) — Bayesian evidence calibration
- rapidfuzz Levenshtein — same library as the scoring server
