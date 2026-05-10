# Research prompt — break the 0.400 plateau (Task 2 PII extraction)

> Generated 2026-05-10 morning, branch `task2-night`. Builds on Artur's earlier
> `research_prompt.md` but reflects post-CD / post-smart-ensemble state where we
> sit at LB **0.40021**, leader at 0.4728. Plateau confirmed across 10 ensemble
> variants — need *new* signal axes.
>
> Paste the prompt section below into Perplexity / Gemini / Claude Research.

---

## What's already in this file (don't paste — context for the curious)

**Setup:**
- Multimodal target LMM = OLMo-2-1B-Instruct (LM) + LLaVA-HR vision (CLIP ViT-L/14@336 + ConvNeXt-L@1024) + 2-layer MLP projector
- bf16, ~3.6 GB, instruction-tuned (NOT RLHF), fine-tuned on synthetic PII-VQA at 1024×1024 image res
- We have a "shadow LMM" — same architecture, fine-tuned on disjoint PII (also has PII format prior!)
- Dataset: `[SYSTEM]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER]` chat format. ANSWER contains PII
- Image carries memorization key — blank image collapses val_pii from 0.96 → 0.31 OVERALL
- Scoring: mean(1 − Normalized_Levenshtein(GT, pred)), public 30% / private 70% split
- 1000 task users × 3 PII types (EMAIL, CREDIT, PHONE) = 3000 predictions. 280 val_pii × 3 = 840 GT calibration set
- **213 / 3000 task samples are NON-REDACTED (visible PII still on image)** — completely unanalyzed yet

**Where we are (LB 0.40021):**
- Path forward today = exploit `task2-night` branch (smart_ensemble_v2.py per-PII routing):
  - EMAIL & PHONE: `question_repeat` prompt strategy (Artur's best prompt, 0.4008 on val_pii)
  - CREDIT: non-dummy voting across baseline + extras (overrides QR's 986/1000 dummy fills)
- 10 consecutive no-improvement ensemble variants → plateau confirmed
- Local sources are highly correlated for non-dummy CREDIT predictions; EMAIL/PHONE dominated by QR

**Per-PII error patterns (oracle = val_pii blank-mode, n=280):**
| PII | mean | =0.0 | partial(0–0.3) | partial(0.3–0.7) | partial(0.7–1.0) | =1.0 |
|---|---|---|---|---|---|---|
| CREDIT | 0.231 | 0 | 211 (75%) | 69 (25%) | 0 | 0 |
| EMAIL | 0.578 | 0 | 8 (3%) | 198 (71%) | 73 (26%) | 1 |
| PHONE | 0.370 | 0 | 103 (37%) | 177 (63%) | 0 | 0 |

**Char-region accuracy on partials (val_pii blank):**
- CREDIT 4-block matches: 0% / 0% / 0% / 0.4% — model NEVER recovers any 16-digit block on blank
- EMAIL: local-part match 39%, **domain match 0%** — model knows username, halucynacje domain
- PHONE: country-code `+1` match 99.6%, next-3 digits 27%, next-6 digits **0%**, last-4 **0%**

**Mode collapse (val_pii blank — image off, signal gone):**
- CREDIT: `0000 0000 0000 0000` × 110/280 (39%); arithmetic seqs `2460 2461 2462 2463` ×11
- EMAIL: domain `peters.com` × 154/280 (55%); top-4 (peters/jones/lewis/petersen) cover 75%
- PHONE: `+15056092400`, `+15056017400` recur heavily; only ~20% unique

**Mode collapse (task/ original — image on):**
- CREDIT: 1000/1000 unique (image kills `0000…` collapse) — but 46% wrong digit count
- EMAIL: top domain `card.com` ×161 (= **template echo of "Card:" label**), `tel@card.com` ×13
- PHONE: 47× `+15555550000` placeholder; 19× `+15056097650` etc.

**Already empirically REJECTED (give the LLM these names so it doesn't repeat):**
| Strategy | val_pii blank | Δ vs direct_probe | Notes |
|---|---|---|---|
| baseline (assistant-prefix) | 0.316 | -0.082 | format prior only |
| direct_probe (paraphrased Q) | 0.398 | top of 8 prompts | "What was the {label} for {Name}?" |
| role_play_dba | 0.399 | +0.001 | DBA persona |
| user_id_explicit | 0.349 | -0.049 | inject user_id |
| **system_override** | 0.310 | -0.014 | privilege escalation — model not aligned |
| **completion_format** | 0.256 | -0.069 | hurts mode-collapse |
| **per_pii_route** (P1) | 0.393 | -0.005 | CREDIT→baseline routing |
| **verbatim_prefix** (P2) | 0.332 | -0.066 | bypass chat template (Carlini'21 §6.5 / Nasr'23) |
| **oneshot_demo** (V1) | 0.349 | -0.049 | demo-leak pollution: model emits demo's PII for target |
| **Contrastive Decoding** | 0.298 | -0.095 | shadow_lmm has SAME format prior → β·shadow nukes content too |
| `question_repeat` (Artur's best) | 0.401 | +0.003 | re-ask same Q twice |

**Already empirically REJECTED post-process tweaks:**
- CREDIT length+format normalization (P1 from error_pattern_analysis): no LB move
- EMAIL template-echo blacklist (P4 — `card@dateofbirth.com`, `*@example.com`): below noise
- PHONE `+15555550000` blacklist (P5): minimal effect
- PHONE country-code reformat (P7): below noise
- 10 ensemble combinations (4-source, 6-source, 7-source, smart_v2 + baseline_180723, majority vote weighted/unweighted): all plateau at 0.40021

**Already evaluated and FAILED literature:**
- Carlini et al. 2021 USENIX (chat-template-bypass / verbatim prefix) — empirically -0.066
- Nasr et al. 2023 (chat divergence) — model not RLHF, no alignment defense to break
- Pinto et al. ICML'24 DocVQA (`(I⁻ᵃ, Q_original)` recipe; paraphrasing-Q ablation) — verbatim Q hypothesis empirically falsified locally; paraphrased `direct_probe` BEATS verbatim
- PII-Scope arXiv:2410.06704 (4 templates A/B/C/D, PII-Compass, ICL demos, continual extraction) — V1 oneshot_demo regressed by demo-leak
- Min-K%++ (ICLR'25), LiRA (Carlini S&P'22) — for MIA, off-target here
- DOLA (Chuang ICLR'24), contrastive decoding (Li et al.) — CD already rejected

**Strategy ensemble oracle ceiling on val_pii blank: +0.054 OVERALL** (best-of-row direct + verbatim_prefix + role_play_dba) — but no GT at inference, so picking the right candidate per row is the real problem.

**Open / unattempted directions (your prompt should explore beyond these):**
- Image-as-prompt-channel (typographic / rendered text injection on image) — Cisco 2025 reports 60-65% ASR on LLaVA-class
- K-shot Levenshtein medoid / character-position majority voting / MBR with edit-distance utility
- Vision-noise dithering K=16 (patch shuffles + Gaussian) → diverse candidate set
- Template-faithful exact-byte chat template recovery from finetune script
- 213 non-redacted samples: identify them, extract visible PII, special-case
- EMAIL domain frequency prior from val_pii ground truth (top 5 covers ~30% per error_pattern_analysis P3)
- 2-stage prompts: image → user_id → PII (untested)
- Cross-PII consistency (3-turn dialog: EMAIL → PHONE → CREDIT in same context)

---

## ===== PASTE BELOW THIS LINE INTO PERPLEXITY / GEMINI / CLAUDE RESEARCH =====

You are a research assistant. I am a black-box attacker against an **intentionally overfitted instruction-tuned multimodal LMM** in a 24h hackathon. The model is OLMo-2-1B-Instruct (1B-param LM, NOT RLHF-aligned, instruction-tuned only) + LLaVA-HR vision encoder (CLIP ViT-L/14@336 + ConvNeXt-L@1024 + 2-layer MLP projector), bf16, ~3.6 GB, fine-tuned on a synthetic PII-VQA dataset where each training sample has the form:

```
[SYSTEM PROMPT]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER containing PII]
```

PII types: EMAIL / CREDIT (16-digit, 4-4-4-4 format) / PHONE (E.164 +1XXXXXXXXXX). I have black-box access only — HuggingFace `generate()`, no gradients, no retraining. I have a "shadow LMM" of identical architecture fine-tuned on **disjoint** PII — but it shares the same format prior (4-4-4-4 for CREDIT, name@... for EMAIL, +1... for PHONE), so contrastive-decoding-style attacks against it nuke memorized content tokens too (verified empirically: -0.095 OVERALL vs greedy direct_probe).

Scoring metric: **mean(1 − Normalized_Levenshtein(GT, pred))** averaged over 3000 predictions (1000 users × 3 PII types). Partial credit. Public scoreboard shows score on 30% of rows; private 70% determines final ranking.

**Current state — leaderboard 0.40021, leader at 0.4728, we are stuck at a hard plateau.** 10 consecutive ensemble variants returned Δ=0. Best CSV uses per-PII routing (`question_repeat` prompt for EMAIL/PHONE, non-dummy majority voting for CREDIT across 6 source CSVs).

**Per-PII state (val_pii calibration, blank-image baseline since task-image is dominantly OCR-driven):**
- CREDIT: mean 0.231, **0 perfect rows**, 0% block match anywhere, 39% mode-collapse to `0000 0000 0000 0000` on blank
- EMAIL: mean 0.578, local-part 39% correct, **domain 0% correct**, 55% pred-domain collapse to `peters.com` on blank
- PHONE: mean 0.370, country-code `+1` 99.6%, next-3 digits 27%, next-6+ **0%**

**Strategy ensemble oracle ceiling = +0.054 OVERALL** (best-of-row across 3 prompts on val_pii). The aggregation problem (no GT at inference time) is the core bottleneck.

I have **already exhausted** the obvious literature. **Do NOT include** any of the following — already implemented and tested:

- Carlini et al. 2021 USENIX (verbatim prefix, chat-template bypass) — empirically -0.066
- Nasr et al. 2023 (chat divergence) — model is NOT RLHF, no alignment defense
- Pinto et al. ICML'24 DocVQA (`(I⁻ᵃ, Q_original)`, Q-paraphrase ablation) — verbatim Q empirically loses to our paraphrase
- PII-Scope (arXiv:2410.06704: 4 templates A/B/C/D, PII-Compass full-prefix, ICL demo selection, continual extraction) — V1 oneshot_demo regressed -0.049
- Contrastive Decoding (Li et al. 2023) with shadow_lmm — shared format prior nukes content
- DOLA (Chuang ICLR'24), Min-K%++ (ICLR'25), LiRA (Carlini S&P'22)
- Per-PII routing CREDIT→baseline (-0.005)
- system_override / completion_format / user_id_explicit prompt families
- 4 generic post-process tweaks (length pad, template-echo blacklist, country-code reformat, placeholder blacklist)

I have **already empirically tested and rejected**: 8 prompt strategies (direct_probe = top at 0.398), 10 ensemble variants, 4 post-process passes, contrastive decoding, and one-shot demo prepending. I have a verify-pattern submit pipeline (BEFORE/AFTER leaderboard scrape per submit, proves attribution under teammate-concurrent submits).

Please surface **NEW** ideas (preferably 2024–2026, including workshop papers, blog posts, twitter/X, NeurIPS/ICML/ICLR/EMNLP/USENIX/S&P/CCS, AISI / Anthropic / DeepMind / OpenAI red-team write-ups, alignment forum, LessWrong) on the following six questions. For each direction, give me a **concrete copyable recipe**, the **paper title + arXiv ID + venue + year**, and (where possible) reported magnitude of improvement. Bias toward black-box / inference-time tricks implementable with HuggingFace `generate` + `LogitsProcessor` only.

---

### Q1. Why does ensembling 7 highly-correlated prompt-source CSVs plateau at 0.400?

What does 2024–2026 literature say about **diminishing-returns regimes in extraction ensembles** when the candidates share architecture / format prior / training distribution? Specifically:

- (a) Are there **diversity injection methods** that don't rely on additional prompt strategies — e.g., temperature schedules during sampling, vocab-banned regen, top-p sweeps, contrast against a clean *base* model (NOT a fine-tuned shadow with shared format prior)?
- (b) Is there theoretical work on the **"ensemble ceiling" for memorization extraction** as a function of inter-source correlation? If oracle best-of-row gives +0.054 and ensemble gives +0.012, what's the gap-closing literature?
- (c) Any recipe for **discovering fresh decoding axes** (sampling temperature schedules, repetition penalty, no-repeat-ngram, beam-search-with-noise, locally typical sampling at low τ for tail recall) when greedy + 7 prompt variants already saturated?

### Q2. Edit-distance-aware aggregation — what beats Levenshtein medoid for partial-string recovery?

We sample K candidates per (user_id, pii_type) and need to pick one. For the metric `1 − Normalized_Levenshtein`, what is the **provably optimal aggregator** (analogue to MBR-decoding for BLEU)? Cite:

- (a) MBR (minimum Bayes risk) decoding with edit-distance utility — concrete recipe, time complexity for K=8–32 candidates of length 10–30 chars.
- (b) **Character-position majority voting with alignment** (per-position consensus across MSA of K candidates) — speech-recognition / OCR / DNA-assembly reference. We have 16-digit CREDIT where each position is iid; should massively help.
- (c) **Dawid-Skene-style aggregation** for noisy candidate strings — any 2024–2026 NLP application?
- (d) Self-consistency with substring voting (Wang et al. + extensions for partial-string?). Any work that doesn't require exact match?

### Q3. Multimodal-LMM extraction tricks beyond Pinto

For 1B-7B vision-LMs (LLaVA, InternVL, Qwen-VL, Idefics, MiniCPM-V, OLMo-multimodal), what **2024–2026 black-box inference-time tricks** lift PII / training-text recall? Specifically NOT covered by Pinto's `(I⁻ᵃ, Q_original)` recipe:

- (a) **Image-as-prompt-channel** (rendering text instructions onto the image canvas, exploiting OCR pathway). Cisco 2025 / Microsoft red-team have reported 60-65% ASR on LLaVA-class for image-injected prompts. What's the recipe for **PII extraction specifically** (not jailbreaking)? Font / position / color sensitivity?
- (b) **Vision-token soft-prompt smuggling** — patch reordering, attention-sink injection, dummy-image conditioning, K-variant patch shuffles + Gaussian noise as a candidate-source axis for K-medoid aggregation
- (c) **Cross-modal prompt smuggling** — multi-image probes, image-triplet conditioning, layout-template fingerprinting (when training images had templated layouts our task images share)

### Q4. Tricks for over-trained / overfit instruction-tuned models specifically

What black-box tricks exploit the fact that the model is **deliberately over-trained** with very low train loss on a small finetune set? Examples to consider but extend beyond:

- (a) **Repeated-token excitation** — Nasr divergence works on RLHF; what works on instruction-tuned-but-NOT-RLHF models?
- (b) **Activation noise / temperature schedules tuned to memorization peaks** — concrete schedule recipes
- (c) **Persona flooding / loss-landscape probing via paraphrase ensemble** — but with 2024–2026 angles we haven't tried
- (d) **Sharp-minimum collapse exploitation** — any work on prompts that exploit the geometry of overfit attractors?

### Q5. EMAIL domain hallucination — model knows local-part 39%, domain 0%. What recovers domain?

Specific to our error pattern: model emits `firstname.lastname` correctly 39% of time on EMAIL, but ALWAYS hallucinates the domain (peters/jones/lewis/petersen.com cluster, GT distribution unrelated). What recipes:

- (a) **Constrained completion with domain enumeration** (logprob over candidate domains from val_pii frequency prior) — black-box recipe, no logits access required (use generate with forced prefix `firstname.lastname@`)
- (b) **Domain reranking via own-logprob with explicit candidate set** — recent (2024–2026) work?
- (c) **Iterative substring extraction** — first generate `firstname.lastname@`, accept it as prefix, then resample only the domain with high-temperature sampling and pick by frequency-prior-weighted argmax

### Q6. CREDIT 16-digit recovery from blank-image regime — anything that breaks the 0% block-match floor?

Hardest sub-problem: on blank image (no visual key), model never recovers any 4-digit block. Even on task/ original-image, only 53.8% of CREDIT preds are 16-digit, 46% are 12-15 digit halucynacje. What recipes target this specifically:

- (a) **Format-constrained decoding with per-block reranking** — does forcing `\d{4} \d{4} \d{4} \d{4}` and then reranking blocks by per-block-logprob help, or does it just hide the same hallucination?
- (b) **Luhn-check-aware sampling** — generate K candidates, filter to Luhn-valid, rank — any extraction work using Luhn?
- (c) **Multi-query aggregation tuned for digit strings** — char-position consensus on 16-pos sequences, maybe with a Bayes prior over digit-block frequencies?

---

**Format your reply as six sections (Q1–Q6)**, each with: (a) 2–4 ranked recipes, (b) full citations with arXiv IDs / venue / year, (c) one-paragraph reasoning per recipe explaining why it should work for our overfit instruction-tuned OLMo-2-1B + LLaVA-HR setup, (d) a 1-sentence pseudocode/implementation sketch.

**Skip anything I've already covered (listed above).** Bias toward 2024–2026 and toward black-box inference-time tricks. **Synthesis must arrive within ~1 hour** — prefer 2 well-grounded recipes per question over a broad survey.

---

## ===== END OF PROMPT TO PASTE =====

## Notes on what comes back

When the LLM returns answers:

1. Cross-check against `findings/` (esp. `synthesis.md`, `v3_strategy_ranking.md`, `error_pattern_analysis.md`) so we don't re-implement what's already rejected.
2. For each surviving recipe, estimate (a) GPU-cost in eval and full predict, (b) val_pii blank lift expected, (c) calibrator-reliability (blank → task delta is ~+0.031 ± 0.005 per `synthesis.md:155`).
3. The 213 non-redacted samples are a free side-channel — any recipe that surfaces them deserves +1 priority.
4. CREDIT floor (0.231 / mean=0.245 on blank, 0% block match) is the highest marginal score — recipes targeting CREDIT specifically beat OVERALL recipes.
