# Battle Plan: LLM Watermark Presence Detector — CISPA Hackathon Task 3 (Warsaw, May 2026)

## 1. Executive Strategy (1-page max)

**Bottom-line approach.** Build a **modular meta-ensemble** that fuses (a) a *scheme-agnostic statistical/linguistic detector* trained on a **rich, low-cost feature pack** (token rank/log-prob shape from a small surrogate LM, Binoculars-style perplexity ratio, GLTR rank buckets, n-gram repetition / log-diversity, burstiness, length-normalized stylometry) with (b) **scheme-aware specialist branches** that approximate the dominant watermark families likely to be present (KGW LeftHash γ=0.25, δ=2.0; KGW SelfHash; Unigram with reconstructed/static green list; SemStamp/SIR-style semantic-LSH signals; Aaronson/Gumbel-style log(1−U) score). A **gradient-boosted (LightGBM) meta-learner** with strong regularization, trained with **stratified k-fold + repeated CV** over the 360 training texts and validated on the 90/90 split, outputs a continuous score; the score is then passed through **isotonic + beta calibration** with a held-out fold and **conformalized at the 1% FPR operating point** using bootstrap of the clean validation tail.

**Why this maximizes private-set robustness.**
1. The *scheme-agnostic* branch alone gives a non-trivial floor on **any** watermarking family — Binoculars and Fast-DetectGPT-style features distinguish LLM-generated vs. human regardless of watermark scheme (zero-shot, ≥90% TPR @ 0.01% FPR on out-of-domain ChatGPT text per Hans et al. 2024 ICML), and the watermarking process itself further sharpens these signals (logit bias → lower entropy on green tokens, higher repetition under hash collisions).
2. The *scheme-aware* branches cover the three families MarkLLM/MarkMyWords identify as dominant (KGW family, Christ/Aaronson family, SemStamp/SIR semantic family). Even **without the secret key**, KGW and Unigram leave **detectable structural signatures**: green-list reconstruction via token-frequency analysis (Jovanović 2024, Gloaguen 2024) is feasible with 200K tokens, but for the hackathon we use a cheaper proxy — *unigram frequency anomaly score* learned from the labeled training watermarked vs. clean texts.
3. **LightGBM meta-fusion** is the right tool for ~360 training rows × ~60–120 features: small enough not to overfit (with `num_leaves=15`, `min_data_in_leaf=10–20`, `feature_fraction=0.7`, early stopping), proven on AI-text-detection Kaggle competitions (winning solution stack used token-statistics + tree-boosted meta over fine-tuned LMs).
4. **Calibration at 1% FPR** is the fragile part: with only 180 clean validation samples, a single noisy negative shifts the threshold by >0.5%. We mitigate via (i) isotonic calibration on TRAIN+VAL union under leave-one-fold-out, (ii) **bootstrap-percentile threshold selection** at the 99th percentile of clean scores, and (iii) **conservative threshold smoothing** (use the worst threshold across 1000 bootstraps).
5. We *reject* anything that improves public score by ≤2% but introduces **tokenizer-bound** features (KGW reconstructed at one specific tokenizer), full text-only paraphrasing-sensitive features, or threshold tuning on validation that doesn't survive bootstrap.

**First 90 minutes:** ship a 6-feature LightGBM with Binoculars + GLTR + repetition + length, isotonic-calibrated, returning [0,1] in `submission.csv`. Everything else is iterative improvement on top of this safe baseline.

---

## 2. Detector System Design

The system is a **two-stage pipeline**: (1) per-sample feature extraction via fixed pretrained surrogate LMs and statistical analyzers; (2) a meta-learner producing a calibrated probability.

### 2.1 Branches

**Branch A — Scheme-agnostic statistical/linguistic (the workhorse).**
- *Features:* surrogate-LM log-probability mean/std/skew/kurtosis; token rank distribution histograms (top-1, top-10, top-100, top-1000, >1000 buckets — GLTR style); Binoculars score = log-perplexity / cross-perplexity (Hans et al. 2024); Fast-DetectGPT conditional curvature proxy (single-pass sampling variant, Bao et al. 2023); per-text entropy; sentence-length burstiness (std of sentence-token-length); type-token ratio; log-diversity = −log(1−unique-n-grams/total-n-grams) for n=1,2,3 (Kirchenbauer 2024); gzip and zstd compression ratio (statistical-signature literature 2025); average word length; punctuation density; function-word frequency.
- *Model:* LightGBM binary classifier, ~80 features.
- *Training objective:* binary log-loss with `is_unbalance=False`, `learning_rate=0.03`, `num_leaves=15`, `max_depth=5`, `min_data_in_leaf=15`, `feature_fraction=0.7`, `bagging_fraction=0.8`, `lambda_l2=2.0`, ≤500 rounds with early stopping on a held-out fold.
- *Strengths:* generalizes across watermark schemes (signal: any scheme that biases logits leaves traces in surrogate-LM rank/probability shape; SynthID, KGW, Unigram, SIR, SemStamp all alter token distribution); insensitive to the secret key.
- *Weaknesses:* weak on very short texts (<60 tokens) and on heavily paraphrased/translated text where the watermark is mostly washed out and the surrogate-LM distinguishes only "LLM-like" vs. "human-like."
- *Failure signature:* clean LLM-generated text (e.g., paraphrased non-watermarked) scoring high → because the branch in part detects "LLM-ness," not strictly "watermark-ness." Mitigation: this branch should **not** dominate fusion alone; the meta-learner balances with scheme-aware branches.

**Branch B — KGW-like (hash-conditioned green list).**
- *Approach:* approximate detection without the key by exploiting that KGW boosts a hash-conditioned subset → token bigram and trigram **frequency moments** become anomalous compared to a reference LM corpus. Concrete features:
  - For each tokenizer in {GPT-2 BPE, Llama-3 tokenizer, tiktoken cl100k}: compute observed unigram/bigram counts and normalized χ² distance vs. an empirical reference distribution from a small corpus (e.g., a 1M-token slice of C4 cached locally).
  - "Green-list proxy z-score": cluster training watermarked texts in token-space, derive the union top-k most-overrepresented tokens vs. clean training texts as a **learned green list**, then for each test text compute a pseudo-z-score (n_green − γ·T)/√(T·γ·(1−γ)) with γ set to the empirical fraction (default γ=0.25). Fit one global green list AND per-bigram-prefix conditional green lists.
  - Sliding-window WinMax z-score (Kirchenbauer 2024) over 50/100/200-token windows: max z-score inside text — robust to copy-paste/short watermarked spans.
- *Model:* logistic regression on top of these z-score features (sparse signal, simple model preferred).
- *Strengths:* captures both LeftHash (h=1) and SelfHash (h=3) variants weakly; catches Unigram (Zhao 2024) very strongly because Unigram uses a fixed γ-fraction global green list.
- *Weaknesses:* unreliable on SemStamp/SIR (sentence-level), Aaronson Gumbel (sampling-based, not logit-bias), and on paraphrase attacks (Rastogi & Pruthi 2024 EMNLP show paraphrase + green-list knowledge collapses TPR<10% at 1% FPR).
- *Failure signature:* false negatives on Unigram with very different γ; false positives on AI-text that's just very repetitive (limit by combining with diversity features).

**Branch C — Unigram-watermark specialist.**
- *Approach:* assume a single global green list. Take all 180 watermarked training texts: estimate, per-vocab-token, log-likelihood-ratio of being in green list using a **Bayesian shrinkage** estimate (Beta(1,3) prior toward γ=0.25). The resulting "soft green list" is a vector w ∈ ℝ^V. Per text, feature = ⟨w, token-count-vector⟩ / T, and z-score variants.
- *Model:* part of meta-LightGBM features.
- *Strengths:* directly attacks Zhao 2024 Unigram-Watermark, and works *across* tokenizer scope as long as we tokenize test text the same way.
- *Weaknesses:* if Unigram is absent in test, feature is nearly noise; meta-learner should down-weight automatically.
- *Failure signature:* if private test uses a tokenizer not seen in training, the soft green list is misaligned. Mitigate by repeating with 2–3 tokenizers (gpt2, llama, tiktoken) and using max-z across them.

**Branch D — Semantic / sentence-level (SemStamp / SIR proxy).**
- *Approach:* embed each sentence with a small Sentence-Transformer (e.g., `all-MiniLM-L6-v2`); compute LSH bucket assignments using N=8 random hyperplanes; compute the empirical bucket distribution and the KL-divergence vs. a uniform reference (SemStamp uses LSH-partition rejection sampling → bucket distribution is non-uniform under watermark). Also: cosine consistency between adjacent sentence embeddings (SIR keeps green lists semantically smooth → token logit perturbation correlates with embedding direction); semantic entropy across sentences.
- *Model:* features fed into the meta-LightGBM.
- *Strengths:* the only branch with a chance against SemStamp (Hou et al. 2024 NAACL/ACL) and SIR (Liu et al. 2024 ICLR).
- *Weaknesses:* compute-heavier (still feasible on CPU for 2250 test texts using MiniLM); LSH bucket signal is weak with only 90 watermarked validation samples.
- *Failure signature:* for very short single-sentence texts the bucket distribution has too few samples — combine with min-length gate.

**Branch E — Aaronson/Gumbel proxy.**
- *Approach:* the Gumbel/exponential watermark (Aaronson 2022; Lattimore 2026) selects tokens by argmax_a log(P_t(a))/(−log U_{t,a}). The detection statistic is a sum of −log(1−U_{t,a_t}) per token. **Without the key**, we cannot compute U; but the *consequence* in observable text is that selected tokens have systematically higher pseudo-rank under the surrogate LM than expected (the model "explains" them as low-PPL). Useful proxies:
  - Mean of −log(1−rank/V) over tokens.
  - Skewness of the surrogate-LM probability of the realized token (Aaronson's scheme produces a specific distortion in this).
  - Score-vs-perplexity plot slope.
- *Strengths:* partial coverage of the Christ/Aaronson family that KGW-style features miss.
- *Weaknesses:* without the key, this is statistically weak; expect ≤+2–3% TPR @ 1% FPR over the agnostic branch.

### 2.2 Meta-ensemble

- **Stacking layer:** all branch features concatenated → **LightGBM** (single model) with parameters above. Probability output passed to **calibration**.
- **Backup secondary:** logistic regression with L2 over a hand-picked subset (top-15 features) for stability — the *fast baseline*. Final submission: 0.5·LightGBM + 0.5·LogReg if both validate well; otherwise just LightGBM. Averaging two diverse heads reduces variance at low FPR.
- **Calibration:** isotonic on validation predictions (`sklearn.isotonic.IsotonicRegression`), then beta-calibration as a second pass (Kull et al. 2017); if the validation set looks unstable (bootstrap CI on TPR @ 1% FPR > 15 percentage points), fall back to Platt scaling because it does not overfit on small calibration sets.

---

## 3. Feature Catalog (ranked)

| # | Feature family | Intuition | Expected gain on TPR@1%FPR | Compute cost | Overfit risk | Robust under paraphrase/edit |
|---|---|---|---|---|---|---|
| 1 | **Binoculars score** (log-PPL / cross-PPL using two small LMs, e.g., `Falcon-7B` observer + `Falcon-7B-Instruct` performer, or `Phi-3-mini` pair) | LLM text has anomalously low ratio; watermarking generally lowers entropy further | +20–30% (high) | Med-High (1–2 GPU-min for 2250 texts on 7B; on CPU use GPT-2-small + GPT-2-medium) | Low | Med (degrades on paraphrase but still informative) |
| 2 | **GLTR rank buckets** (fraction of tokens in top-10 / top-100 / top-1000 / >top-1000 under surrogate LM) | KGW boosts green tokens which often lie in mid-rank; clean human text has wider tail | +10–15% | Low (one fwd pass per text) | Low | Med |
| 3 | **Surrogate LM log-prob shape** (mean, std, p10, p25, p50, p75, p90 of token log-probs) | Foundational stat (Hashimoto/Solaiman/Gehrmann); watermarked text shifts these; cheap | +5–10% | Low | Low | Med-Low |
| 4 | **Fast-DetectGPT conditional curvature** (single-pass variant, Bao 2023) | LLM text sits in negative-curvature regions; watermark intensifies | +5–10% | Med (one extra fwd pass) | Low | Med |
| 5 | **Unigram-soft-green-list z-score** (learned from training watermarked vs. clean) | Direct Unigram-Watermark attack; also fires on KGW with skewed γ | +5–15% (if Unigram present) | Low | **High** — tokenizer-dependent; mitigate with multi-tokenizer | Med (paraphrase weakens) |
| 6 | **WinMax z-score** (max sliding-window z-score over 50/100/200-token windows using global soft green list) | Catches partial / copy-paste watermarks (Kirchenbauer 2024) | +3–7% | Low | Med | Better than full-text z |
| 7 | **N-gram repetition / log-diversity** for n=1,2,3 (Kirchenbauer 2024) | Watermarked text has more pseudo-random collisions → more repetition | +3–5% | Low | Low | High |
| 8 | **Burstiness** (std of sentence lengths; word-rank inter-arrival) | Human text bursty, LLM text uniform; not watermark-specific but useful for fusion | +2–4% | Negligible | Low | High |
| 9 | **Compression ratio** (gzip / zstd ratio after normalization) | Watermarking adds structural regularity → text compresses better | +1–3% | Negligible | Low | High |
| 10 | **Length-normalized features** (all of the above per-token rather than per-text) | At 1% FPR, length is a major confound | +2–4% | Negligible | Low | High |
| 11 | **Stylometric / readability** (avg word length, type-token ratio, function-word freq, Flesch-Kincaid) | Kaggle AI-detection winners use these (NEULIF 2025, Advacheck 2025) | +1–3% | Negligible | Med | Med |
| 12 | **Sentence-LSH bucket KL-divergence** (MiniLM + 8 hyperplanes) | SemStamp/SIR signal | +2–5% (if semantic schemes present) | Med | Med | High |
| 13 | **Adjacent-sentence embedding cosine variance** | SIR signal (semantic-invariant green-list smoothness) | +1–3% | Med | Med | High |
| 14 | **Aaronson-proxy −log(1−rank/V) sum** | Partial Gumbel coverage | +1–3% | Low | Low | Low (needs longer text) |
| 15 | **POS-tag bigram histograms** (lightweight, e.g., spaCy-small) | Stylometric, complements perplexity (Aityan et al. 2025 NEULIF) | +0.5–2% | Med | Med | High |
| 16 | **Multi-tokenizer agreement** (z-score under gpt2 vs. llama tokenizer) | When unknown tokenizer, agreement signals consistent watermark | +0.5–2% | Med | Low | Low |

**Drop-list (do NOT include):** retrieval against any external corpus (no API); fine-tuned RoBERTa classifier (overfits on 180 examples — Kaggle competitions show fine-tuned LMs need thousands of samples); per-token learned embeddings; n-gram bag-of-words at the test sentence level (memorization risk).

---

## 4. Calibration and Low-FPR Optimization

The 1% FPR regime is brittle: with 90 clean validation samples, the empirical 99th percentile is determined by a **single point**. With combined train+val (270 clean) the 99th percentile is determined by ~3 points. Treat threshold selection as a **statistical estimate with uncertainty**, not a single number.

### 4.1 Calibration protocol

1. **Train/validation split for calibration.** Stratified 5-fold over (TRAIN ∪ VAL) = 540 total, preserving class balance and watermark-subtype balance (use `StratifiedKFold` on a derived label = (clean/wm) × subtype-bucket).
2. **Pipeline per fold:** fit feature-pack → fit LightGBM → predict on out-of-fold; collect OOF predictions for all 540 samples.
3. **Calibration model:** fit isotonic regression `IsotonicRegression(out_of_bounds='clip')` on (OOF_pred, label). On 540 samples isotonic is acceptable; if AUC is high but reliability diagram is unstable, switch to **beta calibration** (Kull et al. 2017 — proven more robust than Platt or isotonic on small samples; closed-form 3-parameter sigmoid generalization).
4. **Final score:** retrain LightGBM on full (TRAIN ∪ VAL), apply the calibrator from step 3.

### 4.2 Threshold selection at 1% FPR

We do **not** select a single threshold; we select a *score transform* that maps 1% FPR to a fixed value (e.g., 0.99). This is done implicitly via calibration: after isotonic, the threshold "score = 0.99" should approximately correspond to 1% FPR.

For ranking-only metrics (the leaderboard scores ranked TPR@1%FPR), absolute calibration is less critical — but a calibrated continuous score is robust across distribution shift in the hidden test, while a hard threshold breaks under shift.

**Bootstrap stability check.** Resample (TRAIN ∪ VAL) 1000 times with replacement; for each resample compute the empirical score at 99th percentile of clean. Report the **5th–95th percentile interval** of these thresholds. If the interval width > 0.15 in raw score space, the ensemble is unstable — **add more regularization** (lower `num_leaves`, higher `lambda_l2`, fewer features).

### 4.3 Conformal wrapper (optional, recommended)

Use **split-conformal** with mondrian conditioning on length bin: divide validation into 4 length quartiles; for each, compute empirical 99th-percentile clean score. At test time, look up the threshold for the test text's length quartile. This guards against length-induced FPR inflation.

### 4.4 Tail stability checks (must pass before submission)

- **Top-1% clean inspection:** print the 5 highest-scoring clean validation texts. If any look obviously machine-generated (short, repetitive, formulaic), they are likely OOD and the 1% FPR is being set by anomalies — lower the threshold (more aggressive) and accept the trade-off.
- **Bottom-5% watermarked inspection:** print 5 lowest-scoring watermarked validation texts; identify a watermark subtype that's failing — if SemStamp dominates the failures, reweight Branch D.
- **Cross-fold variance:** TPR@1%FPR std across 5 folds > 8 percentage points → unstable, add features or simplify model.

### 4.5 Safeguards against accidental FPR inflation

- Never tune threshold on test predictions ("public leaderboard probing" is a classic trap).
- Never include a feature that uses the **test set** statistics (e.g., test-set unigram counts to define the green list) — this is a leak even if it's "unsupervised."
- Cap meta-LightGBM `num_leaves` at 15; with 540 training samples and 60+ features, deeper trees memorize.
- Compute validation TPR@1%FPR with **stratified bootstrap** and report the **lower 95% CI**, not the point estimate, when comparing models.

---

## 5. Validation for Hidden-Set Generalization

### 5.1 Validation slices

Compute TPR@1%FPR (and AUC, partial AUC up to FPR=5%) on every slice; require monotonic improvements vs. baseline before shipping changes.

| Slice | How to construct | Why it matters |
|---|---|---|
| **Length bins** | quartiles of token count | watermarks need ≥100 tokens for reliable detection (Mark My Words 2024); short-text TPR will be much lower |
| **Lexical complexity** | TTR quartiles, or avg-log-prob under surrogate quartiles | high-PPL human writing can mimic LLM-low-prob shape |
| **Predicted watermark subtype** | k-means (k=5) on watermarked training features → label each watermarked val sample by closest cluster; compute per-cluster recall | identifies if one scheme is being missed |
| **Paraphrase-perturbed val** | run T5-small or BART-base back-translation on the 90 wm val texts; re-score | tests robustness; expect 30–50% drop, want it ≤50% |
| **Tokenizer mismatch** | re-tokenize with a tokenizer not used at training (e.g., trained with gpt2, test with llama) | sanity-checks that the model isn't just memorizing tokenizer artifacts |
| **Truncation stress** | truncate val texts to 50, 100, 200 tokens | private test may include short replies |

### 5.2 Anti-overfitting criteria

- **Public-leaderboard sanity:** if val→public delta is > +5 percentage points TPR@1%FPR, you are overfitting to the public sub-split; freeze and submit only safe-baseline copies.
- **Feature-importance check:** if any single feature contributes > 30% of LightGBM gain, simplify (split it or drop it). Distributed importance is more robust.
- **Permutation-importance on val:** permute each feature on val; if removing it improves val TPR@1%FPR, drop it.

### 5.3 Ship/no-ship gates

Submit a new best-model only if **all** are true:
1. CV TPR@1%FPR (mean − 1·std across folds) > current submitted CV value.
2. Length-bin lowest quartile TPR@1%FPR > 0.6 × overall (no severe short-text collapse).
3. Bootstrap 5th-percentile threshold within 0.10 of median.
4. Public-leaderboard TPR@1%FPR doesn't *drop* by > 3 percentage points vs. previous submission with same CV claim (sanity check).

### 5.4 Metrics beyond headline

- Partial AUC up to FPR = 5% — a more stable proxy for ranking robustness at low FPR (Bahri 2024 used in Improving Detection of Watermarked LMs).
- Brier score after calibration (proper scoring rule).
- Per-subtype recall at fixed threshold.
- **Reliability diagram** (10-bin) — asymmetric calibration at the high-score end is what kills 1% FPR control.

---

## 6. Ablation Plan (minimal but decisive)

Run ALL ablations as 5×5 repeated stratified CV on (TRAIN ∪ VAL); report mean and 1·std of TPR@1%FPR.

| # | Ablation | Hypothesis | Expected direction | Stop condition |
|---|---|---|---|---|
| A1 | Branch A only (scheme-agnostic) | Forms the floor; ≥40% TPR@1%FPR | Down 5–15% from full | If A1 alone > full → other branches noisy → cut them |
| A2 | + Branch B (KGW proxy) | Catches KGW & Unigram | +5–10% over A1 | If +<2% → drop |
| A3 | + Branch C (Unigram specialist) | Strong on Unigram | +3–8% over A2 | If +<1% → drop |
| A4 | + Branch D (semantic LSH) | SemStamp/SIR coverage | +2–5% over A3 | If +<1% → keep only if length>200 |
| A5 | + Branch E (Aaronson proxy) | Marginal | +0–3% over A4 | Keep if not negative |
| C1 | No calibration vs. isotonic vs. beta | Calibration shouldn't hurt ranking | ≈0% AUC change, but improve threshold stability | Pick most stable across folds |
| C2 | Length-conditional threshold (mondrian conformal) | Fixes FPR inflation on long texts | +1–3% TPR@1%FPR | Keep if positive |
| R1 | Apply DIPPER paraphrase to watermarked val | Quantify worst-case robustness | TPR drops 30–60% | Diagnostic only — do NOT train on paraphrased data unless improves vanilla val |
| R2 | Truncate val texts to 100 tokens | Quantify short-text collapse | TPR drops 20–40% | If catastrophic (<20%), add length-aware features |
| F1 | Single LightGBM vs. LightGBM + LogReg average | Diversity reduces variance | 0.5×LGBM+0.5×LR matches or beats | Use average if matches with lower std |
| F2 | Drop GLTR features | Test if GLTR is redundant given log-prob shape | -1 to -3% | If <1% drop, simplify |

Run A1 first (45 min). If A1 alone gives ≥40% TPR@1%FPR, you have a safe submission. Then A2, A3, A4 in parallel batches. Calibration ablations last (cheapest).

---

## 7. Time-Boxed Execution Plan (24h hackathon)

### Hour 0–2: First baseline submitted

- 0:00–0:15 — Read task spec, confirm CSV format (`id,score`), inspect 5 train + 5 val samples, count tokens.
- 0:15–0:45 — Compute Branch A "core 8" features only: surrogate-LM (use **GPT-2-small** for speed; pre-cache to disk) log-prob mean/std, top-10/top-100 GLTR fractions, log-diversity (n=2,3), gzip ratio, length, burstiness.
- 0:45–1:15 — Train logistic regression on 360 training; CV-evaluate on 180 val. **Submit.** This is the *Minimal Safe Submission Recipe* (see end of report).
- 1:15–2:00 — Add LightGBM with same 8 features; isotonic-calibrate; submit again.

### Hour 2–8: Highest-value improvements

- 2:00–3:30 — Add Binoculars (Phi-3-mini-4k-instruct + Phi-3-mini base, or `gpt2` + `gpt2-medium` if no GPU). Extract observer log-PPL, performer cross-PPL, and the ratio. **Re-train + submit.** Expected jump: +15–20% TPR@1%FPR.
- 3:30–5:00 — Build **Branch C (Unigram specialist)**: fit soft green list from training using shrinkage. Use 3 tokenizers (gpt2, llama-2, tiktoken cl100k); take max-z. Add WinMax over 50/100/200 windows. **Submit.**
- 5:00–6:30 — Add Fast-DetectGPT proxy (single-pass curvature using a sampling step). **Submit.**
- 6:30–8:00 — Add per-tokenizer KGW pseudo-z-score with γ ∈ {0.25, 0.5}; add stylometric/length-normalized features. **Submit.**

### Hour 8–16: Robustness and calibration hardening

- 8:00–10:00 — Add Branch D (sentence-LSH KL-divergence using MiniLM-L6-v2, CPU-OK). Add adjacent-sentence cosine variance. **Submit.**
- 10:00–12:00 — Stress-test on length quartiles, paraphrased val (T5-small back-translation), tokenizer mismatch. Identify weakest slice; add features targeting it.
- 12:00–14:00 — Replace simple isotonic with **isotonic + beta** stack. Implement bootstrap threshold check. Add length-conditional (mondrian) conformal split. **Submit.**
- 14:00–16:00 — Run ablations A1–A5. Drop any branch where contribution < 1.5% TPR@1%FPR with > 3% std. Lock down the feature set.

### Hour 16–24: Stabilization + conservative final selection

- 16:00–18:00 — Add Aaronson proxy and POS-bigram features only if they improve CV (often marginal). Cap improvements by ablation evidence.
- 18:00–20:00 — Build **two final candidates**:
  - *Aggressive*: full LightGBM + LogReg ensemble + all branches.
  - *Conservative*: LightGBM with only Branches A + C + length, isotonic-calibrated.
  - Whichever has higher CV mean − 1·std → primary submission; the other → backup.
- 20:00–22:00 — Final retrain on (TRAIN ∪ VAL). Re-fit calibration via 5-fold OOF. Save weights and feature pipeline as a single pickle.
- 22:00–23:30 — Generate final submission.csv. Sanity-check: 2250 rows, score in [0,1], no NaN, IDs match. Submit.
- 23:30–24:00 — If time, submit one alternative with different ensemble blend (e.g., 0.7·LGBM + 0.3·LR vs. 0.5/0.5) and compare public LB. Pick more robust one if rules allow two finals.

### Compute-limited fallback (no GPU)

- Replace Phi-3 / Falcon Binoculars with **GPT-2 small + GPT-2 medium** Binoculars (works on CPU at ~3 sec/text → ~2 hr for 2250 texts). Drop Branch D's MiniLM if needed (use lighter `paraphrase-MiniLM-L3-v2`). Keep all statistical features which are CPU-cheap.
- If even GPT-2-medium too slow, drop Binoculars entirely and rely on GPT-2-small log-prob shape + GLTR + repetition + Branch C — still a competitive baseline.

---

## 8. Risk Register and Mitigation

| Risk | Early warning | Mitigation | Backup plan |
|---|---|---|---|
| **Scheme-mix blind spot** (private test contains a scheme not in training, e.g., SynthID-Text) | Per-subtype recall drops sharply on one cluster in val | Branch A (scheme-agnostic) is the safety net; keep its weight ≥40% in the meta-blend | Submit "Conservative" ensemble that down-weights specialist branches |
| **Low-FPR calibration instability** | Bootstrap CI on threshold > 0.15 in score; cross-fold TPR@1%FPR std > 8pp | Increase regularization; switch isotonic→beta; mondrian-conformal by length | Use rank-only score (sklearn `predict_proba` then rank-average across LGBM seeds) |
| **Short-text signal collapse** | Lowest length quartile TPR@1%FPR < 0.5× overall | Length-conditional threshold; ensemble with WinMax which works in short windows; clip very-short text features to a length-aware prior | Output 0.5 (uninformative) for texts < 30 tokens — better than wrong-confident |
| **Public/private shift** | Public LB worse than CV by >5pp | Don't tune to public; submit a *more regularized* model in the second slot | Use the conservative model as the second submission |
| **Overfitting via threshold tuning** | Multiple threshold submissions giving different public scores | Lock threshold by calibration once, do not adjust based on public LB | Re-fit calibration on TRAIN+VAL union without looking at public scores |
| **Tokenizer mismatch on private test** | Branch C / KGW-z score behaves erratically | Multi-tokenizer max-z; do not weight Branch C above Branch A in fusion | Drop tokenizer-bound features in conservative submission |
| **Paraphrase / translation on private test** | Val paraphrase-stress shows TPR drop > 60% | Branch A's stylometric features partially survive; keep them | Conservative model that relies less on token-level z-scores |
| **Memory/compute overrun on 2250 inferences** | Surrogate-LM inference > 5 sec/text | Pre-truncate test to 1024 tokens for surrogate-LM features; batch with `torch.no_grad()`; FP16 | Drop heaviest features (Binoculars), keep GLTR + Branch C |
| **Incorrect submission format** | Empty rows, non-numeric scores, mismatched IDs | Validate immediately after first submit; assert `len(df)==2250`, `df.score.dtype==float`, `df.score.between(0,1).all()` | Keep last working submission as backup; never overwrite |
| **Mid-run dependency conflict** (e.g., transformers + sklearn) | Pip install error mid-stretch | Set up a frozen `requirements.txt` in hour 0; use `--no-deps` if conflicts arise | Have a CPU-only `env.lock` ready |

---

## 9. Implementation Starter Pack (pseudocode)

```python
# ============== 0) Imports ==============
import numpy as np, pandas as pd, torch, gzip, zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

# ============== 1) Data prep ==============
def load_split(path):
    df = pd.read_csv(path)  # cols: id, text, [label]
    return df

train = load_split("train.csv")
val   = load_split("val.csv")
test  = load_split("test.csv")
all_lbl = pd.concat([train, val]).reset_index(drop=True)

# ============== 2) Surrogate LMs ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK_OBS = AutoTokenizer.from_pretrained("gpt2")
MOD_OBS = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
TOK_PER = AutoTokenizer.from_pretrained("gpt2-medium")
MOD_PER = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(DEVICE).eval()
SBERT   = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

@torch.no_grad()
def lm_token_logprobs(text, tok, model, max_len=1024):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
    if ids.shape[1] < 2: return np.array([0.0]), np.array([0])
    logits = model(ids).logits[0, :-1]                # T-1 x V
    targets = ids[0, 1:]                              # T-1
    log_probs = torch.log_softmax(logits.float(), -1) # T-1 x V
    tok_lp  = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
    # rank of realized token under model
    ranks = (logits > logits.gather(1, targets.unsqueeze(1))).sum(-1).cpu().numpy()
    return tok_lp, ranks

# ============== 3) Feature extraction ==============
def basic_stats(arr):
    if len(arr)==0: return [0]*7
    return [arr.mean(), arr.std(), np.percentile(arr,10), np.percentile(arr,25),
            np.percentile(arr,50), np.percentile(arr,75), np.percentile(arr,90)]

def gltr_buckets(ranks):
    n = max(len(ranks),1)
    return [(ranks<10).sum()/n, (ranks<100).sum()/n,
            (ranks<1000).sum()/n, (ranks>=1000).sum()/n]

def log_diversity(tokens, n):
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not grams: return 0.0
    return -np.log(1 - len(set(grams))/len(grams) + 1e-9)

def burstiness(text):
    sents = [s for s in text.split(".") if s.strip()]
    lens  = [len(s.split()) for s in sents]
    if len(lens) < 2: return 0.0
    return np.std(lens) / (np.mean(lens)+1e-9)

def gzip_ratio(text):
    b = text.encode("utf-8")
    if len(b)==0: return 1.0
    return len(gzip.compress(b)) / len(b)

def featurize_one(text):
    feats = {}
    # observer / performer LM stats
    lp_obs, rk_obs = lm_token_logprobs(text, TOK_OBS, MOD_OBS)
    lp_per, _      = lm_token_logprobs(text, TOK_PER, MOD_PER)
    feats |= {f"lp_obs_{i}":v for i,v in enumerate(basic_stats(lp_obs))}
    feats |= {f"lp_per_{i}":v for i,v in enumerate(basic_stats(lp_per))}
    feats |= {f"gltr_{i}":v for i,v in enumerate(gltr_buckets(rk_obs))}
    # Binoculars-style ratio
    ppl_obs   = np.exp(-lp_obs.mean()) if len(lp_obs) else 1.0
    cross_ppl = np.exp(-lp_per.mean()) if len(lp_per) else 1.0
    feats["binoculars"] = np.log(ppl_obs+1e-9) / (np.log(cross_ppl+1e-9)+1e-9)
    # token list for diversity
    toks = TOK_OBS.encode(text)[:1024]
    feats["len_tok"] = len(toks)
    feats["logdiv2"] = log_diversity(toks, 2)
    feats["logdiv3"] = log_diversity(toks, 3)
    feats["burst"]   = burstiness(text)
    feats["gzip"]    = gzip_ratio(text)
    feats["ttr"]     = len(set(text.split()))/(len(text.split())+1)
    return feats

# ============== 4) Branch C: Unigram soft green list ==============
def fit_soft_greenlist(train_texts, train_labels, tok, V=50257, gamma=0.25):
    cnt_w = np.ones(V); cnt_c = np.ones(V)
    for t,y in zip(train_texts, train_labels):
        ids = tok.encode(t)[:1024]
        for i in ids:
            if i<V:
                (cnt_w if y==1 else cnt_c)[i] += 1
    p_w = cnt_w/cnt_w.sum(); p_c = cnt_c/cnt_c.sum()
    soft_g = (p_w / (p_w+p_c+1e-12)) - 0.5  # > 0 means watermarked-tilted
    return soft_g

def soft_z(text, soft_g, tok, gamma=0.25):
    ids = tok.encode(text)[:1024]
    if len(ids)<5: return 0.0
    score = np.array([soft_g[i] if i<len(soft_g) else 0 for i in ids])
    pseudo_green = (score>0).sum()
    T = len(ids)
    return (pseudo_green - gamma*T) / np.sqrt(T*gamma*(1-gamma))

# ============== 5) Build feature matrix ==============
def build_X(df, soft_g):
    rows=[]
    for t in df["text"]:
        f = featurize_one(t)
        f["softz_gpt2"] = soft_z(t, soft_g, TOK_OBS)
        rows.append(f)
    X = pd.DataFrame(rows).fillna(0.0)
    return X

# ============== 6) Training with calibration ==============
soft_g = fit_soft_greenlist(train["text"].tolist(),
                            train["label"].tolist(), TOK_OBS)

X_all = build_X(all_lbl, soft_g)
y_all = all_lbl["label"].astype(int).values

# 5-fold OOF for calibration training
oof = np.zeros(len(y_all))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in skf.split(X_all, y_all):
    params = dict(objective="binary", learning_rate=0.03,
                  num_leaves=15, max_depth=5, min_data_in_leaf=15,
                  feature_fraction=0.7, bagging_fraction=0.8,
                  lambda_l2=2.0, verbosity=-1)
    dtr = lgb.Dataset(X_all.iloc[tr_idx], y_all[tr_idx])
    dva = lgb.Dataset(X_all.iloc[va_idx], y_all[va_idx], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=500,
                      valid_sets=[dva], callbacks=[lgb.early_stopping(30)])
    oof[va_idx] = model.predict(X_all.iloc[va_idx])

# Calibrator
iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_all)

# Final model on all labeled data
dall = lgb.Dataset(X_all, y_all)
final_model = lgb.train(params, dall, num_boost_round=300)

# ============== 7) Validation metric: TPR@1%FPR ==============
def tpr_at_fpr(scores, labels, target_fpr=0.01):
    s_neg = np.sort(scores[labels==0])[::-1]
    thr = s_neg[int(np.ceil(target_fpr*len(s_neg)))-1]
    return (scores[labels==1] >= thr).mean()

print("OOF TPR@1%FPR (raw):", tpr_at_fpr(oof, y_all))
print("OOF TPR@1%FPR (cal):", tpr_at_fpr(iso.transform(oof), y_all))

# Bootstrap stability
import scipy.stats as st
boots=[]
for _ in range(1000):
    idx = np.random.choice(len(y_all), len(y_all), replace=True)
    boots.append(tpr_at_fpr(iso.transform(oof[idx]), y_all[idx]))
print("Bootstrap 5/50/95:", np.percentile(boots,[5,50,95]))

# ============== 8) Inference on test ==============
X_test = build_X(test, soft_g)
raw    = final_model.predict(X_test)
cal    = iso.transform(raw)
sub    = pd.DataFrame({"id": test["id"], "score": np.clip(cal, 0.0, 1.0)})
assert len(sub)==2250 and sub["score"].between(0,1).all()
sub.to_csv("submission.csv", index=False)
```

---

## 10. Evidence Ledger

| Claim | Source | Year | Conf. | Transfer rationale |
|---|---|---|---|---|
| KGW (Kirchenbauer et al.) defaults γ=0.25, δ=2.0; z = (n_green − γT)/√(Tγ(1−γ)) | Kirchenbauer et al. ICML | 2023 | High (foundational) | Task spec says KGW-only is insufficient → KGW is present; defaults likely; z-score reconstruction is well-defined |
| KGW LeftHash (h=1) and SelfHash (h=3) are the most popular variants | Gloaguen et al. ICLR | 2025 | High | Likely present in training mix; black-box detection results show both are statistically detectable |
| Unigram-Watermark (Zhao et al.) uses fixed global green list, twice as robust to edits as KGW | Zhao et al. ICLR | 2024 | High (foundational) | Likely present; the *fixed* green list is reverse-engineerable from training watermarked vs. clean — direct attack via Branch C |
| SemStamp (sentence-level LSH partition) and SIR (semantic-invariant) are the canonical "semantic" schemes | Hou et al. NAACL; Liu et al. ICLR | 2024 | High (foundational) | "Semantic-like" branch needed if semantic family present in mix |
| Aaronson/Gumbel watermark uses argmax of P_t(a)/(−log U_{t,a}); detection sums −log(1−U_{t,a_t}) | Aaronson 2022; Lattimore arXiv | 2022/2026 | Medium | Without key, only weak proxies; included as Branch E for partial coverage |
| MarkLLM toolkit (Pan et al.) implements 9 algorithms across KGW & Christ families incl. KGW LeftHash, SelfHash, Unigram, SIR, SemStamp, Exp/Gumbel, DiP, EWD | Pan et al. EMNLP demo | 2024 | High | Best evidence of "what schemes are likely in a hackathon mix" |
| Black-box detection of all three watermark families (Red-Green, Fixed-Sampling, Cache-Augmented) feasible with statistical tests on token frequency | Gloaguen, Jovanović, Staab, Vechev | 2024/2025 | High | Validates that watermark traces survive in observable token statistics — the foundation of Branch B/C |
| Watermark stealing: green list reconstructable from ~200K tokens of watermarked output | Jovanović et al. arXiv | 2024 | High | We have far less data (180 watermarked train), but enough for *coarse* soft green list useful as a feature |
| Binoculars (log-PPL / cross-PPL) achieves ≥90% TPR @ 0.01% FPR zero-shot on ChatGPT text | Hans et al. ICML | 2024 | High | Watermarked text is by definition LLM-generated → Binoculars baseline already provides strong TPR@1%FPR even before any watermark-specific feature |
| Fast-DetectGPT conditional probability curvature is ~75% better than DetectGPT and supports zero-shot detection in a single pass | Bao et al. ICLR | 2024 | High | Cheap to compute and complements Binoculars |
| GLTR rank histograms (top-10/100/1000) discriminate human vs. LLM text | Gehrmann et al. ACL | 2019 | High (foundational) | Cheap, low overfit risk, used in Kaggle AI-detection winners |
| MarkMyWords benchmark: KGW with Llama-2-7B-chat detectable in <100 tokens, robust to simple perturbations | Piet et al. SaTML | 2024/2025 | High | Sets expectation for short-text limit and benchmarking practice |
| WaterBench: identifies V2 KGW best for generation metric; warns watermark detection degrades with short outputs | Tu et al. ACL | 2024 | High | Expect short-text TPR collapse → length-aware features critical |
| Paraphrase (DIPPER) drops DetectGPT to 4.6% TPR @ 1% FPR; watermarks better but still drop materially | Krishna et al. NeurIPS | 2023 | High | Use this to estimate worst-case private TPR; do not panic at 30–50% drops |
| Beta calibration (Kull et al.) more reliable than Platt or isotonic on small samples; Venn-Abers strongest in recent benchmarks but heavier | Kull et al. EJS; Classifier Calibration at Scale (arXiv 2601.19944) | 2017/2026 | Medium-High | With 540 calibration samples, beta is sweet spot |
| WinMax sliding window detection improves robustness to copy-paste and partial watermarks | Kirchenbauer et al. (Reliability) | 2023/2024 | High | Explicit hyperparameter to gain on mixed-source samples |
| n-gram log-diversity = −log(1 − unique-n-grams / total-n-grams) is a watermark-relevant feature; watermarked text has higher diversity bias | Kirchenbauer et al. (Reliability); Survey 2024 | 2024 | Medium | Cheap; complements perplexity |
| LLM-Detect-AI Kaggle winner used CLM-finetuned LMs + token statistics + tree-boosted ensemble | rbiswasfc 1st place writeup | 2024 | Medium | Confirms tree-boosted meta-learner as a robust hackathon-grade fusion choice |
| Tree-boosted meta-learners with `num_leaves=15`, strong L2 prevent overfit on small (≤1k) datasets | LightGBM docs / domain practice | ongoing | Medium-High | Direct hyperparameter guidance |
| SynthID-Text uses tournament sampling + g-function; detection via Bayesian or weighted-mean detector; works best on long, high-entropy outputs | Google DeepMind blog/Nature | 2024/2025 | Medium-High | If SynthID present in mix, expect long-text bias and degraded perf on factual short text — Branch A still helpful |
| 90 watermarked validation samples → TPR@1%FPR uncertainty is ±10pp at 95% — bootstrap mandatory | Standard binomial CI; this report | 2026 | High | Drives all "ship/no-ship" gates |
| MarkLLM github 2024.07 added KGW hashing variants (skip, min, additive, selfhash) → diverse hash schemes likely in any mix | THU-BPM/MarkLLM repo | 2024 | High | Multi-tokenizer + multi-context-size in Branch B |

*All cited works are real and were retrieved during research; no metrics or APIs are invented. Predictions about gain magnitudes are explicitly labeled "expected" and are calibrated against the cited evidence — they are NOT confirmed for this specific hackathon's hidden distribution.*

---

## What NOT to do (common failure patterns)

1. **Do NOT fine-tune a RoBERTa/DeBERTa classifier on 180 watermarked training samples.** Kaggle winners had thousands of texts; here you will overfit catastrophically.
2. **Do NOT rely on a single Kirchenbauer-only z-score detector.** The task spec explicitly rules this out.
3. **Do NOT tune the threshold based on public-leaderboard probing.** Leaderboard is 30%, private is 70% — you will overfit the public split. Tune only on (TRAIN ∪ VAL).
4. **Do NOT use any feature that depends on test-set statistics** (e.g., test-set unigram distribution to define a green list). Even unsupervised, this is a leak.
5. **Do NOT pick a single threshold and snap scores to {0,1}.** The metric is ranking-based at low FPR — emit *continuous, calibrated* scores.
6. **Do NOT include features whose extraction can fail silently** (e.g., sentence segmenter that returns empty list on code). Always fill with safe defaults.
7. **Do NOT submit untested format changes** in the last hour. Validate `submission.csv` schema after every submission. Keep at least one known-good submission as backup.
8. **Do NOT add a feature without an ablation.** With 360 training points, every noisy feature is a potential overfit driver.
9. **Do NOT trust val TPR@1%FPR as a point estimate.** Use bootstrap CI lower bound for decisions.
10. **Do NOT use a deep LightGBM** (`num_leaves > 31`) on 540 rows × 60+ features — guaranteed overfit.

---

## Immediate Action Checklist (first 90 minutes)

- [ ] **0–5 min**: Read task README; confirm submission format; load 3 sample rows.
- [ ] **5–15 min**: Set up Python env; install transformers, lightgbm, sentence-transformers, sklearn; pin versions in `requirements.txt`.
- [ ] **15–30 min**: Download `gpt2` weights (~500MB) once; cache locally.
- [ ] **30–45 min**: Implement `featurize_one()` for the 8 core features (log-prob stats, GLTR buckets, length, logdiv2/3, burstiness, gzip).
- [ ] **45–60 min**: Run featurize over 360 train + 180 val + 5 test samples to confirm pipeline.
- [ ] **60–75 min**: Fit logistic regression; compute val AUC and TPR@1%FPR; print bootstrap CI.
- [ ] **75–90 min**: Run featurize over all 2250 test rows; produce `submission.csv`; verify schema (`id,score`, 2250 rows, scores in [0,1]); **submit**.

---

## Minimal Safe Submission Recipe

A single self-contained recipe that gets you on the leaderboard with format guaranteed correct and decent (likely 30–45% TPR@1%FPR) score, in under 2 hours total CPU-only on a laptop:

```python
import gzip, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

tok = AutoTokenizer.from_pretrained("gpt2")
mod = AutoModelForCausalLM.from_pretrained("gpt2").eval()

@torch.no_grad()
def feats(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    if ids.shape[1] < 2:
        return [0]*8
    logits = mod(ids).logits[0, :-1]
    targets = ids[0, 1:]
    lp = torch.log_softmax(logits.float(), -1).gather(1, targets.unsqueeze(1)).squeeze(1).numpy()
    ranks = (logits > logits.gather(1, targets.unsqueeze(1))).sum(-1).numpy()
    n = len(ids[0])
    toks = ids[0].tolist()
    bigrams = list(zip(toks[:-1], toks[1:]))
    logdiv2 = -np.log(1 - len(set(bigrams))/max(len(bigrams),1) + 1e-9)
    gz = len(gzip.compress(text.encode()))/max(len(text.encode()),1)
    return [lp.mean(), lp.std(), (ranks<10).mean(), (ranks<100).mean(),
            n, logdiv2, gz, len(set(text.split()))/max(len(text.split()),1)]

def build(df):
    return np.array([feats(t) for t in df["text"]])

train = pd.read_csv("train.csv"); val = pd.read_csv("val.csv"); test = pd.read_csv("test.csv")
all_lbl = pd.concat([train, val]).reset_index(drop=True)
X_all = build(all_lbl); y_all = all_lbl["label"].astype(int).values
X_test = build(test)

clf = LogisticRegression(C=1.0, max_iter=500).fit(X_all, y_all)
iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(X_all)[:,1], y_all)
scores = np.clip(iso.transform(clf.predict_proba(X_test)[:,1]), 0.0, 1.0)

sub = pd.DataFrame({"id": test["id"], "score": scores})
assert len(sub) == 2250 and sub["score"].between(0, 1).all() and sub["score"].notna().all()
sub.to_csv("submission.csv", index=False)
```

This recipe has **zero hyperparameter risk**, takes ~30 minutes on CPU for 2250 texts, produces a properly-formatted submission, and provides a known-good fallback for every subsequent attempt. Improve from here with the Branch B/C/D additions detailed above; never let the leaderboard go empty.

---

## Caveats

- The 180/90 split sizes are tight: bootstrap intervals on TPR@1%FPR will be wide (±10pp at 95% CI). Treat any single-fold improvement < 5pp as noise.
- Predictions of feature gain magnitudes (e.g., "+15–20% from Binoculars") are calibrated against published benchmarks on related tasks (general AI-text detection, MarkMyWords KGW detection); they are *not* validated on this specific hidden distribution. The exact mix of watermark families in the private test is unknown — Branch A's scheme-agnostic floor is the only feature group with high-confidence transfer.
- The scheme-aware branches (B–E) assume some scheme overlap between train and test mixes; if the private test introduces a *new* family entirely (e.g., a model-extraction-based watermark), specialist branches will not help and could *hurt* via FPR inflation. The conservative submission (Branch A + length only) is your insurance.
- Black-box watermark detection literature (Gloaguen et al. 2025) presumes thousands of model queries; our setting has only 180 watermarked training texts. Expect Branch B/C signal-to-noise ratio to be weaker than in published black-box detection results — they primarily inform feature design, not absolute performance forecasts.
- Tooling assumption: GPT-2 / Phi-3 / MiniLM weights are accessible. If the hackathon environment is air-gapped, pre-cache models in hour 0 or fall back to pure-statistics features.
- The Lattimore (2026) Gumbel detection refinement is recent and not directly used — only mentioned as an indication that the Aaronson family is still actively studied. Branch E remains a weak proxy.
- All references to "MarkLLM family of schemes" are inferential about the private test composition; we do not have ground truth on which schemes are used. Strategy is robust to this uncertainty by virtue of ensembling.