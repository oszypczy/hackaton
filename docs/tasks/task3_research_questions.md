# Task 3 — Research Questions for Perplexity / Gemini / Claude

> **Context for assistant**: Hackathon task on blind multi-watermark detection
> (KGW/Kirchenbauer + Liu/SIR + Zhao/Unigram). 360 labeled samples (180 clean + 180 watermarked, 60 per type), 2250 test, metric TPR@1%FPR.
> Currently OOF 0.78, leaderboard 0.259, leader 0.387. Gap suggests test distribution differs from train.
>
> When asking these to an external model, paste the relevant question + this context block + add: "Be specific, cite papers, give code snippets where applicable."

---

## Q1. SIR (Liu/Semantic) watermark — how to detect without the official MLP checkpoint?

**Background:** The Liu et al. 2024 SIR watermark uses `compositional-bert-large-uncased` (BERT 1024-d) → MLP (1024→2048→50257) → per-token z-score. Official checkpoint `transform_model_cbert.pth` is at `THU-BPM/Robust_Watermark` on GitHub but downloads keep failing (cluster compute nodes have no internet, login node ssh kept dropping).

**Question(s):**
- Are there alternative hosts/mirrors for `transform_model_cbert.pth`? (PyPI, Zenodo, archive.org, HF Hub mirror)
- Could we train a substitute MLP from scratch? With 360 labeled samples + access to the BERT encoder, what's the simplest way to recover a green-list approximation?
- What's the BEST proxy detector for SIR watermark when you DON'T have the original key? Cite the strongest baseline.
- The paper "Robust Multi-bit Text Watermark with LLM-based Paraphrasers" mentions test-time embedding consistency — can we use BERT cosine of consecutive sentences as proxy?

---

## Q2. KGW (Kirchenbauer) hash function — possible variants

**Background:** KGW seed = `key × hash(prev_token) × ...`. We've tried key=15485863 (paper default), 33554393, 0, 1 with hash functions `v1` (just key×hash) and `v2` (lm_head bias). Also MD5, sha256_str. Results poor (~0.10 z-score separation).

**Question(s):**
- What's the COMPLETE list of hash functions used in real KGW implementations? Specifically:
  - Original Kirchenbauer paper's `extended-hash-scheme.py`
  - Aaronson's variant
  - Yoo et al. multi-bit variant
  - Christ et al. cryptographic
  - WaterBench evaluation suite implementations
- Are there standardized hash_keys beyond 15485863? (some implementations use 1234, 42, 2023, etc.)
- For BIGRAM (h=2) self-hash: what are the known variants? Hash `(prev_token, curr_token, key)` → which order?

---

## Q3. Why does OOF 0.78 → LB 0.26? Generalization gap diagnosis

**Background:** We have 263 features (PPL, watermark z-scores, stylometric, embeddings, judges) and 540 labeled samples. 5-fold OOF gives TPR@1%FPR of 0.78. Leaderboard scores 0.26 on test. Gap is 3x.

**Hypotheses:**
- Train ratios: 60 KGW + 60 Liu + 60 Zhao + 180 clean. Test ratios unknown.
- Train texts ~250-500 tokens. Test texts: unknown.
- Test set may include adversarial paraphrases / truncated / mixed-language.

**Question(s):**
- For a binary AI-text-detection task with LARGE feature space (263) and small N (540), what's the CORRECT regularization strategy?
- Should we use "cross-watermark holdout" (train on KGW+Liu, test on Zhao) for more honest OOF?
- How do recent watermark detection benchmarks (WaterBench, MarkMyWords) handle distribution shift between train/eval?
- Is there a calibration technique that helps (Platt scaling, isotonic)?

---

## Q4. Strongest 2024-2025 techniques for blind multi-watermark detection

**Background:** Papers we've integrated: Fast-DetectGPT (curvature), Binoculars (PPL ratio), GLTR-style features, Liu/SIR proxy, KGW direct, Unigram direct.

**Question(s):**
- What papers from NeurIPS/ICML/ACL 2024 propose "universal" or "scheme-agnostic" watermark detectors?
- Is there a published black-box detector that works ACROSS all 3 schemes (KGW/Liu/Zhao) simultaneously?
- For the "watermark stealing" scenario (we have 60 examples of each watermark, no key), what's SOTA?
- Are there tools like `imdll` or `watermark-detective` we should integrate?

---

## Q5. Better classifier strategies for our setup (263 features, 540 samples, 3 classes pooled)

**Background:** LogReg C=0.05 best (OOF 0.78). LightGBM 0.69. Ensemble 0.75 (no improvement).

**Question(s):**
- For sklearn-pipelines, what's the BEST regularizer for high-dim small-sample binary classifier? (L1+L2 elastic? group lasso? sparse PLS?)
- Should we try `RidgeClassifierCV` with CV-tuned alpha?
- Does `CatBoost` help here vs LightGBM? (CatBoost handles small data better, allegedly)
- Ensemble strategies for low-bias classifiers (LogReg + LDA + Ridge)?
- Multi-class classifier (4-class: clean/KGW/Liu/Zhao) → softmax → 1-P(clean) — better than binary?

---

## Q6. SIR detector substitute — train our own?

**Background:** We have 60 SIR-watermarked + 180 clean labeled. We have BERT cached. We can do feedforward forward passes on labeled data.

**Question(s):**
- Could we train a contrastive embedding model: pull SIR-positive towards each other, push from clean? Triplet loss?
- Simpler: fit a logistic on `BERT_pooled[t] - BERT_pooled[t+1]` differences (consecutive sentence cosine drop)?
- Empirical "green token" recovery via Fisher's exact test on token frequencies in SIR vs clean?
- Anomaly detection: One-class SVM on BERT embeddings of clean, score test outliers?

---

## Q7. (Wild ideas) Steganography artifacts in watermarked text

**Question(s):**
- Do KGW-watermarked texts have detectable byte-level artifacts (specific Unicode chars, formatting)?
- Does watermarked text have detectable "rhythm" via pause/comma distribution that clean LLM text lacks?
- Is `roberta-base-openai-detector` (Hugging Face) trained on watermark-leaked GPT-2 outputs? Would adding its logit help?

---

**Format for assistant**: When pasting these to Perplexity/Gemini/Claude, use this template:

```
Context: [paste from CONTEXT block at top of this file]

Question: [paste Q from this file]

Constraints:
- Be specific (paper titles, GitHub repos, exact code snippets)
- I have ~10 hours remaining in a hackathon
- Cluster has GPU (Pythia-6.9b, OLMo-7B, Phi-2, Mistral-7B already cached)
- I cannot install new HF models from compute nodes (no internet); login node has internet
```
