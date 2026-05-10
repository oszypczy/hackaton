# 🚨 GIGAPROMPT — wklej do Perplexity Pro / Gemini Deep Research / Claude Opus

**WHEN:** 2026-05-09 22:00 UTC, hackathon ends 12:30 next day, ~14h pozostało

---

## CONTEXT (paste this whole block first):

```
I'm in the final 14 hours of a 24h ML security hackathon (CISPA Warsaw 2026).

Task: BLIND multi-watermark detection for LLM-generated text.
Watermarks present in test: Kirchenbauer (KGW), Liu (SIR/Semantic), Zhao (Unigram).
Metric: TPR @ 1% FPR. Score in [0, 1], higher better.
Live leaderboard scored on 30% public test, final ranking on 70% private hidden test.

DATA:
- 360 train (180 clean + 180 watermarked, 60 each type)
- 180 valid (90 + 90)
- 2250 test (mix of all types + clean, distribution unknown)
- Texts ~250-500 tokens, mostly Q&A medical/general topics

CURRENT STATE:
- Leaderboard: 0.259 (#2 of 8 teams), leader: 0.387 (Syntax Terror)
- 263 features extracted: GPT-2 PPL stats, Pythia binoculars (1.4b/2.8b/6.9b),
  Fast-DetectGPT curvature, Liu/SIR proxy via better_liu (15 cols), KGW direct
  multi-tokenizer + LLaMA + v2, Unigram direct (key=9999 sha256_str winner from
  gridsearch), bigram greenlist, RoBERTa-base 768→PCA32, stylometric (TTR/burstiness/etc),
  multi-LM PPL across Phi-2/Qwen2/Llama-chat/Mistral, OLMo-1B+7B-Instruct PPL,
  LM-as-judge zero-shot (OLMo, Phi-2, Mistral)

CLASSIFIER OOF SWEEP (5-fold CV on labeled 540):
- LogReg C=0.05 → 0.7778 (BEST baseline)
- ElasticNet l1=0.1 → 0.7667
- Ensemble logreg+lgbm → 0.7667
- LightGBM → 0.6926
- Multi-seed ensemble (5 seeds) → 0.78 mean
- Calibrated isotonic → 0.7148-0.7519 (HURTS)
- Empirical Fisher green list (watermark stealing) → 0.7037-0.7593 (HURTS)
- Different C values (0.001 to 1.0) all in 0.74-0.78 range

🚀 BREAKTHROUGH discovered 5 min ago:
PSEUDO-LABELING (transductive learning) takes top-X% confident test predictions as
training labels:
- f=0.05 OOF 0.7630 (worse, too few)
- f=0.10 OOF 0.7926
- f=0.20 OOF 0.8111
- f=0.30 OOF 0.8444 ⭐
- f=0.40 OOF 0.9074
- f=0.50 OOF 0.9333 ⭐⭐ (50% of test pseudo-labeled!)
- 5-seed pseudo-rankmean at f=0.30 → mean 0.867
But: OOF is computed by training on labeled+pseudo-test, evaluating on labeled val.
At f=0.50, pseudo-labels (1125) > real labels (540) — could be overfitting to model's own bias.

OOF→LB ratio is consistently 1:3 (OOF 0.78 → LB 0.26). All standard variants give LB 0.26.
Big distribution shift: train+val labeled vs public test set.

CLUSTER: Jülich SLURM, A800 GPUs, models pre-cached: GPT-2, GPT-neo, Pythia (1.4b/2.8b/6.9b),
RoBERTa-base, OPT-1.3b, OLMo-2 (1B+7B Instruct + 13B Instruct), Phi-2, Qwen2-0.5B, Mistral-7B
(base + Instruct), Llama-3-8B, sentence-transformers/all-MiniLM-L6-v2,
perceptiveshawty/compositional-bert-large-uncased.

NOT cached / can't get: Liu et al. official `transform_model_cbert.pth` MLP checkpoint
(GitHub THU-BPM/Robust_Watermark unreachable from compute nodes; download attempts
failed silently). DIPPER paraphraser also not cached.
```

---

## PRIORITY QUESTIONS (most urgent):

### Q1: Pseudo-labeling validation & risk

The OOF jumps from 0.78 (no pseudo) to 0.93 (50% pseudo). Is this real generalization or
self-fulfilling overfitting?

**Specific asks:**
1. What's the SOTA way to validate pseudo-labels in semi-supervised binary classification
   when you can't easily hold out test data? Cite recent papers (2023-2025).
2. Is there a **canary**/saturation check: e.g., shuffle pseudo-labels and see if OOF still
   inflates? If OOF goes up with shuffled labels too, then OOF is meaningless.
3. Any reason to believe **f=0.50 will generalize WORSE on hidden private test** than f=0.30
   even if both score same on public 30%?
4. Is **pi-model / FixMatch / MixMatch** worth implementing for our setup (just 540 labeled +
   2250 unlabeled)?

### Q2: SIR (Liu/Semantic) watermark detector — pragmatic workaround

Liu et al. 2024 watermark uses `compositional-bert-large-uncased` (BERT 1024-d) → MLP
(1024→2048→50257) → per-token z-score. We HAVE the BERT model cached. We DON'T have the
official MLP checkpoint.

**Specific asks:**
1. Mirrors / alternative download URLs for `THU-BPM/Robust_Watermark` `transform_model_cbert.pth`?
   (PyPI, Zenodo, archive.org, HF Hub repo, Wayback)
2. With 60 SIR-watermarked + 60 clean labeled examples, can we **train a substitute MLP from
   scratch** to approximate the green list mapping? What's the simplest architecture and
   training procedure?
3. Are there **2024-2025 papers** describing alternative SIR-style detectors that work
   without the official key? E.g., Christ et al. cryptographic watermarks, Aaronson scheme,
   universal detectors?
4. Since SIR enforces semantic CONSISTENCY between consecutive token contexts, can we use a
   simpler proxy: cosine of consecutive sentence embeddings under compositional-BERT? We have
   features_d (sentence-transformers/all-MiniLM-L6-v2) and features_better_liu — do those
   already cover this? What's missing?

### Q3: Why is OOF→LB ratio 1:3? Distribution shift diagnosis

OOF 0.78 on 540 labeled → LB 0.26 on 30% public test. 3x gap is enormous.

**Specific asks:**
1. For TPR@1%FPR specifically, is there a known calibration issue when training set is
   **balanced** (180 pos / 180 neg) but test set is **imbalanced** (unknown ratio)?
2. If labeled set has 60+60+60 watermark types but test set has different proportions
   (e.g., 30% Liu, 50% KGW, 20% Zhao), what would we expect to happen to TPR?
3. Recent benchmarks **WaterBench, MarkMyWords**: how do they report public/private gaps?
   What's typical generalization?
4. Is there a **simple test for distribution shift** we can run: e.g., density estimation on
   feature space, KS test between train and test marginal distributions?

---

## OUTPUT FORMAT REQUEST:

For each question:
- **Direct answer** (1-2 sentences)
- **Key papers/repos** (with URLs)
- **Code snippets** if applicable (Python, sklearn-compatible preferred)
- **Risk assessment** for our 14-hour deadline

Be specific. Cite arxiv IDs. No fluff. We have GPUs and time for ~3-4 more compute jobs.
