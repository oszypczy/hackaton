# Deep Research Prompt — Task 3: Blind Multi-Type LLM Watermark Detection

> Wklej do Perplexity Deep Research / Gemini Deep Research.
> Cel: uzupełnić luki w implementacji blind watermark detector dla 3 typów jednocześnie.
> Status: score=0.136 (leader ~0.27), 8h pozostało w hackathonie.

---

## Context

We are competing in a 24h security hackathon (CISPA, Warsaw, May 2026). Task 3 requires
building a detector for LLM text watermarks. Key constraints:
- Three watermark types mixed in the test set: Kirchenbauer (KGW), Liu (ICLR 2024),
  Zhao (ICLR 2024 Unigram-Watermark)
- NO per-sample type label is given — we must detect all three types blind
- Train split: 360 samples (180 clean + 60 per watermark type, labels 0/1 but no type info)
- Validation split: 180 samples (90 clean + 30 per type)
- Test split: 2250 samples, predict continuous score in [0,1]
- Metric: TPR@1%FPR (higher is better). Current best: 0.136. Leader: ~0.27
- Cooldown: 5 min per submission, so we need to identify the right approach before submitting

## What we already have (don't re-suggest these)

Current feature set used with LogisticRegression (C=0.05):
1. branch_a: GPT-2 log-prob stats (entropy, variance, rank distribution, GLTR features,
   n-gram diversity, gzip compression, burstiness, TTR)
2. branch_bc: Unigram green-list features (learned from train) + bigram KGW features
3. binoculars_strong: Pythia-1.4b + 2.8b PPL ratio (Binoculars method)
4. fast_detectgpt: Pythia-2.8b analytical curvature (Fast-DetectGPT method)
5. branch_d: sentence-transformer cosine similarity (proxy for Liu semantic watermark)

## Failed approaches (dead ends)

- KGW direct z-score with hash_key=15485863 (default), gpt2 tokenizer, gamma=0.25: NO SIGNAL
- KGW with hash_keys in [0,1,42,100,12345,999,7] and h=2 multigram: also NO SIGNAL
- XL Binoculars (Pythia-2.8b/6.9b): no improvement over 1.4b/2.8b pair
- LightGBM classifier: bimodal collapse in scores → switched to LogReg

## Key unknowns we need answers to

### 1. Liu et al. ICLR 2024 — "A Semantic Invariant Robust Watermark for Large Language Models"

This paper is cited by the organizers as one of the 3 watermark types in the dataset.
We need to know:
- What is the exact detection algorithm? Does it require access to the original generation model?
- What encoder/model does the semantic hash use? (SimCSE? BGE? SONAR? BERT? The LM itself?)
- What are the DEFAULT parameters (gamma, seed, embedding model)?
- Is there a public GitHub repository with detection code?
- Can detection work without knowing the exact embedding model? (black-box detection)
- How does the detection differ from KGW detection?
- ArXiv ID: 2310.06356 (check if this is correct)

### 2. Zhao et al. ICLR 2024 — "Provable Robust Watermarking for AI-Generated Text" (Unigram-Watermark)

- What is the DEFAULT seed for the green list? (GitHub: XuandongZhao/Unigram-Watermark)
- Is the green list the same across different generations, or per-prompt?
- Can we RECOVER the green list from N labeled watermarked samples using frequency analysis?
- What threshold z-score does the paper use for detection?
- Does the green list depend on the generation model (tokenizer)?
- GitHub code: what are the exact default hyperparameters?

### 3. KGW (Kirchenbauer) — why did default detection fail?

We tried hash_key=15485863, gpt2 tokenizer, gamma=0.25, h=1. ALL gave no signal.

Possible reasons and diagnostics:
- Wrong tokenizer? (LLaMA, OPT, Mistral instead of GPT-2?)
- New seeding scheme from "Reliability of Watermarks" paper (ICLR 2024, SelfHash)?
- Custom gamma (0.5 instead of 0.25)?
- The texts have been post-processed (paraphrased)?
- Text too short for z-test to give signal?
- Delta too low?

What diagnostic experiments could we run on the 180 labeled training watermarked samples to
identify which KGW variant was used?

### 4. Multi-type blind detection state of the art

What is the best published approach for detecting watermarked LLM text when:
- You don't know which watermark scheme was used
- You have 60 labeled examples per scheme (but don't know which sample belongs to which scheme)
- The schemes may use different tokenizers and embedding models

Specifically:
- Gloaguen et al. 2024-25 "black-box watermark presence tests" — what are the key ideas?
- Is there a published method for identifying which watermark scheme a sample belongs to
  (clustering/unsupervised)?
- Can we use hypothesis testing to identify the scheme, then apply the targeted detector?

### 5. Empirical green-list recovery (Unigram)

Given 60 labeled watermarked (Unigram type) and 180 labeled clean samples, can we recover
the fixed green list empirically?

The Unigram watermark uses a FIXED green list for all tokens (same green tokens regardless
of context). This means:
- Green tokens appear MORE OFTEN in watermarked text than clean text
- We can compute per-token frequency ratio: watermarked_freq[tok] / clean_freq[tok]
- Tokens with consistently high ratio = likely green tokens

Questions:
- How many samples of the ~60 are needed to reliably recover the green list?
- What statistical test to identify green tokens? (Fisher's exact? Chi-square? Log-ratio?)
- What precision/recall of green list recovery is needed for the z-test to give signal?
- Should we recover it per-tokenizer (gpt2/llama/mistral)?

### 6. Alternative features not yet tried

What additional features might differentiate watermarked from clean text across ALL three
watermark types simultaneously?

Ideas to evaluate:
- Per-token z-scores under multiple tokenizers (gpt2, llama, mistral)
- Sentence-level semantic clustering (is there unusual "semantic stickiness"?)
- Green-ratio consistency across text windows (watermarked = consistent; clean = random)
- RoBERTa/DeBERTa as a zero-shot discriminator (fine-tuned on "human vs AI" data)
- Entropy shift in later parts of the text (watermarks affect entropy uniformly)
- Perplexity under multiple models: mean, variance, min, max across gpt2/pythia/llama
- MPAC-style test statistics

## Research directions we want explored

### Direction A: Targeted detection for each scheme

If we can IDENTIFY which scheme each test sample uses (even approximately), we can apply
the exact z-score test for that scheme:
1. KGW: z-test with the correct hash_key and tokenizer
2. Unigram: z-test with the empirically-recovered green list
3. Liu semantic: z-test using the semantic hash and the correct encoder

Feasibility question: Can unsupervised clustering (k-means, GMM) on the 360 labeled
training samples separate the 3 watermark types into clusters?

If yes: we apply different detectors per cluster, then combine scores.

### Direction B: Better black-box features

Without knowing the scheme, what features best discriminate across all types?
Specifically:
- What does the literature say about features that work for ALL three types simultaneously?
- Are there "universal" watermark signals that all token-level watermarks share?

### Direction C: Fine-tuned classifier

Train a RoBERTa/DeBERTa classifier directly on the 360 train samples as a binary
watermark detector. Questions:
- Is this enough data (360) for fine-tuning a transformer classifier?
- What learning rate, batch size, epochs?
- Better: use the smaller train set as few-shot training, with pre-trained
  "human vs AI" models as initialization?

### Direction D: Mixture-of-experts

Train SEPARATE classifiers for each detected type, then combine:
- Step 1: Cluster training data into 3 types (unsupervised)
- Step 2: Per-type classifier (or z-score detector)
- Step 3: For test data: soft-assign to types, take weighted average of detectors

## What we specifically want the deep research to find

1. **Liu 2024 ICLR "Semantic Invariant Robust Watermark" — detection algorithm in detail**
   - Algorithm pseudocode for detection
   - What embedding model is used as default
   - GitHub code if available
   - arxiv 2310.06356 — confirm details

2. **Zhao 2024 ICLR "Provable Robust Watermarking" (Unigram) — detection code**
   - GitHub: XuandongZhao/Unigram-Watermark
   - Default seed and how green list is generated
   - Full detection procedure

3. **KGW failure diagnosis**
   - Why might default KGW z-score fail on 60 labeled samples?
   - What other seeding schemes exist in the KGW ecosystem?
   - How to enumerate and test possible parameters systematically?

4. **Best published baseline for blind multi-watermark detection**
   - Any paper that evaluates a single detector across multiple watermark schemes
   - AUROC across KGW+Unigram+Liu simultaneously

5. **Empirical green list recovery**
   - Statistical method to recover Unigram green list from labeled samples
   - Sample complexity analysis

6. **Actionable code snippets** for:
   - Liu 2024 detection (Python, using HuggingFace)
   - Unigram detection (Python, with empirical green list)
   - KGW parameter enumeration (try all common hash_keys and tokenizers)

## Additional context

- Our current best submission uses Fast-DetectGPT + Binoculars (PPL ratio) → LogReg
- We have A800 GPUs on a cluster (Jülich Supercomputer), 4× per node
- Time constraint: ~8 hours remaining in the hackathon
- We can run jobs up to 4 hours on the cluster
- The generation model is UNKNOWN (LLaMA? Mistral? OPT? — not stated in task)

## Output format requested

For each of the 6 research questions above:
1. What is the answer based on the literature?
2. What code/algorithm to use?
3. Expected improvement (if any empirical data exists)?
4. Estimated implementation time on an A800 GPU?

Also provide a PRIORITY RANKING of which directions to pursue first given the 8-hour constraint.
