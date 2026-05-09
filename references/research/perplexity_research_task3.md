# Blind Multi-Type LLM Watermark Detection: Deep Research Brief

*CISPA Hackathon — Warsaw, May 2026 | 8-hour constraint*

***

## Executive Summary

This brief answers all six research questions for blind detection of KGW, Liu/SIR (ICLR 2024), and Zhao/Unigram-Watermark (ICLR 2024) when no per-sample type label is available. Key findings: (1) SIR detection **requires** a pre-trained transform model + Compositional-BERT, but the approximate detection can be proxied; (2) Unigram default parameters are `watermark_key=0`, `fraction=0.5`, `strength=2.0` and green-list recovery from ~60 samples is statistically feasible; (3) KGW likely uses `selfhash` with `h=4`, `gamma=0.25`, and a non-default hash key — the `simple_1`/`h=1` scheme you tested is the *old* variant; (4) the best published black-box approach (Gloaguen et al., ICLR 2025) covers all three families but requires LM access; (5) green-list recovery from labeled samples is practical using chi-square or log-ratio tests; (6) the highest-impact untried approach is **multi-tokenizer parallel z-score sweeping + Unigram green-list recovery**, which can plausibly push TPR@1%FPR from 0.136 toward 0.3–0.5.

***

## 1. Liu et al. ICLR 2024 — SIR Watermark Detection

### Algorithm

The **Semantic Invariant Robust (SIR) watermark**  works fundamentally like KGW but replaces the hash-of-previous-tokens with a *semantic* mapping. At each generation step, the system:[^1][^2]

1. Computes a semantic embedding \(e_l = E(t_{:l-1})\) of all preceding tokens using an auxiliary embedding LLM
2. Passes \(e_l\) through a trained watermark transform model \(T\) to produce watermark logits: \(P_W = T(e_l)\)
3. Adds \(\delta \times P_W\) to the base LLM logits before sampling

Detection uses a z-score:[^3]

\[
z = \frac{1}{N} \sum_{j=1}^{N} P_W^{(t_j)}(x_{\text{prompt}}, t_{:j-1})
\]

where each \(P_W^{(t_j)}\) is the watermark logit value at the *actual sampled token position*. Without a watermark, the expected score is 0 (by construction of the normalization loss). With a watermark, the score substantially exceeds 0. The paper notes that after tanh scaling, watermark logits are nearly ±1, making SIR detection **fundamentally equivalent to KGW z-score detection** on a learned green list.[^3]

### Embedding Model (DEFAULT)

The default embedding model is **Compositional-BERT Large** (`perceptiveshawty/compositional-bert-large-uncased` on HuggingFace), with input dimension 1024.  The choice was made because it "better distinguishes dissimilar texts compared to original BERT." The pre-trained watermark transform model `transform_model_cbert.pth` must be loaded — it is a 4-layer fully-connected residual network trained on WikiText-103 embeddings.[^4]

### Default Parameters

From the official repository:[^4]
- `--watermark_type context`
- `--base_model gpt2`
- `--delta 1` (watermark strength)
- `--chunk_size 10` (embedding context window)
- `--embedding_model compositional-bert-large-uncased`
- `--transform_model model/transform_model_cbert.pth`

The vocabulary length for generating mappings is **50257** (GPT-2 tokenizer size).[^4]

### GitHub Repository

**Primary code**: `https://github.com/THU-BPM/Robust_Watermark`[^5][^4]

**Recommended (more user-friendly)**: `https://github.com/THU-BPM/MarkLLM` — the official MarkLLM toolkit contains a cleaner SIR implementation and is the authors' current recommendation.[^6]

### ArXiv ID Confirmation

ArXiv **2310.06356** is **correct** — "A Semantic Invariant Robust Watermark for Large Language Models," ICLR 2024 (accepted), submitted Oct 2023, revised May 2024.[^7][^1]

### Can Detection Work Without the Embedding Model?

**No, not directly.** The transform model \(T\) is required to compute \(P_W\) for each token. Without it, you cannot run the exact SIR z-score.

**However**, two approximate approaches are feasible:

1. **Train a new transform model on the training data** (2–4 hours on A800): Download Compositional-BERT, retrain the 4-layer watermark network on your own STS/WikiText embeddings using the official code. If the organizers used default training, your retrained model will approximate theirs.
2. **Treat SIR as a KGW-style green list** (approximate proxy): Because the tanh-scaled watermark logits are ≈ ±1, SIR reduces to a fixed green/red split *per semantic context*. The EMNLP 2024 paper "Revisiting the Robustness of Watermarking to Paraphrasing Attacks" shows that with ~200K tokens of watermarked output, SIR green lists can be predicted with >0.8 F1 score  — with 60 labeled samples (~12K–18K tokens), you may achieve partial recovery.[^8][^9]

### Python Code Snippet (SIR detection via MarkLLM)

```python
# Install: pip install markllm
from markllm.watermark.sir import SIRDetector
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# detector needs the transform model and embedding model
detector = SIRDetector(
    transform_model_path="model/transform_model_cbert.pth",
    embedding_model_name="perceptiveshawty/compositional-bert-large-uncased",
    device="cuda"
)
tokens = tokenizer.encode(text, return_tensors="pt")
z_score = detector.detect(tokens)  # returns z-score
```

**Estimated A800 implementation time**: 30 min to get code running + 2–4 hours to retrain transform model if needed. If using the pre-trained model from the repo, detection is ~0.34s/text.[^10]

***

## 2. Zhao et al. ICLR 2024 — Unigram-Watermark Detection

### Core Mechanism

Unigram-Watermark  is a **simplified fixed grouping strategy**: unlike KGW (where the green list changes per context), Unigram uses a **single, globally fixed green list** shared across all tokens in all contexts. This is the \(k=0\) limit of KGW, but with a cleaner implementation and formal proofs of robustness.[^11][^12][^13]

### Default Parameters (from official `gptwm.py`)

From the official GitHub code:[^14]

```python
class GPTWatermarkBase:
    def __init__(
        self,
        fraction: float = 0.5,      # gamma: 50% of vocabulary is green
        strength: float = 2.0,       # delta: logit boost
        vocab_size: int = 50257,     # GPT-2 tokenizer
        watermark_key: int = 0       # DEFAULT SEED IS 0
    ):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True]*int(fraction*vocab_size) + 
                        [False]*(vocab_size - int(fraction*vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
```

**Critical defaults**: `watermark_key=0`, `fraction=0.5`, `vocab_size=50257` (GPT-2), `strength=2.0`.[^14]

### Green List Generation

The green list is deterministic given `(watermark_key, fraction, vocab_size)`:
1. Compute `seed = SHA256(watermark_key)[:4]` (little-endian int32)
2. Create array: first `int(fraction * vocab_size)` entries = True, rest = False
3. Shuffle with `numpy.random.default_rng(seed).shuffle(mask)`

The mask is **identical** for every text generated with the same key — this is the defining feature of Unigram.[^11][^14]

### Detection Algorithm

```python
class GPTWatermarkDetector(GPTWatermarkBase):
    def detect(self, sequence: List[int]) -> float:
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return (green_tokens - fraction*len(sequence)) / 
               np.sqrt(fraction*(1-fraction)*len(sequence))
    
    def unidetect(self, sequence: List[int]) -> float:
        # Only count unique tokens (more robust to repetition)
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)
    
    def dynamic_threshold(self, sequence, alpha, vocab_size):
        # Adjusts threshold based on number of unique tokens
        z_score = self.unidetect(sequence)
        factor = np.sqrt(1 - (len(set(sequence))-1)/(vocab_size-1))
        tau = factor * norm.ppf(1-alpha)
        return z_score > tau, z_score
```

The paper recommends using `dynamic_threshold` with `alpha=0.01` for detection at 1% FPR.[^14]

### Does the Green List Depend on the Generation Model?

Yes and no. The green list depends only on `(watermark_key, fraction, vocab_size)`. If the organizers used GPT-2 tokenizer (vocab=50257) with `watermark_key=0`, the list is fixed. However, if they used a different tokenizer (LLaMA=32000, Mistral=32000, OPT=50272), the `vocab_size` would differ, producing a completely different green list structure. **Testing across multiple vocab sizes is essential.**

### Full Detection Code with Key Grid Search

```python
import numpy as np
import hashlib
from scipy.stats import norm
from transformers import AutoTokenizer

def get_green_list(watermark_key: int, fraction: float, vocab_size: int):
    key_bytes = np.int64(watermark_key)
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    mask = np.array([True]*int(fraction*vocab_size) + 
                    [False]*(vocab_size - int(fraction*vocab_size)))
    rng.shuffle(mask)
    return mask

def unigram_zscore(text: str, tokenizer, mask: np.ndarray, fraction: float) -> float:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    green = sum(1 for t in tokens if t < len(mask) and mask[t])
    n = len(tokens)
    if n == 0: return 0.0
    return (green - fraction * n) / np.sqrt(fraction * (1-fraction) * n)

# Grid search
for tokenizer_name in ["gpt2", "huggyllama/llama-7b", "mistralai/Mistral-7B-v0.1"]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    for watermark_key in [0, 1, 42, 100, 1234, 99999]:
        for fraction in [0.25, 0.5]:
            mask = get_green_list(watermark_key, fraction, vocab_size)
            # Score all labeled watermarked samples, compute mean z-score
            scores = [unigram_zscore(text, tokenizer, mask, fraction) 
                      for text in watermarked_train_texts]
            print(f"key={watermark_key}, frac={fraction}, vocab={vocab_size}: "
                  f"mean_z={np.mean(scores):.3f}")
```

**Estimated time**: 10 minutes to run the full grid search on 60 samples.

***

## 3. KGW Failure Diagnosis and Selfhash

### Why Default Detection Failed

The most likely explanations, in order of probability:

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| **Wrong seeding scheme** — you used `simple_1` (h=1), but the recommended scheme is `selfhash` (h=4) | Most likely; the KGW README explicitly recommends `selfhash` as default | Switch to `seeding_scheme="selfhash"` |
| **Non-default hash key** — `15485863` is the published key; organizers likely changed it | Per the README: "do not re-use this key if actually deploying" [^15] | Grid-search keys |
| **Wrong tokenizer** — default code uses whatever tokenizer you pass; LLaMA/Mistral texts tokenized with GPT-2 will give wrong green-list assignments | Symptoms: z-score noise around 0 regardless of key | Test with LLaMA/Mistral tokenizers |
| **Text too short** — KGW z-score is unreliable for <50 tokens; at low entropy it requires even more tokens [^16][^17] | Compute per-sample token lengths; check mean z-score vs. length | Filter by token count |
| **Low entropy text** — when model is "overconfident," green tokens are NOT preferentially sampled because the low-entropy output overrides watermark logits [^18][^16] | Compute per-token entropy of GPT-2; correlate with z-score | Use SWEET-style entropy-weighted z-score |

### SelfHash Explained

SelfHash (ICLR 2024 "Reliability" paper)  uses context of h=4 with min-hash PRF:[^19][^15]
```
seed = min(Hash(y_{t-3}), Hash(y_{t-2}), Hash(y_{t-1}), Hash(y_t)) × Hash(y_t) × key
```
The critical difference: it includes the **current token** in the seed, requiring checking all next-token candidates during generation. The recommended shorthand: `seeding_scheme="selfhash"` = `"ff-anchored_minhash_prf-4-True-15485863"`.[^15]

### KGW Parameter Enumeration Code

```python
from extended_watermark_processor import WatermarkDetector  # jwkirchenbauer/lm-watermarking

for tokenizer_name, vocab in [("gpt2", gpt2_tok), ("llama", llama_tok), ("mistral", mistral_tok)]:
    for seeding_scheme in ["simple_1", "selfhash", "minhash"]:
        for gamma in [0.25, 0.5]:
            for hash_key in [15485863, 0, 1, 42, 100, 33554393, 4294967291]:
                detector = WatermarkDetector(
                    vocab=list(vocab.get_vocab().values()),
                    gamma=gamma,
                    seeding_scheme=seeding_scheme,
                    device="cuda",
                    tokenizer=vocab,
                    z_threshold=4.0,
                    normalizers=[],
                    ignore_repeated_ngrams=True
                )
                scores = []
                for text in watermarked_train_texts:
                    try:
                        result = detector.detect(text)
                        scores.append(result['z_score'])
                    except: pass
                mean_z = np.mean(scores) if scores else 0
                print(f"{tokenizer_name}|{seeding_scheme}|γ={gamma}|key={hash_key}: "
                      f"mean_z={mean_z:.3f}, frac_pos={np.mean([s>4 for s in scores]):.3f}")
```

**Note**: WaterSeeker project lists `key=33554393` for KGW-llama and `key=4294967291` for KGW-mistral as practical examples.  These are worth testing.[^20]

**Estimated A800 time**: 30–60 minutes for full grid over 60 labeled samples.

***

## 4. Black-Box Multi-Watermark Detection State of the Art

### Gloaguen et al. (ICLR 2025) — "Black-Box Detection of Language Model Watermarks"

This is the most relevant paper for your problem.  Key contributions:[^21][^22][^23]

The paper develops **three independent statistical tests** — one per watermark family — that can be applied in a cascade to *identify which scheme was used*:

| Test | Target Family | Signal Exploited | Query Cost |
|------|--------------|-----------------|-----------|
| Red-Green test (§2) | KGW / SIR (both are red-green) | Context-dependent logit bias: varying context changes choice distribution | ~$3 |
| Fixed-Sampling test (§3) | Kuditipudi / Unigram-like | Reduced output diversity (rarefaction curve) | ~$0.30 |
| Cache-Augmented test (§4) | DIPMARK / DeltaReweight | Distribution changes between cached/uncached states | ~$0.10 |

In their experiments across 7 instruction-tuned models (Mistral-7B, LLaMA2 7/13/70B, etc.), the Red-Green test achieved **median p-value < 0.05** for all LEFTHASH and SELFHASH variants (δ=2–4, γ=0.25–0.5), with **p-values = 0 for all correct scheme assignments**.[^22]

**Crucial implication for your task**: KGW/SIR are both Red-Green family; Unigram is effectively a h=0 Red-Green (or Fixed-Sampling in their categorization). The tests can be adapted to **classify test samples** into scheme types without needing LLM access — instead, by analyzing the *statistical properties of the text itself*.

**However**: Their tests require black-box LLM access to re-query the model. You have pre-generated texts, not LLM access. The parameter estimation appendix (App. B.1) is more relevant: given identified scheme, estimate δ and h from text statistics alone.

### EMNLP 2024 — Paraphrasing Attacks and Green List Recovery

"Revisiting the Robustness of Watermarking to Paraphrasing Attacks"  demonstrates that both Unigram and SIR green lists can be **empirically recovered from text**. Key result: with ~200K watermarked tokens, Unigram green list recovery reaches **>0.8 F1**. With 60 samples × ~200 tokens = ~12K tokens, expect F1 ≈ 0.5–0.6 — likely sufficient for z-score improvement when combined with a learned threshold.[^9][^8]

### Blind Multi-Watermark Without Type Labels

No published method directly addresses the **exact setting** of: 60 samples per scheme, labels 0/1, no scheme labels. The closest approaches are:

1. **Feature-based clustering** (practical with your data): Extract per-sample statistics — green-token ratio under multiple (tokenizer, key, gamma) combinations, Binoculars PPL ratio, sentence similarity variance — then cluster with k-means(k=3) or GMM. If watermark types create distinct feature clusters, type-conditional detectors can be trained.

2. **Supervised with type-agnostic features**: Your current approach (LogReg on PPL/GLTR/green-ratio features). The gap to the leaderboard (0.136 vs. 0.27) suggests you're missing ~50% of watermarked samples, likely the SIR-watermarked ones (hardest to detect without the transform model) or KGW with non-default parameters.

***

## 5. Empirical Green List Recovery for Unigram

### Why It Works

Because Unigram uses a **fixed global green list** (same mask for all tokens regardless of context), green tokens appear more frequently in watermarked text than in clean text. This frequency imbalance is statistically detectable.[^8][^9]

### Statistical Method

For each token \(t\) in the vocabulary:

**Step 1**: Count frequency in watermarked samples: \(f_w(t)\) = #occurrences in watermarked corpus / total watermarked tokens

**Step 2**: Count frequency in clean samples: \(f_c(t)\) = #occurrences in clean corpus / total clean tokens

**Step 3**: Compute log-ratio: \(\text{score}(t) = \log\left(\frac{f_w(t) + \epsilon}{f_c(t) + \epsilon}\right)\)

**Step 4**: Apply chi-square test or Fisher's exact test per token to identify statistically significant over-representation:

\[
\chi^2(t) = \frac{(f_w(t) \cdot N_c - f_c(t) \cdot N_w)^2}{(f_w(t) + f_c(t)) \cdot N_w \cdot N_c}
\]

**Step 5**: Rank tokens by score; top `fraction × vocab_size` tokens = estimated green list

### Sample Complexity Estimate

With 60 watermarked samples (~200 tokens each = 12,000 tokens) and 180 clean samples (~36,000 tokens), and assuming GPT-2 vocabulary (50K tokens), most tokens appear rarely. **Expected F1 of green list recovery ≈ 0.4–0.6** based on the EMNLP 2024 result extrapolated to 12K tokens.[^9][^8]

**Key insight**: Precision matters more than recall here. A green list with precision=0.7, recall=0.4 will still give a meaningfully elevated z-score distribution on true Unigram samples vs. clean text.

### Effect on Z-Score Detection

If the empirically recovered green list has precision \(p\) and recall \(r\) vs. the true green list (with true fraction \(\gamma=0.5\)):

- The effective "green fraction" observed on watermarked text is approximately \(\gamma \cdot r + (1-\gamma) \cdot (1-p) \cdot \gamma\)
- Signal is degraded but not eliminated; even F1 ≈ 0.5 recovery gives detectable z-scores

### Code

```python
from collections import defaultdict
import numpy as np
from scipy.stats import chi2_contingency
from transformers import AutoTokenizer

def recover_green_list(watermarked_texts, clean_texts, tokenizer, gamma=0.5):
    wm_counts = defaultdict(int)
    clean_counts = defaultdict(int)
    wm_total = clean_total = 0
    
    for text in watermarked_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for t in tokens:
            wm_counts[t] += 1
        wm_total += len(tokens)
    
    for text in clean_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for t in tokens:
            clean_counts[t] += 1
        clean_total += len(tokens)
    
    vocab_size = tokenizer.vocab_size
    scores = {}
    for t in range(vocab_size):
        wm_f = wm_counts.get(t, 0)
        cl_f = clean_counts.get(t, 0)
        if wm_f + cl_f < 5:  # skip very rare tokens
            scores[t] = 0.0
            continue
        log_ratio = np.log((wm_f / wm_total + 1e-9) / (cl_f / clean_total + 1e-9))
        # chi-square contingency test
        contingency = [[wm_f, wm_total - wm_f], [cl_f, clean_total - cl_f]]
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            scores[t] = log_ratio if p_val < 0.05 else 0.0
        except:
            scores[t] = 0.0
    
    # Select top gamma fraction as estimated green list
    sorted_tokens = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    n_green = int(gamma * vocab_size)
    estimated_green = set(sorted_tokens[:n_green])
    
    # Build binary mask
    mask = np.zeros(vocab_size, dtype=bool)
    for t in estimated_green:
        mask[t] = True
    return mask, scores

# Run for multiple tokenizers
for tok_name in ["gpt2", "huggyllama/llama-7b"]:
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    mask, _ = recover_green_list(watermarked_train, clean_train, tokenizer, gamma=0.5)
    # Now use mask for z-score on test set
```

**Note**: You have 60 labeled watermarked samples (Unigram type unknown), but since labels are 0/1 without type, you must use all 60 watermarked training samples — some are KGW/SIR, not Unigram. This adds noise. Recommended mitigation: cluster training samples first (see Direction D), then apply green-list recovery only to the Unigram-cluster.

**Estimated A800 time**: 15 minutes (tokenization + frequency counting + scoring).

***

## 6. Alternative Features Not Yet Tried

### Priority Features for Immediate Implementation

| Feature | Rationale | Expected Gain | Implementation Effort |
|---------|-----------|---------------|----------------------|
| **Multi-tokenizer green-ratio sweep** (GPT-2 + LLaMA + Mistral) with gamma∈{0.25,0.5} | Unigram/KGW signal is tokenizer-dependent; correct tokenizer gives strong z-scores | High — if correct tokenizer found, z-score gives strong Unigram/KGW signal | 1 hour |
| **KGW selfhash z-score** (seeding_scheme="selfhash", h=4, multiple keys) | Your h=1 tests all failed; selfhash is the recommended scheme in the "Reliability" paper | Medium-High — if the organizer used selfhash | 1–2 hours |
| **Empirical green-list z-score** (from Section 5) | Directly recovers Unigram signal even without knowing the key | Medium — degrades gracefully with imperfect recovery | 1 hour |
| **Token-level entropy filtering + weighted z-score** (SWEET-style) | KGW watermark is invisible at low-entropy tokens; filtering them out improves z-score | Medium — especially relevant if texts have mixed entropy | 2 hours |
| **Green-ratio consistency across text windows** | Watermarked text shows consistent green ratio in all windows; clean text varies randomly | Low-Medium — adds a variance feature rather than a mean feature | 1 hour |
| **SIR approximate z-score via retrained transform model** | Direct SIR detection; mandatory if SIR samples are in the 60 | High — but requires 2–4 hours GPU training | 2–4 hours |
| **RoBERTa/DeBERTa fine-tuned on train set** | With 360 samples, fine-tuning a small classifier is feasible; pre-trained "AI-detection" weights provide initialization | Medium — but risky with only 360 samples and mixed watermark types | 3–4 hours |

### On RoBERTa/DeBERTa Fine-Tuning

Prior work  shows that RoBERTa-large fine-tuned on binary watermark detection has **"strong performance at low token count"** and improves over statistical z-tests at short sequence lengths. The RoBERTa classifier also handles mixed distributions better than a single z-score.[^24]

With 360 training samples: recommended settings are lr=2e-5, batch_size=16, epochs=5–10, with class-balanced sampling. Using a pre-trained "human vs. AI" model (e.g., `Hello-SimpleAI/chatgpt-detector-roberta`) as initialization may provide a useful inductive bias. **However**, the multi-watermark nature means RoBERTa learns both "watermark presence" and "AI-generated text" signals — which may or may not align with your goal.

Estimated A800 time for fine-tuning: **1–1.5 hours** (5 epochs × ~360 samples = fast).

### On Entropy-Based Detection (SWEET / EWD)

The standard KGW z-score fails on low-entropy text because at low entropy, green tokens are *not* preferentially selected (the LM distribution dominates the small watermark boost).  The SWEET/EWD methods weight each token's contribution to the z-score by its entropy:[^16][^17]

\[
z_{\text{entropy}} = \frac{\sum_j w(H_j) \cdot \mathbf{1}[t_j \in G]}{\sqrt{\sum_j w(H_j)^2 \cdot \gamma(1-\gamma)}}
\]

where \(H_j\) is the entropy of the LM distribution at step \(j\). This is already partially captured by your GPT-2 log-prob features (branch_a), but the explicit entropy weighting is different and worth adding as a separate feature.

***

## Priority Ranking for 8-Hour Constraint

Given your current score (0.136) and the leaderboard target (~0.27), here is the recommended action order:

### PRIORITY 1 (0–2 hours): KGW SelfHash Grid Search + Unigram Key Grid Search

**Rationale**: Your entire KGW branch is producing no signal because you used `simple_1` (h=1). Switching to `selfhash` with the correct tokenizer will immediately fix 1 of the 3 watermark types. Simultaneously, Unigram's default key is `0` — try `watermark_key=0` with GPT-2 tokenizer first.

```
Action: Run the parameter grid (Section 3 code) on 60 labeled training watermarked samples.
Goal: Find any (seeding_scheme, gamma, hash_key, tokenizer) that gives mean_z > 2.0 on watermarked vs clean.
Expected: If organizers used default selfhash with key=15485863 or key=0, signal appears immediately.
```

### PRIORITY 2 (2–3 hours): Empirical Unigram Green List Recovery

**Rationale**: If the key grid search doesn't find Unigram's key, empirical recovery from the 60 labeled samples still yields a usable signal. Even partial recovery (F1≈0.4) adds a new feature dimension uncorrelated with your existing features.

```
Action: Run Section 5 code for vocab sizes {50257, 32000}.
Goal: Compute per-sample empirical green-ratio z-scores; add as new feature to LogReg.
Expected: +10–20% TPR@1%FPR if Unigram accounts for ~1/3 of test samples.
```

### PRIORITY 3 (3–4 hours): SIR Transform Model Retraining

**Rationale**: Liu/SIR is currently proxied by sentence-transformer cosine similarity (branch_d), but that's a very weak proxy. Retraining the actual SIR transform model on the official code (2–3 hours on A800) gives exact SIR z-scores.

```
Action: Clone https://github.com/THU-BPM/Robust_Watermark, install deps, run Steps 1-2 from README.
Goal: Get transform_model_cbert.pth trained on STS embeddings.
Expected: Strong SIR signal on the ~1/3 of test samples that are Liu-watermarked.
```

### PRIORITY 4 (4–5.5 hours): RoBERTa Fine-Tuning on All 360 Samples

**Rationale**: If targeted z-scores are now available for at least 2 of the 3 types, adding them as features to a fine-tuned classifier can push scores further. Alternatively, train a RoBERTa-base binary classifier on raw text.

```
Action: Fine-tune roberta-base (or start from Hello-SimpleAI/chatgpt-detector-roberta) on 360 samples.
Expected: +5–15% TPR@1%FPR if the classifier learns scheme-agnostic signals.
```

### PRIORITY 5 (5.5–7 hours): Mixture-of-Experts Combination

**Rationale**: Once targeted detectors exist for all 3 types, combine them via soft-assignment.

1. Cluster the 360 training samples using k-means (k=3) on all extracted features
2. Assign each cluster to a watermark type by checking which detector gives highest mean z-score on that cluster
3. For each test sample: compute all 3 detector scores + soft-assign to types based on proximity to cluster centroids
4. Final score = weighted average of per-type scores

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Build feature matrix: [kgw_zscore, unigram_zscore, sir_zscore, binoculars, fast_detectgpt, ...]
X_train = build_features(train_texts)  
# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_train[watermarked_mask])
# Assign cluster → type by checking which detector has highest mean z-score per cluster
# ...
# For test: soft distances to cluster centers
X_test = build_features(test_texts)
dists = kmeans.transform(X_test)  # shape: (n_test, 3)
soft_weights = softmax(-dists, axis=1)  # closer cluster = higher weight
final_score = (soft_weights[:,0] * kgw_scores + 
               soft_weights[:,1] * unigram_scores + 
               soft_weights[:,2] * sir_scores)
```

***

## Diagnostic Checklist Before Each Submission

1. Confirm mean z-score on labeled watermarked training samples > 2.0 for at least one detector
2. Confirm clean training sample z-scores are centered around 0 (mean < 0.5)
3. Check Spearman correlation between new feature and old features (if corr > 0.9, the new feature adds little)
4. Score distribution on validation set: watermarked mean vs. clean mean separation should be visible

***

## Summary Table

| Research Question | Key Finding | Actionable Code | Priority |
|---|---|---|---|
| Liu/SIR detection | Requires Compositional-BERT + transform model; z-score = avg watermark logit; delta=1, chunk_size=10 | MarkLLM `SIRDetector` | P3 |
| Unigram default params | `watermark_key=0`, `fraction=0.5`, `vocab=50257` (GPT-2); fixed global mask | `gptwm.py` grid search | P1 |
| KGW failure | Used `simple_1`/h=1; should use `selfhash`/h=4; also try keys 33554393, 4294967291 | `extended_watermark_processor.py` grid | P1 |
| Black-box multi-type | Gloaguen et al. ICLR 2025: 3 tests for 3 families; but requires LLM access. From text only: feature clustering | None (LLM access needed) | P4-P5 |
| Green list recovery | Chi-square log-ratio on token frequencies; 60 samples → F1≈0.4–0.6; usable z-score proxy | `recover_green_list()` function above | P2 |
| Alternative features | Multi-tokenizer z-score sweep, entropy-weighted z-score, RoBERTa fine-tune | Grid search + fine-tuning code | P1-P4 |

---

## References

1. [A Semantic Invariant Robust Watermark for Large Language Models](https://arxiv.org/abs/2310.06356) - In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack...

2. [A Semantic Invariant Robust Watermark for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1a2131ebe25bd55e4fc734126ea583ed-Abstract-Conference.html) - In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack...

3. [Published as a conference paper at ICLR 2024](https://openreview.net/pdf/1a74e99e172febdbf13f3718c54ebb254bb73b39.pdf)

4. [GitHub - THU-BPM/Robust_Watermark: Code and data for paper "A Semantic Invariant Robust Watermark for Large Language Models" accepted by ICLR 2024.](https://github.com/THU-BPM/Robust_Watermark) - Code and data for paper "A Semantic Invariant Robust Watermark for Large Language Models" accepted b...

5. [A Semantic Invariant Robust (SIR) Watermark for Large Language ...](https://github.com/thu-bpm/robust_watermark) - We recommend using MarkLLM, our official toolkit for LLM watermarking, which provides a more user-fr...

6. [GitHub - THU-BPM/MarkLLM: MarkLLM: An Open-Source Toolkit for LLM Watermarking.（EMNLP 2024 Demo）](https://github.com/thu-bpm/markllm) - MarkLLM: An Open-Source Toolkit for LLM Watermarking.（EMNLP 2024 Demo） - THU-BPM/MarkLLM

7. [A Semantic Invariant Robust Watermark for Large Language Models](https://openreview.net/forum?id=6p8lpe4MNf) - We propose a semantic invariant watermarking method for large language models that provides both att...

8. [Revisiting the Robustness of Watermarking to ...](https://aclanthology.org/anthology-files/pdf/emnlp/2024.emnlp-main.1005.pdf)

9. [[PDF] Revisiting the Robustness of Watermarking to Paraphrasing Attacks](https://aclanthology.org/2024.emnlp-main.1005.pdf)

10. [[PDF] Token-Specific Watermarking with Enhanced Detectability and ...](https://icml.cc/media/icml-2024/Slides/34750_t6ar8a9.pdf)

11. [Provable Robust Watermarking for AI-Generated Text - OpenReview](https://openreview.net/forum?id=SsmT8aO45L) - We propose a robust and high-quality watermark method, Unigram-Watermark, by extending an existing a...

12. [Provable Robust Watermarking for AI-Generated Text - Hugging Face](https://huggingface.co/papers/2306.17439) - Unigram-Watermark is a robust and high-quality watermark method for ... Code is available at https:/...

13. [Provable Robust Watermarking for AI-Generated Text](https://openreview.net/pdf?id=ucTe1eiLc6) - X Zhao · Cytowane przez 353 — We propose a robust and high-quality watermark method, UNIGRAM-WATERMA...

14. [Unigram-Watermark/gptwm.py at main · XuandongZhao/Unigram-Watermark](https://github.com/XuandongZhao/Unigram-Watermark/blob/main/gptwm.py) - [ICLR 2024] Provable Robust Watermarking for AI-Generated Text - XuandongZhao/Unigram-Watermark

15. [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) - For the context width, h, we recommend a moderate value, i.e. h=4, and as a default PRF we recommend...

16. [An Entropy-based Text Watermarking Detection Method](https://arxiv.org/html/2403.13485v1) - For example, the low-entropy tokens in a machine-generated text could all appear to be red, and caus...

17. [An Entropy-based Text Watermarking Detection Method](https://arxiv.org/pdf/2403.13485.pdf)

18. [[PDF] An Entropy-based Text Watermarking Detection Method](https://aclanthology.org/2024.acl-long.630.pdf)

19. [[PDF] Watermark Stealing in Large Language Models](https://files.sri.inf.ethz.ch/website/papers/jovanovic2024watermarkstealing.pdf) - We are the first to demonstrate a practical attack on the most prominent KGW2-SELFHASH scheme, previ...

20. [GitHub - THU-BPM/WaterSeeker: Code and data for paper "WaterSeeker: Pioneering Efficient Detection of Watermarked Segments in Large Language Models." accepted by NAACL 2025.](https://github.com/THU-BPM/WaterSeeker) - Code and data for paper "WaterSeeker: Pioneering Efficient Detection of Watermarked Segments in Larg...

21. [Black-Box Detection of Language Model Watermarks](http://arxiv.org/pdf/2405.20777.pdf) - ...augmenting LLM generations with later detectable signals. Recent work has
proposed multiple famil...

22. [Published as a conference paper at ICLR 2025](https://openreview.net/pdf?id=E4LAVLXAHW)

23. [Black-Box Detection of Language Model Watermarks](https://openreview.net/forum?id=E4LAVLXAHW&noteId=zPHm3NSEiN) - Watermarking has emerged as a promising way to detect LLM-generated text, by augmenting LLM generati...

24. [[PDF] Improving Detection of Watermarked Language Models - OpenReview](https://openreview.net/pdf/0e19b19cdfbf2072c4f0b328cdc2608a84c2bddf.pdf)

