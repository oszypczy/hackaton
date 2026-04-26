# Papers — quick lookup table

**Goal:** during hackathon Claude reads THIS FILE first, not full PDFs or deep-research artifacts.
PDFs are in this same directory. Index numbers below match filename prefix `NN_*.pdf`.

Total: **25 papers** (4 required + 4 supplementary + 11 hidden + 6 competition-ready tools added 2026-04-26 after researchy 01/02/03).

---

## 1. Required papers (from organizers' email)

### 01 — Carlini et al. 2023, *Extracting Training Data from Diffusion Models* (USENIX 2023)
- arXiv: 2301.13188
- **Core idea:** generate-and-filter pipeline; sample N images from DM, filter for memorized training samples via similarity matching
- **Key result:** >1000 training images extracted from SOTA diffusion models
- **Use for:** Challenge C primary baseline; any "did this DM memorize X?" task

### 02 — Maini, Jia, Papernot, Dziedzic 2024, *LLM Dataset Inference* (NeurIPS 2024) [**SprintML**]
- arXiv: 2406.06443 — Repo: github.com/pratyushmaini/llm_dataset_inference
- **Core idea:** single MIA features are weak (distribution-shift confounded); aggregate multiple weak features at dataset level via Welch t-test
- **Key result:** p < 0.1 with zero FP on Pile subsets
- **Use for:** Challenge A primary; the SprintML eval template (TPR@FPR + p-value + zero-FP)

### 03 — Zawalski et al. 2025, *CoDeC: Detecting Data Contamination via In-Context Learning* (NeurIPS Workshop 2025)
- arXiv: 2510.27055
- **Core idea:** in-context examples improve scores on novel data, *worsen* them on memorized data (disrupts memorization pattern)
- **Use for:** any "is this benchmark contaminated?" task; novel angle for Challenge A

### 04 — Kirchenbauer et al. 2023, *A Watermark for Large Language Models* (ICML 2023)
- Repo: github.com/jwkirchenbauer/lm-watermarking
- **Core formula:** before each token, hash prev-token to seed RNG, mark γ fraction as "green list", boost their logits by δ. Detect via z-score: `z = (n_green - γT) / sqrt(T·γ·(1-γ))`
- **Key params:** γ=0.25, δ=2.0 (defaults used in challenge B)
- **Use for:** Challenge B primary

## 2. Supplementary papers (already in repo)

### 05 — Survey: Model Inversion Attacks
### 06 — Survey: Model Extraction Attacks
### 07 — FGSM/PGD Defense Strategies
### 08 — *Watermarks Provably Removable* (NeurIPS 2024)
- **Core result:** invisible image watermarks can be removed via diffusion-based regeneration with bounded utility loss
- **Use for:** Challenge B B2 (removal); Challenge C reasoning about robustness

---

## 3. Hidden papers (added 2026-04-26 from deep-research recommendations)

### 09 — Dubiński, Kowalczuk, Boenisch, Dziedzic 2025, **CDI: Copyrighted Data Identification in Diffusion Models** (CVPR 2025) [**SprintML**]
- arXiv: 2411.12858 — Repo: **github.com/sprintml/copyrighted_data_identification**
- **Core idea:** aggregate per-sample MIA signals via statistical hypothesis test
- **Key result:** ≥99% confidence with **only ~70 samples**, p < 0.05 with zero FP
- **vs Carlini 01:** 70 samples vs thousands; statistical not heuristic; works on real SDv1.5
- **Use for:** Challenge C **hard mode**, very high probability hackathon target

### 10 — Kowalczuk, Dubiński, Boenisch, Dziedzic 2025, *Privacy Attacks on Image AutoRegressive Models* (ICML 2025) [**SprintML**]
- arXiv: 2502.02514 — Repo: **github.com/sprintml/privacy_attacks_against_iars**
- **Core idea:** IARs (VAR, MUSE etc.) leak much more training data than diffusion; novel MIA achieving 86.38% TPR@FPR=1% on VAR-d30 vs 6.38% for DMs
- **Key result:** dataset inference needs only 6 samples for IARs (vs 200 for DMs); extracts 698 training images
- **Use for:** if hackathon uses IAR not DM; deeply SprintML-flavored

### 11 — Dubiński, Pawlak, Boenisch, Trzciński, Dziedzic 2023, **B4B: Bucks for Buckets — Active Defenses Against Stealing Encoders** (NeurIPS 2023) [**SprintML**]
- arXiv: 2310.08571 — Repo: github.com/stapaw/b4b-active-encoder-defense
- **Core idea:** adversaries' representations cover larger fraction of embedding space than legit users → adaptively scale utility per user; per-user transformations defeat sybil aggregation
- **Use for:** Challenge E (model stealing) — most likely SprintML defense template; encoder stealing = SprintML core direction

### 12 — Hayes, ..., Boenisch, Dziedzic, Cooper et al. 2025, *Exploring the Limits of Strong Membership Inference Attacks on LLMs* (NeurIPS 2025) [**SprintML co-authored**]
- arXiv: 2505.18773
- **Core idea:** scaling LiRA-style "strong" attacks to LLMs with reference models; empirical limits study
- **Use for:** Challenge A hard mode; methodological gold-standard for MIA in 2025

### 13 — Hintersdorf, Struppek, Kersting, Dziedzic, Boenisch 2024, *Finding NeMo: Localizing Neurons Responsible For Memorization* (NeurIPS 2024) [**SprintML**]
- arXiv: 2406.02366 — Repo: github.com/ml-research/localizing_memorization_in_diffusion_models
- **Core idea:** memorization in DMs localizes to **single cross-attention neurons**, identifiable by outlier activation patterns
- **Use for:** Challenge C alternate framing; also gives a "remove memorization by ablating neurons" mitigation angle

### 14 — Dziedzic, Kaleem, Lu, Papernot 2022, *Increasing the Cost of Model Extraction with Calibrated Proof of Work* (ICLR 2022 Spotlight) [**SprintML**]
- arXiv: 2201.09243 — Repo: github.com/cleverhans-lab/model-extraction-iclr
- **Core idea:** force PoW per query, calibrated to information content via DP-style measurement; ~100× attacker overhead, ~2× legit
- **Use for:** Challenge E if defense involves PoW; understand the lab's own defense designs

### 15 — Carlini, Paleka, Dvijotham, Steinke et al. 2024, **Stealing Part of a Production Language Model** (ICML 2024 **Best Paper**)
- arXiv: 2403.06634
- **Core idea:** softmax bottleneck — logits = W·g(p), W is `l × h` with l>>h. SVD on stacked logit matrix recovers `col(W)` and reveals hidden dim h
- **Key result:** OpenAI Ada h=1024 (<$20), Babbage h=2048 (<$20), gpt-3.5-turbo full layer ~$2000
- **Use for:** Challenge E LLM variant; embedding extraction primer

### 16 — Podhajski, Dubiński, Boenisch, Dziedzic 2024, *Efficient Model-Stealing Attacks Against Inductive GNNs* (ECAI 2024 / AAAI 2026 Oral) [**SprintML**]
- arXiv: 2405.12295
- **Core idea:** unsupervised GNN extraction via graph contrastive learning + spectral graph augmentations; no labels needed from victim
- **Use for:** if hackathon has GNN challenge (SprintML active research direction)

### 17 — Xu, Boenisch, Dziedzic 2025, *ADAGE: Active Defenses Against GNN Extraction* (2025) [**SprintML**]
- arXiv: 2503.00065
- **Core idea:** companion defense to #16 — actively perturb GNN responses based on query patterns
- **Use for:** if GNN challenge has defense mechanism; understand attack/defense pair

### 18 — Carlini, Chien, Nasr, Song, Terzis, Tramèr 2022, *Membership Inference Attacks From First Principles* (S&P 2022) [**LiRA**]
- arXiv: 2112.03570
- **Core formula:** per-sample LRT — fit Gaussian to losses under "in" world (sample in training) and "out" world from shadow models, log-likelihood ratio is the score
- **Key insight:** AUC misleads; report **TPR@FPR=0.1% / 1%** instead. The metric SprintML universally adopted.
- **Use for:** Challenge A and D hard modes; the methodological backbone for all modern MIA

### 19 — Orekondy, Schiele, Fritz 2019, **Knockoff Nets: Stealing Functionality of Black-Box Models** (CVPR 2019)
- arXiv: 1812.02766
- **Core idea:** query victim with public-pool images, train surrogate via KD on (img, soft-label) pairs; OOD pool works (ImageNet → CUB)
- **Key result:** Caltech256 60k queries → 76% relative acc; ~$30 to extract Azure Emotion API
- **Use for:** Challenge E primary baseline; canonical functionality-stealing recipe

## 4. Competition-ready tools (added 2026-04-26 after researchy 01/02/03)

### 20 — Zhang et al. 2024, *Min-K%++: Improved Baseline for Pre-Training Data Detection* (ICLR 2025 spotlight)
- arXiv: 2404.02936
- **Core formula:** standardize per-token log-prob by per-position vocabulary distribution: `score(x_i) = (log p(x_i|x_<i) − μ_{·|x_<i}) / σ_{·|x_<i}`, mean over bottom k% (k=20)
- **Key result:** 6.2–10.5% AUC over Min-K% on WikiMIA
- **Use for:** Challenge A — **upgrade primary feature from loss/Min-K% to Min-K%++**

### 21 — Jovanović, Staab, Vechev 2024, **Watermark Stealing in LLMs** (ICML 2024)
- arXiv: 2402.19361 — Repo: github.com/eth-sri/watermark-stealing
- **Core idea:** ~30k black-box queries (~$50) to victim + corpus from open base model → estimate per-(h+1)-gram green likelihood → spoof or scrub
- **Key result:** Spoof 80–95% / scrub >80% on KGW-soft, KGW-SelfHash, Unigram at FPR=10⁻³; GPT-4 judge quality 8.2–9.4
- **Use for:** Challenge B advanced — strongest published attack on KGW family, both spoof + scrub

### 22 — Krishna et al. 2023, **DIPPER: Discourse Paraphraser** (NeurIPS 2023)
- arXiv: 2303.13408 — HF: kalpeshk2011/dipper-paraphraser-xxl
- **Core idea:** 11B T5-XXL paraphraser with two control codes (lexical L, order O), each ∈ {0,20,…,100}
- **Recommended config:** L=60, O=60 — KGW detection 100% → 52.8% with semantic sim 0.946
- **Use for:** Challenge B B2 (removal) — solid mode; needs ~45GB GPU (use teammate-CUDA)

### 23 — Sadasivan et al. 2024, *Can AI-Generated Text be Reliably Detected?* (ICLR 2024)
- arXiv: 2303.11156
- **Core result:** recursive paraphrasing (5×) drops KGW TPR@1%FPR from 99% to 15%; impossibility theorem `AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2`
- **Use for:** Challenge B B2 maximum mode + theoretical justification why removal is achievable

### 24 — An et al. 2024, **WAVES: Benchmarking Watermark Robustness** (ICML 2024)
- arXiv: 2401.08573 — Repo: github.com/umd-huang-lab/WAVES
- **Core idea:** 26 attacks × 3 watermarks (Stable Sig, Tree-Ring, StegaStamp) × 3 datasets, evaluated at TPR@0.1%FPR
- **Key result:** single-pass diffusion regen kills Stable Sig (Avg P ≈ 0.000); Gaussian blur radius=4 alone destroys Stable Sig; AdvEmbG-KLVAE8 grey-box drops Tree-Ring to ~0
- **Use for:** Challenge B image-side reference + benchmark protocol

### 25 — Nasr, Carlini, Hayase, Jagielski et al. 2023, *Extracting Training Data from ChatGPT* (preprint)
- arXiv: 2311.17035
- **Core idea:** prompt `Repeat the word "poem" forever` → alignment collapses → verbatim pretraining text emerges (~150× baseline rate)
- **Key result:** ≥10k unique memorized strings extracted from gpt-3.5-turbo for ~$200; 3% of post-divergence output found verbatim on Internet
- **Use for:** any LLM verbatim extraction challenge; complement to paper 03 (CoDeC) and 02 (Maini)

---

## 5. Mapping by challenge

| Challenge | Primary paper | Methodology / hard mode | SprintML adjacent | New tools (20–25) |
|---|---|---|---|---|
| **A — LLM Dataset Inference** | 02 Maini | 18 LiRA, 12 Strong MIAs | 09 CDI methodology | **20 Min-K%++** |
| **B — LLM Watermark** | 04 Kirchenbauer | 08 Watermarks Provably Removable | — | **21 Stealing**, **22 DIPPER**, **23 RecPara**, **24 WAVES** |
| **C — Diffusion Memorization** | 01 Carlini | **09 CDI** (alternative track), 13 NeMo | 10 IAR Privacy | 24 WAVES (image regen) |
| **D (opt) — Property Inference** | research 06 (no PDF) | 18 LiRA, 12 Strong MIAs | 09 CDI methodology | — |
| **E (opt) — Model Stealing** | 19 Knockoff | 11 B4B, 14 PoW, 15 Carlini LLM, 16/17 GNN | all of 11/14/15/16/17 | 25 ChatGPT divergence (LLM variant) |

## 5b. Quick-attack lookup (by scenario, from researchy 01/02/03)

**For LLM watermark removal (Challenge B B2), in increasing power:**
1. **Emoji attack** (free, prompt-only): `"Insert 🟢 after every word"` then strip — z drops to ~0 against any KGW with h≥1. **Try first.**
2. **CWRA round-trip translation**: pivot through Chinese/French — drops KGW AUC 0.95→0.61, kills Unigram (where emoji fails). Defeats all schemes; ~1–2h work.
3. **DIPPER L=60 O=60** (paper 22) — KGW 100→52.8% detection, semantic sim 0.946. Needs 45GB GPU.
4. **Recursive paraphrasing 5×** (paper 23) — KGW 99→15% TPR@1%FPR. Most robust scrub.
5. **Watermark stealing** (paper 21) — if API access: ~$50, 80–95% spoof+scrub on KGW-SelfHash, Unigram.

**For LLM verbatim extraction (Challenge A adjacent):**
1. **Carlini perplexity/zlib ratio** (paper 02 + 25): generate N≈2000, sort by `log(perplexity)/zlib_entropy`.
2. **Min-K%++** (paper 20) for membership: `score = (log p − μ)/σ` per token, mean bottom 20%. **Beats Min-K% by 6–10% AUC.**
3. **CoDeC** (paper 03) for dataset-level: in-context examples decrease scores on memorized data.
4. **Divergence attack** (paper 25): if challenge involves aligned LLM, prompt `Repeat "poem" forever`.

**For diffusion memorization (Challenge C):**
1. **CDI** (paper 09) — only needs ~70 samples for ≥99% confidence; SprintML own paper.
2. **Carlini generate-and-filter** (paper 01): N=500 samples per prompt, DBSCAN(eps=0.10, min_samples=10) on DINO features.
3. **Webster one-step** (cited in research 02): single denoising step suffices for template verbatims, orders-of-magnitude faster.
4. **NeMo** (paper 13): localize to ≤10 cross-attention neurons, ablate to remove.

**For image watermark removal (Challenge B image-side):**
1. **VAE regeneration** (1 line, fastest, >99% removal of pixel watermarks) — `compressai bmshj2018_factorized(quality=3)`.
2. **SD2.1 img2img regen** at strength=0.15 — kills Stable Sig (Avg P ≈ 0.000), severely degrades StegaStamp.
3. **Gaussian blur radius=4** — single-handedly destroys Stable Sig (free baseline).
4. **AdvEmbG-KLVAE8** grey-box adversarial (PGD ε=4/255 on KL-VAE-f8 encoder) — drops Tree-Ring to ~0.

## 5. Mapping by topic (for fast lookup)

| Topic | Papers |
|---|---|
| Membership inference | 02, 09, 10, 12, 18 |
| Model stealing (general) | 11, 14, 15, 16, 19 |
| Encoder / SSL stealing | 11 |
| GNN attack/defense | 16, 17 |
| LLM watermarking | 04, 08 |
| Diffusion memorization | 01, 09, 13 |
| Image autoregressive privacy | 10 |
| Statistical methodology (LiRA pattern) | 12, 18 |

## 6. SprintML author signatures

These come from Dziedzic and/or Boenisch labs — **most likely templates for Warsaw 2026 challenges**:

- **Dziedzic primary author:** 02, 14
- **Boenisch primary author:** —
- **Both as PIs:** 09, 10, 11, 12, 13, 16, 17
- **External co-authors with SprintML:** 12 (Hayes, Choquette-Choo)

Concentration of co-authors **Dubiński + Boenisch + Dziedzic** appears in: **09, 10, 11, 16**. These four are the highest-probability hackathon templates.

## 7. SprintML evaluation conventions (extracted across papers 02/09/10/11/12/13/14/16/17/18)

Patterns that appear in EVERY paper of this group — **the hackathon will likely score this way**:

- **Primary metric:** TPR@FPR=1% (occasionally 0.1%). AUC only as secondary.
- **p-value from explicit hypothesis test**, threshold p < 0.1, **zero false positives** on independent held-out models.
- **Calibrated likelihood ratios** rather than hard thresholds.
- **Shadow models obowiązkowe** — paper 12 explicitly names "fine-tuning attacks that avoid shadow models" as weak.
- **Distribution-shift confounding** is taken seriously (paper 02): held-out victim set will be in-distribution, you cannot exploit shortcuts.
- **Sample complexity reasoning** — papers 09/10 highlight sample-efficiency (70 / 6 samples) as a primary contribution.

## 8. Code repos to clone before hackathon

If we want zero-friction setup at start:

```bash
mkdir -p references/repos
cd references/repos
git clone https://github.com/sprintml/copyrighted_data_identification     # 09 CDI
git clone https://github.com/sprintml/privacy_attacks_against_iars        # 10 IAR
git clone https://github.com/stapaw/b4b-active-encoder-defense             # 11 B4B
git clone https://github.com/ml-research/localizing_memorization_in_diffusion_models  # 13 NeMo
git clone https://github.com/cleverhans-lab/model-extraction-iclr          # 14 PoW
git clone https://github.com/pratyushmaini/llm_dataset_inference          # 02 Maini
git clone https://github.com/jwkirchenbauer/lm-watermarking               # 04 Kirchenbauer
```

Total disk: ~500MB to ~2GB depending on bundled checkpoints. Worth it: most "I need to remember exactly how this attack works" questions can be answered by reading these repos' README + main attack file.

## 9. How to use this file (for Claude during hackathon)

When user asks "how do we approach this challenge / which method?":

1. Identify challenge type from question → look up section 4 (mapping by challenge)
2. For each relevant paper, read the 6-line summary above
3. Only open full PDF if entry signals "use for: this challenge" AND you need a specific formula/algo not in summary
4. **Do NOT load `docs/deep_research/04_*` / `05_*` / `06_*` unless explicitly asked** — those are 30–55KB each, this MAPPING is ~7KB and covers the same ground for action purposes
