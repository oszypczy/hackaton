# Papers — router (read first)

**Goal:** during hackathon Claude reads THIS FILE first, then greps `txt/*.txt`, then surgical Read with `offset`/`limit`. Never load full PDF when `.txt` exists.

Total: **25 papers** (4 required + 4 supplementary + 11 hidden + 6 competition-ready tools, last batch added 2026-04-26 after researchy 01/02/03). All extracted to `txt/NN_*.txt` via `scripts/extract_papers.sh`.

Per-paper entry format:
- File / token estimate / arXiv / repo
- **Use for** — query routing in one line
- **Key sections** — section names + grep terms unique to that paper
- **Core idea / Key result** — 1 sentence each
- **Cross-refs** — which other papers extend or use this

---

## 1. Required papers (organizers' email)

### 01 — Carlini et al. 2023, *Extracting Training Data from Diffusion Models* (USENIX 2023)
- File: `txt/01_carlini2023_extracting_training_data_diffusion.txt` | ~18k | arXiv: 2301.13188
- **Use for:** Challenge C primary baseline; "did this DM memorize X?"
- **Key sections** (grep in file):
  - "3.1 Threat Model" — terms: `threat model`, `eidetic memorization`
  - "4.2 Extracting Data from Stable Diffusion" — terms: `350 million`, `Stable Diffusion`, `clique`
  - Memorization def — terms: `ℓ2 distance`, `near-duplicate`
- **Core idea:** generate-and-filter — sample N images per prompt, cluster via `ℓ2`, keep clusters whose centroid matches a training example.
- **Key result:** >1000 training images extracted from SOTA diffusion models.
- **Cross-refs:** 09 CDI (statistical alternative), 13 NeMo (neuron-level), 25 Nasr ChatGPT (LLM analogue).

### 02 — Maini, Jia, Papernot, Dziedzic 2024, *LLM Dataset Inference* (NeurIPS 2024) [**SprintML**]
- File: `txt/02_maini2024_llm_dataset_inference.txt` | ~17k | arXiv: 2406.06443 | Repo: github.com/pratyushmaini/llm_dataset_inference
- **Use for:** Challenge A primary; SprintML eval template (TPR@FPR + p-value + zero-FP)
- **Key sections** (grep in file):
  - Method / aggregation — terms: `Welch`, `t-test`, `dataset inference`, `aggregator`
  - Features — terms: `Min-K%`, `zlib`, `loss attack`
  - Pile experiments — terms: `Pile`, `p-value`, `false positive`
- **Core idea:** single MIA features are weak (distribution-shift confounded); aggregate multiple weak features at dataset level via Welch t-test.
- **Key result:** p < 0.1 with zero FP on Pile subsets.
- **Cross-refs:** 18 LiRA (sample-level baseline), 20 Min-K%++ (drop-in upgrade), 12 Strong MIA.

### 03 — Zawalski et al. 2025, *CoDeC: Detecting Data Contamination via In-Context Learning* (NeurIPS Workshop 2025)
- File: `txt/03_zawalski2025_codec_data_contamination.txt` | ~27k | arXiv: 2510.27055
- **Use for:** "is this benchmark contaminated?"; novel angle for Challenge A
- **Key sections** (grep in file):
  - Score definition — terms: `logprob without context`, `logprob difference with context`, `cumulative difference`
  - Method — terms: `in-context examples`, `disrupts memorization`
- **Core idea:** in-context examples improve scores on novel data, *worsen* them on memorized data (disrupts memorization pattern).
- **Cross-refs:** 02 Maini (dataset inference paradigm), 25 Nasr (LLM extraction).

### 04 — Kirchenbauer et al. 2023, *A Watermark for Large Language Models* (ICML 2023)
- File: `txt/04_kirchenbauer2023_watermark_llm.txt` | ~24k | Repo: github.com/jwkirchenbauer/lm-watermarking
- **Use for:** Challenge B primary
- **Key sections** (grep in file):
  - Soft watermark — terms: `green list`, `hash`, `δ`, `γ`, `bias`
  - Detection — terms: `z-score`, `null hypothesis`, `H0`
  - Attacks — terms: `paraphrasing`, `attack`, `robust`
- **Core formula:** before each token, hash prev-token to seed RNG, mark γ fraction as "green", boost their logits by δ. Detect via `z = (n_green − γT) / sqrt(T·γ·(1−γ))`.
- **Key params:** γ=0.25, δ=2.0 (defaults).
- **Cross-refs:** 08 Provably Removable, 21 Watermark Stealing, 22 DIPPER, 23 Recursive Para, 24 WAVES.

## 2. Supplementary papers

### 05 — Survey: Model Inversion Attacks
- File: `txt/05_survey_model_inversion_attacks.txt` | ~35k
- **Use for:** taxonomy lookup only; cite when needing terminology
- **Grep terms:** `model inversion`, `gradient`, `attribute inference`

### 06 — Survey: Model Extraction Attacks
- File: `txt/06_survey_model_extraction_attacks.txt` | ~13k
- **Use for:** Challenge E orientation
- **Key sections:** "3 Taxonomy", "4.1 Functionality Extraction", "4.2 Training Data Extraction"

### 07 — FGSM/PGD Defense Strategies
- File: `txt/07_fgsm_pgd_defense_strategies.txt` | ~9k
- **Use for:** adversarial-example baseline only (low priority — not in SprintML signal)
- **Grep terms:** `FGSM`, `PGD`, `defensive distillation`, `adversarial training`

### 08 — *Watermarks Provably Removable* (NeurIPS 2024)
- File: `txt/08_neurips2024_watermarks_provably_removable.txt` | ~20k
- **Use for:** Challenge B B2 (removal); reasoning about robustness
- **Key sections** (grep in file):
  - "3.1 Identity Embedding with Denoising Reconstruction" — terms: `denoising`, `regeneration`
  - "3.2 VAE Embedding and Reconstruction" — terms: `VAE`, `reconstruction`
- **Core result:** invisible image watermarks removable via diffusion-based regeneration with bounded utility loss.
- **Cross-refs:** 24 WAVES (benchmark), 22/23 (text analogue).

---

## 3. Hidden papers (SprintML 2022–2026, fetched 2026-04-26)

### 09 — Dubiński, Kowalczuk, Boenisch, Dziedzic 2025, **CDI** (CVPR 2025) [**SprintML**]
- File: `txt/09_cdi_dubinski2025.txt` | ~23k | arXiv: 2411.12858 | Repo: github.com/sprintml/copyrighted_data_identification
- **Use for:** Challenge C **hard mode** — top hackathon target
- **Grep terms:** `Members`, `Non-members`, `dataset inference`, `Welch`, `SDv1.5`, `70 samples`
- **Core idea:** aggregate per-sample MIA signals via statistical hypothesis test (CDI = Copyrighted Data ID).
- **Key result:** ≥99% confidence with **only ~70 samples**, p < 0.05 with zero FP. Beats Carlini 01 (thousands of samples → tens).
- **Cross-refs:** 02 Maini methodology, 01 Carlini (replaces), 13 NeMo (alternate framing).

### 10 — Kowalczuk, Dubiński, Boenisch, Dziedzic 2025, *Privacy Attacks on IARs* (ICML 2025) [**SprintML**]
- File: `txt/10_iar_privacy_kowalczuk2025.txt` | ~24k | arXiv: 2502.02514 | Repo: github.com/sprintml/privacy_attacks_against_iars
- **Use for:** if hackathon uses IAR (VAR/MUSE/RAR) not DM
- **Grep terms:** `VAR-d30`, `RAR-XL`, `MinMaxScaler`, `IAR`, `86.38`, `698`
- **Core idea:** IARs leak much more training data than DMs; novel MIA achieving 86.38% TPR@FPR=1% on VAR-d30.
- **Key result:** dataset inference needs only 6 samples for IARs (vs 200 for DMs); extracts 698 training images.
- **Cross-refs:** 09 CDI (DM analogue).

### 11 — Dubiński et al. 2023, **B4B: Bucks for Buckets** (NeurIPS 2023) [**SprintML**]
- File: `txt/11_b4b_dubinski2023.txt` | ~18k | arXiv: 2310.08571 | Repo: github.com/stapaw/b4b-active-encoder-defense
- **Use for:** Challenge E (model stealing) — likeliest SprintML defense template
- **Key sections** (grep in file):
  - "3.1 Threat Model and Intuition" — terms: `coverage`, `embedding space`
  - "3.2 Coverage Estimation" — terms: `Sybil`, `coverage estimation`
  - "3.3 Cost Function Design"
  - "3.4 Per-User Representation Transformations"
- **Core idea:** adversaries cover larger fraction of embedding space than legit users → adaptively scale utility per user; per-user transformations defeat Sybil aggregation.
- **Cross-refs:** 14 PoW (companion defense), 19 Knockoff (attack baseline), 17 ADAGE (GNN analogue).

### 12 — Hayes, ..., Boenisch, Dziedzic, Cooper et al. 2025, *Strong MIAs on LLMs* (NeurIPS 2025) [**SprintML co-authored**]
- File: `txt/12_strong_mia_hayes2025.txt` | ~50k | arXiv: 2505.18773
- **Use for:** Challenge A hard mode; methodological gold standard
- **Key sections** (grep in file):
  - "3.1 How many reference models" — terms: `reference models`, `LiRA`
  - "3.2 Compute-optimal model" — terms: `compute-optimal`
  - "4 Varying compute budget" — terms: `epochs`, `dataset size`
- **Core idea:** scaling LiRA-style strong attacks to LLMs with reference models; empirical limits.
- **Cross-refs:** 18 LiRA (foundation).

### 13 — Hintersdorf et al. 2024, *NeMo* (NeurIPS 2024) [**SprintML**]
- File: `txt/13_nemo_hintersdorf2024.txt` | ~20k | arXiv: 2406.02366 | Repo: github.com/ml-research/localizing_memorization_in_diffusion_models
- **Use for:** Challenge C alternate framing; mitigation via neuron ablation
- **Key sections** (grep in file):
  - "3.1 Quantifying Memorization Strength" — terms: `cross-attention`, `memorization strength`
  - "3.2 Initial Candidate Selection" — terms: `outlier`, `candidate neurons`
  - "3.3 Refinement"
- **Core idea:** memorization in DMs localizes to **single cross-attention neurons**, identifiable by outlier activations.
- **Cross-refs:** 01 Carlini, 09 CDI.

### 14 — Dziedzic, Kaleem, Lu, Papernot 2022, *Calibrated PoW* (ICLR 2022 Spotlight) [**SprintML**]
- File: `txt/14_pow_dziedzic2022.txt` | ~22k | arXiv: 2201.09243 | Repo: github.com/cleverhans-lab/model-extraction-iclr
- **Use for:** Challenge E if defense involves PoW; lab's own defense designs
- **Grep terms:** `proof of work`, `PoW`, `calibration`, `information content`, `differential privacy`
- **Core idea:** force PoW per query, calibrated to information content via DP-style measurement; ~100× attacker overhead, ~2× legit.
- **Cross-refs:** 11 B4B, 19 Knockoff (attack to bypass).

### 15 — Carlini et al. 2024, **Stealing Part of a Production LM** (ICML 2024 **Best Paper**)
- File: `txt/15_carlini_stealing_llm2024.txt` | ~24k | arXiv: 2403.06634
- **Use for:** Challenge E LLM variant; embedding extraction primer
- **Grep terms:** `softmax bottleneck`, `SVD`, `hidden dim`, `logit`, `gpt-3.5`, `ada`, `babbage`
- **Core idea:** softmax bottleneck — `logits = W·g(p)`, W is `l × h` with l>>h. SVD on stacked logit matrix recovers `col(W)` and reveals hidden dim.
- **Key result:** OpenAI Ada h=1024 (<$20), Babbage h=2048 (<$20), gpt-3.5-turbo full layer ~$2000.
- **Cross-refs:** 25 ChatGPT divergence (companion attack).

### 16 — Podhajski, Dubiński, Boenisch, Dziedzic 2024, *Efficient GNN Stealing* (ECAI 2024 / AAAI 2026 Oral) [**SprintML**]
- File: `txt/16_gnn_extract_podhajski2024.txt` | ~14k | arXiv: 2405.12295
- **Use for:** GNN challenge (SprintML active direction)
- **Key sections** (grep in file):
  - "4.1 Attack Taxonomy" — terms: `Type I`, `Type II`, `inductive`
  - "5.2 Spectral Graph Augmentations" — terms: `spectral`, `t-SNE`, `contrastive`
  - "6.1/6.2 Evaluation"
- **Core idea:** unsupervised GNN extraction via graph contrastive learning + spectral augmentations; no labels from victim.
- **Cross-refs:** 17 ADAGE (defense), 19 Knockoff.

### 17 — Xu, Boenisch, Dziedzic 2025, *ADAGE* [**SprintML**]
- File: `txt/17_adage_xu2025.txt` | ~26k | arXiv: 2503.00065
- **Use for:** GNN challenge with defense mechanism
- **Key sections** (grep in file):
  - "2.1 Notations", "2.2 Preliminaries"
  - "3.3 Design of ADAGE" — terms: `query rate`, `τ`, `perturbation`, `Algorithm 1`
- **Core idea:** companion defense to #16 — actively perturb GNN responses based on query patterns.
- **Cross-refs:** 16 (paired).

### 18 — Carlini et al. 2022, *MIA From First Principles* (S&P 2022) [**LiRA**]
- File: `txt/18_lira_carlini2022.txt` | ~21k | arXiv: 2112.03570
- **Use for:** Challenge A and D hard modes; backbone for all modern MIA
- **Grep terms:** `TPR @ 0.1% FPR`, `shadow model`, `Gaussian`, `likelihood ratio`, `online attack`, `offline`
- **Core formula:** per-sample LRT — fit Gaussian to losses under "in" world (sample in training) and "out" world from shadow models, log-likelihood ratio is the score.
- **Key insight:** AUC misleads; report **TPR@FPR=0.1% / 1%** instead. SprintML's universal metric.
- **Cross-refs:** 12 Strong MIA (LLM extension), 02 Maini.

### 19 — Orekondy, Schiele, Fritz 2019, **Knockoff Nets** (CVPR 2019)
- File: `txt/19_knockoff_orekondy2019.txt` | ~15k | arXiv: 1812.02766
- **Use for:** Challenge E primary baseline
- **Grep terms:** `knockoff`, `random`, `KD`, `victim`, `surrogate`, `Caltech256`, `truncated`
- **Core idea:** query victim with public-pool images, train surrogate via KD on (img, soft-label) pairs; OOD pool works (ImageNet → CUB).
- **Key result:** Caltech256 60k queries → 76% relative acc; ~$30 to extract Azure Emotion API.
- **Cross-refs:** 11 B4B (defense), 14 PoW (defense), 16 GNN.

## 4. Competition-ready tools (added 2026-04-26)

### 20 — Zhang et al. 2024, *Min-K%++* (ICLR 2025 spotlight)
- File: `txt/20_minkpp_zhang2025.txt` | ~14k | arXiv: 2404.02936
- **Use for:** Challenge A — **upgrade primary feature from loss/Min-K% to Min-K%++**
- **Grep terms:** `Min-K%++`, `μ`, `σ`, `WikiMIA`, `Zlib`, `bottom`, `standardize`
- **Core formula:** standardize per-token log-prob: `score(x_i) = (log p(x_i|x_<i) − μ_{·|x_<i}) / σ_{·|x_<i}`, mean over bottom k% (k=20).
- **Key result:** 6.2–10.5% AUC over Min-K% on WikiMIA.
- **Cross-refs:** 02 Maini (drop-in feature), 18 LiRA.

### 21 — Jovanović, Staab, Vechev 2024, **Watermark Stealing in LLMs** (ICML 2024)
- File: `txt/21_watermark_stealing_jovanovic2024.txt` | ~25k | arXiv: 2402.19361 | Repo: github.com/eth-sri/watermark-stealing
- **Use for:** Challenge B advanced — strongest published attack on KGW family (spoof + scrub)
- **Grep terms:** `spoof`, `scrub`, `green list`, `30k queries`, `KGW-soft`, `KGW-SelfHash`, `Unigram`, `s⋆`, `FPR⋆`
- **Core idea:** ~30k black-box queries (~$50) to victim + corpus from open base model → estimate per-(h+1)-gram green likelihood → spoof or scrub.
- **Key result:** Spoof 80–95% / scrub >80% on KGW-soft, KGW-SelfHash, Unigram at FPR=10⁻³; GPT-4 judge quality 8.2–9.4.
- **Cross-refs:** 04 Kirchenbauer (target).

### 22 — Krishna et al. 2023, **DIPPER** (NeurIPS 2023)
- File: `txt/22_dipper_krishna2023.txt` | ~26k | arXiv: 2303.13408 | HF: kalpeshk2011/dipper-paraphraser-xxl
- **Use for:** Challenge B B2 (removal) — solid mode; needs ~45 GB GPU (use CUDA-teammate)
- **Key sections** (grep in file):
  - "3 Building a controllable discourse paraphraser" — terms: `lexical`, `order`, `L=`, `O=`
  - "4.3 Attacking AI-generated text detectors"
  - "4.4 Alternative paraphrasing attacks"
- **Recommended config:** L=60, O=60 — KGW detection 100% → 52.8% with semantic sim 0.946.
- **Cross-refs:** 04 Kirchenbauer (target), 23 RecPara (escalation).

### 23 — Sadasivan et al. 2024, *Recursive Paraphrasing* (ICLR 2024)
- File: `txt/23_recursive_paraphrase_sadasivan2024.txt` | ~27k | arXiv: 2303.11156
- **Use for:** Challenge B B2 maximum mode + theoretical justification
- **Key sections** (grep in file):
  - "2.3 Paraphrasing Attacks on Watermarked AI Text"
  - "3 Spoofing Attacks on Generative AI-text Models"
  - "4 Hardness of Reliable AI Text Detection" — terms: `AUROC`, `TV(M,H)`, `impossibility`
- **Core result:** recursive paraphrasing (5×) drops KGW TPR@1%FPR from 99% to 15%; impossibility theorem `AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2`.
- **Cross-refs:** 22 DIPPER (paraphraser), 04 Kirchenbauer.

### 24 — An et al. 2024, **WAVES** (ICML 2024)
- File: `txt/24_waves_benchmark_an2024.txt` | ~24k | arXiv: 2401.08573 | Repo: github.com/umd-huang-lab/WAVES
- **Use for:** Challenge B image-side reference + benchmark protocol
- **Grep terms:** `Tree-Ring`, `Stable Sig`, `StegaStamp`, `Regen-VAE`, `Regen-Diff`, `AdvEmbG-KLVAE8`, `26 attack`
- **Core idea:** 26 attacks × 3 watermarks × 3 datasets, evaluated at TPR@0.1%FPR.
- **Key result:** single-pass diffusion regen kills Stable Sig (Avg P ≈ 0.000); Gaussian blur radius=4 alone destroys Stable Sig; AdvEmbG-KLVAE8 grey-box drops Tree-Ring to ~0.
- **Cross-refs:** 08 Provably Removable.

### 25 — Nasr, Carlini, Hayase, Jagielski et al. 2023, *Extracting Training Data from ChatGPT* (preprint)
- File: `txt/25_chatgpt_divergence_nasr2023.txt` | ~68k | arXiv: 2311.17035
- **Use for:** any LLM verbatim extraction; complement to 03 (CoDeC) and 02 (Maini)
- **Key sections** (grep in file):
  - "3.2 Attack Methodology" — terms: `Repeat`, `poem`, `divergence`
  - "3.4 Estimating Total Memorization"
  - "3.5 Discoverable vs Extractable Mem"
  - "4 Extracting Data from Semi-closed Models"
- **Core idea:** prompt `Repeat the word "poem" forever` → alignment collapses → verbatim pretraining text emerges (~150× baseline rate).
- **Key result:** ≥10k unique memorized strings extracted from gpt-3.5-turbo for ~$200; 3% of post-divergence output found verbatim on Internet.
- **Cross-refs:** 15 Stealing Part (companion), 01 Carlini DM.

---

## 5. Quick-attack lookup (by scenario)

**For LLM watermark removal (Challenge B B2), in increasing power:**
1. **Emoji attack** (free, prompt-only): `"Insert 🟢 after every word"` then strip — z drops to ~0 against any KGW with h≥1. **Try first.**
2. **CWRA round-trip translation**: pivot through Chinese/French — drops KGW AUC 0.95→0.61, kills Unigram (where emoji fails). ~1–2h work.
3. **DIPPER L=60 O=60** (paper 22) — KGW 100→52.8% detection, semantic sim 0.946. Needs 45 GB GPU.
4. **Recursive paraphrasing 5×** (paper 23) — KGW 99→15% TPR@1%FPR. Most robust scrub.
5. **Watermark stealing** (paper 21) — if API access: ~$50, 80–95% spoof+scrub on KGW-SelfHash, Unigram.

**For LLM verbatim extraction (Challenge A adjacent):**
1. **Carlini perplexity/zlib ratio** (paper 02 + 25): generate N≈2000, sort by `log(perplexity)/zlib_entropy`.
2. **Min-K%++** (paper 20): `score = (log p − μ)/σ` per token, mean bottom 20%. **Beats Min-K% by 6–10% AUC.**
3. **CoDeC** (paper 03): in-context examples decrease scores on memorized data.
4. **Divergence attack** (paper 25): if challenge involves aligned LLM, prompt `Repeat "poem" forever`.

**For diffusion memorization (Challenge C):**
1. **CDI** (paper 09) — only ~70 samples for ≥99% confidence; SprintML own paper.
2. **Carlini generate-and-filter** (paper 01): N=500/prompt, DBSCAN(eps=0.10, min_samples=10) on DINO features.
3. **Webster one-step** (cited in research 02): single denoising step suffices for template verbatims.
4. **NeMo** (paper 13): localize to ≤10 cross-attention neurons, ablate to remove.

**For image watermark removal (Challenge B image-side):**
1. **VAE regeneration** — `compressai bmshj2018_factorized(quality=3)`. 1 line, fastest, >99% removal of pixel watermarks.
2. **SD2.1 img2img regen** at strength=0.15 — kills Stable Sig (Avg P ≈ 0.000), severely degrades StegaStamp.
3. **Gaussian blur radius=4** — single-handedly destroys Stable Sig (free baseline).
4. **AdvEmbG-KLVAE8** grey-box adversarial (PGD ε=4/255 on KL-VAE-f8 encoder) — drops Tree-Ring to ~0.

---

## 6. Mapping by challenge

| Challenge | Primary | Hard mode / methodology | New tools |
|---|---|---|---|
| **A — LLM Dataset Inference** | 02 Maini | 18 LiRA, 12 Strong MIA, 09 CDI methodology | **20 Min-K%++** |
| **B — LLM Watermark** | 04 Kirchenbauer | 08 Provably Removable | **21 Stealing**, **22 DIPPER**, **23 RecPara**, **24 WAVES** |
| **C — Diffusion Memorization** | 01 Carlini | **09 CDI**, 13 NeMo, 10 IAR | 24 WAVES |
| **D (opt) — Property Inference** | research 06 | 18 LiRA, 12 Strong MIA, 09 CDI | — |
| **E (opt) — Model Stealing** | 19 Knockoff | 11 B4B, 14 PoW, 15 Carlini LLM, 16/17 GNN | 25 ChatGPT divergence |

## 7. Mapping by topic

| Topic | Papers |
|---|---|
| Membership inference | 02, 09, 10, 12, 18 |
| Model stealing (general) | 11, 14, 15, 16, 19 |
| Encoder / SSL stealing | 11 |
| GNN attack/defense | 16, 17 |
| LLM watermarking | 04, 08, 21, 22, 23 |
| Image watermarking | 08, 24 |
| Diffusion memorization | 01, 09, 13 |
| Image autoregressive privacy | 10 |
| LiRA-style methodology | 12, 18 |

## 8. SprintML author signatures

These come from Dziedzic and/or Boenisch labs — **most likely templates for Warsaw 2026**:

- **Dziedzic primary author:** 02, 14
- **Both as PIs:** 09, 10, 11, 12, 13, 16, 17
- **External co-authors with SprintML:** 12 (Hayes, Choquette-Choo)

Concentration of **Dubiński + Boenisch + Dziedzic** appears in: **09, 10, 11, 16**. Highest-probability hackathon templates.

## 9. SprintML evaluation conventions

Patterns appearing in EVERY SprintML paper (02/09/10/11/12/13/14/16/17/18) — **the hackathon will likely score this way**:

- Primary metric: **TPR@FPR=1%** (occasionally 0.1%). AUC only as secondary.
- p-value from explicit hypothesis test, threshold p < 0.1, **zero false positives** on independent held-out models.
- Calibrated likelihood ratios rather than hard thresholds.
- Shadow models obowiązkowe (paper 12 explicitly names shadow-free attacks as weak).
- Distribution-shift confounding taken seriously (paper 02): in-distribution held-out victim set.
- Sample-complexity reasoning (papers 09/10 highlight 70 / 6 samples as primary contribution).

## 10. Code repos to clone before hackathon

```bash
mkdir -p references/repos && cd references/repos
git clone https://github.com/sprintml/copyrighted_data_identification     # 09 CDI
git clone https://github.com/sprintml/privacy_attacks_against_iars        # 10 IAR
git clone https://github.com/stapaw/b4b-active-encoder-defense            # 11 B4B
git clone https://github.com/ml-research/localizing_memorization_in_diffusion_models  # 13 NeMo
git clone https://github.com/cleverhans-lab/model-extraction-iclr         # 14 PoW
git clone https://github.com/pratyushmaini/llm_dataset_inference          # 02 Maini
git clone https://github.com/jwkirchenbauer/lm-watermarking               # 04 Kirchenbauer
```

Total disk: ~500 MB to ~2 GB depending on bundled checkpoints.

## 11. How to use this file (for Claude during hackathon)

When user asks "how do we approach this challenge / which method?":

1. Identify challenge type → look up section 6 (mapping by challenge).
2. For each relevant paper: Grep the listed terms in `txt/NN_*.txt`.
3. Read `txt/NN_*.txt` with `offset`/`limit` for surgical reads. Never load full PDF when `.txt` exists.
4. Open `docs/deep_research/0N_*.md` only if MAPPING + grep + offset-Read didn't answer (those are 30–55 KB, ~7–14k tokens).
