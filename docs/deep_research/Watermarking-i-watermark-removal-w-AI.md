# AI watermarking attacks and detection: a hackathon technical brief

**Every published invisible watermark — for both LLM text and AI-generated images — has been successfully attacked in the academic literature, usually with black-box recipes that need no detector or model access.** For text, the strongest attacks are watermark stealing (Jovanović et al., ICML 2024) at ~$50 cost and recursive paraphrasing (Sadasivan et al., ICLR 2024) which drops KGW true-positive rate from 99% to 15%. For images, a single Stable-Diffusion regeneration pass removes 93–99% of pixel-level watermarks (Zhao et al., NeurIPS 2024) with PSNR ≥ 28 dB, and a paper-sized theoretical result proves this destruction is unavoidable for sufficiently invisible schemes. This brief gives you the math, the parameters, the GitHub repos, and a copy-paste code path to do all of it. The ordering follows the schemes you need to recognize, then the attacks that beat them, then the toolkits that compress this whole pipeline into a few function calls.

---

## 1. The KGW family — green-list watermarks for LLM text

The dominant LLM watermarking scheme is **Kirchenbauer et al. (ICML 2023, arXiv:2301.10226), "A Watermark for Large Language Models" (KGW)**. At each generation step *t*, a hash of the previous *h* tokens (default *h*=1) seeds a PRNG that pseudorandomly partitions the vocabulary *V* into a green list *G* of size γ|V| and a red list *R*. The **soft watermark** simply biases green-token logits by adding a constant δ before softmax:

$$\hat p^{(t)}_k \;\propto\; \exp(l^{(t)}_k + \delta\cdot\mathbb{1}[k\in G]).$$

Recommended defaults from the official repo (`jwkirchenbauer/lm-watermarking`) are **γ=0.25, δ=2.0**. The **hard watermark** sets the red-list probability to zero — strong detection but visible quality damage on low-entropy continuations like "Barack" → "Obama". Detection requires only the tokenizer and the hashing key. Count the green tokens *|s|*ᴳ in *T* generated tokens and compute the one-proportion z-score:

$$\boxed{\;z = \dfrac{|s|_G - \gamma T}{\sqrt{T\gamma(1-\gamma)}}\;}$$

The conventional decision threshold is **z > 4**, corresponding to a one-sided p-value of ≈ 3.17×10⁻⁵ under the null that text is unwatermarked. Empirically, with δ=2, γ=0.5, *T*=200 multinomial-sampled tokens, KGW reports type-II error ≤ 1.6% and zero observed type-I errors. Theorem 4.2 of the paper bounds expected green-token count via a "spike entropy" of the distribution; Theorem 4.3 bounds perplexity inflation by (1+(eᵟ-1)γ).

The **follow-up paper "On the Reliability of Watermarks for LLMs" (Kirchenbauer et al., arXiv:2306.04634, ICLR 2024)** generalizes the seeding scheme. Two flavors matter: **LeftHash** (the original *h*=1 scheme, where the green list at *t* depends only on token *t-1*) and **SelfHash** (Algorithm 1, where the green list depends on the candidate token itself plus prior *h*-1 tokens via an anchored-min-hash PRF). The currently recommended configuration is **γ=0.25, δ=2.0, h=4, PRF=selfhash, ignore_repeated_ngrams=True**. The repo string `seeding_scheme="selfhash"` expands to `"ff-anchored_minhash_prf-4-True-15485863"`. Crucially, larger *h* improves stealing-resistance but *worsens* edit-robustness: a single token edit randomizes the next *h* green lists.

## 2. Other LLM watermarking schemes worth knowing

**Unigram-Watermark (Zhao, Ananth, Li, Wang, ICLR 2024, arXiv:2306.17439)** simplifies KGW to *h=0*: the green list is **drawn once at the start of generation** and held fixed; logits get +δ for green tokens regardless of context. Defaults γ=0.5, δ=2.0, threshold often z>6. This trades stealing-resistance for paraphrase-robustness — a single edit only randomizes one green-status flag instead of *h*+1, so Unigram is roughly twice as robust to paraphrase as h=1 KGW. The code lives at `XuandongZhao/Unigram-Watermark`. Its weakness is exactly its strength: a fixed green list is easy to reverse-engineer from frequency analysis on watermarked outputs.

**The Aaronson / EXP scheme**, introduced in Scott Aaronson's 2022 OpenAI talk, uses **exponential-minimum (Gumbel-trick) sampling**: the previous tokens hash to seed a PRNG that produces ξ_{t,k} ~ Uniform(0,1) for each *k*∈*V*, then the next token is **x_t = argmax_k log(ξ_{t,k})/p_k**. The marginal token distribution is exactly *p* — the watermark is **distortion-free**. Detection uses Aaronson's score *S = Σ log(1/(1-ξ_{t,xt}))*, which is Gamma(*T*, 1)-distributed under the null. The scheme has no peer-reviewed primary source; details circulate via Aaronson's blog and through reproductions.

**EXP-edit / Kuditipudi-Thickstun-Hashimoto-Liang (TMLR 2024, arXiv:2307.15593)** keeps EXP's distortion-free property but uses a **fixed random key sequence** Ξ=(ξ₁,…,ξₙ) of length 256 by default, not a per-token hash. Detection is by **Levenshtein-aligned cost** between the candidate text and the key, with significance computed by a permutation test over surrogate keys. This is the most edit-robust distortion-free scheme published — detection from 35 tokens at *p*≤0.01 even after 40–50% random substitutions, and survives French/Russian round-trip translation. Code: `jthickstun/watermark`.

**SynthID-Text (Dathathri et al., Google DeepMind, Nature 634:818–823, October 2024)** is the production watermark deployed in Gemini. Its core innovation is **tournament sampling**: at each step, draw 2ᵐ candidates from the LM (*m*=30 layers in production), then run *m* pairwise rounds where each round's winner is decided by pseudorandom g-values g_ℓ(token, r_t) ∈ {0,1} keyed by an Aaronson-style hash *r_t* of the previous *H*=5 tokens. Configured with Bernoulli(0.5) g-values it is **distortion-free in expectation**. Detection is either a **Mean detector** (z-test against E[S]=0.5 under null) or a trained **Bayesian detector**. SynthID supports speculative decoding for zero-latency deployment. Code: `google-deepmind/synthid-text`, integrated natively into `transformers ≥ 4.46`.

**Christ-Gunn-Zamir (COLT 2024, arXiv:2306.09194)** achieves **cryptographic undetectability**: no PPT adversary without the secret key can distinguish watermarked from unwatermarked outputs. Construction reduces the LM to a binary stream and seeds each Bernoulli draw with a PRF *F_sk*. Distortion-free is *computational* (PRF security). Detection by empirical correlation has exponentially-small false-positive rate. The trade-off: any constant-rate edit destroys correlation, so robustness is poor.

**SemStamp (Hou et al., NAACL 2024, arXiv:2310.03991)** moves to **sentence-level** watermarking: embed each generated sentence via a paraphrase-invariant Sentence-T5/SimCSE encoder, partition embedding space via LSH (or *k*-means in k-SemStamp), and **rejection-sample sentences** until one falls in a γ-fraction "valid region" determined by the previous sentence. Detection is the same one-proportion z-test on the sentence count. Because semantic embeddings are paraphrase-invariant by design, SemStamp survives DIPPER better than token-level schemes. Code: `abehou/SemStamp`.

A useful **summary table** for implementation:

| Scheme | Defaults | Detection | Distortion-free | Repo |
|---|---|---|---|---|
| KGW soft (LeftHash) | γ=0.25, δ=2.0, h=1 | z-test, z>4 | No | jwkirchenbauer/lm-watermarking |
| KGW SelfHash | γ=0.25, δ=2.0, h=4 | same | No | same repo |
| Unigram | γ=0.5, δ=2.0 | z-test, z>6 | No | XuandongZhao/Unigram-Watermark |
| EXP/Aaronson | h=1–4 PRF | Σ log(1/(1-ξ)) Gamma test | Yes (statistical) | (in synthid-text baseline) |
| EXP-edit (KTH) | n=256 key | Levenshtein + permutation test | Yes | jthickstun/watermark |
| SynthID-Text | m=30, H=5 | mean or Bayesian detector | Configurable | google-deepmind/synthid-text |
| Christ-Gunn-Zamir | sec param λ | empirical correlation | Yes (cryptographic) | (no canonical) |
| SemStamp / k-SemStamp | d=3–8 LSH or k=8, γ=0.25 | sentence z-test | No | abehou/SemStamp |

## 3. Attacks on LLM watermarks ranked by power

**Watermark stealing (Jovanović, Staab, Vechev, ICML 2024, arXiv:2402.19361, repo `eth-sri/watermark-stealing`)** is the strongest published attack on the KGW family. With **only black-box query access** (no detector, no key), the attacker queries the victim with ~30,000 RealNewsLike prompts (~$50 OpenAI cost), generates a matched corpus from an open base model (Mistral-7B/Llama-2-7B), then for every (h+1)-gram estimates a green-likelihood score s(T, ctx) = clip(min(p̂_w/p̂_b, c), 0, 1). At generation time the attacker boosts logits by δ_att·s* to **spoof** human text into watermarked-looking, or by -δ_att to **scrub** watermarked text. Headline numbers at the strict FPR=10⁻³: **80–95% spoof success on KGW-soft, KGW2-SelfHash, KGW2-Sum, and Unigram**; >80% scrubbing on KGW2-SelfHash where prior baselines hit <25%. GPT-4-judge quality stays at 8.2–9.4/10 with perplexity comparable to clean generation. The result overturns the previously-believed spoof-vs-scrub robustness tradeoff: stealing breaks both simultaneously.

**Recursive paraphrasing (Sadasivan et al., ICLR 2024, arXiv:2303.11156)** applies DIPPER (or similar) *k* times in sequence. Against KGW soft-watermark, **TPR@1%FPR drops from 99% to 15% after 5 rounds**; against retrieval-based defenses, 100% → 25%; DetectGPT collapses 96.5% → 25.2% with even a single T5 paraphrase. The paper also proves an impossibility-style result: AUROC(D) ≤ 1/2 + TV(M, H) − TV(M, H)²/2 — as LMs approach the human distribution in total-variation distance, no detector can do much better than chance.

**DIPPER (Krishna et al., NeurIPS 2023, arXiv:2303.13408)** is the workhorse paraphraser: an 11B-parameter T5-XXL fine-tuned on PAR3 with two control codes *lexical diversity* (L) and *order diversity* (O), each in {0,20,…,100}. At the strongest setting **L=60, O=60**, KGW detection drops from 100% to 52.8% (OPT-13B base, 300-token generations) while semantic similarity P-SP stays at 0.946. The HF model `kalpeshk2011/dipper-paraphraser-xxl` requires ~45 GB GPU memory; a smaller substitute is `humarin/chatgpt_paraphraser_on_T5_base`.

**Cross-lingual round-trip translation (CWRA, He et al., ACL 2024, arXiv:2402.14007)** translates the prompt to a pivot language (Chinese/French), queries the victim, translates back. **Single round-trip drops KGW AUC from 0.95 to 0.61 and Unigram AUC to 0.54** — translation switches the entire vocabulary so *n*-gram overlap with watermarked text collapses.

The **emoji attack (Goodside, cited in KGW §5)** is essentially free. Prompt the watermarked LLM with: *"Insert 🟢 after every word."* Strip emojis from the output. With *h*=1 LeftHash, the watermarker seeds each green list using the *previous* token, which is now an emoji. After stripping, the detector reseeds using the actual content tokens, which produces a different green list — green-token rate collapses to ~γ and **z drops to ≈0**. Trivially defeats any *h*≥1 context-dependent KGW variant; harmless against Unigram (no context dependence).

**Color-Aware Substitution / SCTS (Wu & Chandrasekaran, ACL 2024, arXiv:2403.14719)** is a black-box token substitution that prompts the LLM itself to reveal its green tokens, then targets exactly those for swap. It achieves higher attack success rate than DIPPER under matched edit budgets (~10–20% edits). Code: `Qi-Pang/LLM-Watermark-Attacks`.

**T5 span replacement** at ε=0.3 drops KGW AUROC from 0.998 to 0.696 but at +15 perplexity (visible quality damage). **BERT-MLM synonym swap at 30%** breaks KGW heavily but Unigram only loses ~5% AUC — illustrating Unigram's edit-robustness advantage. **Copy-paste attacks** (10–25% watermarked text inside human writing) get AUC down to 0.78–0.93 but defenders can use a windowed *WinMax* z-score detector.

A consolidated effectiveness ranking for hackathon planning:

| Rank | Attack | Target | Detection drop | Quality | Access |
|---|---|---|---|---|---|
| 1 | Watermark Stealing (Jovanović) | KGW-Soft, SelfHash, Sum, Unigram | Spoof 80–95% / Scrub >80% @ FPR 1e-3 | GPT-4 judge 8.2–9.4 | Black-box (~$50) |
| 2 | Recursive paraphrasing (5×) | KGW family | TPR 99→15% | small PPL ↑ | Black-box |
| 3 | DIPPER L=60 O=60 | KGW family | 100→52.8% | semantic sim 0.94 | Black-box, 40GB GPU |
| 4 | CWRA round-trip translate | KGW, Unigram, SIR | AUC 0.95→0.54–0.67 | high | Black-box + MT |
| 5 | Color-aware (SCTS) | KGW LeftHash/SelfHash, Unigram | > DIPPER at matched budget | high | Black-box |
| 6 | Emoji attack | KGW h≥1 | z → ~0 | minor | Prompt-only |
| 7 | T5 span ε=0.3 | KGW soft | AUC 0.998→0.696 | PPL +15 | Black-box |

All major attacks are **black-box** — no published attack on KGW requires model weights or the secret key. The empirical state of the field as of 2024–2025 is pessimistic: every deployed scheme has a published defeat at academic-publication-quality.

## 4. Statistical detection without knowing the scheme

When the watermarking algorithm is unknown, the attacker (or auditor) can still detect *that* a watermark is present. **N-gram frequency analysis** trivially exposes Unigram-Watermark — green tokens have unnaturally high frequencies relative to a Common Crawl reference. KGW with *h*=1 needs conditional bigram analysis. Generic AI-text detectors like **DetectGPT, Fast-DetectGPT, GPTZero, Binoculars** detect AI-generated-ness rather than watermark-specifically, and are themselves vulnerable to paraphrasing (DetectGPT 70.3% → 4.6% under DIPPER). **Gloaguen et al. 2024-25** develops black-box presence-and-parameter tests for KGW, EXP, and Christ-Gunn-Zamir families — useful as a reconnaissance step before stealing.

The general decision framework everywhere uses the **one-proportion z-test** on a Bernoulli(γ) null with z = (|s|ᴳ − γT)/√(Tγ(1−γ)). Standard thresholds: z=4 (one-sided p=3.17×10⁻⁵, KGW default), z=6 (p≈10⁻¹⁰, Unigram default). For multi-category detectors (k-SemStamp clusters, multi-bit watermarks), use a **chi-square test** with *C*-1 d.f.. For Aaronson, the test statistic *S* is **Gamma(T,1)**-distributed under the null. For EXP-edit, the Levenshtein cost has no closed-form distribution; use a **permutation test** over ~10³ surrogate keys. **Repeated-n-gram filtering** (`ignore_repeated_ngrams=True`) is essential to avoid false positives on boilerplate.

---

## 5. Image watermarking — from LSB to diffusion-rooted signatures

Classical post-hoc watermarking divides into spatial methods (LSB substitution, additive spread-spectrum *I_w = I + αW*) and frequency methods (DCT, DWT, DFT). The **invisible-watermark** Python library used by Stable Diffusion implements two classical hybrids: **DwtDct** (2-level DWT → 4×4 block DCT → max-coefficient bit embed; ~300 ms on 1080p) and **DwtDctSvd** (adds SVD-based modulation of the first singular value). Both operate on the YUV U-channel and provide a 32–40 bit payload at PSNR ~42 dB. **Both are trivially broken by a 90° rotation followed by un-rotation, or by JPEG at low Q.**

DNN-based watermarks dominate the modern literature. **HiDDeN (Zhu, Kaplan, Johnson, Fei-Fei, ECCV 2018)** introduced the canonical **Encoder–Noise Layer–Decoder** architecture: an encoder concatenates a tiled message tensor with image features; a differentiable noise layer simulates JPEG/blur/crop/dropout during training; a decoder predicts the message. With a small adversarial discriminator pushing encoded images toward natural distribution, HiDDeN encodes 30 bits in 128×128 images at PSNR 32–35 dB and survives crops retaining only 3.5% of pixels. Code: `ando-khachatryan/HiDDeN`. **StegaStamp (Tancik, Mildenhall, Ng, CVPR 2020)** extends this with a *physical-world* noise model (perspective warp, motion/defocus blur, color manipulation), enabling **100-bit payloads recoverable from photographs of printed images**. Code: `tancik/StegaStamp`. **RivaGAN (Zhang et al. 2019)** uses an attention-based encoder to embed 32 bits surviving MPEG/scaling/cropping; the model ships in `ShieldMnt/invisible-watermark` as the `rivaGan` backend. **SSL Watermarking (Fernandez et al., ICASSP 2022 — note: ICASSP, not CVPR)** abandons end-to-end training and instead optimizes a perturbation δ at embedding time so that **DINO features** of *I*+δ project onto secret directions; this is resolution-agnostic and unusually robust to rotation and screenshots.

For diffusion models, **generation-time watermarks** are now standard. **Tree-Ring (Wen, Kirchenbauer, Geiping, Goldstein, NeurIPS 2023, arXiv:2305.20030)** modifies the **initial Gaussian noise z_T** in Fourier space: replace values inside a small ring mask with a fixed concentric-ring key pattern *K*, inverse-FFT, then run normal DDIM sampling. The watermark ends up *baked into the semantic content* rather than imposed on pixels. Detection runs **DDIM inversion** for ~50 steps to recover ẑ_T, FFTs it, and computes L1 distance to *K* inside the mask. Empirical robustness on SDv2.1 with 1000 prompts is striking: **TPR ≈ 100% under 75% center crop, TPR ≈ 95–98% under JPEG Q=25, TPR ≈ 92% under Gaussian noise σ=0.1**, all at FPR=1%, with FID change ≈0.2 versus baseline. The original paper is **zero-bit** (presence only); RingID extends to multi-key. Code: `YuxinWenRick/tree-ring-watermark`.

**Stable Signature (Fernandez et al., ICCV 2023, arXiv:2303.15435)** "roots" a 48-bit signature in the **VAE decoder of the latent diffusion model**: pre-train a HiDDeN-style extractor *W_D* once, then fine-tune the LDM decoder *D'* for ~1 minute per signature with loss BCE(W_D(D'(z)), m) + λ·perceptual. Every image emerging from *D'* now carries *m*, recoverable in one forward pass. Bit accuracy is 0.92 under combined crop50%+brightness+JPEG80, with ~40% successful 1-of-1000 user identification at FPR=10⁻⁶ and zero false-accusation. PSNR ≈ 30 dB, FID change ≤ 0.3. Code: `facebookresearch/stable_signature`.

**Gaussian Shading (Yang et al., CVPR 2024, arXiv:2404.04956)** is **provably performance-lossless**: it represents the watermark bits in a 4×64×64 latent (256-bit default capacity), randomizes them with ChaCha20, then samples each latent dimension from the *correct half* of N(0,1) — preserving the exact standard-Gaussian marginal. Since the latent distribution is unchanged, FID is unchanged. Detection inverts to the latent and threshold-decodes. Robustness exceeds Tree-Ring and Stable Signature on most attacks — bit accuracy 0.97 at JPEG Q=25, 0.93 at 60% crop, 0.95 at noise σ=0.05. Code: `bsmhmmlf/Gaussian-Shading`. **WaDiff (Min et al., ECCV 2024)** generalizes this with user-conditional input concatenation to the U-Net.

| Image scheme | Type | Bits | Detect | Quality | Survives |
|---|---|---|---|---|---|
| DwtDctSvd | Post-hoc, freq | ~32 | extractor | ~42 dB | mild JPEG; broken by crop+rotate |
| HiDDeN | Post-hoc, DNN | 30 | extractor | 32–35 dB | JPEG, 3.5% crop |
| StegaStamp | Post-hoc, DNN | 100 | extractor | 28–30 dB | print→photo |
| SSL Watermark | Post-hoc, DNN | 0/30 | DINO sign-test | 40 dB | rot, crop, screenshot |
| Tree-Ring | Gen-time (latent init) | 0 | DDIM inversion + FFT | FID Δ≈0.2 | crop 75%, JPEG 25, blur |
| Stable Signature | Gen-time (decoder fine-tune) | 48 | extractor | 30 dB | crop10%@FPR 10⁻⁶ |
| Gaussian Shading | Gen-time (latent sampling) | 256 | DDIM inversion | provably 0 quality loss | strong across the board |
| WaDiff | Gen-time (U-Net input) | 48 | extractor | similar to Stable Sig | comparable |

## 6. Image attacks: regeneration, adversarial, and benchmark results

The defining theoretical result is **Zhao, Zhang, Su, Vasan, Grishchenko, Kruegel, Vigna, Wang, Li, "Invisible Image Watermarks Are Provably Removable Using Generative AI" (NeurIPS 2024, arXiv:2306.01953, repo `XuandongZhao/WatermarkAttacker`)**. The **regeneration attack** is *x̂ = A(φ(x_w) + N(0, σ²I))*: embed (identity, VAE, or LDM latent), add Gaussian noise, reconstruct. For latent diffusion they use SDv2.1-base + PNDM with **noise steps = {10, 30, 50, 100, 150, 200}** corresponding to **σ = {0.10, 0.17, 0.23, 0.34, 0.46, 0.57}**. **Theorem 4.3 proves** that for any Δ-invisible watermark in ℓ₂, with σ ≈ 1.16·Δ, type-I+type-II error of *any* detector exceeds ≈1, i.e. detection becomes random. The proof connects to Gaussian Differential Privacy (Dong, Roth, Su 2022). Empirically, **regeneration removes 93–99% of DwtDctSvd, RivaGAN, and SSL watermarks while keeping PSNR ≥ ~30 and SSIM ≥ ~0.85** — versus ≤3% removal at the same quality from brightness/contrast/JPEG/noise/blur baselines. Stable Signature TPR collapses to 0; StegaStamp survives single regeneration better because it embeds larger perturbations.

**WAVES (An et al., ICML 2024, arXiv:2401.08573, repo `umd-huang-lab/WAVES`)** is the canonical benchmark: 26 attacks × 3 watermarks (Stable Signature, Tree-Ring, StegaStamp) × 3 datasets (DiffusionDB, MS-COCO, DALL·E-3, 5000 images each), evaluated at the strict **TPR@0.1% FPR**. Findings:

- **Single-pass diffusion regeneration (`Regen-Diff`, `Regen-DiffP`) destroys Stable Signature** (Avg P ≈ 0.000) and severely degrades StegaStamp (Avg P ≈ 0.94). Tree-Ring is more resistant to single-pass regeneration but **multi-pass `Rinse-2xDiff`/`Rinse-4xDiff` brings it down**.
- **Geometric combinations (rotation + resized-crop + erase, "DistCom-Geo") are the best classical-only attack on Tree-Ring** (Avg P 0.277).
- **Gaussian blur radius=4 single-handedly destroys Stable Signature** (Avg P 0.000) — a free attack against the ICCV 2023 scheme.
- Erase, brightness, contrast alone are useless against modern watermarks.
- The most stealth-efficient attacks are **adversarial embedding attacks**: PGD with ℓ∞=4/255 on a *surrogate* encoder. The **grey-box AdvEmbG-KLVAE8** (assumes victim uses the public KL-VAE-f8 from Stable Diffusion, which Tree-Ring/Stable Signature/DALL·E-3 all do) drops Tree-Ring TPR@0.1%FPR to ~0 with imperceptible perturbation and quality degradation < 0.15.

**Saberi et al. (ICLR 2024, arXiv:2310.00076, repo `mehrdadsaberi/watermark_robustness`)** introduced surrogate-model attacks: collect watermarked + non-watermarked images, train a ResNet-18 classifier, run PGD on it, transfer to the real detector. Tree-Ring AUROC drops to **0.47 at ε=4/255** — worse than random. WAVES extends this with three surrogate configurations: AdvCls-UnWM&WM (Saberi), AdvCls-Real&WM (largely fails — surrogate latches on AI-vs-real features), and the novel **AdvCls-WM1&WM2** that trains on watermarked images from two different user keys and effectively disrupts user identification.

**VAE-based regeneration via CompressAI** is the cheapest universal attack: load `bmshj2018_factorized` (Ballé et al., ICLR 2018) or `cheng2020-anchor` (Cheng et al., CVPR 2020) at compression quality 3, run one forward pass, and >99% of DwtDctSvd/RivaGAN/SSL watermarks are gone at PSNR 30–34. **Lukas et al. (ICLR 2024, arXiv:2309.16952)** and **WEvade (Jiang et al., arXiv:2305.03807)** provide adaptive white-box attacks for cases where the detector itself is queryable.

A practical attack ranking for image-watermark removal:

| Rank | Attack | Recipe | Removal | Quality |
|---|---|---|---|---|
| 1 | SD regeneration | SDv2.1-base, PNDM, t*=60–100 (σ≈0.23–0.34) | 93–99% on RivaGAN/DwtDctSvd/SSL; Stable Sig→0 | PSNR 28–32 |
| 2 | Rinse-2xDiff | repeat above twice, t*≈50 each | Tree-Ring P 1.0→0.5 | slightly lower |
| 3 | VAE regeneration (Bmshj2018 q=3) | one forward pass | >99% pixel WMs | PSNR 30–34 |
| 4 | AdvEmbG-KLVAE8 PGD | ε=4/255 on shared SD KL-VAE | Tree-Ring TPR→0 | imperceptible |
| 5 | DistCom-Geo (rot+crop+erase) | best classical | Tree-Ring Avg P 0.28 | moderate |
| 6 | Gaussian blur r=4 | single op | kills Stable Signature | heavy degradation |
| 7 | JPEG Q=25 | one save | bit-flips on weak schemes only | drops fast |

**Tree-Ring's robustness is contested in the literature**: Zhao et al. 2024 reports it survives single regeneration (TPR ≈ 1.0 in Table 2) because the Fourier ring lives in the semantic latent that diffusion sampling reconstructs; WAVES reports rinsed regeneration *plus* adversarial embedding breaks it. Both are correct under their threat models — the operational lesson is that single-pass *latent-noise* attacks miss the ring while *VAE-encoder* adversarial perturbations hit it.

---

## 7. Code, repos, and the hackathon toolchain

The single most useful starting point on the LLM side is **`THU-BPM/MarkLLM` (Pan et al., EMNLP 2024 demo, arXiv:2405.10051)** — a unified Python toolkit covering KGW, SWEET, EWD, Unbiased, DipMark, Unforgeable/SIR, EXP, and built-in attacks (WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, **DipperParaphraser**, RandomWalkAttack). One pattern fits everything:

```python
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

tc = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b"),
    tokenizer=AutoTokenizer.from_pretrained("facebook/opt-1.3b"),
    vocab_size=50272, device="cuda")
wm = AutoWatermark.load("KGW", algorithm_config="config/KGW.json", transformers_config=tc)
text = wm.generate_watermarked_text("The quick brown fox")
print(wm.detect_watermark(text))   # {'is_watermarked': True, 'score': 6.4}
```

If you'd rather avoid extra dependencies, **Hugging Face Transformers ≥ 4.46 has KGW and SynthID built in**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig
tok   = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
cfg   = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
out = model.generate(**tok(["This is the beginning of a long story"], return_tensors="pt"),
                     watermarking_config=cfg, do_sample=True, max_length=100)
det = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=cfg)
print(det(out, return_dict=True))   # contains z-score, prediction
```

For SynthID specifically, `pip install synthid-text` gives the reference implementation; the HF integration uses `SynthIDTextWatermarkingConfig(keys=[...30 ints...], ngram_len=5)`. For Kuditipudi-EXP-edit use `jthickstun/watermark` (note: pin `transformers<=4.30.1`). For sentence-level semantic watermarks use `abehou/SemStamp` with the encoders `AbeHou/SemStamp-c4-sbert` and `AbeHou/SemStamp-booksum-sbert`.

**A KGW detector code path that works against any HF tokenizer**:

```python
from transformers import AutoTokenizer
from extended_watermark_processor import WatermarkDetector  # from jwkirchenbauer repo

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
detector  = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25, seeding_scheme="selfhash",
    device="cpu", tokenizer=tokenizer,
    z_threshold=4.0, normalizers=[], ignore_repeated_ngrams=True)
print(detector.detect("Suspected machine-generated text..."))
# {'z_score': ..., 'p_value': ..., 'prediction': True/False}
```

**A DIPPER paraphrasing attack** (45 GB GPU; substitute `humarin/chatgpt_paraphraser_on_T5_base` for low-VRAM):

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
mdl = T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl").cuda().eval()
def dipper(text, lex=60, order=60, sent_interval=3):
    lex_code, ord_code = 100-lex, 100-order
    sents = sent_tokenize(text); out = ""
    for i in range(0, len(sents), sent_interval):
        chunk = " ".join(sents[i:i+sent_interval])
        prompt = f"lexical = {lex_code}, order = {ord_code} <sent> {chunk} </sent>"
        ids = tok(prompt, return_tensors="pt").to("cuda")
        o = mdl.generate(**ids, do_sample=True, top_p=0.75, max_new_tokens=256)
        out += " " + tok.batch_decode(o, skip_special_tokens=True)[0]
    return out
```

**The full image-watermarking ecosystem in one place**: `ShieldMnt/invisible-watermark` (DwtDct/DwtDctSvd/RivaGAN — used by Stable Diffusion itself; `pip install invisible-watermark`), `tancik/StegaStamp`, `ando-khachatryan/HiDDeN`, `facebookresearch/ssl_watermarking`, `YuxinWenRick/tree-ring-watermark`, `facebookresearch/stable_signature`, `bsmhmmlf/Gaussian-Shading`, `rmin2000/WaDiff`, plus Meta's umbrella suite `facebookresearch/meta-seal` (with `watermark-anything` for localized watermarks and `videoseal` for video). The **MarkDiffusion** sister-toolkit at `THU-BPM/MarkDiffusion` (`pip install markdiffusion`) covers Tree-Ring, RingID, ROBIN, WIND, SFW, Gaussian-Shading, PRC, SEAL, VideoShield, GaussMarker, VideoMark with 31 evaluation tools.

**`invisible-watermark` encode/decode** for any post-hoc watermarking experiments:

```python
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
bgr = cv2.imread("test.png")
enc = WatermarkEncoder(); enc.set_watermark("bytes", b"hackathon-id-001")
wm_img = enc.encode(bgr, "dwtDct")           # or "dwtDctSvd", "rivaGan"
cv2.imwrite("test_wm.png", wm_img)
dec = WatermarkDecoder("bytes", 16*8)
print(dec.decode(cv2.imread("test_wm.png"), "dwtDct").decode("utf-8"))
```

**A diffusion regeneration attack** in 12 lines using HF Diffusers:

```python
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.enable_model_cpu_offload()
init = Image.open("watermarked.png").convert("RGB").resize((512,512))
attacked = pipe(prompt="", image=init, strength=0.15, guidance_scale=1.0,
                num_inference_steps=50).images[0]
attacked.save("regenerated.png")
# strength≈0.06 mirrors Zhao's noise_step=60; raise to 0.3 for Tree-Ring
```

For VAE-only regeneration (cheaper, often sufficient against pixel watermarks):

```python
from compressai.zoo import bmshj2018_factorized
net = bmshj2018_factorized(quality=3, pretrained=True).cuda().eval()
out = net(img)["x_hat"]    # one forward pass = strong WM removal
```

For trivial baselines — JPEG attack `Image.save(buf, "JPEG", quality=25)`, Gaussian noise `arr + np.random.normal(0, sigma, arr.shape)`, geometric `torchvision.transforms.RandomResizedCrop(scale=(0.5,0.8))` plus `RandomRotation(±25°)` plus `RandomErasing(p=1.0)`.

The dedicated **attack repos** are `XuandongZhao/WatermarkAttacker` (regeneration: `DiffWMAttacker(noise_step=60)`), `mehrdadsaberi/watermark_robustness` (DiffPure + adversarial + spoofing across Tree-Ring/StegaStamp/dwtDct/dwtDctSvd/rivaGan/watermarkDM/MBRS), `eth-sri/watermark-stealing` (LLM stealing; run `bash setup.sh`, set `OAI_API_KEY`), and `umd-huang-lab/WAVES` (full benchmark with surrogate detector zoo at `huggingface.co/furonghuang-lab/WAVES-Models`).

## 8. Operational hackathon strategy

If you only have access to the **watermarked artifact and nothing else**: for images, try VAE regeneration with Bmshj2018 q=3 first (one line, fastest, >99% removal of pixel watermarks); escalate to SD2.1 regeneration with t*=60–100 if that fails; for Tree-Ring specifically, combine Rinse-2xDiff with PGD on the KL-VAE-f8 encoder at ε=4/255. For text, the **emoji attack is free and instantly defeats any KGW with h≥1** — start there if you suspect KGW; otherwise apply DIPPER at L=60 O=60 (or recursive DIPPER for paranoid robustness). If the watermark survives, paraphrase across a translation pivot (CWRA) — this defeats Unigram where emoji fails.

If you can **query the victim model**: invest in watermark stealing à la Jovanović. With ~$50 of API budget and a few hours of GPU time you can spoof or scrub at >80% success on KGW2-SelfHash, the strongest deployed text watermark before SynthID. The same data lets you train a surrogate classifier (Saberi-style) to attack image detectors.

If you have **detector-API access**: WEvade-style PGD with assigned messages on a local surrogate, then verify via the oracle. For LLM watermarks with a published z-score endpoint, Pang et al.'s exploit-the-strengths attacks become viable.

The published consensus across NeurIPS 2024, ICML 2024, ICLR 2024, ACL 2024 and Nature is uniform: **invisible watermarks for AI content are not security primitives.** They are forensic signals robust to incidental processing, not adversarial ones. For a hackathon CTF this means almost every challenge has a documented attack path, and the optimal strategy is to identify which scheme you face (n-gram analysis for text, latent inversion for diffusion) and pull the corresponding tool from the playbooks above.

## Conclusion

The 2023–2025 watermarking literature has settled into a stable picture: post-hoc pixel watermarks fall to a single VAE pass; latent-rooted diffusion watermarks fall to surrogate-encoder PGD or rinsed regeneration; KGW-family text watermarks fall to $50 of black-box stealing; even distortion-free schemes have been broken by adaptive prompting. **The strongest theoretical results — Zhao 2024's certified removability via Gaussian DP, Sadasivan 2023's TV-distance impossibility, Christ-Gunn-Zamir's robustness limits — all point to fundamental tradeoffs rather than implementation flaws.** The most surprising practical findings are inversions of intuition: Unigram beats KGW on paraphrase-robustness but loses to frequency analysis, smaller context width *h* makes scrubbing harder but stealing easier, single-pass diffusion regeneration spares Tree-Ring (because the watermark lives in the semantic latent the diffusion model reconstructs) but rinsed multi-pass attacks defeat it. For hackathon offense, the workflow that maximizes expected points per hour is: (1) identify the watermark family with a few black-box probes, (2) apply the matching first-line attack from §3 or §6, (3) escalate to stealing or rinsed regeneration only if needed. For defense, the practical lesson is that watermarking buys forensic traceability against accidental edits, not adversarial security — designing a CTF challenge as if it did invites trivial defeats.