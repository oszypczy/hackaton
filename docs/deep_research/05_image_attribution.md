# Generative-model image attribution: a deep technical playbook for CISPA 2026

The state of the art in AI-image attribution is now a layered ensemble problem rather than a single-classifier problem: low-level forensic primitives (frequency peaks at VAE-stride harmonics, neighbor-pixel residuals, autoencoder reconstruction error) carry the architectural fingerprint, while frozen foundation backbones (CLIP-ViT-L/14, DINOv3, MetaCLIP-2) supply the semantic prior that lets detectors generalize across unseen generators. **For a 24-hour hackathon, the dominant strategy is a HEDGE-style ensemble of UniversalFakeDetect (Ojha CVPR 2023), NPR (Tan CVPR 2024), and an AEROBLADE-style VAE-reconstruction branch, all trained with aggressive JPEG/blur augmentation and calibrated with conformal energy-based OOD rejection.** This combination consistently lands in the 90–95% cross-generator accuracy band on GenImage and survives the JPEG/resize chain that cripples single-model baselines. The wider field has moved past the GAN-versus-real binary toward open-set attribution across diffusion, rectified-flow (FLUX/SD3), and autoregressive (VAR/MUSE) models, with FLUX and AR-image models constituting the current forensic blind spots.

The report below is organized as the eight requested sections plus a deployable hackathon pipeline. Citations follow Author/Year/Venue convention; GitHub URLs and dataset links are inline.

---

## 1. Artifact taxonomy: what each generator family leaves behind

### 1.1 GAN artifacts: the upsampling-replication signature

Every standard GAN generator (DCGAN, ProGAN, StyleGAN1/2/3, BigGAN, CycleGAN, StarGAN, GauGAN) reaches output resolution by stacking transposed convolutions or nearest-neighbour upsampling plus convolution. **Zhang, Karaman & Chang (WIFS 2019)** proved that the zero-insertion stage of an up-convolution is equivalent to a multiplication by a 2-D Dirac comb in the spatial domain, which by the convolution theorem is equivalent to *spectral replication* in frequency space. Formally, after a 2× zero-insertion upsample, the output spectrum is X̂(u,v) = Σ_{m,n∈{0,1}} X(u−m·f_c, v−n·f_c) — the original spectrum plus three replicas centered at f_c/2, the Nyquist frequency. The downstream convolution kernel is too small to suppress these replicas, producing visible spectral peaks at Nyquist multiples and a periodic "checkerboard" autocorrelation.

**Durall, Keuper & Keuper (CVPR 2020)** generalized this with the azimuthally-averaged 1-D power spectrum P(f) = (1/N_f) Σ_{‖(u,v)‖=f} |F[I](u,v)|², showing GAN-generated images systematically *over-estimate* magnitudes at high spatial frequencies relative to natural images, whose spectra follow a 1/f^α decay. **Frank et al. (ICML 2020)** extended the analysis to block-DCT coefficients; tiny linear classifiers on log-DCT features hit ≥99% on multiple GAN datasets. **Wang, Wang, Zhang, Owens & Efros (CVPR 2020)** then showed a single ResNet-50 with aggressive blur+JPEG augmentation generalizes near-zero-shot from ProGAN to 11 unseen GANs, suggesting all CNN-based generators share a common low-level fingerprint.

Critical caveat: **Chandrasegaran, Tran & Cheung (CVPR 2021)** built counter-example GANs with the final transposed conv replaced by nearest/bilinear upsampling, removing high-frequency anomalies entirely. Frequency-only detectors are necessary but not sufficient against an informed adversary.

Color-statistics anomalies (**McCloskey & Albright, ICIP 2019**) and noise-residual fingerprints (**Marra et al., MIPR 2019**; **Yu, Davis & Fritz, ICCV 2019**) supply complementary cues. Marra averages denoising residuals r(I) = I − f_denoise(I) over many images of one model to recover a reproducible per-architecture fingerprint, directly transplanting Lukas-Fridrich-Goljan PRNU forensics from cameras to GANs.

### 1.2 Diffusion model artifacts: VAE strides and brightness bias

**Ricker, Damm, Holz & Fischer (VISAPP 2024)** found that pre-trained GAN detectors fail on diffusion images (AUROC drops ~15 pp) because DM images do not exhibit GAN-style grid peaks; instead they *under-estimate* high-frequency magnitudes, attributable to the L2 noise-prediction objective which favors perceptual smoothness.

**Lin, Liu, Li & Yang (WACV 2024)** identified the **noise-schedule fingerprint**: SD-1.x's terminal SNR is ≈0.068 rather than 0, so the model cannot generate very dark or very bright images — mean luminance is biased toward 0.5. SDXL partly mitigates; FLUX/SD3 (rectified flow) bypass via straight-line ODE training. This is a directly observable brightness-histogram fingerprint.

The most architecturally-specific diffusion artifact is the **VAE-stride spectral peak**. **Corvi, Cozzolino, Poggi, Nagano & Verdoliva (CVPRW 2023)** documented spectral peaks at multiples of 1/8 Nyquist in LDM noise residuals, exactly matching the 8× upsampling factor of the SD KL-VAE-f8 decoder. VQ-GAN-based models (Taming Transformers, DALL·E-Mini) show peaks at multiples of 1/4 and 1/16. **Pixel-space DDPMs (ADM, Imagen, DALL·E 2) lack these peaks entirely** — they have only the iterative-denoising bias.

**Wang, Bao, Zhou et al. (ICCV 2023)** introduced DIRE: invert the input via DDIM to x_T, denoise back to x'_0, and compute |x − x'_0|. Generated samples reconstruct accurately (low DIRE), real images do not. **Ricker, Lukovnikov & Fischer (CVPR 2024)** refined this into AEROBLADE: skip the iterative inversion, use only encode-then-decode through a public LDM autoencoder. Their measure Δ_AE = LPIPS(x, D(E(x))) hits mAP 0.992 across SD1/SD2/SDXL/Kandinsky/Midjourney without training.

### 1.3 Autoregressive image models: a forensic frontier

The forensic literature on AR-image models (Parti, MUSE, VAR, MaskGIT, LlamaGen, Janus-Pro, Switti, Open-MAGVIT2, Infinity) is essentially nascent. **Zhang, Yu, Zheng et al. (ICCV 2025), "D³QE"** is the first systematic AR-specific detector, exploiting a **Discrete Distribution Discrepancy**: real images, when re-encoded through a public VQ-VAE, exhibit long-tail token distributions; AR-generated images concentrate probability mass on high-frequency codebook entries. They release ARForensics with samples from 7 AR models. **PRADA (Damm et al., 2025)** uses next-token probability ratios for attribution.

Architectural traces include 16-pixel grid periodicity (VQ-VAE-f16 stride for VAR/MaskGIT/MUSE), 8-pixel periodicity (VQ-GAN-f8 / Parti), and anisotropic spectral peaks from raster-scan generation. Most "diffusion detectors" silently subsume VQ-GAN-based AR images because the convolutional VQ decoder shares upsampling-spectral-peak signatures.

### 1.4 Model-specific differentiators

| Family | Key spectral artifact | Spatial AC periodicity | Brightness bias | AE reconstruction error |
|---|---|---|---|---|
| GAN (StyleGAN/ProGAN) | Peaks at f_c/2 | 2-px checkerboard | none | n/a |
| DDPM/ADM (pixel-space) | High-f under-estimation, no peaks | none | none | high |
| Stable Diffusion 1.x/2.x (LDM-f8) | Peaks at 1/8 Nyquist | period 8 | mean ≈0.5 | very low under SD1/SD2 AE |
| SDXL | 1/8 peaks, different magnitude | period 8 | partially fixed | low under SDXL AE |
| FLUX / SD3 (rectified flow + LDM AE) | f8 peaks; smoother high-f | period 8 | corrected | low under SD3 AE |
| Imagen / DALL·E 2 (pixel cascade) | weak; SR-cascade tell | none | none | high |
| DALL·E 3 | distinct new peaks (Synthbuster) | ~periodic | n/a | n/a (closed) |
| MidJourney | SD2-like AE + custom grading | period 8 | mild | low under SD2 AE |
| VQ-GAN/AR (VAR/MUSE/LlamaGen) | Peaks at 1/4 or 1/16 Nyquist | period 4 / 16 | none | codebook-discrepancy |

Foundational attribution references: **Sha, Li, Yu & Zhang (CCS 2023) "DE-FAKE"** (DALL·E 2, SD, GLIDE, LDM); **Bammey (IEEE OJSP 2024) "Synthbuster"** (9-model benchmark including MJ v5, DALL·E 3, Firefly); **Wißmann et al. (MAD 2024) "Whodunit"**; **Cazenavette et al. (CVPR 2024) "FakeInversion"**.

---

## 2. Passive fingerprinting methods

### 2.1 Spectral and DCT analysis

**Zhang et al. (WIFS 2019, "AutoGAN")** trains a CNN on log-magnitude FFT spectra, also providing a generator simulator. GitHub: github.com/ColumbiaDVMM/AutoGAN. **Frank et al. (ICML 2020)** uses block 2-D DCT with classical (kNN/SVM) and deep classifiers on log-DCT coefficients. GitHub: github.com/RUB-SysSec/GANDCTAnalysis. **Dzanic, Shah & Witherden (NeurIPS 2020)** fits a two-parameter decay to the radial reduced spectrum and runs kNN on the coefficients — 99.2% accuracy from as few as 8 training images. **Wang et al. (CVPR 2020), "CNNSpot"** — ResNet-50 with blur+JPEG augmentation; ~98% mAP across 11 GANs, but performance collapses on diffusion. GitHub: github.com/PeterWang512/CNNDetection.

### 2.2 Noise residual / PRNU-style

**Marra et al. (MIPR 2019)** averages denoising residuals to produce per-architecture fingerprints, ~90% attribution accuracy on five GANs. **Yu, Davis & Fritz (ICCV 2019)** learns model fingerprints as classifier weight vectors and per-image fingerprints as pre-FC features, achieving >99% on five-way attribution and demonstrating that a single training-set difference produces a distinguishable fingerprint. GitHub: github.com/ningyu1991/GANFingerprints. **Cozzolino & Verdoliva (TIFS 2020), "Noiseprint"** provides a model-agnostic CNN-based camera-noise fingerprint widely reused as a feature backbone (now Noiseprint++ in TruFor).

### 2.3 Co-occurrence / SRM rich models

**Fridrich & Kodovsky (TIFS 2012)**'s Spatial Rich Model — a bank of 30+ high-pass filters with co-occurrence matrices of truncated quantized residuals — remains the standard pre-processing in modern AIGI detectors (AIDE, SSP, F3-Net). **Nataraj et al. (Electronic Imaging 2019)** uses pixel-domain co-occurrence matrices directly with a moderate CNN, hitting >99% on CycleGAN and StarGAN.

### 2.4 Reconstruction-based methods: DIRE, AEROBLADE, LaRE², DRCT, FIRE

**DIRE (Wang et al., ICCV 2023)** — DDIM-inversion + reconstruction error trained on ADM-LSUN-Bedroom; generalizes to 11+ diffusion models. Expensive (one DDIM trajectory per inference) and reconstruction-model-dependent. GitHub: github.com/ZhendongWang6/DIRE. **AEROBLADE (Ricker et al., CVPR 2024)** — training-free LPIPS on encode-then-decode through public LDM autoencoders; mAP 0.992 across SD1/SD2/SDXL/Kandinsky/Midjourney. **LaRE² (Luo et al., CVPR 2024)** — latent-space reconstruction error with Error-Guided Refinement; 8× faster than DIRE and +11.9% Acc on GenImage. **DRCT (Chen et al., ICML 2024 Spotlight)** — diffusion reconstruction contrastive training; releases DRCT-2M (16 generators); +10–15% cross-set accuracy. **FIRE (Chu et al., CVPR 2025, arXiv:2412.07140)** — frequency-decomposed reconstruction error.

### 2.5 Gradient and pixel-relationship methods

**LGrad (Tan et al., CVPR 2023)** — feeds the gradient of a frozen pretrained CNN's score w.r.t. input to a ResNet-50; +11.4% over FrePGAN. GitHub: github.com/chuangchuangtan/LGrad. **NPR (Tan et al., CVPR 2024)** — captures local pixel inter-dependencies induced by upsampling via NPR(I)^c_{i,j} = I^c_{i,j} − I^c_{a,b}; trained only on 4-class ProGAN, generalizes to 28 generators including diffusion at ~92.5% mean Acc, +12.4% over UnivFD. GitHub: github.com/chuangchuangtan/NPR-DeepfakeDetection. **FreqNet (Tan et al., AAAI 2024)** — frequency convolutional layers operating on FFT amplitude and phase separately; 1.9 M parameters, +9.8% over baselines.

### 2.6 Diffusion noise feature, PatchCraft, AIDE

**DNF (Zhang & Xu, arXiv:2312.02625)** uses partial DDIM-inverse trajectory noise residuals; faster than DIRE but with poor OOD calibration in independent verification. **PatchCraft / RPTC (Zhong et al., 2023/2024)** — Smash & Reconstruction sorting patches by texture diversity, then 30 SRM filters on rich- vs poor-texture reconstructions. GitHub: github.com/Ekko-zn/AIGCDetectBenchmark. **AIDE (Yan et al., ICLR 2025)** — mixture-of-experts combining DCT-scored highest/lowest-frequency patches passed through SRM filters with frozen ConvNeXt-OpenCLIP semantic embeddings; +4.6% on GenImage (86.88% absolute) and introduces the **Chameleon** in-the-wild benchmark on which all nine evaluated detectors fail. GitHub: github.com/shilinyan99/AIDE.

### 2.7 CLIP-based universal detectors

**UniversalFakeDetect / UnivFD (Ojha, Li & Lee, CVPR 2023)** — frozen CLIP-ViT-L/14 features with linear probe or kNN; +15.07 mAP, +25.90% Acc over CNNSpot on unseen diffusion models. The signature paper for foundation-backbone forensics. GitHub: github.com/WisconsinAIVision/UniversalFakeDetect. **RINE (Koutlis & Papadopoulos, ECCV 2024)** — concatenates CLS tokens from all 24 CLIP transformer blocks with a Trainable Importance Estimator; only 6.3 M trainable params, 1-epoch training (~8 min), +10.6% over CLIP-only baseline. GitHub: github.com/mever-team/rine. **FatFormer (Liu et al., CVPR 2024)** — CLIP-ViT + frequency adapter + language-guided alignment via text prompts; 98% on unseen GANs, 95% on unseen diffusion. GitHub: github.com/Michel-liu/FatFormer. **C2P-CLIP (Tan et al., AAAI 2025)** — LoRA fine-tunes CLIP image encoder with Category Common Prompt injection; +12.4% over plain CLIP. **AntifakePrompt (Chang et al., 2023)** — soft-prompt-tunes ~4K parameters on InstructBLIP for VQA-style detection.

### 2.8 Lightweight and 2024–2026 entries

**SSP/ESSP (Chen et al., 2024)** — single simplest patch + SRM + ResNet-50; +14.6% on GenImage. **SAFE (Li et al., KDD 2025)** — image-transformation-centric training pipeline; 90.3% on UnivFD-bench. **HFI (Choi et al., 2024)** — quantifies high-frequency aliasing from LDM autoencoders for training-free attribution. **RIGID (He et al., NeurIPS 2024)** — perturbation stability of CLIP/DINO features. **Forensic Self-Descriptions (arXiv:2503.21003, CVPR 2025)** — self-supervised predictive filters trained on real images only.

---

## 3. Active fingerprinting and watermarking-based attribution

### 3.1 Stable Signature (Fernandez et al., ICCV 2023)

Two-stage method that (1) pre-trains a HiDDeN-style 48-bit watermark encoder/extractor on natural images, then (2) **fine-tunes only the LDM VAE decoder** so every generated image carries the signature. Loss combines Watson-VGG/LPIPS perceptual + BCE on bits; training takes ~10 min on 500 COCO images. Detection runs the extractor and a binomial test. Reports >90% accuracy at 10% crop with FPR<10⁻⁶. Vulnerable to autoencoder regeneration (Saberi 2024). GitHub: github.com/facebookresearch/stable_signature.

### 3.2 Tree-Ring Watermarks (Wen et al., NeurIPS 2023)

Zero-bit, training-free. Embeds concentric ring patterns directly into the **FFT of the initial Gaussian latent x_T** before DDIM sampling. Detection inverts the suspect image with empty-prompt DDIM, FFTs the recovered x_T, and measures L1 to the ring template. Ring patterns in Fourier space are invariant to convolutions, scaling, modest crops, flips, and rotations. ~99% TPR@1%FPR vs distortions in WAVES. GitHub: github.com/YuxinWenRick/tree-ring-watermark.

### 3.3 LoRA-based fingerprinting

**AquaLoRA (Feng et al., ICML 2024, arXiv:2405.11135)** — first white-box-protection scheme. Two stages: (1) latent watermark pre-training with Peak Regional Variation Loss for 48-bit imperceptible patterns; (2) Prior-Preserving Fine-Tuning integrates the pattern via a Watermark LoRA module (rank ~320) with a scaling matrix permitting per-user message changes without retraining. Survives LoRA merging into base weights. GitHub: github.com/Georgefwt/AquaLoRA. Related: **WMAdapter (Ci et al. 2024)**, **SleeperMark** (resilient to downstream LoRA fine-tuning), **AuthenLoRA**, **MaXsive**. (Note: "FI-LoRA" does not appear as a standalone published method; AquaLoRA is the canonical reference for this line.)

### 3.4 Gaussian Shading (Yang et al., CVPR 2024)

Training-free, plug-and-play, **distribution-preserving**. Watermark bits are diffused (replicated for redundancy), randomized via ChaCha20, and mapped to the initial Gaussian latent via quantile sampling so the marginal distribution of x_T is identical to a fresh N(0,I) — provably performance-lossless under steganographic indistinguishability. Detection inverts via DDIM and majority-votes the recovered bits. TPR>0.99 and bit accuracy >0.97 under JPEG Q=25, σ=0.1 noise, blur. **Gaussian Shading++ (arXiv:2504.15026)** strengthens the security proof and adds soft-decision PRC-style decoding. GitHub: github.com/bsmhmmlf/Gaussian-Shading.

### 3.5 Yu et al. — Artificial Fingerprinting (ICCV 2021)

A StegaStamp-like fingerprint encoder embeds a binary fingerprint into all training images of a GAN; the fingerprint **transfers** through training so any generated image carries it, decodable by the pre-trained extractor. Closes the responsibility loop without modifying GAN architecture or losses. The follow-up **Yu et al. (ICLR 2022)** scales to >10³⁸ identifiable instances via fingerprint-conditioned filter modulation. GitHub: github.com/ningyu1991/ArtificialGANFingerprints.

### 3.6 C2PA / Content Credentials

The Coalition for Content Provenance and Authenticity (Adobe, Microsoft, Sony, BBC, Intel, Google, Meta) defines a signed CBOR/COSE manifest containing assertions about capture device, edits, AI-generation declarations, and ingredient references, signed by an X.509 PKI. Two binding modes: hard binding (SHA-256 of asset bytes) and soft binding (perceptual fingerprint or invisible watermark). Adoption: Leica M11-P, Sony Alpha 9III, Nikon Z6 III, Adobe Firefly, OpenAI DALL·E 3, Microsoft Designer. Critical limitations: **not tamper-proof, only tamper-evident** — stripping the manifest by re-encoding/screenshot leaves no mark; absence ≠ "fake"; signer can lie in assertions; re-digitization breaks hard binding. **Must be paired with active watermarking (SynthID, Stable Signature) for AI-attribution.**

### 3.7 2024–2026 advances

**WaDiff (Min et al., ECCV 2024)** — watermark concatenated as extra UNet input channel, fine-tuned on per-user bits with consistency loss; scales to large user counts. **RingID (Ci et al., ECCV 2024)** — multi-channel heterogeneous Tree-Ring rings allowing thousands of distinct keys. **ZoDiac (Zhang et al., NeurIPS 2024)** — zero-shot watermarking of *existing* images by optimizing a latent z_T that DDIM-decodes back to the original and carries a Tree-Ring; WDR>98% on COCO/DiffusionDB/WikiArt under combined attacks including Zhao23 SD-regeneration. **PRC Watermarks (Gunn, Zhao & Song, ICLR 2025; built on Christ-Gunn CRYPTO 2024)** — initial latents sampled via pseudorandom error-correcting codes; provides *cryptographic undetectability* plus error-correction-based robustness; up to 512 robust bits. GitHub: github.com/XuandongZhao/PRC-Watermark. **ROBIN (Huang et al., NeurIPS 2024)** — implants watermark at intermediate diffusion step then adversarially optimizes a hidden text prompt to conceal it; AUC 0.998 under VAE-Bmshj18 attack. **Latent Watermark / LW (Meng et al., 2024)** — injection and detection inside LDM latent space. **VINE (Lu et al., ICLR 2025)** — first watermark explicitly hardened against generative editing (uses SDXL-Turbo as encoder prior, trained with random-blur surrogate attacks); releases W-Bench. **WOUAF (Kim et al., CVPR 2024)** — UNet weight-modulation by user fingerprint. **TrustMark (Bui et al., Adobe CAI)** — universal multi-resolution GAN watermark.

---

## 4. Classification approaches

### 4.1 Backbones

| Backbone | Use in forensics | Typical performance |
|---|---|---|
| ResNet-50 | CNNSpot baseline; standard on GenImage | ~98.5% in-generator, ~55–75% cross-generator |
| EfficientNet | Face-forensics workhorse (DFDC) | Plateaus on diffusion without frequency conditioning |
| ViT / DeiT / Swin-T | Backbones for IML-ViT, FatFormer | Plain ViT trained from scratch underperforms foundation models |
| ConvNeXt | AIDE semantic branch | +2–3 pp over ResNet-50 |
| CLIP-ViT-L/14 (frozen) | UnivFD, RINE, FatFormer, C2P-CLIP | +15 mAP, +26% Acc over CNNSpot on diffusion |
| DINOv2 / DINOv3 | DFF-Adapter, MoE-FFD | DINOv3 baseline F1=0.774 vs prior 0.530 SOTA on MVSS |
| MAE | CINEMAE, SAFE | Reconstruction-aware features for fine-tuning |

### 4.2 Augmentation, loss, and training recipes

The CNNSpot recipe (random JPEG Q∈[30,100], Gaussian blur σ∈[0,3], random resize, ColorJitter, RandomErasing, hflip) is *the* driver of cross-generator generalization — without it the network overfits semantics. **Grommelt et al. (2024)** showed real/fake JPEG-Q distributions and resolutions must be matched in GenImage; aligning them adds +11 pp cross-generator (ResNet-50: 71.7%→82.7%). NPR/FreqNet recommend *avoiding* aggressive color jitter because it suppresses upsampling traces. AIDE advocates DCT-score patch sampling instead of random crop.

Loss functions: cross-entropy with label smoothing remains the default; **ArcFace** margin softmax helps for fine-grained attribution (Yu ICCV 2019, Yang TIFS 2023); **SupCon / NT-Xent** is used in RINE (BCE + contrastive) and DRCT (contrastive between original/reconstructed pairs, +10% Acc cross-set).

### 4.3 Open-set OOD detection for unseen generators

Standard OOD methods adapted to attribution: **MSP (Hendrycks & Gimpel, ICLR 2017)**, **ODIN (Liang et al., ICLR 2018)**, **Energy (Liu et al., NeurIPS 2020)**, **Mahalanobis (Lee et al., NeurIPS 2018)**. The energy score E(x) = −T·logsumexp(logits/T) is the most reliable in practice — lower means more in-distribution. In CLIP/DINOv2 feature space, FRIDA-200 (arXiv:2510.27602) reports 88% average attribution Acc on GenImage with 200-image support sets. Forensic-specific work: **Girish et al. (CVPR 2021)** for open-world GAN attribution with Winner-Take-All clustering; **OmniDFA (arXiv:2509.25682)** for open-set + few-shot attribution paradigm with the OmniFake dataset (1.17 M images, 45 generators).

### 4.4 Few-shot adaptation

**MoE-FFD (Kong et al., NeurIPS 2024)** and **DFF-Adapter (AAAI 2026)** apply multi-head LoRA on DINOv2 with 3.5 M trainable parameters; **AdaptPrompt** extends VLM prompt-tuning to deepfake detection. ProtoNet-style few-shot heads on top of frozen CLIP work well when only 50–200 examples per generator are available.

### 4.5 Self-supervised foundation features

UnivFD (frozen CLIP linear probe) remains the strongest single zero-effort baseline. **Cozzolino et al. (CVPR 2024) "Raising the bar of AI-generated image detection with CLIP"** systematizes prompt-tuning vs adapter vs linear-probe choices. **DINOv3 (Siméoni et al. 2025)** beats specialized IML detectors as a simple baseline (arXiv:2604.16083). **MAE-based forensics** (CINEMAE, SAFE) offers reconstruction-aware features when fine-tuned with manipulation augmentation.

### 4.6 Ensembles

Top recent systems all stack heterogeneous experts: **AIDE** combines DCT-patch ResNet-50 with ConvNeXt-OpenCLIP. **LaRE²** combines latent-reconstruction-error with image features via Error-Guided Refinement. **TruFor (Guillaro et al., CVPR 2023)** fuses RGB with Noiseprint++ via SegFormer-style transformer. NTIRE 2026 winning entries (HEDGE, FeatDistill) use DINOv3 + MetaCLIP-2 ensembles with logit-space gating.

---

## 5. Robustness and adversarial considerations

### 5.1 Effect of common transformations

The standard robustness suite (WAVES, W-Bench, ZoDiac, VINE benchmarks) covers JPEG Q∈{50,75,90}, resize ±50%, random crops 0.1–0.9, Gaussian blur σ 1–3, color/brightness/contrast jitter, rotation, and "screenshot" (compounded crop+resize+JPEG).

| Distortion | StegaStamp | Stable Signature | Tree-Ring | Gaussian Shading | PRC | VINE |
|---|---|---|---|---|---|---|
| JPEG Q=90 | ~100% | ~99% | ~99% | ~99% | ~100% | ~100% |
| JPEG Q=50 | ~92% | ~85% | ~98% | ~98% | ~98% | ~96% |
| JPEG Q=25 | ~70% | drops | ~92% | ~95% | ~93% | ~85% |
| Center crop 25% | ~95% | ~90% | ~98% | ~96% | ~95% | ~95% |
| Gaussian blur σ=2 | ~90% | ~80% | ~96% | ~96% | ~95% | ~96% |
| Screenshot compound | ~70% | ~75% | ~92% | ~93% | ~91% | ~88% |
| Rotation 90° | severe drop | severe drop | partial | drops | drops | drops |

**Latent-domain / FFT-domain methods systematically beat pixel-domain methods** because their signal lives in semantic/global frequency space.

### 5.2 Adversarial perturbations

**Carlini & Farid (CVPRW 2020)** showed five attacks reducing a SOTA forensic CNN's AUC from 0.95 to 0.0005 (LSB flip), 0.08 (1% pixel perturbation), 0.17 (latent noise), 0.22 (black-box transfer). **Hussain et al. (WACV 2021)** achieved per-frame PGD attacks defeating XceptionNet/MesoNet across DFDC top-3 detectors with Expectation-over-Transformations to survive H.264. Generic deepfake CNNs collapse to chance under ε=4/255 L∞ PGD. **Saberi et al. (ICLR 2024)** proves a fundamental Wasserstein trade-off between TPR on real-vs-AI separation and adversarial robustness.

### 5.3 Laundering / regeneration attacks

**Zhao et al. (arXiv:2306.01953)** — pixel-level invisible watermarks are *provably* removable by adding bounded Gaussian noise then denoising via diffusion or VAE. **Saberi et al. (ICLR 2024)** demonstrate three attack families: diffusion-purification (σ=0.1 noise + SD-2.0 denoise drops Stable Signature/RivaGAN/SSL/DwtDct to near-random while changing PSNR <2 dB); model-substitute adversarial attack (reduces TreeRing AUC from ~1.0 to ~0.55); spoofing attack (forge a "watermark noise image" that makes real images detected as AI-generated). VAE laundering (Bmshj18, KL-VAE-f8 quality 1–8) sharply drops StegaStamp/Stable Signature but barely affects Tree-Ring. img2img re-diffusion at strength 0.3–0.5 reduces TrustMark/StegaStamp/MBRS/HiDDeN to <5% bit accuracy. **Controllable Regeneration from Clean Noise (Zhang et al., ICLR 2025)** uses DINOv2-conditioned SD-1.5 to regenerate from clean noise, defeating both low-perturbation and high-perturbation watermark families.

### 5.4 Composite / partial generation

Inpainting datasets: **HiFi-IFDL**, **GIM (Chen 2024)**, **DMID**, **COCO-Inpaint (Yan 2025, 258k images, 6 SOTA inpainters)**. Detection: legacy splicing methods (PSCC-Net, MVSS-Net, ObjectFormer) generalize poorly to diffusion inpainting; **InpDiffusion (Wang et al., AAAI 2025)** treats localization as conditional-diffusion mask generation; **DeFI-Net** uses dense feature interaction; **IIS module** uses self-similarity matrices. Image-blending α·watermarked + (1−α)·clean reduces decoding bit-accuracy proportional to α; multi-bit methods need spatial redundancy.

### 5.5 Net robustness ranking (2026)

1. **PRC + Gaussian Shading++ (provable / cryptographic)** — best under classical distortions.
2. **Tree-Ring family (Tree-Ring, RingID, ZoDiac, GaussMarker)** — robust to distortions and modest regen; vulnerable to adversarial embedding attacks on KL-VAE.
3. **VINE / ROBIN** — best against current text-driven editing.
4. **AquaLoRA / WaDiff / Stable Signature** — solid for classical distortions; defeated by VAE laundering.
5. **Post-hoc pixel watermarks (StegaStamp, TrustMark, RoSteALS, HiDDeN)** — broken by any diffusion-based regeneration.
6. **C2PA hard-binding alone** — broken by re-encoding/screenshot; requires pairing.

Benchmarks: **WAVES (An et al., ICML 2024)** — 26 attacks × 3 datasets × 5000 images. **W-Bench (VINE, ICLR 2025)** — first targeting editing-based attacks. **NeurIPS 2024 "Erasing the Invisible" Challenge** confirmed watermarks remain removable under restrictive threat models.

---

## 6. Datasets and benchmarks

### 6.1 GenImage (Zhu et al., NeurIPS 2023)

The default training/evaluation benchmark. **arXiv:2306.08571**. Repo: github.com/GenImage-Dataset/GenImage. Project: genimage-dataset.github.io. Download: Google Drive folder 1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS or Baidu Yunpan (code `ztf1`), ~200 GB. Composition: ≈1.33 M real ImageNet-1k + ≈1.35 M generated balanced per generator and class with prompt template `"photo of {class}"`. **8 generators**: SD v1.4, SD v1.5, ADM, GLIDE, Midjourney, VQDM, BigGAN, Wukong. Standard protocol trains on SD v1.4 and tests on the 7 others.

**Critical biases**: (1) JPEG bias — real ImageNet is JPEG Q≈70–100, fakes are PNG; detectors shortcut on this. Bias-controlled splits at unbiased-genimage.org recompress everything to JPEG Q=96, yielding ~+11 pp cross-generator. (2) Resolution bias — fakes are fixed 256/512, reals span the multi-modal ImageNet distribution. (3) Prompt-template bias — trivial `"photo of {class}"`.

### 6.2 AIGCDetectBenchmark (Zhong et al., PatchCraft)

Repo: github.com/Ekko-zn/AIGCDetectBenchmark. Implements 9 detectors uniformly (CNNSpot, FreDect, Fusing, Gram-Net, UnivFD, LGrad, LNP, DIRE, PatchCraft) over 16 generators (ProGAN/StyleGAN/StyleGAN2/BigGAN/CycleGAN/StarGAN/GauGAN/AttGAN/DeepFake + ADM/Glide/LDM/DALL·E/SD v1.4/v1.5/VQ-Diffusion). Training follows ForenSynths Protocol-I or AntifakePrompt Protocol-II.

### 6.3 DiffusionForensics (Wang et al., ICCV 2023, DIRE)

Repo: github.com/ZhendongWang6/DIRE. Subsets: lsun_bedroom, imagenet, celebahq with train/val/test triplets (source/reconstructed/DIRE) at 256×256, ~50k images per generator. 8 diffusion generators per subset (ADM, DDPM, iDDPM, PNDM, LDM, SD v1.4, SD v1.5, VQ-Diffusion).

### 6.4 ForenSynths / CNNDetection (Wang et al., CVPR 2020)

Repo: github.com/PeterWang512/CNNDetection. Project: peterwang512.github.io/CNNDetection. ProGAN over 20 LSUN classes for training; 11–13 generators in the test set including ProGAN, StyleGAN/2, BigGAN, CycleGAN, StarGAN, GauGAN, DeepFakes, CRN, IMLE, SAN, SITD, whichfaceisreal.

### 6.5 Other notable benchmarks

- **Synthbuster (Bammey, IEEE OJSP 2024)** — github.com/qbammey/synthbuster + zenodo.org/records/10066460. 9000 images across 9 generators (DALL·E 2/3, Firefly, MJ v5, SD 1.3/1.4/2, SDXL, GLIDE), pristine via RAISE-1k.
- **DRCT-2M (Chen et al., ICML 2024)** — github.com/beibuwandeluori/DRCT, ~2 M images, 16 diffusion generators with paired real-rec reconstructed counterparts as hard negatives.
- **WildFake (Hong & Zhang, AAAI 2025, arXiv:2402.11843)** — hierarchical 4-level structure (cross-generator, cross-architecture, cross-weight, cross-version).
- **Chameleon (AIDE, ICLR 2025)** — github.com/shilinyan99/AIDE. 11k+ AI images sourced from ArtStation/Civitai/LiblibAI that pass human Turing tests; **all 9 evaluated detectors collapse**.
- **CIFAKE** — github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images. 60k real + 60k fake at 32×32; sanity check only.
- **Fake2M / Sentry MPBench** — github.com/Inf-imagine/Sentry. ~2 M images with leaderboard.
- **DiffusionDB** — huggingface.co/datasets/poloclub/diffusiondb. 14 M SD images + prompts.
- **ArtiFact (Rahman et al., ICIP 2023)** — ~2.5 M images, 25 generators.
- **Community Forensics (Park & Owens, CVPR 2025)** — huggingface.co/datasets/OwensLab/CommunityForensics. **4,803 distinct generators**, 2.7 M images (1.1 TB full / 278 GB subset). Best dataset for diversity.
- **GenImage++ (Zhou et al., NeurIPS 2025, arXiv:2506.00874)** — designed to defeat GenImage shortcuts; long-form prompts, Flux/SD3.
- **AI-GenBench (IJCNN 2025)** — github.com/MI-BioLab/AI-GenBench. **Time-ordered continual** evaluation across many generators.
- **NTIRE 2026 Robust AIGI Detection** — Codabench competition 12761; **42 generators** + 36 transformations, 108k real + 186k generated; current frontier benchmark.
- **OpenFake (arXiv:2509.09495)** — 30+ models including GPT-Image-1, Grok-2.

### 6.6 Leaderboards (April 2026)

Papers with Code shut down its leaderboard portal in July 2025; historical JSON survives at paperswithcode/paperswithcode-data on GitHub. Active leaderboards:

- **NTIRE 2026** (Codabench live) — top entries use DINOv3 + MetaCLIP-2 ensembles with logit-space gating (HEDGE, FeatDistill).
- **GenImage cross-generator**: PatchCraft, AIDE (86.88%), NPR (88.6%), FreqNet (86.8%), FatFormer (88.9%), DRCT (87.7–89.5%), SAFE (90.3%), C2P-CLIP (95.8%), AIGI-Holmes (99.2% mean Acc on extended unseen test).
- **Chameleon**: All published methods <75% on fakes — open frontier.
- **Sentry-Image**: Active.

Caveat: there is no single official maintained leaderboard for GenImage; method-by-method numbers come from each paper's own table with slightly different protocols.

---

## 7. Practical hackathon pipeline

Target: single A100 (80 GB) or RTX 4090 (24 GB), PyTorch 2.3+. Task: multi-class generator attribution (N+1 classes including "unknown") over GenImage with possible unseen test-time generators.

### 7.1 Repository skeleton

```
hackathon/
├── data/                 # symlinks to GenImage, ForenSynths, Synthbuster, NTIRE2026
├── models/
│   ├── feature_extractor.py
│   ├── heads.py
│   └── ensemble.py
├── train.py
├── infer.py
├── tta.py
├── calibration.py
└── utils/
```

### 7.2 Hybrid feature extractor (FFT + CLIP + ResNet + AEROBLADE-DIRE)

```python
# models/feature_extractor.py
"""
Hybrid feature extractor combining:
 (1) Radial FFT spectrum (256 bins) — frequency-domain forensic cue
 (2) CLIP ViT-L/14 patch-token features (768-D) — UnivFD-style semantic prior
 (3) ResNet-50 features (2048-D) — local up-sampling / NPR-style cue
 (4) Fast DIRE: SD VAE encode/decode reconstruction error (AEROBLADE-style,
     ~50x faster than full DDIM DIRE while preserving signal for LDM-family fakes)
Output: (768 + 2048 + 256 + 256) = 3328-D vector
"""
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from torchvision.models import resnet50, ResNet50_Weights
from diffusers import AutoencoderKL


def radial_spectrum(x: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    g = x.mean(1)                                    # luminance (B,H,W)
    F2 = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
    mag = torch.log1p(F2.abs())
    B, H, W = mag.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins = (r / r.max() * (n_bins - 1)).long()
    out = torch.zeros(B, n_bins, device=x.device)
    for b in range(B):
        out[b].scatter_add_(0, bins.flatten(), mag[b].flatten())
    cnt = torch.bincount(bins.flatten(), minlength=n_bins).clamp(min=1)
    return out / cnt


class HybridExtractor(nn.Module):
    def __init__(self, vae_id="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        self.clip.visual.requires_grad_(False); self.clip.eval()
        self.rn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.rn.fc = nn.Identity()
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
        self.vae.requires_grad_(False).eval()

    @torch.no_grad()
    def dire_map(self, x_512: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            z = self.vae.encode(x_512).latent_dist.mean
            recon = self.vae.decode(z).sample
        err = (x_512 - recon).abs()
        return torch.stack([torch.histc(e, bins=256, min=0, max=2) for e in err])

    def forward(self, x224, x512):
        with torch.no_grad():
            clip_feat = self.clip.encode_image(x224)
        rn_feat = self.rn(x224)
        fft_feat = radial_spectrum(x224, 256)
        dire_feat = self.dire_map(x512)
        return torch.cat([clip_feat.float(), rn_feat,
                          fft_feat, dire_feat.float()], dim=1)
```

### 7.3 Training script with proper augmentation

```python
# train.py
import torch, torch.nn as nn, random, io, math, numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
from torch.optim.lr_scheduler import LambdaLR
from models.feature_extractor import HybridExtractor

CLASSES = ["real", "sdv1.4", "sdv1.5", "adm", "glide",
           "midjourney", "vqdm", "biggan", "wukong"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


class GenImageMulti(Dataset):
    def __init__(self, manifest, train=True):
        self.items = [l.strip().split('\t') for l in open(manifest)]
        self.train = train
        self.norm = T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                [0.26862954, 0.26130258, 0.27577711])

    def __len__(self): return len(self.items)

    def aug(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 3)))
        if random.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(30, 95))
            img = Image.open(buf).convert("RGB")
        if random.random() < 0.3:
            s = random.randint(160, 320)
            img = img.resize((s, s), Image.BICUBIC)
        if random.random() < 0.3:
            arr = np.array(img).astype(np.float32)
            arr += np.random.randn(*arr.shape) * random.uniform(2, 8)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return img

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.train:
            img = self.aug(img)
            img = TF.resized_crop(img, *T.RandomResizedCrop.get_params(
                img, (0.6, 1.0), (0.9, 1.1)), (224, 224))
            if random.random() < 0.5: img = TF.hflip(img)
        else:
            img = TF.center_crop(TF.resize(img, 256), 224)
        x224 = self.norm(TF.to_tensor(img))
        x512 = TF.resize(TF.to_tensor(Image.open(path).convert("RGB")),
                         512)[:, :512, :512] * 2 - 1
        return x224, x512.half(), CLS2IDX[label]


def cosine_warmup(opt, total, warmup):
    def fn(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(opt, fn)


def main():
    ds_tr = GenImageMulti("data/train.tsv", train=True)
    counts = np.bincount([CLS2IDX[l] for _, l in
        [x.strip().split('\t') for x in open('data/train.tsv')]])
    w_per_cls = 1.0 / counts
    sample_w = [w_per_cls[CLS2IDX[l]] for _, l in
        [x.strip().split('\t') for x in open('data/train.tsv')]]
    sampler = WeightedRandomSampler(sample_w, len(ds_tr), replacement=True)
    dl = DataLoader(ds_tr, batch_size=64, sampler=sampler, num_workers=8,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)

    feat = HybridExtractor().cuda()
    head = nn.Sequential(nn.Linear(3328, 1024), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(1024, len(CLASSES))).cuda()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(list(head.parameters()) + list(feat.rn.parameters()),
                            lr=3e-4, weight_decay=0.05)
    total_steps = 8 * len(dl); sched = cosine_warmup(opt, total_steps, 500)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(8):
        for x224, x512, y in dl:
            x224, x512, y = x224.cuda(non_blocking=True), x512.cuda(), y.cuda()
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = head(feat(x224, x512))
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        torch.save({'head': head.state_dict(),
                    'rn': feat.rn.state_dict()}, f"ckpt_e{epoch}.pt")
```

Optional ArcFace head for fine-grained attribution:

```python
class ArcFaceHead(nn.Module):
    def __init__(self, dim, n, s=30.0, m=0.50):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m
    def forward(self, x, y=None):
        x = F.normalize(x); W = F.normalize(self.W)
        cos = x @ W.T
        if y is None: return cos * self.s
        theta = torch.acos(cos.clamp(-1+1e-7, 1-1e-7))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.size(1)).float()
        return (onehot * target + (1 - onehot) * cos) * self.s
```

### 7.4 Test-time augmentation

```python
# tta.py
import torch, io
from PIL import Image
from torchvision.transforms import functional as TF

@torch.no_grad()
def tta_predict(model_fn, img_pil, device='cuda'):
    views = []
    base = TF.resize(img_pil, 256)
    for crop in (TF.center_crop(base, 224),
                 TF.crop(base, 0, 0, 224, 224),
                 TF.crop(base, 0, 32, 224, 224),
                 TF.crop(base, 32, 0, 224, 224),
                 TF.crop(base, 32, 32, 224, 224)):
        views.append(crop); views.append(TF.hflip(crop))
    for q in (85, 95):
        b = io.BytesIO(); img_pil.save(b, "JPEG", quality=q)
        c = TF.center_crop(TF.resize(Image.open(b).convert("RGB"), 256), 224)
        views.append(c)
    batch = torch.stack([TF.normalize(TF.to_tensor(v),
                          [0.48145466,0.4578275,0.40821073],
                          [0.26862954,0.26130258,0.27577711]) for v in views]).to(device)
    return torch.softmax(model_fn(batch), -1).mean(0)
```

### 7.5 Calibration and unknown-class rejection

```python
# calibration.py
import torch, torch.nn as nn, numpy as np

class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.T
    def fit(self, logits_val, y_val, lr=0.01, iters=200):
        opt = torch.optim.LBFGS([self.T], lr=lr, max_iter=iters)
        crit = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = crit(self.forward(logits_val), y_val)
            loss.backward(); return loss
        opt.step(closure); return self.T.item()

def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=-1)

def conformal_threshold(scores_val_id, alpha=0.1):
    n = len(scores_val_id)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores_val_id.cpu().numpy(), q_level))

def predict_with_unknown(logits, T_scaler, q_thr):
    z = T_scaler(logits); e = energy_score(z)
    pred = z.argmax(-1); pred[e > q_thr] = -1
    return pred, e
```

Workflow: train base model → on val of *known* generators compute energy scores → set `q_thr = conformal_threshold(...)` for desired coverage (α=0.05) → at test time, anything above the threshold is rejected as "unknown".

### 7.6 Batch inference at scale

```python
# infer.py
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models.feature_extractor import HybridExtractor

def build_engine(head, feat, compile=True):
    model = torch.nn.Sequential(feat, head).eval().cuda()
    if compile:
        model = torch.compile(model, mode='max-autotune', fullgraph=False)
    return model

def main(rank, world_size, manifest):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    feat = HybridExtractor().cuda(rank)
    head = torch.nn.Linear(3328, 9).cuda(rank)
    head.load_state_dict(torch.load("ckpt_best.pt", map_location=f'cuda:{rank}')['head'])
    model = DDP(build_engine(head, feat).module, device_ids=[rank])
    ds = TestDataset(manifest)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    dl = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=6,
                    pin_memory=True, prefetch_factor=4, persistent_workers=True)
    out = []
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for x224, x512, ids in dl:
            x224 = x224.cuda(rank, non_blocking=True)
            x512 = x512.cuda(rank, non_blocking=True)
            logits = head(feat(x224, x512))
            for i, l in zip(ids, logits.softmax(-1).cpu()):
                out.append((i, l.tolist()))
    torch.save(out, f"preds_rank{rank}.pt")
```

Launch with `torchrun --nproc_per_node=4 infer.py`. For CPU-bottlenecked deployments, export ResNet-50 + linear head via `torch.onnx.export` then `trtexec --onnx=... --fp16` for ~3× speedup. Keep CLIP and SD-VAE in PyTorch.

### 7.7 Pre-built feature pipeline to deploy at the hackathon

A turnkey three-checkpoint ensemble achievable in the first 8 hours:

1. **UnivFD baseline** (hour 0–2): Clone github.com/WisconsinAIVision/UniversalFakeDetect, load pretrained CLIP-ViT-L/14 + linear FC weights, run zero-shot. Expect mAP ≈ 0.92 cross-generator out of the box.
2. **NPR drop-in** (hour 2–4): Clone github.com/chuangchuangtan/NPR-DeepfakeDetection, load ProGAN-trained ResNet-50 weights. Generalizes to ~92% mean Acc across 28 generators.
3. **AEROBLADE branch** (hour 4–6): Use the SD-VAE encode-then-decode LPIPS distance as a third score. Training-free, mAP 0.992 on LDM-family fakes.

Fuse via logit-space weighted average:

```python
def hedge_fuse(logits_list, weights=None):
    if weights is None: weights = [1.0/len(logits_list)] * len(logits_list)
    z = sum(w * l for w, l in zip(weights, logits_list))
    return torch.softmax(z, -1)
```

### 7.8 24-hour timeline

| Hours | Task | Deliverable |
|---|---|---|
| 0–2 | Download GenImage SD-v1.4 subset (~30 GB), build manifest, train ResNet-50 binary head with CNNSpot augmentation | First submission, AUC ≈ 0.8 |
| 2–4 | Add UnivFD baseline (frozen CLIP + linear) | Submission 2, AUC ≈ 0.92 |
| 4–6 | Switch to multi-class head (9 classes), full augmentation suite | Multi-class baseline |
| 6–10 | Plug NPR + FreqNet members; train AEROBLADE-fast branch | 4-model ensemble |
| 10–14 | Run on bias-controlled GenImage; identify JPEG/size leakage; retrain with size-matched aug | Compression-robust ensemble |
| 14–18 | Integrate TTA, fit temperature scaling, set conformal threshold; validate on Synthbuster + Chameleon | Calibrated submission |
| 18–22 | Final HEDGE-style ensemble with logit-space gating; inject NTIRE 2026 transformations as augmentation | Best single submission |
| 22–24 | Deterministic inference (`torch.use_deterministic_algorithms(True)`); seed everything; submit best ensemble + UnivFD-only safety net | Final two submissions |

### 7.9 Common pitfalls

1. **Train/test resolution mismatch** — GenImage real ImageNet is huge JPEGs while fakes are 256/512 PNGs. Always resize *both* sides identically and re-compress to identical JPEG quality before the network. Grommelt et al. (2024) report +11 pp once this is fixed.
2. **JPEG bias** — most published ">95%" results collapse to ~60% at JPEG Q=80 without JPEG augmentation. Add `JPEG(p=0.5, Q∈[30,95])` to every training pipeline.
3. **Single-generator overfitting** — train on ≥2 generators across architectures (one GAN + one diffusion); hold out generators with different architectures for validation, never just a different version.
4. **CLIP semantic over-reliance** — CLIP nails "this is an SDXL anime portrait" but fails on photorealistic content. Combine with low-level NPR/FreqNet/DIRE features.
5. **Validation leakage** — Chameleon, GenImage++, Synthbuster, NTIRE-2026 test should *never* be trained on; use only for OOD calibration.
6. **Class imbalance** — real is typically 50% of data while each fake generator is 5–10%. Use `WeightedRandomSampler` or per-class loss weighting.
7. **Submission determinism** — set seeds for `torch`/`numpy`/`random`; `torch.use_deterministic_algorithms(True)`; pin cuDNN benchmark off.
8. **Generator-specific watermarks** — DALL·E 3 and Imagen sometimes embed signatures the model latches onto, which won't transfer.

### 7.10 Realistic targets

On a single A100 within 24 hours: binary real/fake AUC ≥ 0.97 on held-out NTIRE-style transformed test; multi-class top-1 ≥ 80% on seen-generator subset; unknown-class rejection rate ≥ 70% at α=0.05 conformal coverage on Chameleon / GenImage++. The points are won not by exotic backbones but by **augmentation + calibration + ensembling**.

---

## 8. Recent work (2024–2026)

### 8.1 GenImage SOTA progression

| Method | Venue | GenImage 8-gen mean Acc |
|---|---|---|
| ResNet-50 / CNNSpot | CVPR 2020 | ~70–73% |
| F3Net | ECCV 2020 | ~70% |
| Spec | WIFS 2019 | ~64% |
| UniFD (Ojha) | CVPR 2023 | ~78% |
| DIRE | ICCV 2023 | ~73–80% |
| LaRE² | CVPR 2024 | 86–88% |
| NPR | CVPR 2024 | 88.6% |
| FreqNet | AAAI 2024 | 86.8% |
| FatFormer | CVPR 2024 | 88.9% |
| DRCT | ICML 2024 | 87.7–89.5% |
| AIDE | ICLR 2025 | 86.88% |
| **C2P-CLIP** | **AAAI 2025** | **95.8%** |
| SAFE | KDD 2025 | 90.3% |
| **AIGI-Holmes** | **ICCV 2025** | **99.2% mean Acc on extended unseen test** |

C2P-CLIP and AIGI-Holmes are the most-cited near-top methods. **Caveat**: Yan et al. (ICLR 2025) Chameleon study shows almost all 9 evaluated detectors classify in-the-wild AI images as real; arXiv:2602.07814 finds off-the-shelf detector accuracy collapses to 18–31% on Flux Dev / Firefly v4 / MJ v7 / Imagen 4 / DALL·E 3.

### 8.2 Universal detectors generalizing across generators

**RINE (Koutlis & Papadopoulos, ECCV 2024)** — intermediate CLIP CLS tokens + Trainable Importance Estimator; +10.6% over CLIP-only. **FatFormer (Liu et al., CVPR 2024)** — CLIP + frequency adapter + language-guided alignment; 98% GAN, 95% diffusion. **AIDE (Yan et al., ICLR 2025)** — DCT-patch + CLIP MoE. **C2P-CLIP (Tan et al., AAAI 2025)** — LoRA-tuned CLIP with category-common-prompt injection. **DRCT (Chen et al., ICML 2024 Spotlight)** — diffusion reconstruction contrastive. **NPR / FreqNet (Tan et al., CVPR/AAAI 2024)** — pixel and frequency primitives. **SAFE (Li et al., KDD 2025)** — image-transform-centric.

### 8.3 Local edit / inpainting detection

**TruFor (Guillaro et al., CVPR 2023)** — RGB + Noiseprint++ via SegFormer fusion. **IML-ViT (Ma et al., AAAI 2024)** — first pure ViT for image manipulation localization with high-resolution input + multi-scale + edge supervision; surpasses MVSS-Net, CAT-Net, ObjectFormer, PSCC-Net. GitHub: github.com/SunnyHaze/IML-ViT. **MGQFormer (Zeng et al., AAAI 2024)** — query-based transformer. **InpDiffusion (Wang et al., AAAI 2025)** — conditional-diffusion mask generation with edge supervision. **DeFI-Net** — dense feature interaction. **IMDL-BenCo (Ma et al., NeurIPS 2024 D&B)** — comprehensive IML benchmark codebase. **LEGION (ICCV 2025)** — chain-of-thought localization with MLLMs.

### 8.4 Venue index 2024–2026

**CVPR 2024**: NPR, FatFormer, LaRE², "Raising the bar with CLIP" (Cozzolino et al.), AEROBLADE (Ricker et al.), FakeInversion (Cazenavette et al.). **ICCV 2023 baselines**: DIRE, UnivFD. **ECCV 2024**: RINE, "Common Sense Reasoning for Deep Fake Detection". **NeurIPS 2024**: Zhang et al. "Breaking Semantic Artifacts" (+18.85% / +10.59% open-world over NPR/Ojha), IMDL-BenCo, MoE-FFD. **ICML 2024**: DRCT (Spotlight). **AAAI 2024/2025**: FreqNet, C2P-CLIP, MGQFormer. **ICLR 2025**: AIDE (Sanity Check), LOKI multimodal benchmark. **ICCV 2025**: AIGI-Holmes, ATTSD, ForgeLens, LEGION, D³QE, LOTA, Forensic-MoE, CatAID. **KDD 2025**: SAFE. **NeurIPS 2025**: Dual Data Alignment, MLEP, training-free cropping-robustness detection.

2025–2026 arXiv preprints worth tracking: arXiv:2503.21003 (Forensic Self-Descriptions), arXiv:2509.25682 (OmniDFA), arXiv:2504.20865 (AI-GenBench), arXiv:2507.10236 ("Navigating the Challenges in the Wild"), arXiv:2604.16083 (DINOv3 baseline beating specialized detectors), arXiv:2511.23158 (REVEAL), arXiv:2512.10248 (RobustSora).

### 8.5 Emerging trends

**Foundation models for forensics** — frozen CLIP-ViT and DINOv2/v3 dominate cross-generator generalization. DINOv3 (Siméoni et al., arXiv:2508.10104) pushed dense-prediction transfer further; arXiv:2604.16083 reports a simple DINOv3 baseline beating specialized IML detectors on MVSS protocol (F1=0.774 vs prior 0.530).

**Multimodal detection** — FatFormer uses text prompts as contrastive supervision; C2P-CLIP injects category prompts; CLIPping the Deception (Khan & Dang-Nguyen, ICMR 2024) compares fine-tune / linear / prompt-tune / adapter; ForenX (arXiv:2508.01402) provides explainable AIGI detection with MLLMs and forensic prompts.

**AI-video detection (Sora / Veo / Runway / Pika / Kling)** — "Turns Out I'm Not Real" (Liu et al., arXiv:2406.09601); Deepfake-Eval-2024 (arXiv:2503.02857) shows GenConViT/FTCN drop 21.3% Acc on Sora-style; GenVidBench (Ni et al. 2025) hits 79.90% cross-generator; AEGIS (Li et al. 2025) for hyper-realistic AIGC video; DuB3D (Ji et al. 2024) at 96.77% in-domain via motion features; BusterX++ (Wen et al. 2025) for cross-modal AIGC video; RobustSora (arXiv:2512.10248, AAAI 2026) reveals 2–8 pp Sora 2 watermark dependency; EDVD-LLaMA (arXiv:2510.16442) explainable via fine-grained multimodal CoT. Reality Defender and TrueMedia.org added Sora 2 detection in late 2025.

**LLM-as-judge / MLLM forensics** — **AIGI-Holmes (ICCV 2025, Zhou et al.)** uses 3-stage pipeline (Visual Expert pre-training with CLIP+NPR, SFT, DPO with Holmes-DPO data), collaborative decoding fuses visual expert + MLLM, 99.2% mean Acc on unseen diffusion + autoregressive (incl. VAR, FLUX). **ThinkFake (Bai 2025)** — Qwen2.5-VL-7B + chain-of-thought + GRPO RL with UnivFD/AIDE as expert agents. **REVEAL (arXiv:2511.23158)** — Chain-of-Evidence + R-GRPO. "Towards Explainable Fake Image Detection" (arXiv:2504.14245) reports GPT-4o at 93.4% Acc, beating CNNSpot (91.8%), AEROBLADE (85.2%), and the most accurate human annotator (86.3%). **AIFo (arXiv:2511.00181)** — multi-agent LLM forensic framework with debate module.

---

## Conclusion: what wins this hackathon

The decisive insight is that **no single artifact family suffices**: GAN-era spectral peaks (Frank ICML 2020) are removed by a single architectural tweak (Chandrasegaran CVPR 2021); CLIP semantic priors (Ojha CVPR 2023) miss photorealistic content; DIRE/AEROBLADE reconstruction error (Wang ICCV 2023, Ricker CVPR 2024) is LDM-specific. Robust attribution requires layering — **frozen CLIP for cross-family generalization + NPR/FreqNet for upsampling forensics + AEROBLADE for LDM signature + heavy JPEG/blur augmentation against compression shortcuts + conformal energy-based OOD rejection for unseen generators**. C2P-CLIP and AIGI-Holmes mark the current SOTA on canonical benchmarks, but Chameleon and GenImage++ confirm the field is far from solved on in-the-wild content.

For the CISPA hackathon specifically, the security framing matters: any single watermark is removable (Saberi ICLR 2024, Zhao 2023), C2PA hard-binding is broken by re-encoding, and adversarial perturbations defeat passive classifiers (Carlini-Farid CVPRW 2020). The defensible posture is **layered detection** — combine active fingerprints (Stable Signature, Tree-Ring, PRC) with passive detectors (UnivFD, NPR, AEROBLADE) and provenance metadata (C2PA) so that any single bypass leaves the others intact. The forensic frontiers worth keeping in mind for novel attacks: rectified-flow models (FLUX, SD3) have minimal published forensic literature; autoregressive image models (VAR, MUSE, MaskGIT) have only D³QE (ICCV 2025) and PRADA as dedicated detectors; AI-video (Sora 2, Veo 3) is essentially uncovered. Prioritize a clean, calibrated, well-augmented HEDGE ensemble over exotic novelty — that is where measurable points are won within a 24-hour budget.