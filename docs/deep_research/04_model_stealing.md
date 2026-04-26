# Stealing the SprintML stage: a hackathon-grade model-extraction playbook

**Bottom line up front.** Model extraction against black-box ML APIs is an information-theoretic problem in disguise: every byte the API returns — hard label, top-K softmax, full logits, embeddings — has a precise leakage budget, and the strongest attacks (Tramèr 2016 equation solving, Carlini–Jagielski–Mironov CRYPTO 2020 cryptanalytic extraction, Carlini et al. ICML 2024 logit-bias SVD) saturate that bound for restricted architectures while functionality-stealing attacks (Knockoff Nets CVPR'19, ActiveThief AAAI'20, DFME CVPR'21, StolenEncoder CCS'22, B4B NeurIPS'23) trade orders of magnitude more queries for fidelity on deep nets. For the **CISPA European AI Security Hackathon (May 2026)** organized by Adam Dziedzic and Franziska Boenisch (SprintML Lab, CISPA), challenges are designed from the lab's own research portfolio — encoder stealing under active defenses, GNN extraction under hard-label budget caps, dataset/membership inference, watermark radioactivity, and Carlini-style logit-bottleneck attacks are all on the menu. Past iterations of the championship (Paris, Vienna, Munich, Barcelona) confirm a CTF format: surrogate uploaded to a scoring server, **fidelity-style metric on a hidden test set**, live leaderboard, larger re-evaluation at the end. **Optimize fidelity (not clean accuracy), pre-cache every query, and submit early and often.**

The rest of this report dives into theory, attack families, domain specifics, query efficiency, defenses, a concrete PyTorch pipeline, scoring, and a 2024–2026 update — closing with a one-page cheat sheet of what to memorize and pre-implement before the 24-hour clock starts.

---

## 1. Theoretical foundations: what leaks, how much, and which metric to chase

### 1.1 The leakage hierarchy of an API response

A prediction API leaks information in tiers (Jagielski et al., USENIX Security 2020; Tramèr et al., USENIX Security 2016):

- **Hard label (top-1):** one bit-per-class of information; corresponds to PAC / membership-query learning bounds. Every black-box classifier exposes at least this.
- **Top-k labels with confidences:** ranked list of K labels and posteriors.
- **Full softmax** `p ∈ Δ^K`: K−1 effective scalar leakages per query (probabilities sum to 1).
- **Logits / log-probs** (pre-softmax): for a ReLU network these are *piecewise-linear* in the input — Carlini–Jagielski–Mironov (CRYPTO 2020) exploit gradient discontinuities of the logits to read weights off directly.
- **Embeddings / penultimate features:** one query reveals an entire d-dimensional vector — equivalent to many label queries and amenable to direct linear regression on the head (StolenEncoder, Liu et al. CCS 2022).

For binary logistic regression `p = σ(w·x+b)`, inverting the sigmoid gives **one linear equation per query** in the weights, so `d+1` linearly independent labelled queries uniquely recover `(w,b)` — Tramèr et al. (2016) confirm 100% recovery from an average of **41 queries** across UCI datasets.

### 1.2 Information-theoretic lower bounds

Three formal hardness results bound the query count required for *functional* extraction of deep nets:

- **Hardness of functional equivalence (Jagielski et al. 2020, Thm. 1):** there exists a family of width-3k, depth-2 ReLU networks on `[0,1]^d` (`d ≥ k`) with p-bit precision requiring `Θ(p^k)` queries — such networks can encode a "spike" supported on a `p^{−k}` fraction of input space.
- **NP-hardness of equivalence testing (Thm. 2):** even given white-box access to two networks, deciding `f₁ ≡ f₂` is NP-hard via reduction to subset-sum.
- **SQ lower bound for fidelity (Das–Gollapudi–Kumar–Panigrahy 2020; Jagielski Thm. 3):** any Statistical-Query learner — which includes SGD, PCA, and most learning-based extractors — needs `exp(O(h))` samples to fidelity-extract a depth-h random network. **This is the formal reason classical "train-a-substitute" approaches plateau on deep targets.**

Decision trees admit a *path-finding* attack (Tramèr 2016) whose query complexity is proportional to the number of leaves, with each query returning a leaf identifier that uniquely fingerprints a path.

### 1.3 Three targets, one metric to rule them all

Let `O: X → Y` be the victim oracle and `Ô` the extracted model. Following Jagielski et al. (2020):

1. **Exact extraction** `Ô_θ = O_θ`. Impossible for ReLU nets due to permutation/scaling symmetries.
2. **Functionally-equivalent extraction** `∀x, Ô(x) = O(x)`. Necessary for cryptanalytic claims; only attainable for restricted architectures (linear, low-depth ReLU).
3. **Fidelity** `Fid_{S,D_F}(Ô, O) = Pr_{x ∼ D_F}[S(Ô(x), O(x))]`. With `S(p,q) = 𝟙[argmax p = argmax q]` and `D_F` = natural data, this is **label-agreement on the test distribution**. The metric Knockoff Nets, ActiveThief, DFME, B4B and (almost certainly) the CISPA scoring server use.
4. **Task accuracy** `Acc_{D_A}(Ô) = Pr[argmax Ô(x) = y]`. A high-accuracy extract may *exceed* the victim by correcting its mistakes — and consequently *lose* on a fidelity-scored server.

**Hackathon directive: optimize fidelity, not task accuracy.** A surrogate with `Acc=92%, Fid=78%` loses to one with `Acc=85%, Fid=88%` on every leaderboard the SprintML lab has ever run. Train with KL on the victim's full softmax (or temperature-scaled distillation if soft labels are available); never use label smoothing; bias query selection toward the victim's decision boundary.

---

## 2. Attack families: the canonical algorithms with reported numbers

### 2.1 Equation-solving attacks (Tramèr et al., USENIX Security 2016)

For multinomial softmax with K classes, take logs and subtract a reference class:

`log(p_k / p_K) = (w_k − w_K)·x + (b_k − b_K)`

producing a linear system in `(K−1)·(d+1)` unknowns, solved via OLS:

```
1. Generate n = α(K−1)(d+1) random inputs (α ≈ 1.1)
2. Query P = [O(x_i)] ∈ R^{n×K}
3. For k = 1..K−1: form L_k = log(P[:,k] / P[:,K])
4. Solve [W_k − W_K, b_k − b_K] = (X̃ᵀX̃)⁻¹ X̃ᵀ L_k  where X̃ = [X | 1]
5. Fix W_K = 0 (gauge); reconstruct biases b
```

Reported: **100% match in `d+1` queries** for UCI Adult LR (1485 queries), 12 queries for Iris softmax LR (`d=4, K=3`), exact recovery of BigML decision trees up to **1880 nodes in ~7416 queries** via path-finding, **~$30** to pwn an Amazon ML / BigML / Microsoft Azure model. Decision-tree path-finding works by varying one feature at a time (binary search for thresholds, value enumeration for categorical) until the leaf identifier changes.

### 2.2 Jacobian-Based Data Augmentation — JBDA (Papernot et al., AsiaCCS 2017)

Hard-label oracle, small in-distribution seed S₀:

```
S ← S₀
for ρ = 0..ρ_max-1:
   D ← {(x, Õ(x)) : x ∈ S}
   F̂ ← train(F̂, D)                    # supervised, hard labels
   for x ∈ S:
       g ← sign( ∂F̂(x)[Õ(x)] / ∂x )    # class-conditional Jacobian sign
       S ← S ∪ {x + λ_ρ · g}
   λ_ρ ← λ · (-1)^{⌊ρ/τ⌋}              # periodic step flip prevents cycling
```

`λ = 0.1, τ = 3`, S₀ ≈ 100–150 hand-collected samples; |S| doubles per round so total queries `≈ |S₀|·2^{ρ_max}`. Reported: MNIST oracle (MetaMind API) reached **81.2% substitute accuracy from 6 400 queries** with 84.24% FGSM transfer rate; **>96% transferability** against Amazon ML / Google Prediction API even with gradient-masked oracles.

### 2.3 Knockoff Nets (Orekondy, Schiele, Fritz, CVPR 2019)

The canonical functionality-stealing attack. Threat model: posteriors returned, attacker has no access to victim's training data, label semantics, or architecture. Two strategies — **random** sampling from a public pool (ImageNet, OpenImages) and **adaptive** RL over a coarse-to-fine label hierarchy with reward = certainty + diversity + loss:

```
H_{t+1}(z_t) = H_t(z_t) + α(r_t − r̄_t)(1 − π_t(z_t))
H_{t+1}(z')  = H_t(z')  − α(r_t − r̄_t) π_t(z')
π_t(z) = exp(H_t(z)) / Σ exp(H_t(z'))
```

Headline numbers at `B = 60 000` queries with ImageNet thief data:

| Victim | Random | Adaptive | Victim acc |
|---|---|---|---|
| Caltech256 | 75.4% | 76.2% | 78.8% |
| CUBS200 | 68.0% | 69.7% | 76.5% |
| Indoor67 | 66.5% | 69.9% | 74.9% |

A real Microsoft Azure Emotion API was extracted to ~76–82% relative accuracy for **≈$30**. The **out-of-distribution surrogate trick** is robust: querying a CUB-200 victim with ImageNet images works fine despite zero label-semantics overlap.

### 2.4 ActiveThief (Pal et al., AAAI 2020)

Active learning over a public unlabelled pool with four strategies — uncertainty (entropy), k-center (Sener & Savarese 2018), DFAL (Ducoffe & Precioso, adversarial-distance), and DFAL+k-center hybrid:

```
S ← random_seed (200) from unlabelled pool U
for i = 0..N-1:
   D_i ← {(x, F_V(x)) : x ∈ S}
   F̂  ← train(F̂, ⋃ D_j)
   if strategy == 'k-center':                # coreset
      C ← S
      while |C| < |S|+k:
         x* = argmax_{x∈U\C} min_{c∈C} ‖φ(x) − φ(c)‖
         C ← C ∪ {x*}
   elif strategy == 'DFAL+k-center':
      first compute DFAL scores, then k-center over the ε-most-uncertain pool
   S ← S ∪ selected
```

Reported: CIFAR-10 victim 80.8% → **agreement 81.7% at K=10k** with k-center alone, **4.7× improvement over uniform-noise baseline**, evades PRADA at the time. ActiveThief established that **`DFAL+k-center` (uncertainty filter then diversity step)** is the strongest single recipe — it's the spiritual ancestor of BADGE.

### 2.5 CopyCat CNN, BADGE, and the active-learning extractors

**CopyCat** (Correia-Silva et al., IJCNN 2018) is essentially Knockoff-random with hard labels: VGG-16 ImageNet-pretrained, fine-tuned on (random Internet image, victim hard label) pairs. With ~3M random images it reached **97.3% relative accuracy** vs. Microsoft Azure Emotion.

**BADGE** (Ash, Zhang, Krishnamurthy, Langford, Agarwal, ICLR 2020) computes the *gradient embedding* of each pool sample using its hallucinated label, then runs k-MEANS++ seeding (D²-sampling) over `{g_x}`; the magnitude `‖g_x‖` lower-bounds true uncertainty while D²-sampling enforces diversity — **no hyperparameters**, dominates uncertainty/coreset across batch sizes. Use BADGE in any hackathon with a public pool you can score.

### 2.6 Knowledge distillation as extraction (Hinton et al. 2015 = the simplest viable attack)

When the API returns probabilities, ordinary KD *is* extraction:

```
p_T = softmax(z_T(x) / T)
p_S = softmax(z_S(x) / T)
L = T² · KL(p_T || p_S) + α · CE(y_true, softmax(z_S(x)))
```

Temperature `T ∈ [2, 20]`, `α = 0.1` if any ground truth is available else 0. Jagielski et al. (2020) used T=1.5 distillation over ~130k ImageNet queries to extract Facebook's WSL-1B ResNet-50 with **+1.0 pp absolute accuracy gain** over training directly on ImageNet labels — a "labels worth more than data" demonstration.

### 2.7 Cryptanalytic / functionally-equivalent extraction (Carlini, Jagielski, Mironov, CRYPTO 2020)

For ReLU networks with real-valued logits, recover parameters exactly. *Critical points* are inputs where one ReLU flips sign — they encode the equation of one neuron's hyperplane. Algorithm:

1. **Critical-point search:** binary-search a line segment for piecewise-linear breakpoints (gradient discontinuities).
2. **Weight-direction recovery:** at a critical point, the difference of finite-difference gradients on either side of the kink is proportional to the row weight `w_i`; querying along basis directions reads off ratios.
3. **Sign disambiguation:** probe second derivatives (sign of slope-jump determines neuron sign).
4. **Layer peeling:** subtract layer 1 analytically, recurse.

Reported: **100k-parameter MNIST net extracted in 2²¹·⁵ queries (≈3M) under 1 hour, worst-case error 2⁻²⁵** — 2²⁰× more precise and 100× fewer queries than Jagielski's prior best. Follow-ups in 2024 made this *polynomial-time* (Canales-Martínez et al., EUROCRYPT 2024) and extended to hard-label (Chen et al., ASIACRYPT 2024) and PReLU/RNN (2025–2026).

### 2.8 Reported numbers — the master table

| Attack | Output type | Victim | Queries | Result |
|---|---|---|---|---|
| Tramèr eq-solving (LR) | softmax | UCI Adult LR | 1 485 | 100% match |
| Tramèr path-finding | conf+id | BigML IRS tree (1880 nodes) | 7 416 | 100% match |
| Papernot JBDA | hard label | MNIST DNN (MetaMind) | ~6 400 | 84.2% FGSM transfer |
| Carlini cryptanalytic | logits | 100k-param MNIST | 2²¹·⁵ | error ≤ 2⁻²⁵ |
| CopyCat CNN | hard label | Azure Emotion | ~3M | 97.3% rel acc |
| Knockoff random | softmax | Caltech256 RN-34 | 60 000 | 75.4% (vs 78.8) |
| Knockoff adaptive | softmax | Caltech256 | 60 000 | 76.2% |
| ActiveThief k-center | softmax | CIFAR-10 CNN | 10 000 | 81.7% agree |
| Jagielski distillation | softmax | WSL-1B IN RN-50 | 130 000 | 76.0% (+1.0 pp) |
| MAZE (data-free) | softmax | CIFAR-10 | 30M | 0.91× |
| DFME (data-free) | logits | SVHN | 2M | 0.99× |
| DFME | logits | CIFAR-10 | 20M | 0.92× (88.1%) |
| MARICH (active) | softmax | CIFAR-10 | 1k–7k | near victim |
| SEEKER | softmax | CIFAR-10 | 100K | 93.97% |
| Carlini logit-bias | top-K logprobs + bias | OpenAI Ada | <$20 | full projection layer |
| Carlini est. | top-K logprobs + bias | gpt-3.5-turbo | ~$2 000 | full final layer |

---

## 3. Domain-specific methods: what changes when the API isn't a CIFAR classifier

### 3.1 Image classifiers (CNN / ViT)

The hierarchy from Knockoff Nets onward shows three robust facts: **out-of-distribution surrogate datasets work** (ImageNet for Caltech, CIFAR-100 for CIFAR-10); **architecture mismatch is fine** (ResNet-34 victim, VGG copycat); **adaptive sampling adds 10–15% query efficiency at low budgets**. ViTs follow the same playbook with one bonus — public ImageNet-pretrained ViT-Small / DeiT / MAE checkpoints make excellent warm starts, dropping the labelled-query budget by 2–4×. Pure OOD random *noise* underperforms OOD natural images by a wide margin: noise queries push victim outputs toward uniform, killing gradient signal.

### 3.2 LLMs — the Carlini 2024 logit-bias SVD attack

Carlini, Paleka, Dvijotham, Steinke, Hayase, Cooper, Lee, Jagielski, Nasr, Conmy, Yona, Wallace, Rolnick, Tramèr — *Stealing Part of a Production Language Model*, **ICML 2024 (Best Paper)** [arXiv 2403.06634] — exploits the **softmax bottleneck**: a transformer's logits are `f(p) = W · g_θ(p)` where `W ∈ R^{l×h}` is the unembedding matrix with `l ≫ h` (e.g., l=100 277 vocab, h=4 096 hidden). All logit vectors live in the h-dimensional column space of W, so:

```
SVD(Y) = U Σ Vᵀ          # Y = stacked logit vectors
h ≈ argmax_i (σ_i − σ_{i+1})    # spike at hidden dim
```

The first h left-singular vectors span `col(W)`, recovering the projection matrix up to an unknowable basis change in the residual stream. Carlini et al. confirmed **h = 1 024 for OpenAI Ada and h = 2 048 for Babbage** at <$20 each, and (under agreement with OpenAI) gpt-3.5-turbo's hidden dim at ~$200, with full-projection extraction estimated at ~$2 000.

For APIs that hide raw logits but expose `top-K logprobs + logit_bias`, the attacker biases target token i by a constant B so it enters the top-K, then solves softmax algebra across two queries (bias 0, bias B) to recover logit *differences* `z_i(p) − z_r(p)`. **Defense actually deployed**: OpenAI and Google now disallow simultaneous `logit_bias + logprobs` (March 2024). Anthropic was never vulnerable.

**Finlayson, Ren, Swayamdipta (ICLR 2024 / COLM 2024)** — *Logits of API-Protected LLMs Leak Proprietary Information* [arXiv 2403.09539] — independently showed the same bottleneck, treating the model's image as a *signature* for accountability and unauthorized-clone detection.

### 3.3 LLMs — embedding inversion (Vec2Text family)

Morris, Kuleshov, Shmatikov, Rush — *Text Embeddings Reveal (Almost) As Much As Text*, **EMNLP 2023** [arXiv 2310.06816] — train an iterative T5-base corrector `ψ_θ(x_t, φ(x_t), e*) → x_{t+1}` that minimizes `‖φ(x̂) − e*‖`:

```
x_0 ← argmax_x p(x | e*; θ_0)
for t = 0..T-1:
    e_t ← φ(x_t)
    x_{t+1} ← ψ_θ(x_t, e_t, e*)
```

50 correction steps + beam search recover **92% of 32-token sequences exactly, BLEU 97** on `text-embedding-ada-002` and GTR-base. Demonstrated extraction of patient names from clinical notes embeddings. **Morris et al. ICLR 2024** (*Language Model Inversion*) extended this to inverting next-token logit distributions back to the prompt — recovering hidden system prompts.

### 3.4 LLMs — imitation, decoding theft, prompt theft

- **Wallace, Stern, Song (EMNLP 2020)**: imitate Google/Bing translation APIs with monolingual queries to within **0.6 BLEU** of production; the stolen model serves as a white-box surrogate for HotFlip-style adversarial transfer attacks.
- **Krishna et al. *Thieves on Sesame Street* (ICLR 2020)**: BERT-fine-tuned classifiers stealable with random word sequences for a few hundred dollars — the *transfer-learning paradigm itself* is the vulnerability.
- **Naseh, Krishna, Iyyer, Houmansadr (CCS 2023, distinguished paper)**: extract decoding hyperparameters (temperature, top-p, top-k).
- **Yang et al. *PRSA* (USENIX Security 2025)** [arXiv 2402.19200]: prompt-mutation + pruning attacks on PromptBase / OpenAI GPT Store, ASR 17.8% → 46.1% at 1.3–12.3% of original prompt cost.

### 3.5 Tabular / linear / decision trees

Tramèr 2016 § 2.1 covers everything; the cryptanalytic line (Carlini 2020 → Canales-Martínez 2024 → Chen 2024) handles MLPs with logit access, including hard-label settings as of ASIACRYPT 2024.

### 3.6 Graph neural networks — likely SprintML hackathon material

GNNs have a unique attack surface: graph *structure* is part of the secret, and aggregation prevents IID queries. The taxonomy:

- **He, Jia, Backes, Gong, Zhang — *Stealing Links from GNNs* (USENIX Security 2021)**: infer adjacency, not the model, with **AUC > 0.95** on Cora/Citeseer/PubMed using pairwise posterior similarity.
- **DeFazio & Ramesh (AAAI-DLG 2020)**: insert adversarial nodes into the graph, train surrogate from returned labels — ~80% fidelity.
- **Wu, Yang, Pan, Yuan — *Model Extraction Attacks on GNNs* (AsiaCCS 2022)**: 7-attack taxonomy (Type I/II/III on attribute / structure / shadow-graph knowledge), 73–96% fidelity.
- **Shen, He, Han, Zhang (S&P 2022)**: stealing inductive GNNs, 80–95% fidelity.
- **Zhuang et al. — STEALGNN (USENIX Security 2024)**: data-free, hard-label-only extraction.
- **Podhajski, Dubiński, Boenisch, Dziedzic et al. (ECAI 2024 + AAAI 2026 Oral)**: graph contrastive learning + spectral augmentations; AAAI 2026 paper extracts under *severely restricted query budget + hard-label* — **bootstraps the model backbone without direct queries to the victim**, using queries only for highest-information samples.

### 3.7 Embedding / encoder extraction (the SprintML core)

Encoders return high-dim vectors instead of K-class posteriors — bandwidth per query is much higher, so stealing is correspondingly cheaper.

- **Liu, Jia, Liu, Gong — *StolenEncoder* (CCS 2022)** [arXiv 2201.05889]: train surrogate `φ_A` to match victim `φ_V`, using augmentation invariance to inflate effective query count:

  `L = Σ_x Σ_{a ∈ Augs} ‖φ_A(a(x)) − φ_V(x)‖_2 + λ ‖φ_A(x) − φ_V(x)‖_2`

  Stolen CLIP reaches ≥93% of victim downstream accuracy using **<0.03% of victim's training data** (113K STL-10 images vs. 400M image-text pairs).

- **Dziedzic et al. — *On the Difficulty of Defending SSL against Model Extraction* (ICML 2022)** and **Dataset Inference for SSL (NeurIPS 2022)** establish the SprintML stealing playbook: contrastive losses (InfoNCE, SoftNN) outperform MSE; stealing requires `<1/5` of victim pre-training data; ~$72 to steal a SimCLR ResNet-50 vs. ~$5 714 to pre-train.

- **Sha, He, Yu, Backes, Zhang — *Cont-Steal* (CVPR 2023)**: adds explicit cross-pair negatives between surrogate and victim batches, beating StolenEncoder by 3–10 pp downstream.

- **Dubiński, Pawlak, Boenisch, Trzciński, Dziedzic — *Bucks for Buckets (B4B)* (NeurIPS 2023)**: first **active** defense against encoder stealing — adversaries' representations cover a much larger fraction of embedding space than legitimate users; B4B adaptively scales utility and applies **per-user transformations** to defeat sybil aggregation. Code: github.com/adam-dziedzic/B4B. **High-probability hackathon target.**

---

## 4. Query efficiency: making 10k queries do the job of 100k

### 4.1 Synthetic data — GAN-based (data-free)

**MAZE** (Kariyappa, Prakash, Qureshi, CVPR 2021) plays a min-max game between clone `C_θ` and generator `G_φ` over disagreement loss `L_dis = KL(f_V(x) || C_θ(x))`, with a NES-style zeroth-order gradient estimator for `f_V`:

`∇_x L̂ ≈ (1/(m·μ)) Σ_i [L(x+μu_i) − L(x)] u_i, u_i ∼ N(0,I)`

CIFAR-10 normalized accuracy 0.91× at ~30M queries; partial-data MAZE-PD reaches 0.97–1.0× at 1.25–15M.

**DFME** (Truong, Maini, Walls, Papernot, CVPR 2021): same setup with `ℓ_1` disagreement (beats KL/`ℓ_2`) and per-example logit recovery from softmax via the additive-constant trick. **SVHN 0.99× at 2M queries; CIFAR-10 0.92× at 20M, 0.94% at 30M.**

**Verdict for a hackathon**: data-free joint generator training is **rarely worth it for a 24h budget-constrained CTF** — millions of queries required. Use a *frozen* off-the-shelf generator (BigGAN, Stable Diffusion) only for query selection diversity, not online co-training.

### 4.2 Diffusion-based query synthesis (2024–2026)

- **AEDM** (Hong et al., ICONIP 2024): steers Stable-Diffusion latents to maximize victim/clone disagreement — **1/200th** the query budget of GAN-based attacks while remaining semantically plausible (evades OOD detectors).
- **Stealix** (Zhuang, Wang, Nicolae, Fritz, **ICML 2025**) [arXiv 2506.05867]: prompt evolution via genetic algorithm over text prompts → diffusion-synthesized queries, **no predefined prompts or class names needed**. Code: boschresearch/stealix.
- **DiMEx** (2025–2026): REMBO-style Bayesian optimization in Stable-Diffusion latent space; addresses cold-start of GAN-based attacks.
- **MARICH** (Karmakar & Basu, NeurIPS 2023): public data + active learning, near-victim accuracy at **1 070–6 950 queries**.
- **SEEKER** (ICLR 2024 submission): self-supervised pretraining + public-data queries, **93.97% CIFAR-10 at 100K queries**, >50× efficiency over DFMS.

### 4.3 MixUp augmentation for stealing

Linear mix `x̃ = λx_i + (1−λ)x_j, λ ∼ Beta(α,α)` before querying f_V (a) inflates effective diversity, (b) interpolates posteriors so the clone learns a smoother boundary, and (c) defeats simple OOD detectors via intermediate softmax entropy. Jagielski et al.'s MixMatch-based extraction made label-only access nearly as effective as logit access.

### 4.4 Decision-boundary coverage / active strategies

- **Uncertainty / entropy:** `argmax H(C_θ(x))`.
- **Margin sampling:** `argmin (p_{(1)} − p_{(2)})`.
- **K-center / coreset** (Sener & Savarese ICLR 2018): greedy farthest-point in feature space, 2-OPT approximation.
- **DFAL** (Ducoffe & Precioso 2018): adversarial-distance based.
- **BADGE** (Ash et al. ICLR 2020): gradient-embedding k-MEANS++ — uncertainty + diversity in one shot, no hyperparameters.
- **Cluster-Margin** (Citovsky et al., NeurIPS 2021): million-scale alternative to BADGE.

The empirical winner in extraction settings is **uncertainty filter (margin or DFAL) followed by k-center diversity step** (ActiveThief AAAI'20; spiritually BADGE).

### 4.5 Concrete query-vs-accuracy tradeoffs

CIFAR-10 normalized clone accuracy (×victim): 0.5–0.8× at B=10k, **0.85–0.95× at B=100k with ImageNet queries**, 0.9–0.99× for fully data-free at B=1M–20M. ImageNet-class victims rarely reach >0.7× even at 10⁶ queries with current methods. LLM final-layer recovery: <$20 for OpenAI Ada, ~$2 000 for gpt-3.5-turbo (Carlini 2024).

---

## 5. Defenses and how to bypass them

### 5.1 Output perturbation — ineffective passive defenses

Top-1 / top-k truncation, softmax rounding, and additive Gaussian/Laplacian noise on logits all fail against KL-trained surrogates: the *expected* gradient remains aligned with the true target, so extraction degrades only modestly. Orekondy et al. (ICLR 2020) explicitly demonstrate that all passive perturbation defenses "are ineffective against DNN stealing attacks."

### 5.2 Active perturbation — MAD / MAZE-defense / GRAD²

**Prediction Poisoning (Orekondy et al., ICLR 2020)** selects `y′` in an `ε`-ball around `y` to maximize the angular deviation between attacker's gradient on `y′` and on `y`:

`max_{y′: ‖y′−y‖_p ≤ ε, y′ ∈ Δ^K} ∠(g(y′), g(y))`

Solved via convex relaxation; reported up to **2× attacker error** at modest utility cost.

**Adaptive Misinformation (Kariyappa & Qureshi, CVPR 2020)** smoothly mixes `f_V` with a misinformation network `f_M` weighted by an OOD score:

`y′ = α·f_V(x) + (1−α)·f_M(x), α = σ(τ·(s_OOD − ν))`

Up to **40% drop in clone accuracy with <0.5% benign accuracy hit** on Flowers-17/GTSRB. **Bypass:** Stealix / AEDM / SEEKER produce semantically plausible, in-distribution queries that evade the OOD trigger.

**GRAD² — Gradient Redirection (Mazeika, Li, Forsyth, ICML 2022)** [arXiv 2206.14157] minimizes `⟨g(y), g(y′)⟩` (a strict generalization of MAD) via a closed-form linear program, can implant arbitrary behavior into the stolen model. **Bypass:** ensemble of diverse-architecture surrogates so the defender's single-surrogate gradient is misaligned with at least one clone; or hard-label distillation that ignores the perturbed soft labels.

### 5.3 Watermarking — forensic, not preventive

**Adi et al. (USENIX Security 2018)**: trigger set `T = {(x_i, y*_i)}` of OOD samples baked into f_V at training; verification by querying suspect model on T. **Critical limitation:** watermark inputs are OOD and not queried by the attacker, so extraction simply removes them. Shafieinejad et al. (IH&MMSec 2021) show fine-tuning + regularization removes Adi watermarks while preserving accuracy.

**DAWN — Dynamic Adversarial Watermarking (Szyller et al., ACM MM 2021)** [arXiv 1906.00830]: for ~0.5% of queries return a *deliberately wrong* label drawn from a PRF of the input — the trigger set is generated dynamically *from the attacker's own queries*, so the surrogate inherits them. Reported confidence `>1 − 2⁻⁶⁴` at 0.03–0.5% benign loss. **Bypass:** (i) query each input twice and majority-vote (DAWN's wrong labels are deterministic per input but the attacker can detect inconsistency by re-issuing equivalent queries with augmentation); (ii) confidence-based filtering of high-entropy outputs; (iii) ensemble distillation across multiple stolen models.

**Entangled Watermarks (EWE) — Jia, Choquette-Choo, Chandrasekaran, Papernot (USENIX Security 2021)**: uses Soft Nearest-Neighbor Loss to *entangle* trigger and task representations in the same hidden manifold:

`L_SNNL(X,T) = − (1/|X|) Σ log [Σ_{j: y_j=y_i} exp(−‖h_i−h_j‖²/T) / Σ_{k≠i} exp(−‖h_i−h_k‖²/T)]`

Trains with `L_CE − κ·L_SNNL` (κ negative on legit, positive on watermark). Extraction transfers **38–96% of watermark accuracy** vs. <10% for Adi-style. **Bypass:** scale-up to deeper architectures degrades transfer; pruning + fine-tuning still partially erases it.

### 5.4 PRADA detection — bypassed by natural-image queries

**Juuti, Szyller, Marchal, Asokan (EuroS&P 2019)**: stateful per-client detector; maintains pairwise `ℓ_2` distance distribution `{d_{ij}}` of queries, fires when Shapiro–Wilk normality statistic drops below threshold:

```
For each new query x_new from client c:
    Update D_c ← D_c ∪ {‖x_new − x‖ : x in past queries of c}
    W = ShapiroWilk(D_c)
    if W < δ: raise alarm
```

**Detects all Papernot-style attacks at 100% TPR / 0% FPR** in 210–710 queries. **Bypass:** ActiveThief queries only natural images from a public pool, leaving the pairwise-distance distribution Gaussian; SEEKER reports W=0.95 at B=100K; sybil distribution defeats per-client state; injecting innocuous random natural images alongside extraction queries restores normality.

### 5.5 Differential privacy on outputs and proof-of-work

**Yan et al. *Monitoring-based DP* (TDSC 2022)**: per-query budget allocation that decays as cumulative leakage grows. Empirically, DP that actually reduces stealing fidelity (`ε ≤ 1` on outputs) drops benign utility by 5–15 pp. **DP-SGD on training does NOT directly stop extraction** — extraction is post-processing-invariant.

**Dziedzic et al. — *Calibrated Proof of Work* (ICLR 2022 Spotlight)** [SprintML]: forces clients to compute PoW calibrated to query *information content*; ~100× compute overhead for attackers, ~2× for legit users. **High-probability hackathon target — bypassing or efficiently completing PoW.**

### 5.6 Defense summary

| Defense | Class | Acc cost | Reduction | Adaptive bypass |
|---|---|---|---|---|
| Top-k / argmax | Passive | 0% | 1–5 pp | Hard-label DFMS |
| Logit noise σ | Passive | low | small | Average over repeats |
| MAD (Orekondy 2020) | Active | <1 pp | 2× error | Surrogate ensemble |
| Adaptive Misinfo (Kariyappa 2020) | OOD-cond | <0.5% | 40% drop | In-dist diffusion queries |
| GRAD² (Mazeika 2022) | Active | low | dominates MAD | Hard-label distillation |
| Adi (2018) | Watermark | 0% | detection only | Fine-tune, distill, prune |
| DAWN (2021) | Dyn. WM | 0.03–0.5% | conf >1−2⁻⁶⁴ | Query repetition, ensemble |
| EWE (2021) | Entangled WM | 1–3 pp | 38–96% transfer | Deeper arch degrades |
| PRADA (2019) | Detection | 0% | 100% TPR Papernot | Natural-image pool, sybils |
| MDP (2022) | DP outputs | moderate | ext-error ↑ | Query budget, sybils |
| EDM (Kariyappa 2021 ICLR) | Architectural | small | 32 pp | Continuous-relaxation adaptive |
| PoW (Dziedzic 2022) | Compute cost | 0% | 100× cost | Distributed computing, sybils |
| B4B (Dubiński 2023) | Active encoder | small | strong | Sybil aggregation, transform inversion |

**Bottom line:** as of 2026, no defense gives a strong utility/security tradeoff against an adaptive adversary willing to spend substantial query budget. The most reliable operational defense in deployed LLM APIs (Carlini 2024) was simply *removing the conjunction of `logit_bias + logprobs`*.

---

## 6. The hackathon pipeline: a working PyTorch codebase

This section is opinionated and tuned for the 24-hour CISPA format: API wrapper, query strategies, surrogate trainer, eval harness, GPU tips. Drop into one repo; ship a baseline within hours, iterate on the leaderboard.

### 6.1 API wrapper with rate limit, dedup cache, retry, budget

```python
# attacker/api.py
from __future__ import annotations
import hashlib, sqlite3, pickle, time, threading, logging, io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import torch
import requests

log = logging.getLogger("victim_api")

@dataclass
class APIConfig:
    endpoint: Optional[str] = None        # HTTPS endpoint
    local_fn: Optional[Callable] = None   # fallback callable(np)->np
    cache_path: str = "cache.sqlite"
    qps: float = 5.0                      # token-bucket rate
    burst: int = 16
    max_retries: int = 5
    timeout: float = 10.0
    budget: int = 50_000                  # cap on UNIQUE billed queries

class VictimAPI:
    """Black-box wrapper: rate-limit + dedup-cache + retry + budget."""

    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
        self._tokens = float(cfg.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(cfg.cache_path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (h BLOB PRIMARY KEY, y BLOB)")
        self._conn.commit()
        self.queries_used = 0

    @staticmethod
    def _hash(x): return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).digest()

    def _cache_get(self, h):
        r = self._conn.execute("SELECT y FROM cache WHERE h=?", (h,)).fetchone()
        return pickle.loads(r[0]) if r else None

    def _cache_put(self, h, y):
        self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (h, pickle.dumps(y)))
        self._conn.commit()

    def _take_token(self):
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.cfg.burst, self._tokens + (now - self._last)*self.cfg.qps)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0; return
                time.sleep((1.0 - self._tokens)/self.cfg.qps)

    def _raw_query(self, batch):
        backoff = 1.0
        for attempt in range(self.cfg.max_retries):
            try:
                self._take_token()
                if self.cfg.local_fn is not None:
                    return np.asarray(self.cfg.local_fn(batch), dtype=np.float32)
                buf = io.BytesIO(); np.save(buf, batch); buf.seek(0)
                r = requests.post(self.cfg.endpoint, data=buf.read(), timeout=self.cfg.timeout)
                r.raise_for_status()
                return np.load(io.BytesIO(r.content))
            except Exception as e:
                log.warning("query failed (try %d): %s", attempt, e)
                time.sleep(backoff); backoff *= 2
        raise RuntimeError("victim API unreachable")

    def query(self, x):
        out = [None]*len(x); miss_idx, miss_h = [], []
        for i, xi in enumerate(x):
            h = self._hash(xi); hit = self._cache_get(h)
            if hit is None: miss_idx.append(i); miss_h.append(h)
            else: out[i] = hit
        if miss_idx:
            if self.queries_used + len(miss_idx) > self.cfg.budget:
                raise RuntimeError(f"budget exceeded: {self.queries_used}+{len(miss_idx)}>{self.cfg.budget}")
            ys = self._raw_query(x[miss_idx])
            for j, i in enumerate(miss_idx):
                self._cache_put(miss_h[j], ys[j]); out[i] = ys[j]
            self.queries_used += len(miss_idx)
            log.info("budget used: %d/%d", self.queries_used, self.cfg.budget)
        return np.stack(out)
```

SHA-1 hash of float32 bytes is collision-safe at our scale, SQLite gives ACID + crash survival, and the bucket is thread-safe so a multi-worker DataLoader can call `query` concurrently.

### 6.2 Unified query strategy interface

```python
# attacker/strategies.py
from abc import ABC, abstractmethod
import numpy as np, torch, torch.nn.functional as F

class QueryStrategy(ABC):
    @abstractmethod
    def select(self, pool, surrogate, labeled_idx, k): ...

class RandomStrategy(QueryStrategy):
    def select(self, pool, surrogate, labeled_idx, k):
        rest = list(set(range(len(pool))) - set(labeled_idx))
        return list(np.random.choice(rest, size=k, replace=False))

@torch.no_grad()
def _probs(model, x, bs=256):
    model.eval(); device = next(model.parameters()).device
    return torch.cat([F.softmax(model(x[i:i+bs].to(device)), -1).cpu()
                      for i in range(0, len(x), bs)])

class MarginStrategy(QueryStrategy):
    def select(self, pool, surrogate, labeled_idx, k):
        p = _probs(surrogate, pool)
        top2 = p.topk(2, dim=-1).values
        margin = top2[:,0] - top2[:,1]
        margin[labeled_idx] = float("inf")
        return (-margin).topk(k).indices.tolist()

class EntropyStrategy(QueryStrategy):
    def select(self, pool, surrogate, labeled_idx, k):
        p = _probs(surrogate, pool)
        H = -(p * p.clamp_min(1e-12).log()).sum(-1)
        H[labeled_idx] = -1
        return H.topk(k).indices.tolist()

class BALDStrategy(QueryStrategy):
    """BALD via MC-Dropout (Houlsby 2011)."""
    def __init__(self, T=10): self.T = T
    def select(self, pool, surrogate, labeled_idx, k):
        surrogate.train()
        device = next(surrogate.parameters()).device
        ps = []
        with torch.no_grad():
            for _ in range(self.T):
                ps.append(torch.cat([F.softmax(surrogate(pool[i:i+256].to(device)), -1).cpu()
                                     for i in range(0, len(pool), 256)]))
        P = torch.stack(ps); mp = P.mean(0)
        H_mean = -(mp * mp.clamp_min(1e-12).log()).sum(-1)
        E_H = -(P * P.clamp_min(1e-12).log()).sum(-1).mean(0)
        bald = H_mean - E_H; bald[labeled_idx] = -1
        return bald.topk(k).indices.tolist()

class KCenterStrategy(QueryStrategy):
    """Greedy k-center on penultimate features (Sener & Savarese 2018)."""
    @torch.no_grad()
    def _features(self, model, x, bs=256):
        device = next(model.parameters()).device; feats = []
        for i in range(0, len(x), bs):
            f = model.forward_features(x[i:i+bs].to(device)) if hasattr(model, "forward_features") \
                else model(x[i:i+bs].to(device))
            feats.append(F.adaptive_avg_pool2d(f, 1).flatten(1).cpu() if f.ndim == 4 else f.cpu())
        return torch.cat(feats)
    def select(self, pool, surrogate, labeled_idx, k):
        F_all = self._features(surrogate, pool)
        if not labeled_idx: labeled_idx = [int(np.random.randint(len(pool)))]
        d = torch.cdist(F_all, F_all[labeled_idx]).min(dim=1).values
        d[labeled_idx] = -1; chosen = []
        for _ in range(k):
            i = int(d.argmax()); chosen.append(i)
            d = torch.minimum(d, torch.cdist(F_all, F_all[i:i+1]).squeeze(1))
            d[i] = -1
        return chosen

def mixup_pairs(x, alpha=1.0):
    lam = np.random.beta(alpha, alpha, size=len(x)).astype("float32")
    perm = torch.randperm(len(x))
    lam_t = torch.from_numpy(lam).view(-1, *([1]*(x.ndim-1)))
    return lam_t * x + (1 - lam_t) * x[perm]
```

### 6.3 Surrogate training: KD + CE + CutMix + EMA + AMP

```python
# attacker/train.py
import copy, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import timm

def kd_loss(s_logits, t_probs, T=4.0):
    s = F.log_softmax(s_logits / T, dim=-1)
    t = F.softmax(t_probs.clamp_min(1e-12).log() / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T*T)

def mixed_loss(logits, soft, hard, alpha=0.7, T=4.0):
    return alpha*kd_loss(logits, soft, T) + (1-alpha)*F.cross_entropy(logits, hard)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()): s.copy_(p)

def cutmix(x, y_soft, alpha=1.0):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[-2:]
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    cw, ch = int(W*math.sqrt(1-lam)), int(H*math.sqrt(1-lam))
    x1, x2 = max(cx-cw//2, 0), min(cx+cw//2, W)
    y1, y2 = max(cy-ch//2, 0), min(cy+ch//2, H)
    x[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
    lam_eff = 1 - (x2-x1)*(y2-y1)/(W*H)
    return x, lam_eff*y_soft + (1-lam_eff)*y_soft[perm]

def train_surrogate(loader, num_classes, epochs=30, arch="resnet18",
                    lr=3e-4, wd=5e-4, T=4.0, device="cuda"):
    model = timm.create_model(arch, num_classes=num_classes, pretrained=True, drop_rate=0.1).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(loader))
    scaler = GradScaler()
    for ep in range(epochs):
        model.train()
        for x, y_soft in loader:
            x = x.to(device, non_blocking=True); y_soft = y_soft.to(device, non_blocking=True)
            if torch.rand(1) < 0.5: x, y_soft = cutmix(x, y_soft)
            y_hard = y_soft.argmax(-1)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
                loss = mixed_loss(logits, y_soft, y_hard, T=T)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); ema.update(model); sched.step()
        print(f"[ep {ep:02d}] loss={loss:.4f} lr={sched.get_last_lr()[0]:.2e}")
    return ema.shadow  # ALWAYS submit the EMA model
```

`timm.create_model(...)` gives ImageNet-pretrained backbones for free — a 5–10 pp fidelity bump over from-scratch. **CutMix on soft labels** correctly interpolates teacher signals. **EMA**: at the end, return the shadow weights, which consistently score higher on held-out fidelity.

### 6.4 Fidelity evaluation harness

```python
# attacker/eval.py
import torch, torch.nn.functional as F
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def predictions(model, loader, device="cuda"):
    model.eval(); ys = []
    for x, _ in loader: ys.append(model(x.to(device)).argmax(-1).cpu())
    return torch.cat(ys)

def fidelity(s, v): return float((s == v).float().mean())
def task_accuracy(p, gt): return float((p == gt).float().mean())

def fgsm(model, x, y, eps=8/255):
    x = x.clone().requires_grad_(True)
    g = torch.autograd.grad(F.cross_entropy(model(x), y), x)[0]
    return (x + eps*g.sign()).clamp(0, 1).detach()

def pgd(model, x, y, eps=8/255, alpha=2/255, steps=10):
    delta = torch.empty_like(x).uniform_(-eps, eps).requires_grad_(True)
    for _ in range(steps):
        g = torch.autograd.grad(F.cross_entropy(model(x+delta), y), delta)[0]
        delta = (delta + alpha*g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()

@torch.no_grad()
def transfer_rate(surrogate, victim_api, x, y, attack="pgd", **kw):
    surrogate.eval()
    x_adv = (pgd if attack=="pgd" else fgsm)(surrogate, x, y, **kw)
    v_clean = victim_api.query(x.cpu().numpy()).argmax(-1)
    v_adv   = victim_api.query(x_adv.cpu().numpy()).argmax(-1)
    fooled = (v_adv != v_clean) & (v_clean == y.cpu().numpy())
    return float(fooled.mean())
```

A high transfer rate is a strong "is it really the same model?" sanity check — historically, hackathons reward models that mirror **decision boundaries**, not just labels.

### 6.5 GPU efficiency tips

```python
loader = torch.utils.data.DataLoader(
    dataset, batch_size=256, shuffle=True, num_workers=8,
    pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True,
)

# bf16 on Hopper/Ampere — no scaler needed
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = mixed_loss(model(x), y_soft, y_hard) / ACCUM
# torch.compile (PT 2.x)
model = torch.compile(model, mode="max-autotune")
```

Cardinal rules: (1) **pre-cache every query to disk** so architecture changes don't burn budget; (2) `bfloat16` autocast on Hopper/Ampere — numerically stable, no gradient scaler; (3) `persistent_workers=True` removes 5–10 s of fork overhead per epoch; (4) start with `resnet18` (\<2 min/epoch on CIFAR-sized data on A100), scale to `convnext_tiny` or `vit_small_patch16_224` only after fidelity plateaus; (5) `torch.compile(..., mode="reduce-overhead")` is safe; `"max-autotune"` after architecture is fixed.

---

## 7. Metrics, scoring, and the 24-hour playbook

### 7.1 Formal metric definitions

For victim `f` and surrogate `f̂` over distribution `𝒟`:

- **Fidelity** `Fid(f̂, f) = E_{x∼𝒟}[𝟙{argmax f̂(x) = argmax f(x)}]` — the canonical metric.
- **Task accuracy** `Acc(f̂) = E_{(x,y)∼𝒟}[𝟙{argmax f̂(x) = y}]`.
- **High-fidelity extraction** (Jagielski et al.): match the victim's *errors* too.
- **Adversarial transferability** `T(f̂→f, A) = E[𝟙{f(A(f̂,x)) ≠ f(x)}]` for an attack A.

### 7.2 Likely scoring functions on the CISPA server

Past iterations of the CISPA Hackathon Championship (Paris, Vienna, Munich, Barcelona) confirm a CTF format with hidden test set, live leaderboard, larger held-out re-evaluation. Expect one of:

| Pattern | Formula | Optimization target |
|---|---|---|
| Pure fidelity | `score = Fid(f̂, f)` | Match labels, full stop. |
| Fidelity + budget bonus | `score = Fid − λ·queries_used/budget_max` | Save queries; submit early. |
| Fidelity + transfer | `score = α·Fid + β·T(f̂→f)` | Match decision boundaries. |
| KL-based | `score = −E_x KL(f(x)∥f̂(x))` | Distill at low temperature. |
| Per-class macro fidelity | `(1/K) Σ_c Fid_c` | Class-balance your queries. |

For encoder-stealing tasks (B4B-style), expect downstream-task accuracy on a hidden classifier head, or representation alignment metrics like CKA/cosine similarity over a held-out probe set.

### 7.3 The 24-hour playbook

1. **Hour 0–1**: read the task; identify the exact scoring function. Ping the server with a random network to confirm I/O format.
2. **Hour 1–3**: stand up the `VictimAPI` wrapper. Burn 5% of budget on uniform random queries from a public pool (ImageNet/CC12M for vision, the C4 pool or random ASCII strings for NLP) — your warm cache.
3. **Hour 3–8**: train a `resnet18` surrogate on the 5% cache with KD loss, T=4. Submit. This is your floor.
4. **Hour 8–14**: iterate — `MarginStrategy` ∪ `KCenterStrategy`, query, retrain from cached pool with warm-start. Submit every cycle.
5. **Hour 14–20**: stronger architecture (`convnext_tiny`) on the same cache; longer training + EMA.
6. **Hour 20–23**: ensemble — average logits of best 3 surrogates. Often +1–2 fidelity points.
7. **Hour 23–24**: stop. Run sanity checks. Submit best EMA model. **Don't experiment.**

### 7.4 Time-wasting traps

- **DFME-style joint generator training**: needs millions of queries; way over budget.
- **GAN-from-scratch query synthesis**: cold-start problem (per DiMEx 2025).
- **Hyper-parameter sweeps**: AdamW @ 3e-4, WD 5e-4, cosine, T=4 is fine — spend the time on more queries.
- **Confidence calibration**: useless for fidelity unless server scores KL.
- **AutoAttack for measuring transfer**: FGSM at ε=8/255 is enough.

---

## 8. The 2024–2026 frontier and the SprintML-specific signal

### 8.1 Carlini ICML 2024 and the softmax bottleneck

The dominant 2024 result. Production LLMs leak their final-layer projection matrix (up to a basis) via SVD on stacked logit vectors collected through `top-K logprobs + logit_bias`. Confirmed hidden dims: **OpenAI Ada h=1024, Babbage h=2048**, gpt-3.5-turbo redacted; cost <$20 / ~$2 000 respectively. Finlayson et al. (ICLR 2024) showed the model's "image" (low-dim subspace) doubles as a forensic signature for unauthorized-clone detection.

### 8.2 Embedding inversion goes mainstream

Vec2Text (EMNLP 2023), LMI (ICLR 2024), PILS (multi-step prompt inversion 2024–2025), GEIA/ALGEN (few-shot embedding alignment 2025). Composing **steal-then-invert** on encoder APIs is now a standard pipeline.

### 8.3 Stealing diffusion models and using them to steal others

- **Stealix (ICML 2025)**: prompt-evolution genetic algorithm + diffusion-synthesized queries; no class names required.
- **AEDM (ICONIP 2024)**: latent-steering for ~1/200 query budget.
- **DataStealing (NeurIPS 2024)**: multi-trojan exfiltration of training data from federated diffusion training.
- **Dual Student Networks (ICLR 2023)** + descendants **DisGUIDE / DualCOS / DFDS / MGCT (Cybersecurity 2025) / DFHL-RS (AAAI 2024)**: two-student data-free extraction.
- **HoneypotNet (AAAI 2025)** "attack-as-defense": injects backdoor into responses so distilled clones inherit ownership claim.
- **Model-Guardian (2025), MISLEADER (arXiv 2506.02362), MisGUIDE, HODA (TIFS 2024)**: detection / output-perturbation for data-free extraction.

### 8.4 SprintML / Dziedzic / Boenisch 2024–2026 — the high-probability hackathon material

The lab's interview record explicitly states tasks are designed from their own research (CISPA hackathon interview). Direct candidates:

- **Encoder stealing under B4B active defense** (Dubiński et al. NeurIPS 2023). Bypass strategies: sybil orchestration with controlled diversity, embedding-space coverage minimization, learning per-user transformations.
- **GNN extraction under restricted budget + hard label** — Podhajski, Dubiński, Boenisch, Dziedzic et al., **AAAI 2026 Oral**; and **ADAGE active defense** (Xu, Boenisch, Dziedzic, 2025, arXiv 2503.00065) and **ECAI 2024** efficient extraction paper [arXiv 2405.12295].
- **Calibrated Proof-of-Work bypass** (Dziedzic et al., ICLR 2022 Spotlight).
- **Dataset / membership inference** on diffusion or image-autoregressive models — **CDI (Dubiński, Kowalczuk, Boenisch, Dziedzic, CVPR 2025)** detects with ≥99% confidence from 70 samples; **Privacy Attacks on IARs (Kowalczuk, Dubiński, Boenisch, Dziedzic, ICML 2025)** TPR@FPR=1% of 86.38%, extracts 698 training images from VAR-d30; **LLM Dataset Inference (Maini, Jia, Papernot, Dziedzic, NeurIPS 2024)**; **Strong MIAs (Hayes, Dziedzic, Cooper, Choquette-Choo, Boenisch et al., NeurIPS 2025)**; **Curation Leaks (Wahdany, Jagielski, Dziedzic, Boenisch, ICLR 2026)**; **NIDs (Rossi, Marek, Boenisch, Dziedzic, ICLR 2026)**.
- **Watermark radioactivity / breakage** — **BitMark (Kerner, Meintz, Zhao, Boenisch, Dziedzic, NeurIPS 2025)**, **SERUM (ICLR 2026)**, *Are Watermarks for Diffusion Models Radioactive?* (Dubiński et al., ICLR 2025 Workshop) — answer: **no, current diffusion watermarks are not radioactive**. **Data Provenance for IAR Generation (Zhao et al., ICLR 2026)** — post-hoc detection without watermarking.
- **Memorization localization** — **NeMo (Hintersdorf, Struppek, Kersting, Dziedzic, Boenisch, NeurIPS 2024)** localizes diffusion memorization to individual cross-attention neurons; **Localizing Memorization in SSL (Wang et al., NeurIPS 2024)**; **CLIPMem (Wang et al., ICLR 2025)**; **Precise Parameter Localization for Text in DM (Staniszewski et al., ICLR 2025)**.
- **Auditing private LLM adaptations** — **Open LLMs are Necessary (Hanke et al., NeurIPS 2024)**; **POST private soft prompts (Wang et al., ICML 2025)**; **Benchmarking Empirical Privacy (Marek et al., ICLR 2026 Oral)**.

The unifying signal: SprintML hackathon tasks are evaluated **automatically and quantitatively** ("solution sent to our servers, compared to other solutions" — Dziedzic). Master the metric definitions in **B4B, Carlini 2024, CDI, Kowalczuk 2025, ADAGE, Strong MIAs** and you have a real edge.

---

## 9. What to memorize / pre-implement before the hackathon — the cheat sheet

**Memorize these formulas and numbers:**

- KD loss: `T² · KL(softmax(z_T/T) ‖ softmax(z_S/T))` + α·CE.
- Logit-bias SVD trick (Carlini 2024): rank of stacked logit matrix = hidden dim; left singular vectors span column space of unembedding W.
- Softmax inversion (Tramèr 2016): `log(p_k/p_K) = (w_k−w_K)·x + (b_k−b_K)` — linear in weights.
- BADGE: gradient embedding `g_x = ∂L(C_θ(x), ŷ)/∂W_last`, then k-MEANS++ seeding.
- k-center / coreset: greedy farthest-point in feature space, 2-OPT.
- **CIFAR-10 reference numbers**: Knockoff random 60k → 0.97×; ActiveThief k-center 10k → 0.82; DFME 20M → 0.92×; SEEKER 100k → 93.97%.
- **LLM Carlini cost**: <$20 to recover OpenAI Ada projection; ~$2 000 for gpt-3.5-turbo full final layer.
- **Encoder stealing cost**: ~$72 (SimCLR RN-50) vs. ~$5 714 to pre-train.

**Pre-implement these (have them tested, in a git repo, with a Dockerfile, before May 2026):**

1. `VictimAPI` wrapper with SQLite cache, token-bucket, retry, budget enforcement.
2. `RandomStrategy`, `MarginStrategy`, `EntropyStrategy`, `KCenterStrategy`, `BALDStrategy`, plus `mixup_pairs` helper.
3. KD trainer with mixed loss (KD+CE), CutMix on soft labels, EMA, AMP (bf16), cosine LR, AdamW, `torch.compile`, gradient clipping at 5.0.
4. timm-based architecture zoo: `resnet18`, `convnext_tiny`, `vit_small_patch16_224`, `efficientnet_b0` — all ImageNet-pretrained.
5. Eval harness: `fidelity`, `task_accuracy`, `transfer_rate(FGSM, PGD)`, per-class fidelity, confusion-matrix divergence.
6. Submission utilities: state_dict packaging, safetensors export, model-size sanity check.

**Pre-stage these datasets on disk:**

- ImageNet-1K (subset OK), CIFAR-100, STL-10, CC12M (subset), MS-COCO captions.
- A small frozen Stable-Diffusion pipeline (`runwayml/stable-diffusion-v1-5`) for synthetic queries.
- A pretrained CLIP encoder (`openai/clip-vit-base-patch32`) and a sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`) for warm-start of encoder-stealing tasks.

**Read these papers in this order:**

1. Tramèr et al., USENIX Security 2016 (theory backbone).
2. Orekondy et al., CVPR 2019 (Knockoff Nets — the canonical recipe).
3. Pal et al., AAAI 2020 (ActiveThief — active learning recipe).
4. Jagielski et al., USENIX Security 2020 (definitions + lower bounds).
5. Carlini et al., CRYPTO 2020 (cryptanalytic baseline).
6. Truong et al., CVPR 2021 (DFME) and Kariyappa et al., CVPR 2021 (MAZE).
7. Dziedzic et al., ICLR 2022 (PoW), ICML 2022 (SSL stealing), NeurIPS 2022 (Dataset Inference).
8. Liu et al., CCS 2022 (StolenEncoder); Sha et al., CVPR 2023 (Cont-Steal); Dubiński et al., NeurIPS 2023 (B4B).
9. Carlini et al., ICML 2024 (logit-bias SVD); Finlayson et al., ICLR 2024.
10. Morris et al., EMNLP 2023 + ICLR 2024 (vec2text + LMI).
11. SprintML 2024–2026: NeMo, CDI, Privacy Attacks on IARs, Stealix, ADAGE, AAAI 2026 GNN paper, BitMark, Curation Leaks.

**Mental rules at the keyboard:**

- Optimize **fidelity, never task accuracy**, unless the task page says otherwise.
- **Pre-cache every query**, then iterate on architecture / loss / strategy without paying budget twice.
- **Submit early and often**; the leaderboard is your best diagnostic.
- **EMA model > live model** for submission, consistently.
- **`uncertainty filter then k-center`** is the strongest single active-learning recipe.
- When a defense fires (PRADA, AM, B4B), **switch to natural in-distribution / diffusion-generated queries** before adding query volume.
- **Ensemble averaging of 3 best surrogates** at the end → +1–2 fidelity points for free.
- **Don't experiment in the last hour.** Submit the EMA model that scored highest, with checkpoint-5 as backup.

Good hunting at CISPA — may your fidelity exceed the leaderboard threshold, your queries stay under budget, and your EMA model survive the held-out re-evaluation.