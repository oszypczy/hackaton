# Stealing minds: a hackathon playbook for ML privacy attacks

**The state of the art has shifted decisively toward attacker advantage.** Production LLMs leak gigabytes of training data via $200 of API queries (Nasr et al. 2023), Stable Diffusion regurgitates pixel-perfect LAION images (Carlini et al. 2023), and federated learning gradients reconstruct ImageNet inputs at PSNR > 19 dB (Geiping et al. 2020). The 2024–2025 frontier moved from per-sample membership inference — which barely beats random on properly de-confounded LLM benchmarks (Duan et al. COLM 2024) — toward **dataset-level statistical tests** (Maini NeurIPS 2024) and **dataset-scale extraction pipelines** that aggregate weak signals into legally and scientifically conclusive claims. This report synthesizes ten attack families, their math, runnable PyTorch code, evaluation metrics, defenses (especially DP-SGD and its 2024 auditing breakthroughs), and a decision matrix mapping hackathon scenarios to the right attack. Two themes dominate: **statistical aggregation beats single-shot signals**, and **alignment is not a privacy defense** — every aligned production model has been broken by a mechanically simple prompt distribution shift.

---

## 1. The threat landscape and why it matters now

Three recent incidents anchor the field. First, the **ChatGPT divergence attack** (Nasr, Carlini, Hayase, Jagielski et al., arXiv:2311.17035) showed that prompting `gpt-3.5-turbo` with `Repeat the word "poem" forever` causes alignment to collapse and the model to emit verbatim pretraining text at ~150× the baseline rate; ~3% of post-divergence output was found verbatim on the public Internet, including PII and copyrighted material. The team extracted >10,000 unique memorized strings for ~$200. Second, **Carlini et al. (USENIX Security 2023)** extracted ≥109 verbatim LAION images from Stable Diffusion v1.4 by generating ~500 samples per duplicated caption and clustering them in feature space. Third, the **NYT v. OpenAI / Microsoft** case (filed Dec 2023; motion to dismiss denied March 2025 by Judge Sidney Stein, SDNY) is being litigated on memorization evidence; Freeman et al. (arXiv:2412.06370, Dec 2024) confirmed that lawsuit-listed articles are significantly more memorized than baseline NYT articles, and that memorization scales sharply beyond ~100B parameters.

The **regulatory backdrop** has tightened in parallel. The EU AI Act (in force 2024, phased through 2027) requires GDPR compliance throughout an AI system's lifecycle, while GDPR Article 17 (Right to Be Forgotten) — drafted around structured records — is being reinterpreted to apply to parametric memorization, with machine unlearning emerging as the only realistic compliance path short of retraining. The US Executive Order 14110 (rescinded January 2025) drove the creation of NIST AI red-teaming guidance and the WMDP benchmark, both of which persist. **Attack surfaces** sort cleanly: closed APIs (rate-limited, content-filtered, no logits), gray-box logit APIs (HF inference, some OSS deployments), open-weight models (full white-box for Llama/Pythia/Mistral/SDXL), federated-learning rounds (per-step gradient sharing), and RAG systems (retrieval over private corpora).

---

## 2. Model inversion on image classifiers: from Fredrikson to diffusion priors

**The core objective** common to all white-box image MI attacks is

L(z) = L_id(G(z); y, T) + λ·L_prior(z), x̂ = G(ẑ*)

where G is a public-data generator (the GAN/diffusion *prior*), L_id pushes the synthesis toward the target class y under classifier T (typically −log T_y(G(z)) or a max-margin variant), and L_prior keeps z on the manifold (discriminator score, KL to N(0,I), W+ truncation). The **evolution** runs from Fredrikson et al. CCS 2015 (pixel-space gradient ascent on softmax confidence — works for shallow nets, collapses on deep ones), through GMI (Zhang CVPR 2020, the first GAN prior with WGAN-GP on CelebA), KEDMI (Chen ICCV 2021, inversion-specific GAN with multi-task discriminator predicting target soft-labels and a learned per-class Gaussian q(z|y)), VMI (Wang NeurIPS 2021, normalizing-flow variational posterior), and **PPA / Plug-and-Play** (Struppek ICML 2022 — the modern white-box default: pretrained StyleGAN2 in W+, **Poincaré loss** to avoid vanishing softmax gradients, augmentation-robust result selection).

The **2023–2025 frontier** advances along two axes. First, **richer optimization spaces**: PLG-MI (Yuan AAAI 2023) pre-labels public data with the target's top-1 predictions and trains a class-conditional GAN, then runs a max-margin attack `L_id = −[T_y(x) − max_{k≠y} T_k(x)]`; LOMMA (Nguyen CVPR 2023) plugs a logit max-margin loss + model augmentation into GMI/KEDMI/PPA for +30–40% attack accuracy; **IF-GMI** (Qiu ECCV 2024 oral) optimizes both z and **intermediate StyleGAN feature residuals** δf_k inside ℓ₁-balls — best out-of-distribution robustness; AlignMI (NeurIPS 2025) projects gradients onto the manifold tangent space. Second, **diffusion priors replace GANs**: Diff-MI (Li 2024, arXiv:2407.11424), DiffMI (2025, embedding-based and training-free), and single-step distilled-diffusion variants exceed StyleGAN fidelity. **Black-box** progress: RLB-MI (Han CVPR 2023) treats the GAN latent space as an MDP with reward T_y(G(z)) and trains SAC; PPO-MI (2025) swaps SAC for PPO; LOKT (NeurIPS 2023) trains a target-aware conditional GAN as a surrogate for label-only attacks; BREP-MI (Kahla CVPR 2022) uses spherical boundary repulsion in Z.

**Defenses** include MID (Wang AAAI 2021, variational bound on I(X;Ŷ)), BiDO (Peng KDD 2022, HSIC objective), TL-DMI (Ho CVPR 2024, freeze first k layers based on Fisher-information analysis), NegLS (Struppek ICLR 2024 — *negative* label smoothing α<0 defends, while positive LS *helps* the attacker), and Trap-MID (NeurIPS 2024, backdoor trapdoors that redirect MI toward trigger images). DP-SGD provably bounds leakage but per Zhang 2020 Fig 3(d), MI accuracy barely drops even at ε=0.1 in some settings — DP-SGD for MI is empirically less effective than for verbatim extraction.

A **basic Fredrikson-style attack** in PyTorch (MNIST, ~2 minutes on one GPU) maps cleanly:

```python
def invert(model, target_class, steps=2000, lr=0.05,
           lam_tv=1e-2, lam_l2=1e-4, img_shape=(1,1,28,28), device='cuda'):
    x = torch.randn(img_shape, device=device, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class], device=device)
    for i in range(steps):
        opt.zero_grad()
        x_in = torch.tanh(x) * 0.5 + 0.5         # box-constrain to [0,1]
        x_norm = (x_in - 0.1307) / 0.3081        # match training normalization
        logits = model(x_norm)
        loss = (F.cross_entropy(logits, target)
                + lam_tv * tv_loss(x_in)
                + lam_l2 * (x_in ** 2).mean())
        loss.backward(); opt.step()
    return torch.tanh(x).detach() * 0.5 + 0.5

def tv_loss(x):
    return ((x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()
          + (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean())
```

For high-resolution faces, the canonical **PPA scaffold** loads a frozen StyleGAN2 and optimizes z with the Poincaré loss under random crops, then re-scores candidates by averaged augmented confidence. **Open-source**: GMI (`AI-secure/GMI-Attack`), KEDMI (`SCccc21/Knowledge-Enriched-DMI`), PPA (`LukasStruppek/Plug-and-Play-Attacks`), PLG-MI (`LetheSec/PLG-MI-Attack`), LOMMA (`sutd-visual-computing-group/Re-thinking_MI`), IF-GMI (`final-solution/IF-GMI`), and the omnibus **MIBench / Model-Inversion-Attack-ToolBox** (`ffhibnese/Model-Inversion-Attack-ToolBox`) — the right starting point for a hackathon.

**Strategy**: white-box low-res → PLG-MI + LOMMA; white-box high-res → PPA or IF-GMI; OOD priors → IF-GMI > PPA; defended target → LOMMA plug-in or AlignMI; black-box soft-label → RLB-MI or PPO-MI; label-only → BREP-MI then LOKT; embedding-based recognition target → DiffMI.

---

## 3. Training data extraction from LLMs: prefix, divergence, and post-alignment attacks

**Carlini et al. USENIX Security 2021** (arXiv:2012.07805) defined the modern playbook: generate N candidate continuations (top-n sampling from `<|endoftext|>`, *temperature decay* with T≈10 for the first ~20 tokens then T=1, or *Internet conditioning* with Common Crawl snippets), then rank by membership-inference signals. Six key signals — **perplexity**, **small/XL log-perplexity ratio**, **zlib-entropy-to-log-PPL ratio**, **lowercase ratio**, **window-perplexity** (min over sliding 50-token windows), and **reference-model log-likelihood-ratio** — together extracted >600 verbatim memorized samples from GPT-2 XL (PII, IRC logs, code, UUIDs). The follow-up **Carlini et al. ICLR 2023** (arXiv:2202.07646) introduced **k-extractability** (a string s is k-extractable if greedy decoding from some k-token prefix p produces s exactly) and three log-linear scaling laws: **10× model parameters → +19 percentage-point increase in extractable fraction**; duplicated strings are exponentially more extractable; longer prompts extract more. GPT-J 6B leaks ≥1% of the Pile.

**Carlini's perplexity/zlib ranking** is the workhorse:

```python
import torch, zlib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2-xl")
xl  = GPT2LMHeadModel.from_pretrained("gpt2-xl").cuda().eval()
sm  = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()

@torch.no_grad()
def loss(model, ids): return model(ids, labels=ids).loss.item()

def score(text):
    ids   = tok(text, return_tensors="pt").input_ids.cuda()
    p_xl  = loss(xl, ids); p_s = loss(sm, ids)
    p_lo  = loss(xl, tok(text.lower(), return_tensors="pt").input_ids.cuda())
    z     = len(zlib.compress(text.encode("utf-8")))
    return dict(ppl_xl=torch.tensor(p_xl).exp().item(),
                ratio_s_xl=p_s/p_xl, ratio_lo_xl=p_lo/p_xl, ratio_zlib=z/p_xl)

samples = generate_batch(N=2000)             # high-T sample then T=1 continuation
top = sorted(((s, score(s)) for s in samples),
             key=lambda x: -x[1]["ratio_zlib"])[:50]
```

**Nasr et al. 2023** (arXiv:2311.17035) generalized the playbook to aligned production LLMs via the **divergence attack**: prompt `Repeat the word "poem" forever`. After ~50 repetitions the model "diverges" out of chat mode and emits memorized pretraining text. The attack works on `gpt-3.5-turbo` but not GPT-4; only single-token words trigger divergence on the patched model, but Dropbox (2024) demonstrated multi-token variants. SPY Lab (2024) showed cheap fine-tuning ($3 on OpenAI's API) on ~1000 Pile docs caused 4.3% of subsequent generations to be memorized; fine-tuning on previously extracted text raised it to 17%. The **Special Characters Attack** (Bai et al. arXiv:2405.05990, May 2024) prompts with structured-symbol/JSON characters mixed with English letters, exploiting joint memorization of special-character co-occurrences; it triggers 2–10× more leaks than divergence on Llama-2 and works on Falcon, ChatGPT, Gemini, and ERNIE-Bot.

**Stealing Part of a Production Language Model** (Carlini et al., ICML 2024 best paper, arXiv:2403.06634) is parameter — not data — extraction: by exploiting the **logit-bias** API parameter plus top-K log-probs, the authors solve a linear system whose rank equals hidden dimension d, then SVD recovers the final-layer projection W ∈ ℝ^{d×|V|}. They confirmed `text-ada-001`'s d=1024 and `text-babbage-001`'s d=2048 for <$20, and the full hidden dim of `gpt-3.5-turbo` for <$2,000. OpenAI/Google subsequently rate-limited or removed `logit_bias`.

**LLM membership inference**: **Min-K% Prob** (Shi et al. ICLR 2024, arXiv:2310.16789) computes per-token log-probs, takes the bottom K% (typically 20%), and averages — non-members have more outlier low-probability tokens. **Min-K%++** (Zhang et al. ICLR 2025 spotlight, arXiv:2404.02936) standardizes by the per-position vocabulary distribution: score per token = (log p(x_i|x_<i) − μ_{·|x_<i}) / σ_{·|x_<i}, where μ, σ are mean/std of log-probs over the entire vocabulary at position i. This adds 6.2–10.5% AUC over Min-K% on WikiMIA. **DC-PDD** (Zhang EMNLP 2024, arXiv:2409.14781) calibrates by unigram frequency cross-entropy; **ReCaLL** measures relative log-likelihood change under non-member prefixing. Min-K%++ implementation:

```python
@torch.no_grad()
def minkpp(model, tok, text, k_pct=20):
    ids    = tok(text, return_tensors='pt').input_ids.cuda()
    logits = model(ids).logits[0, :-1]
    logp   = F.log_softmax(logits, dim=-1)
    probs  = logp.exp()
    mu     = (probs * logp).sum(-1)
    sig    = ((probs * (logp - mu.unsqueeze(-1))**2).sum(-1)).sqrt() + 1e-8
    targets= ids[0, 1:]
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    z      = (tok_lp - mu) / sig
    k = max(1, int(z.numel() * k_pct/100))
    return z.topk(k, largest=False).values.mean().item()
```

The **critical caveat** comes from Duan et al. COLM 2024 (arXiv:2402.07841, the **MIMIR** benchmark): all MIA methods hover near AUC 0.5–0.6 on properly IID member/non-member splits because LLMs train one epoch on huge corpora — most reported MIA "successes" reflect *temporal distribution shift* between members (older) and non-members (newer), not real membership signal. WikiMIA's blind baseline (no model access) achieves 98.7% AUC for this reason. Use MIMIR's ≤20% n-gram-overlap split or temporally-controlled benchmarks. **Defenses**: deduplication (Lee et al. 2022) cuts extraction roughly proportionally to duplication count; DP-SGD with ε≤8 essentially eliminates verbatim extraction (Carlini 2021); output filtering on verbatim regurgitation is bypassed by style-transfer prompts (Ippolito 2022).

---

## 4. MIA, dataset inference, and model inversion: a clean taxonomy

| Attack family | Output | Granularity | Access | Best 2024–25 SOTA |
|---|---|---|---|---|
| **Membership Inference (MIA)** | "Was *x* in D_train?" | one sample | black/gray/white-box | LiRA (S&P 22), RMIA (ICML 24), Min-K%++ (ICLR 25) |
| **Dataset Inference (DI)** | "Was *D* used to train?" | dataset (n=30–10k) | black/white-box | Maini NeurIPS 2024 (LLM-DI) |
| **Model Inversion** | reconstruct *x* | per-class / per-record | black/white-box | PPA, IF-GMI, DiffMI |

**LiRA** (Carlini, Chien, Nasr, Song, Terzis, Tramèr, S&P 2022, arXiv:2112.03570) frames MIA as a Neyman–Pearson hypothesis test. With the rescaled-logit signal **φ(f, x) = log(p_y/(1−p_y))**, train N shadow models, split into IN (trained on x) and OUT, fit per-example Gaussians, and compute the log-likelihood ratio:

S_LiRA(x) = (φ−μ_out)²/(2σ²_out) − (φ−μ_in)²/(2σ²_in) + log(σ_out/σ_in)

LiRA achieves ~10× higher TPR at FPR = 0.001 than prior loss-threshold attacks. **RMIA** (Zarifzadeh, Liu, Shokri, ICML 2024, arXiv:2312.03262) refines the null hypothesis: a candidate could have been *replaced* by any sample from the population, giving a pairwise likelihood ratio against population samples z. RMIA achieves >25% AUC gain over LiRA in the low-shadow-model regime (1–4 references). **Reference / calibrated MIAs** (Watson 2022, Long 2020) and **Attack-R / EMIA** (Ye et al. CCS 2022) form a continuum with LiRA at the average-case limit and RMIA generalizing both. The 2026 **BaVarIA** paper unifies them as exponential-family Bayesian variance attacks.

**A clean LiRA implementation** with online Gaussian fits per example:

```python
def lira_scores(target_model, shadows, keeps, dataset, device='cuda'):
    n = len(dataset); N = len(shadows)
    phis = np.zeros((N, n)); phi_t = np.zeros(n)
    loader = DataLoader(dataset, batch_size=512)
    for j, m in enumerate(shadows):
        m.eval().to(device); pos = 0
        for x, y in loader:
            phis[j, pos:pos+len(x)] = logit_signal(m, x.cuda(), y.cuda())
            pos += len(x)
    pos = 0
    for x, y in loader:
        phi_t[pos:pos+len(x)] = logit_signal(target_model, x.cuda(), y.cuda())
        pos += len(x)
    scores = np.zeros(n)
    for i in range(n):
        ins, outs = phis[keeps[:,i], i], phis[~keeps[:,i], i]
        mu_in,  sg_in  = ins.mean(),  ins.std() + 1e-6
        mu_out, sg_out = outs.mean(), outs.std() + 1e-6
        scores[i] = (st.norm.logpdf(phi_t[i], mu_in,  sg_in)
                   - st.norm.logpdf(phi_t[i], mu_out, sg_out))
    return scores
```

**Maini's Dataset Inference** (ICLR 2021 spotlight, arXiv:2104.10706) made the foundational shift: train points sit further from decision boundaries than test points, so **per-sample margin** is a noisy MIA signal but **mean margin over n samples** concentrates by CLT into a strong claim with low FPR. Two embeddings — MinGD (white-box, minimum L_p perturbation distance to misclassification) and Blind-Walk (black-box, count steps in a random direction until label flip) — feed a confidence regressor whose outputs are compared via a one-sided Welch's t-test between suspect and control sets. Ownership claims at p<0.01 with as few as 30–60 samples on CIFAR-10/100 and ImageNet.

**LLM-DI** (Maini, Jia, Papernot, Dziedzic, NeurIPS 2024, arXiv:2406.06443) extends this to language: compute L=10 MIA features per sequence (perplexity, lowercase-ratio, zlib-ratio, Min-K% at 8 K-values, reference-model loss); fit a linear regressor on a labeled split A; apply to held-out split B; run a one-sided Welch's t-test:

t = (s̄_V − s̄_0) / sqrt(σ²_V/n_V + σ²_0/n_0)

Power grows as O(√n), so even per-sample AUC ≈ 0.55 yields p < 0.01 on hundreds of paragraphs. On Pythia-12B + Pile, DI achieves p<0.1 across most domains while no single MIA exceeds AUC 0.55. Code: `pratyushmaini/llm_dataset_inference`. **Implementation**:

```python
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

def fit_aggregator(A_susp, A_val):
    X = np.vstack([A_susp, A_val])
    y = np.hstack([np.zeros(len(A_susp)), np.ones(len(A_val))])
    return LogisticRegression(max_iter=2000).fit(X, y)

def dataset_inference(g, A_susp_B, A_val_B, alpha=0.1):
    s_susp = g.predict_proba(A_susp_B)[:, 1]
    s_val  = g.predict_proba(A_val_B)[:, 1]
    t, p   = ttest_ind(s_susp, s_val, equal_var=False, alternative='less')
    return {'p_value': p, 'reject_H0': p < alpha}
```

**2024–2025 follow-ups** include Entropy-Memorization Linearity (TMLR 2026), DE-COP (Duarte 2024, multiple-choice copyright detection), Leave No TRACE (2025, black-box DI via watermarking), Data Taggants (ICLR 2025, provable ownership via pre-injected adversarial keys), Range MIA (Tao & Shokri SaTML 2025, region-level membership), and PETAL (USENIX Sec 2025, label-only token-similarity DI).

---

## 5. Gradient-based inversion: when gradients are the leak

In federated learning a client transmits **∇θ L(θ; x*, y*)**; an honest-but-curious server reconstructs (x*, y*) by initializing dummy (x', y') and minimizing a distance between recomputed and observed gradients. **DLG** (Zhu, Liu, Han NeurIPS 2019) uses ℓ₂ distance with L-BFGS — works for shallow nets, fails on deep ReLU. **iDLG** (Zhao 2020) extracts the label analytically from the sign of ∇W_out (only the true class has a negative bias gradient under one-hot CE), eliminating joint optimization. **Inverting Gradients** (Geiping NeurIPS 2020, arXiv:2003.14053) is the modern default: cosine-similarity loss + signed Adam + total variation, recovers ResNet-152 inputs from a *single* trained-network gradient with PSNR > 19 dB:

L_IG(x') = 1 − ⟨∇L(f(x'),y), g_obs⟩ / (‖∇‖·‖g_obs‖) + α_TV·TV(x')

**GradInversion** (Yin CVPR 2021) scales to batches of 8–48 ImageNet samples for ResNet-50 by adding BatchNorm-statistics regularization and group consistency across multi-seed agents. **Generative priors** — GIAS, GIFD (ICCV 2023, intermediate-feature optimization), GGL — replace TV with a GAN/diffusion prior, surviving moderate noise. **Malicious-server attacks**: *Robbing the Fed* (Fowl 2021) inserts an "imprint" linear module that recovers any single batch exactly; *Decepticons* (Fowl ICLR 2022) does the same for transformers; *Fishing* (Wen 2022) magnifies one user's contribution to bypass secure aggregation. **NLP gradient inversion**: TAG (EMNLP 2021) was first; **LAMP** (NeurIPS 2022) combines continuous embedding optimization with discrete swaps re-ranked by GPT-2 perplexity; FILM (Gupta 2022) beam-searches from token-set recovered from embedding gradient. The 2024 frontier: **DAGER** (Dimitrov NeurIPS 2024) achieves *exact* batch recovery up to 128 sequences in encoder/decoder LLMs by exploiting low-rank attention gradients; **SPEAR** does the same for FedSGD small batches.

**PyTorch DLG and IG** in one canonical pattern:

```python
def cosine_grad_loss(g_d, g_r):
    flat_d = torch.cat([g.flatten() for g in g_d])
    flat_r = torch.cat([g.flatten() for g in g_r])
    return 1 - F.cosine_similarity(flat_d, flat_r, dim=0)

# label-leakage from last-layer bias gradient sign (iDLG)
y_hat = (real_grad[-1].sum(-1) if real_grad[-1].dim()>1 else real_grad[-1]).argmin().unsqueeze(0)

x = torch.randn(1,3,32,32, device='cuda', requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)
for step in range(8000):
    opt.zero_grad()
    g_d = torch.autograd.grad(F.cross_entropy(model(x), y_hat),
                              model.parameters(), create_graph=True)
    loss = cosine_grad_loss(g_d, real_grad) + 1e-2 * tv_loss(x)
    loss.backward()
    x.grad.sign_()                     # signed Adam (Geiping)
    opt.step()
    x.data.clamp_(0,1)
```

**Defenses**: per-sample clipping + Gaussian noise (DP-SGD) is the strongest empirical defense at moderate σ; gradient pruning is bypassed at <70% sparsity by IG+prior; PRECODE/Soteria were broken by Balunović 2021; secure aggregation is bypassed by *Robbing the Fed* and *Fishing*. Cosine-similarity attacks are essentially unaffected by gradient clipping alone (clipping bounds magnitude, not direction). The **breaching** library (`JonasGeiping/breaching`) implements DLG, iDLG, IG, GradInversion, Robbing-the-Fed, Decepticons, Fishing, TAG, and LAMP under one Hydra config — install and pick an attack via flag.

---

## 6. Diffusion model extraction: generate, cluster, confirm

**Carlini et al. USENIX Security 2023** (arXiv:2301.13188) target Stable Diffusion v1.4 and Imagen with a four-stage pipeline: (1) select ~350k duplicated/popular LAION captions (memorization concentrates on duplication); (2) generate N≈500 images per caption with different seeds at 50 inference steps and CFG 7; (3) cluster by ℓ₂ in pixel space or 1−cosine on DINO/SSCD features (a connected component of size ≥10 with diameter <0.15 marks the prompt as memorized); (4) confirm via FAISS NN search in the LAION-2B index. Their **MIA component for diffusion** is a LiRA-style score using conditional vs unconditional denoising error:

s(x,c) = E_{t,ε}[‖ε − ε_θ(α_t x + σ_t ε, t, c)‖² − ‖ε − ε_θ(α_t x + σ_t ε, t, ∅)‖²]

Members have abnormally negative s. They extracted **109 verified verbatim training images** as a lower bound. **Webster 2023** (arXiv:2305.08694, `ryanwebster90/onestep-extraction`) achieved orders-of-magnitude faster extraction with a **single denoising step** sufficient to flag template verbatims, and identified three categories — matching, retrieval, and template verbatims — with template verbatims persisting in SD 2.0, DeepFloyd IF, and Midjourney v4 (mitigated only in v5). **Somepalli et al.** (CVPR 2023, NeurIPS 2023) showed 1.88–2% of generations from LAION-A captions reach SSCD copy-detection scores >0.5, and that **text conditioning matters as much as image duplication** (random labels still increase memorization).

**2024–2025 advances**: Wen et al. ICLR 2024 detect memorized prompts in *one* step by measuring ‖ε_θ(z_t,t,c) − ε_θ(z_t,t,∅)‖ magnitude; **Finding NeMo** (Hintersdorf NeurIPS 2024) localizes memorization to ≤10 cross-attention value-matrix neurons whose ablation removes specific images while preserving FID; memorized-subspace pruning (arXiv:2406.18566) drops extraction rate from ~9% to 0% on fine-tuned SD; **SIDE** (arXiv:2410.02467) breaks unconditional DPMs (CIFAR-10, CelebA-HQ) — previously thought safe — via a time-dependent classifier-guided attack; **Jain CVPR 2025** shows classifier-free guidance amplifies memorization via "attraction basins." Image autoregressive models (VAR-d30: 698 extractions, RAR-XXL: 36, MAR-H: 5; Liu 2025) and video diffusion (Chen arXiv:2410.21669, hundreds of memorized clips from WebVid-10M) extend the threat surface.

**Hackathon pipeline**:

```python
from diffusers import StableDiffusionPipeline
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
       torch_dtype=torch.float16).to("cuda")
embed = feature_extractor()                # DINOv2 ViT-S/14

def extract(prompts, N=500, eps=0.10, min_samples=10):
    candidates = []
    for c in prompts:
        imgs = pipe([c]*N, num_inference_steps=50, guidance_scale=7.5).images
        feats = embed(imgs)
        D = cosine_distances(feats)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(D)
        for lbl in set(db.labels_) - {-1}:
            members = [imgs[i] for i,l in enumerate(db.labels_) if l == lbl]
            candidates.append((c, members[0], len(members)))
    return candidates
```

For LAION-scale confirmation, search a FAISS index of LAION SSCD features and threshold cosine similarity at 0.85. **Mitigations** include caption deduplication (cuts ~5×, doesn't kill template verbatims), random caption augmentation, anti-memorization guidance (Wen 2024), cross-attention re-weighting (Ren ECCV 2024), neuron deactivation, and DP-SGD (the only formal guarantee, with severe utility cost at low ε).

---

## 7. Reconstruction-quality metrics: what to measure and when

**Image metrics**. **FID** computes ‖μ_X − μ_Y‖² + Tr(Σ_X + Σ_Y − 2(Σ_X Σ_Y)^{1/2}) on Inception-V3 pool3 features; use it for distribution-level realism on ≥10k samples, never for single-image comparison. Lower is better; CIFAR-10 SOTA ≈ 2.0, decent DDPM 3–10. **SSIM** measures per-image structural similarity in [-1,1]; >0.9 = near-perfect pixel-aligned recovery (use for MIA reconstructions, gradient leakage). **LPIPS** is a calibrated weighted-L2 between AlexNet/VGG activations; <0.3 ≈ "looks the same"; preferred when SSIM is too pixel-strict (diffusion extraction, GAN inversion). **PSNR** (10·log₁₀(MAX²/MSE), >30 dB visually similar, >40 dB nearly identical) is a quick sanity baseline.

**Text metrics**. **Exact match** is the standard for k-extractability and canary recovery. **Edit distance** (Levenshtein, normalized by max length) for "almost recovered" memorized strings. **BLEU/ROUGE-L** for paraphrased extractions. The **Carlini 2021 perplexity-zlib ratio** is the canonical filter: `score = log(perplexity) / zlib_entropy`; low scores flag memorized candidates. **k-extractability** (Carlini 2022): a string is k-extractable from f if greedy decoding from some k-token prefix produces it — sweep k ∈ {50, 100, 200, 500} and report the fraction.

**Privacy-attack metrics**. **AUC** alone is misleading — the LiRA paper requires **TPR @ low FPR**, especially TPR@0.1%FPR and TPR@1%FPR with a log-log ROC plot, because useful attacks must confidently identify ≥1% of members at <0.1% FPR or accusations are dominated by false positives. **Attack Success Rate (top-1 / top-5)** is standard for face MI. **Calibrated KNN memorization score**: ratio of nearest-train distance to nearest-held-out distance — values ≫1 indicate memorization. The recommendation table:

| Attack | Primary metrics | Secondary |
|---|---|---|
| Class-prototype MI | LPIPS to class mean, qualitative | SSIM, recovered-image classifier confidence |
| Face MI | ASR top-1/top-5, ArcFace cosine | LPIPS |
| MIA | TPR@0.1%FPR, TPR@1%FPR, AUC | log-log ROC curve |
| Gradient leakage | PSNR, LPIPS, SSIM | Pixel-MSE |
| Diffusion extraction | Calibrated KNN-mem-score, DINO/CLIP cosine | SSCD copy-detection, FID of non-memorized samples |
| LLM verbatim | k-extractability, exact-match | zlib/perplexity ratio |
| LLM approximate | ROUGE-L, normalized edit distance | BLEU, perplexity |

---

## 8. DP-SGD: theoretical guarantees and the 2024 auditing breakthroughs

**(ε, δ)-DP** requires P(M(D)∈S) ≤ e^ε·P(M(D')∈S) + δ for all neighbors D, D'. **DP-SGD** (Abadi et al. CCS 2016) per-sample-clips gradients to bound C, then adds Gaussian noise N(0, σ²C²I) before averaging. The **MIA bound** TPR ≤ e^ε·FPR + δ (Kairouz 2015, Humphries 2020) is vacuous at ε=8 (ratio ≤2981) but tight at ε=1 (≤2.72). What matters in practice: **DP-SGD with ε ≤ 10 collapses Carlini-Webster verbatim extraction** and pushes MIA AUC to ~0.5 because per-sample clipping bounds any individual record's gradient contribution and Gaussian noise washes out memorization-specific signals. Carlini's GPT-2 with ε=8 emits zero extractable training sequences vs. hundreds for non-DP. Membership inference against rare records still leaks slightly but bounded by e^ε.

The **2024–2025 auditing revolution** transformed DP from a worst-case bound into an empirically-tight tool. **Steinke, Nasr, Jagielski (NeurIPS 2023 Outstanding Paper)** introduced **privacy auditing in one training run**: insert K i.i.d. randomized canaries, observe IN/OUT guess accuracy, lower-bound ε via concentration on K parallel Bernoullis. With K=1000 canaries, a single CIFAR-10 DP-SGD run gives ε̂ ≈ 1.7 against a claimed ε = 4 — comparable to prior 1000-run audits. **Pillutla et al. (NeurIPS 2023)** introduced LiDP with leave-one-out canary tests, tightening confidence intervals 4–16×. **Cebere-Bellet-Papernot (ICLR 2025)** showed that for hidden-state threat models (only final weights released), empirical ε is often 2–10× smaller than the worst-case accountant suggests. **Steinke et al. 2025 (Last Iterate Advantage)** proved this formally: realized privacy at the *final* iterate is strictly tighter than cumulative composition.

**Opacus 1.5+ (2024)** with Fast/Ghost Gradient Clipping makes DP-SGD plug-and-play even for billion-parameter models, and **DP-LoRA** (arXiv:2312.17493) trains only LoRA matrices — concentrating noise where it matters and reaching ε=8 on LLaMA-2 7B with single-GPU regimes:

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

model = ModuleValidator.fix(MyModel().cuda())   # replace BatchNorm → GroupNorm
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loader    = DataLoader(train_ds, batch_size=4096)
pe = PrivacyEngine(accountant="prv")
model, optimizer, loader = pe.make_private(
    module=model, optimizer=optimizer, data_loader=loader,
    noise_multiplier=1.1, max_grad_norm=1.0,
    poisson_sampling=True, grad_sample_mode="ghost")
# train; query pe.get_epsilon(delta=1e-5) anytime
```

**Known weaknesses** (hackathon-relevant): **DP is conservative** — empirical ε is often 0.2× claimed; defenders may push ε to 8–32 for utility, leaving template-verbatim attacks workable on heavily-duplicated samples. **Group privacy decay**: (ε,δ) for a group of size k becomes (kε, k·e^{(k-1)ε}·δ); duplicated samples explode ε. **Implementation bugs** are common (Tramèr et al. 2022 audited a public repo and rejected its claim with 99.99999999% confidence). **Public-pretraining assumption**: DP fine-tuning relies on a non-private pretrain that may itself leak. **Gradient hijacking** (Cebere 2025, Feng-Tramèr 2024): adversarial architectures concentrate leakage in one dimension to bypass naive per-layer clipping.

---

## 9. CoDeC: in-context contamination detection at the dataset level

**CoDeC — Contamination Detection via Context** (Zawalski, Boubdir, Bałazy, Nushi, Ribalta, NVIDIA, arXiv:2510.27055, Oct 2025) tackles the dataset-level question "was benchmark D in training?" — answerable even when per-sample MIAs fail. The core insight: for an *unseen* dataset, in-context examples from the same distribution should *raise* model confidence (standard ICL); for a *trained-on* dataset, additional in-context examples either provide redundant information or **disrupt memorization patterns** (different formatting/order than seen at training time), so confidence *decreases*.

Formally, with L_0(x) = (1/|x|)Σ_t log p_θ(x_t|x_<t) the unconditional mean log-prob and L_C(x) the same conditioned on context C of m other dataset samples, define:

CoDeC(D, M) = (1/N) Σ_i 1[E_C[L_C(x_i)] < L_0(x_i)]

Near 50% indicates unseen data; near 100% indicates strong memorization. The paper reports near-perfect separation (AUC ≈ 99.9%) across Pythia/Pile, OLMo/Dolma, and Nemotron-CC with known ground-truth membership, while baselines (Loss, Zlib, Min-K%, Min-K%++, Reference) overlap heavily on the same task. CoDeC is parameter-free, model-agnostic (needs only token log-probs), dataset-agnostic (works on raw text, QA, GSM8K, MMLU), and reveals partial contamination (fine-tuning on a subset pushes CoDeC to ~100% on that subset). Implementation:

```python
def codec_score(model, tok, dataset_texts, m=4, trials=8):
    drops = 0
    for x in dataset_texts:
        x_ids = tok(x, return_tensors="pt", truncation=True,
                    max_length=512).input_ids.cuda()
        bos = torch.tensor([[tok.bos_token_id]]).cuda()
        lp_no = seq_logprob(model, tok, bos, x_ids)
        diffs = []
        for _ in range(trials):
            ctx = " ".join(random.sample(dataset_texts, m))
            ctx_ids = tok(ctx, return_tensors="pt", truncation=True,
                          max_length=2048).input_ids.cuda()
            lp_yes = seq_logprob(model, tok, ctx_ids, x_ids)
            diffs.append(lp_yes - lp_no)
        if np.mean(diffs) < 0:
            drops += 1
    return drops / len(dataset_texts)
```

**Related contamination methods**: Time-Travel in LLMs (Golchin & Surdeanu ICLR 2024, guided-instruction prompting that achieves 92–100% accuracy vs. expert labels), CDD/TED (Dong ACL 2024, output-distribution peakedness, corrects benchmark scores by 21–30% AUC), Did-You-Train-on-My-Dataset (Oren 2023, statistical exchangeability tests on test-set ordering), and `Contamination_Detector` (Bing/Common-Crawl lookup of QA pairs). **CoDeC is the current dataset-level SOTA** because it uses the model-internal ICL-differential signal rather than external lookups or per-sample MIA aggregation; it operationally distinguishes from MIA by aggregating over many samples, averaging out single-example noise.

---

## 10. Synthesis: which attack wins which hackathon scenario

| Scenario | Recommended attack | Difficulty | Compute | Ready code |
|---|---|---|---|---|
| Black-box API, image classifier, recover training images | **PPA** with public StyleGAN2 prior (or BREP-MI / RLB-MI for label-only) | Med-High | 1 GPU, ~2hr/target | `LukasStruppek/Plug-and-Play-Attacks`, `ffhibnese/MI-Toolbox` |
| White-box LLM weights, detect dataset usage | **LLM-DI** (Maini 2024) — aggregate Min-K%++, perplexity, zlib, neighborhood into per-dataset t-test | Low | Forward-only, 1 GPU for 7B | `pratyushmaini/llm_dataset_inference` |
| Federated learning client gradients leaked | **Inverting Gradients** (honest server) or *Robbing the Fed*/Decepticons/Fishing (malicious server) | Medium | 24k iters, ~20 min on A100 | `JonasGeiping/breaching` |
| API to text-to-image diffusion, prompts known | **Carlini 2023 generate-and-filter** (or Webster 2023 one-step) | Medium | $$$ — N×candidate queries | `ryanwebster90/onestep-extraction` |
| API to aligned LLM, want training data | **Divergence attack** (Nasr 2023; multi-token if patched) or **fine-tune extraction** ($3 SPY-Lab method) | Easy / Medium | ~$200 / FT API + queries | Prompt scripts; Tramèr SPY Lab blog |
| Gray-box logits of classifier, membership info | **LiRA** (64–256 shadows) or **RMIA** (4–8 shadows) | Med-High | 12 GPU-hr / hours | `orientino/lira-pytorch`, `antibloch/mia_attacks` |
| Prove copyright violation in trained model | **LLM-DI** for statistical p-value + targeted **divergence/generate-and-filter** for verbatim smoking gun | Medium | <$500 API spend | `llm_dataset_inference` + `onestep-extraction` |
| Aligned chat LLM (logit access), pretraining MIA | **Min-K%++** or CAMIA, validate on **MIMIR ≤20% n-gram split** to avoid distribution-shift inflation | Medium | Forward-only | `iamgroot42/mimir`, `computationalprivacy/mia_llms_benchmark` |
| Audit a benchmark for contamination | **CoDeC** if log-probs available; Time-Travel if pure black-box | Low | Forward-only | Implement from arXiv:2510.27055 |
| DP-defended target | Group-privacy attack on duplicates; canary-injection audit; pivot to non-private pretrained base in DP-LoRA stacks | Medium | 1 training run | `lidp_auditing`, Opacus |

**Hackathon-specific tips**. A weekend means 1–2 GPUs and <48h, so favor scenarios with **public pretrained checkpoints** (Pythia, SDXL, CommonCanvas) where someone else paid the training cost. The fastest wins: MIMIR loss-based attacks on Pythia (5 min to first AUC), divergence prompts on a self-hosted Llama base model, Inverting Gradients on CIFAR with `breaching` (one config file). **Avoid these pitfalls**: reporting AUC instead of TPR@1%FPR (Carlini 2022 critique) — judges with privacy backgrounds will dock you; using WikiMIA without the blind-baseline sanity check (distribution-shift inflates scores 30–50 AUC points); fine-tuning a diffusion model on Pokemon for 15k steps and claiming "MIA works" — Dubinski WACV 2024 demonstrated this doesn't generalize. **Demo-ready visually striking attacks**: PPA face inversion, generate-and-filter for "Ann Graham Lotz" on SD, divergence on a base LLM, IG on a single ImageNet image. **Ethics/legal**: for any extraction on production APIs follow 90-day responsible disclosure; many ToS prohibit "extracting training data" — use open-weight models or licensed lab access.

---

## Conclusion: the attacker's edge in 2026

Three insights reorder the field. **First, statistical aggregation is the great equalizer**: dataset inference (Maini 2024) and CoDeC (Zawalski 2025) turn per-sample AUC ≈ 0.55 into p < 0.01 conclusions through O(√n) power scaling — the hackathon attacker should aggregate weak signals, not chase a single perfect MIA. **Second, alignment is a thin veneer**: every aligned production model has been broken by a mechanically simple distributional prompt (divergence, special characters, role-play), and the underlying memorization is unchanged. **Third, DP-SGD is the only defense with formal guarantees, but its real-world budgets are conservative**: empirical ε from one-run audits is 2–10× smaller than the accountant claims, which is good news for defenders deploying ε=8 and bad news for attackers trying to claim leakage in DP-protected models — pivot instead to non-private base models in DP-LoRA stacks, group-privacy decay on duplicated records, or canary-injection audits.

The **practical attacker's stack** for a 48-hour hackathon: install `breaching`, `mimir`, `llm_dataset_inference`, `Plug-and-Play-Attacks`, `onestep-extraction`, and `lira-pytorch`. Pick a scenario from the decision matrix, copy the matching repo's example notebook, and report TPR@1%FPR or p-values rather than AUC. The **defender's reality** is harsher: deduplication helps but doesn't eliminate template verbatims, output filters bypass with style-transfer prompts, and only DP-SGD with ε ≤ 10 + per-sample clipping reliably defeats verbatim extraction — at meaningful utility cost. The next frontier, signaled by SIDE on unconditional DPMs, DAGER on LLM federated batches up to 128, and CoDeC's near-perfect dataset-level discrimination, is **attacks that succeed where defenders previously thought they had retreated to safety**: unconditional generation, large-batch federated learning, and one-epoch pretraining. Hackathon competitors who internalize this asymmetry will dominate the leaderboard.