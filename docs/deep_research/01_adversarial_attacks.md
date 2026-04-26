# The 24-hour adversarial attacks playbook

**This guide is built around a single principle: in a hackathon, query efficiency and correct normalization win more points than algorithmic novelty.** The fastest path to scoring is library-based PGD/AutoAttack for white-box, transfer-attack-then-Square for black-box, and aggressive sanity checks for gradient masking. Every section below is organized for grep-and-skim. Bolded gotchas are where teams routinely lose hours.

The research underlying this document covers FGSM through 2025-era attacks (BSR, TESSER, MALT, DiffAttack), the three dominant Python libraries (torchattacks 3.5.1, ART 1.19.1, Foolbox 3.3.4), defense-aware techniques (BPDA, EOT, randomized smoothing), and ViT-specific methods (PatchFool, TGR, PNA). Use the decision tree in §1 to pick attacks; use §11 as a reference card during the run.

---

## 1. First 4 hours playbook (priority-ordered)

**Hour 0 — Setup and environment (all three teammates working in parallel).**

```bash
pip install torch torchvision timm
pip install torchattacks==3.5.1
pip install adversarial-robustness-toolbox==1.19.1
pip install foolbox==3.3.4
pip install git+https://github.com/fra31/auto-attack
pip install git+https://github.com/RobustBench/robustbench
```

Set up a shared repo with: `models.py` (NormalizedModel wrapper from §5), `attacks.py` (a thin dispatcher), `eval.py` (ASR computation), `data.py` (loaders for CIFAR-10 and ImageNet val subset). **Verify clean accuracy matches the model card before attacking anything** — if your ResNet-50 reports 30% clean acc on ImageNet, you have a normalization bug, not a hard challenge.

**Hour 1 — Quick wins on white-box challenges.** For every white-box target, immediately run `torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)` with 5 restarts. This breaks all undefended models in seconds. If success rate is below 99%, the model is adversarially trained — escalate to AutoAttack standard.

**Hour 2 — Transfer attack baseline for every black-box challenge.** Build an ensemble surrogate (ResNet-50 + DenseNet-121 + ViT-B/16 from timm), run **MI-DI-TI-SI-FGSM** (or its torchattacks shorthand `SINIFGSM` plus `DIFGSM`/`TIFGSM` chained) at ε=16/255 for ImageNet, ε=8/255 for CIFAR-10. Submit the resulting adversarial examples to the API with one query each. Anything that succeeds on the first query costs zero query budget.

**Hour 3 — Score-based black-box for survivors.** For each sample where transfer failed, run **Square Attack** with `n_queries=5000, p_init=0.05`. This is the single highest-value attack for black-box challenges; it is hyperparameter-robust, gradient-free, and breaks most undefended models in fewer than 400 queries on ImageNet.

**Hour 4 — Gradient masking sanity check.** For any defended target where attacks underperform, run the five-test obfuscated-gradients diagnostic in §9. Decide before hour 5 whether you're dealing with a real defense (escalate to AutoAttack rand + EOT + BPDA) or just gradient masking (switch to Square or SPSA, ignore the gradients).

**Decision tree (memorize this).** White-box undefended → PGD-20. White-box adversarially trained → AutoAttack standard. White-box randomized defense → AutoAttack rand with EOT=20. White-box preprocessing defense → BPDA + PGD. Black-box with score access → transfer first, then Square. Black-box label-only → HopSkipJump for L₂, RayS for L∞. Black-box with surrogate prior → P-RGF or GFCS. Targeted on black-box → multiply your query budget by 5–10× and use DLR loss, not cross-entropy.

---

## 2. White-box attacks: the workhorses

### FGSM — your sanity check, never your final attack

Goodfellow et al. 2014. The single-step formulation is **x_adv = x + ε · sign(∇_x L(f(x), y))**. Use it only to confirm your pipeline works; it leaves 30–80% of samples unbroken on undefended models because of the linear approximation. **The most common pitfall is computing the gradient with respect to a normalized input but applying the perturbation in [0,1] space**, which silently scales ε by 1/std. Always operate the attack in [0,1] and let the model wrapper handle normalization.

### PGD — the workhorse

Madry et al. 2018. Iterative FGSM with random initialization and projection onto the ε-ball after each step. The canonical configuration is **ε=8/255, α=2/255, steps=20, random restarts=5** for CIFAR-10 L∞, and ε=4/255, α=1/255, steps=20 for ImageNet. Step size α should satisfy α ≈ 2.5·ε/steps so you can traverse the ball; with 20 steps the standard choice α=ε/4 gives just enough headroom for the random start. **Restarts matter most against adversarially trained models** — increase to 10 restarts and 100 steps when robust accuracy is the metric.

The clipping order matters: **project onto the ε-ball first, then clip to [0,1]**. Reversing this silently violates the budget at image boundaries. In a custom implementation, detach between iterations or your autograd graph grows and you OOM by step 50.

```python
def pgd_linf(model, x, y, eps=8/255, alpha=2/255, steps=20, restarts=5):
    """PGD-L∞ with random restarts. x in [0,1]. Model wraps normalization internally."""
    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), -float('inf'), device=x.device)
    for _ in range(restarts):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        delta = (x + delta).clamp_(0, 1) - x   # ensures x+delta in [0,1]
        delta.requires_grad_(True)
        for _ in range(steps):
            logits = model(x + delta)
            loss = F.cross_entropy(logits, y, reduction='none')
            grad = torch.autograd.grad(loss.sum(), delta)[0]
            with torch.no_grad():
                delta = delta + alpha * grad.sign()
                delta = torch.clamp(delta, -eps, eps)        # 1) project onto ε-ball
                delta = (x + delta).clamp_(0, 1) - x          # 2) then to valid pixels
            delta.requires_grad_(True)
        # keep best per-sample across restarts (untargeted: highest loss)
        with torch.no_grad():
            cur_loss = F.cross_entropy(model(x + delta), y, reduction='none')
        better = cur_loss > best_loss
        best_adv[better] = (x + delta)[better].detach()
        best_loss[better] = cur_loss[better]
    return best_adv
```

### C&W — when you need minimum-norm perturbations

Carlini & Wagner 2017. Used when the metric is L₂ distortion at fixed success, not success at fixed ε. The trick is the **change-of-variables x_adv = ½(tanh(w) + 1)** which removes box constraints from the optimizer; you then minimize **‖x_adv − x‖₂² + c · max(max_{j≠y} Z(x_adv)_j − Z(x_adv)_y, −κ)** over w using Adam. Two non-obvious knobs: **κ (kappa) controls confidence** (raise to evade detectors that flag low-confidence predictions; values 10–40 are common); **c is found by binary search**, typically 9 outer steps doubling/halving from c=0.01.

**Torchattacks' CW does NOT implement the binary search** (it fixes c=1 by default, which is too small for many models); for proper minimum-norm results use Foolbox's `L2CarliniWagnerAttack(binary_search_steps=9)` instead. C&W is roughly 100× slower than PGD per sample; reserve it for evaluation, not optimization.

### DeepFool — fast minimum-norm geometry

Moosavi-Dezfooli et al. 2016. Linearizes the decision boundary at each step and projects the input onto the closest hyperplane. **It produces smaller perturbations than FGSM at comparable computational cost** but is dominated by C&W in solution quality. Use it as a free L₂-distance probe: torchattacks `DeepFool(model, steps=50, overshoot=0.02)` returns minimum-norm perturbations in seconds.

### AutoAttack — the gold standard

Croce & Hein ICML 2020. A parameter-free ensemble of **APGD-CE, APGD-DLR (targeted, 9 target classes), FAB-T, and Square Attack (5000 queries)**. The DLR loss `−(z_y − max_{j≠y} z_j) / (z_{π_1} − z_{π_3})` is scale-invariant in logits, which fixes a failure mode of cross-entropy on networks with very large output magnitudes. **APGD's adaptive step-size schedule eliminates the need to tune α** — it halves the step size when progress stalls.

Use the official `fra31/auto-attack` repo for any leaderboard claim. Three versions:

| Version | Use case | Wall-clock (CIFAR-10 WRN-28-10, 1000 imgs) |
|---|---|---|
| `standard` | Deterministic models, leaderboard eval | 10–25 min |
| `plus` | Stronger eval (5 restarts × all attacks) | 60–120 min |
| `rand` | **Randomized defenses, EOT=20** | 5–10 min |

**ImageNet ResNet-50 standard runs 30–90 min for 1000 images** — budget accordingly. AutoAttack flags potentially unreliable evaluations when later attacks recover much more accuracy than earlier ones; treat that flag as evidence of gradient masking and switch to `rand`.

### Momentum variants — MI-FGSM and NI-FGSM

Dong et al. CVPR 2018, Lin et al. ICLR 2020. PGD with normalized momentum **g_{t+1} = μ·g_t + ∇L/‖∇L‖₁** smooths the gradient direction and roughly doubles black-box transfer success. **Always use μ=1.0**; the paper's ablation shows a unimodal max there. NI-FGSM adds a Nesterov look-ahead computed at `x + α·μ·g_t` before taking the gradient. These are critical for transfer attacks (§3) but rarely improve white-box performance against the same model.

---

## 3. Black-box attacks: where the hackathon is won or lost

### The transfer-attack-then-query workflow

The single most query-efficient strategy is: **build a strong transfer adversarial first, submit one query to verify, and only refine with queries when transfer fails.** Roughly half of an ImageNet test set falls to a good ensemble transfer attack at zero query cost.

### Transfer attacks: building the surrogate

**Always use an ensemble of diverse architectures.** Liu et al. 2017 established that **logit-level ensembling** (sum the pre-softmax logits, then take cross-entropy) outperforms loss-level and prediction-level ensembling. Equal weights `w_k = 1/K` work fine. Include both CNN and ViT in the ensemble — cross-architecture transfer is the bottleneck and a mixed ensemble narrows it from 30–55% to 60–80% on ImageNet.

```python
class EnsembleLogitModel(nn.Module):
    def __init__(self, models): super().__init__(); self.models = nn.ModuleList(models)
    def forward(self, x):
        return sum(m(x) for m in self.models) / len(self.models)   # logit average

# usage with timm: ResNet-50 + DenseNet-121 + ViT-B/16, all wrapped with NormalizedModel
ensemble = EnsembleLogitModel([resnet, densenet, vit]).eval().cuda()
atk = torchattacks.VMIFGSM(ensemble, eps=16/255, alpha=1.6/255, steps=10, decay=1.0, N=20, beta=1.5)
adv = atk(x, y)
```

### The transfer attack stack (ε=16/255, T=10, α=ε/T)

The 2025 practical recipe combines four orthogonal tricks. **MI-FGSM** with μ=1 stabilizes optimization. **DI-FGSM** randomly resizes-and-pads the input each iteration with `resize_rate=0.9, diversity_prob=0.5`. **TI-FGSM** convolves the gradient with a fixed Gaussian kernel (size 15, σ=3 by default) to find translation-invariant perturbations. **SI-FGSM** averages gradients over m=5 scaled copies `x/2^i`. The composite **SI-NI-TI-DIM** attack reaches ~93.5% transfer on adversarially trained ImageNet models.

For 2023+ improvements: **VMI-FGSM** (Wang & He CVPR 2021) adds a variance-tuning term over N=20 neighbor samples within β·ε=1.5·ε radius, smoothing the gradient further. **SSA** (Long et al. ECCV 2022) augments in the DCT spectrum with σ=16, ρ=0.5, N=20; it reaches 95.4% average success across nine SOTA defenses. **BSR** (Wang et al. CVPR 2024) splits the input into n×n blocks and randomly shuffles + rotates them across K=20 draws, particularly disrupting attention heatmaps for ViT targets. **Use SSA or BSR when targeting ViTs from a CNN-heavy ensemble.**

torchattacks-version-compatible chaining:

```python
# stacked transfer attack: MI + DI + TI + SI in one go
atk = torchattacks.SINIFGSM(ensemble, eps=16/255, alpha=1.6/255, steps=10, decay=1.0, m=5)
# DI and TI can be added by composing or using TIFGSM(...) alone:
atk = torchattacks.TIFGSM(ensemble, eps=16/255, alpha=1.6/255, steps=10, decay=1.0,
                          kernel_name='gaussian', len_kernel=15, nsig=3,
                          resize_rate=0.9, diversity_prob=0.5)   # this version includes DI inside
adv = atk(x, y)
```

### Score-based black-box: Square Attack is your default

Andriushchenko et al. ECCV 2020. Random search with localized square updates: at each step, pick a random square of side h in the image and replace its pixels with ±ε on each channel. **No gradient estimation, robust to gradient masking, hyperparameter-light.** Median queries on undefended ImageNet ResNet-50 are 70–340 with 0% failure rate at ε=0.05 L∞. The squared-update locality matters because CNNs are sensitive to spatially correlated perturbations.

Hyperparameters: **`p_init=0.05` for L∞ (proportion of pixels in the initial square), `p_init=0.1` for L₂**, halved at iterations [10, 50, 200, 1000, 2000, 4000, 6000, 8000] when total budget is 10000 (rescale schedule for other budgets via `resc_schedule=True`). **Use the margin loss `max_{j≠y} z_j − z_y`, not cross-entropy** — torchattacks' default `loss='margin'` is correct; ART defaults vary by version.

```python
# torchattacks (cleanest)
atk = torchattacks.Square(model, norm='Linf', eps=8/255, n_queries=5000,
                          n_restarts=1, p_init=0.05, loss='margin', resc_schedule=True)
adv = atk(x, y)

# ART (also fine, NumPy I/O)
from art.attacks.evasion import SquareAttack
adv = SquareAttack(estimator=cls, norm=np.inf, eps=8/255, max_iter=5000,
                   p_init=0.05, batch_size=64).generate(x=x_np, y=y_onehot)
```

A from-scratch Square Attack core loop is 30 lines and worth keeping in the repo for non-standard architectures or unusual loss surfaces:

```python
def square_attack_linf(predict, x, y, eps, n_queries=5000, p_init=0.05):
    """predict: callable returning logits. x in [0,1] of shape (N,C,H,W)."""
    N, C, H, W = x.shape
    # Stripe initialization: vertical bars of ±ε per channel
    init = (torch.randint(0, 2, (N, C, 1, W), device=x.device).float() * 2 - 1) * eps
    x_adv = (x + init).clamp_(0, 1)
    margin = lambda logits: logits.gather(1, y[:,None]).squeeze() - \
                            (logits.scatter(1, y[:,None], -1e9).max(1).values)
    L_best = margin(predict(x_adv))                     # negative margin = adversarial
    for i in range(n_queries):
        p = p_init * 0.5 ** sum(i >= t for t in [10, 50, 200, 1000, 2000, 4000, 6000, 8000])
        h = max(int(round((p * H * W) ** 0.5)), 1)
        r = torch.randint(0, H - h + 1, (N,)); c = torch.randint(0, W - h + 1, (N,))
        x_new = x_adv.clone()
        for n in range(N):
            sign = (torch.randint(0, 2, (C, 1, 1), device=x.device).float() * 2 - 1) * eps
            patch = (x[n:n+1, :, r[n]:r[n]+h, c[n]:c[n]+h] + sign).clamp_(0, 1)
            x_new[n:n+1, :, r[n]:r[n]+h, c[n]:c[n]+h] = patch
        L_new = margin(predict(x_new))
        improved = L_new < L_best                       # untargeted: smaller margin is better
        x_adv[improved] = x_new[improved]; L_best[improved] = L_new[improved]
    return x_adv
```

### Score-based alternatives and when to use them

**NES** (Ilyas et al. ICML 2018) estimates the gradient by antithetic Gaussian sampling: `g ≈ (1/(σ·n)) Σ [F(x+σδᵢ) − F(x−σδᵢ)] · δᵢ`. Use `σ=1e-3, n=50, lr=0.01`, total ~10000 query budget. Brittle against gradient masking. **SPSA** (Spall 1992; Uesato 2018) uses Rademacher ±1 vectors instead of Gaussians, two queries per estimate; preferred when you suspect obfuscated gradients because the larger perturbation magnitude produces signal even when local gradients vanish. **Bandits-TD** (Ilyas ICLR 2019) adds time and data priors to NES, ~1100 mean queries on ImageNet vs ~1735 for vanilla NES. **P-RGF / GFCS** use a surrogate gradient as a prior and reach ~70–280 average queries on ImageNet — best in class when you have any half-decent surrogate. **SimBA-DCT** (Guo ICML 2019) is a 20-line baseline reaching ~582 median queries on ResNet-50; keep it as a debugging tool.

```python
def nes_grad(predict, x, y, sigma=1e-3, n=50):
    """Antithetic NES gradient estimate. predict returns logits."""
    deltas = torch.randn(n // 2, *x.shape[1:], device=x.device)
    pos = predict(x + sigma * deltas).gather(1, y.expand(n // 2, 1)).squeeze()
    neg = predict(x - sigma * deltas).gather(1, y.expand(n // 2, 1)).squeeze()
    weights = (pos - neg).view(-1, 1, 1, 1)             # use margin loss for stability
    return (weights * deltas).sum(0) / (sigma * n)
```

### Decision-based (label-only): HopSkipJump for L₂, RayS for L∞

Chen-Jordan-Wainwright IEEE S&P 2020. **HopSkipJump** estimates the gradient direction at the decision boundary using only label feedback via `ĝ ≈ (1/B)·Σ sign(φ_x(x_t + δ_b u_b)) · u_b`, where φ is the misclassification indicator. Three-step iteration: binary search to project onto the boundary, sign-based gradient direction estimation with batch size B that grows like B₀√t, geometric step search along ĝ. Hyperparameter-light (B₀=100, T set by budget). **5–10× more query-efficient than the original Boundary Attack.** Median ImageNet untargeted L₂ in 1k–10k queries.

**RayS** (Chen & Gu KDD 2020) reformulates L∞ hard-label attack as discrete ray search over sign vectors θ ∈ {±1}^d. Hyperparameter-free, no gradient estimation, no adversarial loss. Strongest sanity check for "falsely robust" L∞ models — if RayS breaks a model that AutoAttack didn't, the model has gradient masking. **QEBA** (Li CVPR 2020) is HopSkipJump in a low-frequency DCT subspace, saving ~10× queries.

Foolbox has the canonical HopSkipJump implementation; ART has both and works with `BlackBoxClassifier` for true API-only access:

```python
# ART pattern for label-only API
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
from art.utils import to_categorical

def predict_fn(x_batch):                                # x_batch: NHWC numpy in [0,1]
    out = []
    for img in x_batch:
        cls = api_call((img * 255).astype(np.uint8))    # returns int class
        out.append(to_categorical([cls], nb_classes=1000)[0])
    return np.array(out)

cls = BlackBoxClassifier(predict_fn=predict_fn, input_shape=(224,224,3),
                         nb_classes=1000, clip_values=(0., 1.))
atk = HopSkipJump(classifier=cls, targeted=False, norm=2,
                  max_iter=50, max_eval=10000, init_eval=100, init_size=100)
x_adv = atk.generate(x=x_np)                            # iterate with x_adv_init for budget control
```

### Hybrid and surrogate-guided attacks

**P-RGF** (Cheng NeurIPS 2019) biases NES sampling toward the surrogate gradient direction; ~270 average queries on ImageNet at 99.3% success. **GFCS** (Lord ICLR 2022) and **LeBA** (Yang NeurIPS 2020, ~178 average queries on ImageNet) jointly use surrogate priors for direction selection and learning. **ODS** (Tashiro NeurIPS 2020) initializes black-box attacks by maximizing diversity in the surrogate's logit outputs, cutting queries by ~2×. The practical workflow: transfer attack first → if fails, use the transfer perturbation as initialization for Square Attack → if Square stalls in 500 queries, switch to P-RGF with the same surrogate.

### Model stealing as preparation

**Knockoff Nets** (Orekondy CVPR 2019) trains a surrogate by querying the victim with proxy data (ImageNet works for unrelated tasks). Roughly **60k queries** for fine-grained classifiers, 100k–1M for ImageNet-grade victims. Functional similarity ≥ 75–85% of victim accuracy, sufficient for transfer attacks to bounce back at 30–60% success. **Jagielski et al. USENIX 2020** uses active learning (query points near the surrogate's current decision boundary), reaching >95% functional similarity at 10⁵–10⁶ queries. In a 24-hour hackathon, only invest in stealing if the API is generous and you have ≥4 hours; otherwise rely on public pretrained models in an ensemble.

### Query budget reference card

| Attack | Type | CIFAR-10 (avg) | ImageNet (avg) | Notes |
|---|---|---|---|---|
| Transfer (MI-DI-TI-SI) | 0 query | n/a | 0–1 | 80–95% on CNN, 30–60% on ViT |
| NES | score | ~500 | 1500–2000 | Brittle to masking |
| Bandits-TD | score | ~300 | ~1100 | NES + priors |
| **Square** | score | <200 | **70–400** | First choice |
| SimBA-DCT | score | ~300 | ~600 median | 20-line backup |
| P-RGF / GFCS / LeBA | score+surr | <100 | **70–280** | Needs surrogate |
| HopSkipJump | decision | 1k–3k | 1k–10k | First decision-based choice |
| QEBA | decision | ~500 | 1k–5k | HSJ in DCT subspace |
| RayS | decision (L∞) | ~1k | 5k–10k | L∞ sanity check |
| Boundary | decision | 1k–5k | 30k+ | Inferior to HSJ |

### Detecting rate limiting and API quirks

Sudden drops in success rate as queries pile up suggest input preprocessing (JPEG, resize) is destroying pixel-level perturbations — switch to low-frequency or DCT-band perturbations that survive resampling. **Output scores that don't change for small ε** indicate quantized confidences; switch to sign-based methods (Square, RayS, Sign-OPT). **Repeated identical queries returning different scores** indicate a stochastic defense — average n_eot=10–20 queries per evaluation. Watch for HTTP 429 and latency jumps; throttle to 1 QPS, parallelize across sessions only if TOS allows. Defenses like **Blacklight** detect adversarial queries via probe similarity — perturb at the patch level rather than pixel level to evade.

---

## 4. Targeted vs untargeted: when targeted costs 5–10× more

Implementation differences are minor: untargeted maximizes loss with respect to the true label (`+sign(∇L)`); targeted minimizes loss with respect to the target label (`−sign(∇L_t)`). The **target class strategy** is what matters. **Least-likely class** (argmin of logits) is the hardest target and produces the strongest attacks but bloats query counts. **Second-most-likely class** is by far the cheapest — often achievable in 2–3× the untargeted budget. **Specific class targeting** sits in the middle.

For black-box query budgets, expect **targeted to cost 5–10× more queries than untargeted**: Square targeted needs ~100k queries on ImageNet vs ~300 untargeted; HopSkipJump targeted needs 10k–25k vs 1k–10k. Targeted transfer is the worst case — success rates drop from 80–95% (untargeted) to <20% (targeted) for cross-architecture transfer; this gap closes only with very strong methods like Logit-Margin or Po+Trip targeted attacks.

**Use the DLR loss for targeted attacks, not cross-entropy.** Cross-entropy saturates as the target logit grows; DLR (`(z_t − max_{j≠t} z_j) / (z_{π_1} − z_{π_3})`) remains scale-invariant. APGD-T uses DLR-T loss and is the targeted component in AutoAttack. In torchattacks: `atk.set_mode_targeted_least_likely(kth_min=1)` or `atk.set_mode_targeted_by_label()` then pass target labels as the second argument to `atk()`.

---

## 5. Perturbation tricks per Lp ball

**L∞ (the default).** Step-size warmup helps for high-step PGD: start at α=ε/2 and decay to ε/8 over the first 25% of iterations. Overshooting then projecting back (`step_size = 1.5·α` followed by ε-ball clip) accelerates progress on flat loss landscapes. The clip-after-projection order is non-negotiable. Random initialization within the ε-ball is essential against adversarial training — it's the primary reason PGD-AT was discovered.

**L₂.** Gradient normalization (`g / ‖g‖₂`) is mandatory; otherwise step size is dimension-dependent. Trust-region methods (DDN, Decoupled Direction and Norm) decouple the direction from the magnitude and converge to smaller perturbations than vanilla L₂-PGD. C&W with binary search remains the gold standard for minimum L₂ distortion.

**L₀ (sparse attacks).** **JSMA** (Papernot et al. 2016) computes per-pixel Jacobian saliency `S(x,y_t)[i] = (∂Z_t/∂x_i) · |Σ_{j≠t} ∂Z_j/∂x_i|` and modifies one pixel at a time. **One-pixel attack** (Su et al. 2019) uses differential evolution and breaks ~70% of CIFAR-10 ResNets with a single pixel. **EAD** (Chen et al. 2018) adds an L₁ regularizer to C&W. Use sparse attacks only when the challenge specifically scores L₀ — they're 100× slower than PGD.

**Universal perturbations.** Moosavi-Dezfooli et al. CVPR 2017 — a single δ that fools many inputs. Compute by iterating over a held-out batch and accumulating DeepFool-style updates. Useful for batch attack scoring but rarely needed in hackathons.

### EOT for randomized defenses

Athalye et al. ICML 2018. For a randomized defense f with transformation distribution T, the gradient is `E_{t~T}[∇L(f(t(x)))]`, estimated by averaging gradients over n samples per PGD step. **Typical n=20** for image transforms; AutoAttack `rand` uses n=20. Heavy randomized defenses (BaRT, GradDiv) need n=80–500. **DiffPure-style purification** needs n=20–40 plus the segment-wise backprop trick from DiffAttack.

```python
def pgd_eot(model, x, y, eps=8/255, alpha=2/255, steps=20, n_eot=20):
    delta = torch.empty_like(x).uniform_(-eps, eps); delta = (x+delta).clamp_(0,1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        grads = 0
        for _ in range(n_eot):                          # average gradient across stochastic forward
            loss = F.cross_entropy(model(x + delta), y)
            grads = grads + torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta + alpha * (grads / n_eot).sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = (x + delta).clamp_(0, 1) - x
        delta = delta.detach()
    return x + delta
```

### BPDA for non-differentiable preprocessing

Athalye, Carlini, Wagner ICML 2018 — the seminal "Obfuscated Gradients" paper. For a non-differentiable layer f_i, **use true f_i on the forward pass** (preserving the defense's effect) and a differentiable approximation g on the backward pass — most often **g = identity** when f_i is approximately identity (JPEG compression, bit-depth reduction, total variance minimization). Then run more PGD iterations than normal because gradients are inexact.

```python
class BPDAIdentity(torch.autograd.Function):
    """Forward applies the real (non-differentiable) defense; backward is identity."""
    @staticmethod
    def forward(ctx, x, defense_fn):
        return defense_fn(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None                       # straight-through estimator

def jpeg_defense(x):  # non-differentiable, but call inside BPDAIdentity.apply
    ...

def bpda_pgd(model, x, y, eps, alpha, steps, defense_fn):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        x_def = BPDAIdentity.apply(x + delta, defense_fn)
        loss = F.cross_entropy(model(x_def), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)
    return x + delta
```

For randomized + non-differentiable preprocessing (TVM, image quilting), combine BPDA with EOT as in the original paper.

---

## 6. Library cheat sheet (April 2026 versions)

| Library | Latest | Status | Strength |
|---|---|---|---|
| torchattacks | 3.5.1 (Oct 2023) | Inactive but stable | Cleanest PyTorch white-box |
| ART | 1.19.1 (Jan 2025) | Active | Black-box + API attacks via `BlackBoxClassifier` |
| Foolbox | 3.3.4 (Mar 2024) | Stable | Best decision-based, multi-framework |
| auto-attack | git HEAD | Active | Canonical AutoAttack used by RobustBench |
| robustbench | 0.2.1 | Active | Pretrained robust models + `benchmark()` |
| timm | 1.0.x | Active | 600+ ImageNet classifiers |

### Key gotchas

**torchattacks `AutoAttack` defaults to `eps=0.3`** — always pass `eps=8/255` explicitly. **torchattacks `CW` does not implement binary search** over c; default c=1 is too small for many models. **ART works on NumPy arrays in NHWC by default** (use `channels_first=True`); some attacks require one-hot labels via `art.utils.to_categorical`. **Foolbox returns `(raw, clipped, is_adv)`** — use `clipped` for ε-projected outputs.

### The normalization wrapper (the single most important piece of code)

```python
class NormalizedModel(nn.Module):
    """Lets every attack lib treat the model as if it natively eats raw [0,1] images."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)
    CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465); CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    def __init__(self, model, mean, std):
        super().__init__(); self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std ).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)               # x must be in [0,1]
```

**Always wrap your model this way** and feed [0,1] images to attacks. Skipping this is the #1 source of "my attack succeeds at 5%" frustration. timm ViT/CLIP variants often use `(0.5,0.5,0.5)/(0.5,0.5,0.5)` rather than ImageNet means — read `timm.data.resolve_model_data_config(m)` before attacking.

### From-scratch C&W with tanh trick and binary search

```python
def cw_l2(model, x, y_target, c_lo=1e-3, c_hi=1e10, bs_steps=9, max_iter=1000,
          lr=0.01, kappa=0.0):
    """Targeted CW-L2. y_target: target labels (LongTensor)."""
    N = x.size(0); device = x.device
    x_atanh = torch.atanh((x * 2 - 1).clamp(-1+1e-6, 1-1e-6))
    best = x.clone(); best_dist = torch.full((N,), float('inf'), device=device)
    c = torch.full((N,), 1.0, device=device)
    c_lo_t = torch.full((N,), c_lo, device=device); c_hi_t = torch.full((N,), c_hi, device=device)
    for bs in range(bs_steps):
        w = x_atanh.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        local_best = best.clone(); local_dist = best_dist.clone()
        for _ in range(max_iter):
            x_adv = 0.5 * (torch.tanh(w) + 1)                       # tanh trick: box-free
            logits = model(x_adv)
            real = logits.gather(1, y_target[:,None]).squeeze()
            other = logits.scatter(1, y_target[:,None], -1e9).max(1).values
            f = torch.clamp(other - real + kappa, min=0)             # targeted: real should win
            dist = ((x_adv - x) ** 2).flatten(1).sum(1)
            loss = (dist + c * f).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                succ = (logits.argmax(1) == y_target) & (dist < local_dist)
                local_best[succ] = x_adv[succ]; local_dist[succ] = dist[succ]
        # binary search update on c per sample
        with torch.no_grad():
            improved = local_dist < best_dist
            best[improved] = local_best[improved]; best_dist[improved] = local_dist[improved]
            c_hi_t = torch.where(improved, torch.minimum(c_hi_t, c), c_hi_t)
            c_lo_t = torch.where(~improved, torch.maximum(c_lo_t, c), c_lo_t)
            c = torch.where(c_hi_t < 1e9, (c_lo_t + c_hi_t) / 2, c * 10)
    return best
```

---

## 7. Architecture-specific guidance

**ResNet (ResNet-50 ImageNet, ResNet-18 CIFAR).** The reference architecture for adversarial work. PGD-20 at ε=8/255 (CIFAR) or ε=4/255 (ImageNet) breaks undefended models 100% of the time. ResNet-to-ResNet transfer is the best-case ~99% with the SI-NI-TI-DIM stack. **Skip-Gradient-Method** (Wu et al. ICLR 2020) explicitly downweights skip connections during attack, improving ResNet-source transfer by 5–10%.

**VGG.** Less robust than ResNet (no skip connections to bypass during attack). Square Attack on VGG-16-BN frequently succeeds with the stripe initialization alone (median = 1 query!). Transfer attacks to and from VGG work normally with the standard stack.

**Vision Transformers (ViT, DeiT, Swin).** **More robust to standard L_p attacks than CNNs at comparable parameter count** due to global feature aggregation, but **more vulnerable to patch-concentrated perturbations**. PatchFool (Fu et al. ICLR 2022) selects the most attention-influential patch and concentrates perturbation there — overturning the "ViTs are robust" narrative. **Use Adam optimizer and slightly smaller step sizes (α=ε/10) and more iterations (30–100) for ViTs** because gradient magnitudes vary more across tokens. **For ViT-source transfer, use TGR** (Zhang et al. CVPR 2023), which regularizes per-token gradient variance in Attention/QKV/MLP components, or **PNA** (Wei et al. AAAI 2022), which skips attention block gradients on backprop.

Cross-architecture transfer is the bottleneck: **CNN→ViT and ViT→CNN drop to 30–55% with vanilla MI-FGSM** but reach 60–80% with BSR, SIA, MIG, or TESSER (2025). **Always include a ViT in your transfer-attack ensemble** when targeting unknown architectures. Targeted cross-architecture transfer remains <20% even with SOTA methods.

**EfficientNet.** Behaves like a slightly more efficient ResNet variant; standard PGD/AutoAttack work without modification. Compound scaling does not provide adversarial robustness benefits.

**MLP-Mixer.** Less robust than ViT in white-box L∞ tests; behaves more like a CNN. Standard attacks work.

**ConvNeXt + ConvStem.** The current SOTA on RobustBench ImageNet (Singh, Croce, Hein NeurIPS 2024): ~58–59% robust accuracy at ε=4/255. Replacing ViT's PatchStem with ConvStem dramatically improves AT generalization across L₁/L₂/L∞ unseen threat models.

---

## 8. Defense-aware attacks

**Adversarially trained models (Madry, TRADES, MART).** Standard PGD-20 underestimates robustness; use **AutoAttack standard** for evaluation. PGD-100 with 10 restarts at α=ε/10 approaches AutoAttack quality at ~5× lower wall-clock. TRADES with poorly tuned hyperparameters can exhibit gradient masking — flagged when AutoAttack robust acc << PGD-10 robust acc (Lin et al. arXiv:2410.07675, 2024). Current RobustBench leaders: CIFAR-10 L∞ ε=8/255 ~71% (Peng 2023); CIFAR-100 ε=8/255 ~42–45%; ImageNet ε=4/255 ~58–59%.

**Input preprocessing defenses (JPEG, bit-depth reduction, TVM).** Use **BPDA** with identity backward pass; the forward pass keeps the real defense intact. Athalye 2018 showed this drives most preprocessing defenses to <30% robust accuracy.

**Randomized defenses (random resize/pad, stochastic activation pruning, RND).** Use **EOT with n=20** integrated into PGD or use **AutoAttack `version='rand'`** which already does this. SAP fully broken by correctly implemented BPDA (Dhillon erratum 2020 — drives accuracy to 0.1%).

**Diffusion-based purification (DiffPure).** Originally claimed strong robustness via SDE-based denoising; **broken in 2023–2024** by DiffAttack (Kang et al. NeurIPS 2023 — deviated reconstruction loss + segment-wise backprop), DiffHammer (Wang et al. NeurIPS 2024), and Lee & Kim ICCV 2023 (PGD+EOT outperforms AutoAttack by 16–26% ASR). **The lesson generalizes: any randomized defense evaluated with AutoAttack standard rather than `rand` is suspect.**

**Certified randomized smoothing (Cohen et al. ICML 2019).** The certificate covers L₂ within radius r of the *smoothed* classifier g — outside r there is no guarantee. The base classifier f has zero certified robustness; standard PGD on f succeeds normally. Attacking g requires EOT through Gaussian noise (SmoothAdv-style). At ImageNet σ=0.5, certified accuracy at L₂ ≤ 0.5 is ~49%. Accept the certificate and target unsupported regimes (L∞, larger budgets, the base classifier).

**Detection-based defenses.** Use high-confidence C&W (κ=10–40) so detectors that flag low-confidence predictions don't catch your adversarials. For logit-distribution detectors (Tramèr NeurIPS 2020), build adversarials whose logit-noise behavior mimics clean inputs.

---

## 9. Identifying gradient masking — the five red flags

Athalye et al. 2018 codified five tests. **(1) If iterative attacks (PGD) are weaker than single-step (FGSM)**, gradients are unreliable. **(2) If black-box attacks (Square, transfer) are stronger than white-box** on the same model, the gradient signal is masked. **(3) If unbounded ε does not reach 100% success**, the optimization is broken. **(4) If random noise transfers better than crafted perturbations**, the model is exploiting locally non-smooth behavior. **(5) If gradient values are vanishing, exploding, or identically zero**, gradients are shattered.

The three categories with corresponding fixes: **shattered gradients (non-differentiable preprocessing) → BPDA**; **stochastic gradients (randomized defense) → EOT**; **vanishing/exploding gradients → reparameterization or attack the surrogate**. AutoAttack has built-in flags ("potentially unreliable evaluation") when later attacks recover much more accuracy than earlier ones — treat as evidence and switch to `version='rand'` with EOT.

---

## 10. 2023–2025 developments worth knowing

**Faster AutoAttack alternatives.** **MALT** (Melamed et al. NeurIPS 2024) re-orders APGD-DLR target classes by Jacobian-normalized confidence ratio — 5× faster than AutoAttack on ImageNet, beats AA on CIFAR-100/ImageNet for several robust models. **A³** (Liu CVPR 2022) adapts direction initialization and discards examples online — ~10× speedup. **PMA+** (Ma et al. 2024) uses a probability-margin loss in an ensemble that exceeds AutoAttack white-box effectiveness. **AutoAttack remains the de facto evaluation standard despite these.**

**Diffusion-based attacks.** **DiffAttack** (Kang NeurIPS 2023) breaks DiffPure with deviated-reconstruction loss at intermediate sampling steps and segment-wise backprop, dropping CIFAR-10 robust acc by >20%. **AdvDiff** (Dai et al. 2023) generates unrestricted adversarial examples via diffusion-model reverse process with adversarial gradient injection — useful when your challenge allows unbounded perturbations.

**New transfer attacks for ViTs.** **BSR** (Wang CVPR 2024) splits images into n×n blocks with random shuffling and rotation, K=20 draws — best for cross-architecture transfer. **SIA** (Wang ICCV 2023) applies a different random transformation per block; ε=16/255 with ~20 transformations per iteration. **DeCoWA** (Lin AAAI 2024) uses elastic-deformation augmentation specifically tuned for CNN↔ViT cross-genus transfer. **TESSER** (arXiv:2505.19613, 2025) combines feature-sensitive gradient scaling with spectral smoothness regularization — +10.9% ASR on CNNs, +7.2% on ViTs.

**Foundation model attacks.** **Robust CLIP / FARE** (Schlarmann et al. ICML 2024) provides a plug-and-play robust CLIP encoder that protects LLaVA/OpenFlamingo. **CGNC** (Fang ECCV 2024) uses CLIP-guided generative networks for transferable targeted attacks. PGD/APGD on CLIP encoders at ε=4/255 produces targeted captions in LVLMs.

**Defenses to be aware of.** **MixedNUTS** (Bai 2024) is training-free accuracy-robustness mixing. **DUCAT** (Cao 2024) re-thinks AT label assignment. **AAS-AT** (Jain CVPR 2024) fixes attention-block FP underflow that caused gradient masking in ViT AT. **Smoothed-ViT** (Salman CVPR 2022) is broken by multi-patch attacks (Filtered-ViT AAAI 2025). **ADBM** (Li 2024) replaces DiffPure's reverse process — +4.4% over DiffPure under reliable adaptive attack.

---

## 11. Wall-clock estimates and what NOT to waste time on

**Speed ranking (CIFAR-10 batch of 128, modern GPU).** FGSM: ~50 ms. PGD-20: ~2 s. PGD-100: ~10 s. C&W (1000 iters, no binary search): ~30 s. C&W with full 9-step binary search: ~5 min. AutoAttack standard (1000 imgs): 10–25 min. AutoAttack plus: 60–120 min. Square Attack 5000 queries (1000 imgs, score-based local): ~5 min. HopSkipJump 10k queries (1000 imgs local): ~30 min.

**ImageNet ResNet-50 batch of 64.** FGSM: ~200 ms. PGD-20: ~5 s. AutoAttack standard 1000 imgs: 30–90 min. Square Attack against an API: query-rate-limited, plan 1k–10k queries per sample.

**What NOT to waste time on in 24 hours.** **Don't tune PGD hyperparameters** beyond the canonical ε=8/255, α=2/255, steps=20, 5 restarts — diminishing returns vs switching to AutoAttack. **Don't run AutoAttack `plus` or `rand`** unless you have evidence of randomization or are submitting a leaderboard-quality robustness number. **Don't write a from-scratch boundary attack** when Foolbox's HopSkipJump exists. **Don't try to beat Square Attack with a custom score-based method** — it's been the SOTA baseline for five years for a reason. **Don't pursue model stealing for transfer attacks** unless the API is generous and you have ≥4 hours; public timm models in an ensemble are usually sufficient. **Don't attempt diffusion-based attacks (DiffAttack, AdvDiff)** unless the challenge specifically involves diffusion purification — their setup overhead consumes hours. **Don't tune C&W's binary search range** by hand — let it run 9 steps and accept the result. **Don't attempt new ViT-transfer methods (TESSER, GNS-HFA, MuMoDIG)** before the standard MI-DI-TI-SI + BSR ensemble has been tried.

---

## 12. Common debugging pitfalls (skim before you panic)

**Wrong normalization.** Symptom: clean accuracy near random. Fix: wrap the model with `NormalizedModel` and feed [0,1] images. Verify with `timm.data.resolve_model_data_config(m)` for ImageNet models — ViT/CLIP variants often use (0.5, 0.5, 0.5) means.

**Clipping order.** Always project onto the ε-ball first, then clip to [0,1]. The reverse silently violates the ε-budget at image boundaries.

**Gradient accumulation.** Forgetting `delta = delta.detach()` or `requires_grad_(True)` between PGD iterations grows the autograd graph unboundedly and causes OOM by step 50. Or worse, `delta.grad` accumulates across iterations and the attack diverges.

**`model.eval()` not set.** BatchNorm uses batch statistics rather than running statistics in train mode, making attacks unstable and non-reproducible. Dropout adds noise that masquerades as gradient masking. Always call `model.eval()` before attacks; torchattacks and ART do this automatically but custom code does not. **For RNN models that need `train()` mode for autograd**, use `atk.set_model_training_mode(model_training=True, batchnorm_training=False, dropout_training=False)`.

**Float precision.** Accumulating α=2/255 in fp16 over 100 iterations drifts. Run attacks in fp32 explicitly: `with torch.cuda.amp.autocast(enabled=False):`. For reproducibility set `torch.backends.cudnn.deterministic=True` and `torch.use_deterministic_algorithms(True)`.

**NaNs in C&W.** c too small (default c=1 is often inadequate) or learning rate too high on highly confident logits. Start with κ=0; raise c until success rate jumps; then raise κ for higher-confidence adversarials.

**ART one-hot labels.** Some ART attacks expect `(N, nb_classes)` one-hot labels via `art.utils.to_categorical(labels, nb_classes)`. torchattacks and Foolbox use class indices.

**AutoAttack `eps=0.3` default.** torchattacks' `AutoAttack` defaults to `eps=0.3` (an MNIST-era default) — pass `eps=8/255` explicitly for CIFAR-10 / ImageNet.

**Wrong attack success rate.** Compute ASR as `(misclassified ∧ originally_correct).sum() / originally_correct.sum()` for untargeted, or `(pred == target).mean()` for targeted. Including originally-misclassified samples in the denominator inflates ASR.

---

## Conclusion

The attacks landscape in 2026 has stabilized around a small number of robust workhorses: PGD and AutoAttack for white-box, the SI-NI-TI-DIM-MI transfer stack for surrogates, and Square Attack / HopSkipJump for query-based black-box. **The hackathon-winning insight is that algorithmic novelty matters less than execution discipline: correct normalization wrappers, transfer-then-query workflows, and aggressive gradient-masking diagnostics.** Spend the first 4 hours on coverage with libraries, the next 12 hours on hard cases with custom code, and the last 8 hours on defense-aware techniques (BPDA, EOT) for whatever resists. Reserve from-scratch implementations for the three places they pay back: PGD with bells and whistles for full hyperparameter control, C&W with proper binary search when minimum-norm matters, and Square Attack when you need to plug in a custom loss or non-standard input pipeline.

The deepest 2023–2025 insight is that **defenses are increasingly broken by adaptive attacks the original papers didn't run**. DiffPure, Smoothed-ViT, and several TRADES variants all looked strong under their own evaluations but fell to PGD+EOT, multi-patch attacks, and AutoAttack respectively. When your challenge involves a defended model, your competitive advantage is running the *right* adaptive attack — `rand` mode plus n=20 EOT for randomized, BPDA for preprocessing, RayS as an L∞ sanity check, Square Attack as the gradient-free gold standard. Trust those tools, and the only remaining variable is how cleanly your team executes.