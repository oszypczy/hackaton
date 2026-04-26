# Auditing medical AI for hidden training-data bias: a black-box property inference playbook

**Bottom line up front.** The CISPA Warsaw 2026 challenge will almost certainly mirror the Barcelona task: given black-box query access to a set of medical models, decide which were trained on demographically skewed patient populations and ideally estimate the skew ratio. The winning strategy is well-defined by the literature — train a small bank of shadow models on a public medical dataset with controlled demographic ratios, query both shadow and target models on a carefully constructed probe set, and apply a likelihood-ratio or KL-divergence statistical test calibrated on the shadow distribution. SNAP (S&P 2023) and the Suri–Evans KL attack (SaTML 2023) are the algorithmic state-of-the-art; Maini–Dziedzic-style aggregated-MIA dataset inference (NeurIPS 2024) provides the statistical-rigor template the organizers favor. Differential privacy will *not* protect the victim because property inference is definitionally orthogonal to record-level DP. The decisive technical edge for a 24-hour team comes from query efficiency (active selection à la BAFA 2026), proper calibration (Brier/ECE/conformal), and TPR@1%FPR-style reporting that matches the Boenisch–Dziedzic group's evaluation conventions.

---

## 1. The threat model and what "property" means

Property inference, distribution inference, dataset inference, and membership inference form a hierarchy distinguished by *granularity of the secret*. Let `D` be a public data distribution, `S ~ D^n` a training set, and `M = T(S)` the target model accessed through a black-box oracle `O(M)`.

**Membership inference (MI)** asks whether a *specific record* `x*` was in `S`. The cryptographic-game definition (Yeom et al., CSF 2018; Shokri et al., S&P 2017) samples `b ∈ {0,1}` and feeds the adversary `x*` drawn either from `S` (b=1) or `D\S` (b=0); advantage is `|Pr[b̂=b|b=1] − Pr[b̂=b|b=0]|`.

**Property / distribution inference (PI/DI)** asks about a *global statistic* `g(D_S)` of the training distribution. Suri & Evans (PETS 2022) formalize this as: trainer picks `b ∈ {0,1}`, samples `S ∼ G_b(D)` for two public distribution-transformation functions `G_0, G_1` (e.g., `G_0` keeps 30% female records, `G_1` keeps 50%), trains `M`, and the adversary outputs `b̂ = H(M)`. **The two distributions differ on essentially every record, which is exactly why DP — bounding influence of one record — does not defend.**

**Subpopulation inference** (Mahloujifar et al., S&P 2022; Jagielski et al., CCS 2021) is the special case where the property is the *prevalence* `Pr_x[χ(x)=1]` of a filter `χ` defining a subpopulation; this is the natural framing for "fraction of female patients" or "fraction of dark-skinned patients."

**Dataset inference (DI)** (Maini, Yaghini, Papernot, ICLR 2021) asks whether a *specific dataset* `S_V` was used. The victim retains private samples and runs a hypothesis test (Welch t-test on prediction-margin embeddings) comparing the suspect model's behavior on private vs. public data. This is the framework most heavily used by the SprintML lab.

The **threat model for the hackathon** is, with high confidence: black-box query access (probably probability vectors, possibly label-only); a hard query budget (likely 100–10,000); knowledge of the modality and architecture family but not weights; access to a public auxiliary dataset of the same modality; and a held-out suite of 10–100 victim models to score. The "property" is a demographic ratio — sex was named explicitly in CISPA's Barcelona writeup — possibly extended to age, race, or intersectional groups.

## 2. Core attack techniques and the meta-classifier paradigm

The shadow-model paradigm originates with **Ateniese et al. (IJSN 2015)**: train `n` shadow classifiers on datasets with and without property `P`, extract a feature representation `Φ(C_i)` of each, and train a meta-classifier `MC: Φ → {0,1}` that you then apply to the target's representation. **Ganju et al. (CCS 2018)** sharpened this for fully-connected networks by using a DeepSets permutation-invariant representation: each layer is treated as a *set* of neurons embedded by a small MLP `φ_l` and aggregated by sum, which respects the network's invariance under within-layer neuron permutation and improves attack accuracy by 20–40 percentage points over naïve flattening. Their CelebA case study showed a "smile detector" leaking the **gender ratio** of its training data with >95% accuracy — the canonical demonstration that downstream models leak about properties uncorrelated with their nominal task.

**Suri & Evans (PETS 2022, "Formalizing and Estimating Distribution Inference Risks")** is the formal framework the field now uses. Beyond the cryptographic game above, they introduce the **`n_leaked` metric**: the number of samples an "optimal sampling adversary" — given direct access to the training distribution — would need to match the inference attack's distinguishing accuracy. For ratios `α₀, α₁` with attack accuracy `ω`,
`n_leaked = log(4ω(1−ω)) / log(min(α₀,α₁)/max(α₀,α₁))`.
Distinguishing α=0.5 from α=1.0 with ω=0.95 yields `n_leaked ≈ 3`; distinguishing 0.5 from 0.52 with the same accuracy demands `n_leaked ≈ 42`. This is a much more honest leakage metric than 50/50-prior accuracy.

The **state-of-the-art black-box attack is SNAP** (Chaudhari, Abascal, Oprea, Jagielski, Tramèr, Ullman, S&P 2023, arXiv:2208.12348). SNAP combines subpopulation poisoning with a logit-distribution test: the adversary inserts `pn` poisoned samples (label-flipped points from a chosen subpopulation), trains `k=4` shadow models per candidate ratio, queries them on a clean probe set drawn from the same subpopulation, and fits per-ratio Gaussians to the logits of the *poisoned target label*. The optimal Bayes threshold between the two Gaussians is determined by their KL divergence, which the poisoning rate is chosen to maximize. The key theoretical result (their Theorem 4.1) is that under calibration the poisoned logit obeys

`φ̃(x)_ṽ = log[ p / (π_v(1−p)t) + e^{φ(x)_ṽ} · (1 + p / (π_v(1−p)t)) ]`

where `t = Pr_x[f(x)=1]` is the property's prevalence; smaller `t` produces a larger logit shift, so worlds with rarer property are more separable. Empirically SNAP distinguishes 1% vs 3.5% female prevalence on Census with 96% accuracy at 0.4% poisoning, and is **34% more accurate and 56.5× faster** than the Mahloujifar et al. baseline.

**Without poisoning capability** (the most likely hackathon constraint), the best attack is the **black-box KL-divergence test from Suri, Lu, Chen, Evans (SaTML 2023, "Dissecting Distribution Inference")**. For each candidate ratio `b ∈ {0,1}`, train reference models, average their softmax outputs on probes to get `p^{(b)}(x)`, then decide `b̂ = argmin_b Σ_x KL(p^{target}(x) ‖ p^{(b)}(x))`. This black-box approach matches or beats expensive white-box DeepSets meta-classifiers on Census, RSNA Bone Age, and Texas-100X.

For the LLM/text variant (a possible Warsaw twist), **PropInfer** (Huang, Yadav, Chaudhuri, Wu, arXiv:2506.10364, v4 Feb 2026) demonstrates property inference on **ChatDoctor**, a medical fine-tuned LLM, using either prompt-based generation (sample, classify, aggregate) or shadow-model attacks with word-frequency features. ChatDoctor as the victim domain would be a very natural fit for a Boenisch–Dziedzic challenge given their NeurIPS 2024 LLM Dataset Inference paper.

## 3. Query design — what inputs leak the most

The information-theoretic principle is that the adversary should choose query inputs `x` to **maximize KL divergence between the predicted output distributions under the two training distributions**, since `n ≥ Ω(log(1/ε)/KL(P₀‖P₁))` queries suffice to distinguish at error `ε`.

Five probe families dominate the literature, in roughly increasing sophistication:

**Random in-distribution probes** drawn from a public auxiliary dataset are the default. Suri & Evans's *Loss Test* and *Threshold Test* use only this and still recover `n_leaked` of tens for `Δα = 0.1`. **Counterfactual paired probes** generate `(x, x')` differing only on the protected attribute via StyleGAN gender swaps, FairFace augmentations, dermatology Fitzpatrick re-shading, or EHR insurance/ZIP swaps. The pair-difference `f(x) − f(x')` reduces variance by orders of magnitude over independent samples and isolates the demographic signal — the foundation of **Themis** (Galhotra–Brun–Meliou, FSE 2017) and **AEQUITAS** (Udeshi et al., ASE 2018) individual-discrimination testing. **Boundary / adversarial probes** following Choquette-Choo et al. (ICML 2021) measure the L2 distance to the nearest adversarial example or the fraction of probes still correctly classified after Gaussian noise σ; this fraction shifts with training composition. **Synthetic / OOD / interpolated probes** generated by diffusion or by linear interpolation between class manifolds explore boundary regions where training-distribution-specific artifacts amplify; a model trained mostly on female faces will be less confident on androgynous OOD samples. **SNAP-style trigger probes** are clean samples drawn from the poisoned subpopulation, queried after poisoning has biased the model; logits on these inputs are bimodal across the two worlds.

For a tight query budget, **active selection** wins. Yan & Zhang (ICML 2022) prove that auditing demographic-parity gap reduces to property testing whose query complexity is governed by the disagreement coefficient of the hypothesis class. **BAFA — "Audit Me If You Can"** (Hartmann et al., arXiv:2601.03087, Jan 2026) maintains a version space of BERT surrogates consistent with observed scores, computes uncertainty bounds on `ΔAUC` via constrained ERM, and selects queries by Bayesian optimization in regions of maximum surrogate disagreement — achieving the same error as stratified sampling **with up to 40× fewer queries**. For a hackathon with a 1k-query cap this is the difference between a confident answer and a noisy one.

## 4. Fairness metrics estimable from a black-box API

With features `X`, protected attribute `A`, label `Y`, score `S = f(X)`, and decision `Ŷ = 1[S ≥ t]`:

- **Statistical parity** (Dwork et al., ITCS 2012): `P(Ŷ=1|A=a) = P(Ŷ=1|A=a')`; gap `Δ_DP = |P(Ŷ=1|A=1) − P(Ŷ=1|A=0)|`.
- **Equalized odds / equal opportunity** (Hardt, Price, Srebro, NeurIPS 2016): `P(Ŷ=1|A=a, Y=y)` constant across `a`. Operationally: `ΔTPR`, `ΔFPR`.
- **Calibration parity** (Pleiss et al., NeurIPS 2017; Chouldechova, Big Data 2017): `P(Y=1|S=s, A=a) = s` for every `s, a`; group ECE `ECE_a = Σ_b (|B_b|/n) · |acc_b − conf_b|`.
- **Counterfactual fairness** (Kusner et al., NeurIPS 2017): `P(Ŷ_{A←a}|X=x, A=a) = P(Ŷ_{A←a'}|X=x, A=a)` under do-intervention.

The **Chouldechova / Kleinberg–Mullainathan–Raghavan impossibility** is critical: when base rates `P(Y=1|A)` differ, no non-trivial classifier can satisfy predictive parity, equal FPR, and equal FNR simultaneously. Pleiss et al. strengthen this to show calibration is compatible with at most a single error-rate constraint. **The practical consequence for auditing: asymmetric miscalibration across subgroups is a robust indicator of training-distribution skew**, because a calibrated model trained on a balanced distribution cannot exhibit such asymmetry without violating these impossibility results.

When the API does not expose `A`, the auditor estimates demographics via three routes: (i) train an auxiliary head on a public encoder's frozen embedding to predict `A` from the same input the API receives — if recoverable, the API likely uses `A` implicitly; (ii) **BISG / BIFSG** (Bayesian Improved Surname Geocoding, CFPB 2014) imputes race from surname and ZIP for U.S. EHR data; (iii) Singh et al. (ACM JRC 2024) show that exact fairness disparities are recoverable from aggregate proxy tables when `Z ⊥ Y | A`. Kallus, Mao, Zhou (2022) provide bounds on the bias of disparity estimates under noisy proxies: `|TPR-disparity_proxy − TPR-disparity_true| ≤ ε(P(Â≠A))`.

## 5. Statistical tests, sample complexity, and multiple-testing control

A small set of tests covers the hackathon's needs. **Permutation tests** are distribution-free and require no assumptions: compute `T_obs`, pool predictions, randomize labels `B = 10,000` times, and report `p̂ = (1 + #{T_b ≥ T_obs})/(B+1)`. **Two-sample Kolmogorov–Smirnov** tests detect *shape* shifts in continuous score distributions, useful when the two worlds shift the variance rather than the mean. **Chi-square** tests handle label-only outputs over `k` classes. **Bootstrap BCa intervals** (`B = 10,000`) report uncertainty on every gap. The **likelihood-ratio test à la Carlini's LiRA** (S&P 2022), instantiated for property inference, fits Gaussians to per-probe shadow scores under each world, computes `Λ(x) = log[φ(s; μ₀,σ₀²)/φ(s; μ₁,σ₁²)]`, and aggregates over the probe set; LiRA achieves up to 10× higher TPR at FPR=0.1% than naive thresholding.

The **two-proportion z-test sample-complexity heuristic** to memorize: at α=0.05, power=0.8, **n ≈ 16·p(1−p)/Δ²** per group. Detecting Δ=0.10 around p=0.5 needs ~400 per group (≈ 800 total queries); Δ=0.05 needs ~1,569 per group; Δ=0.20 needs ~100 per group. The **Hoeffding budget** `n ≥ ln(2/α)/(2ε²)` is distribution-free and gives ~740 queries for ε=0.05, α=0.05 — about 18% wider than CLT but applicable to any bounded model output of unknown variance. The **information-theoretic lower bound** from Suri & Evans is `n ≥ Ω(log(1/ε)/KL(P₀‖P₁))`, which is the metric the adversary actually optimizes by query design.

For **multiple testing** across a fairness-metric battery, **Benjamini–Hochberg FDR at q=0.05** dominates Bonferroni in power and is the appropriate choice for the ~50 per-attribute tests typical of an intersectional audit. **Always-valid p-values** (Howard–Ramdas, *Annals* 2021; Johari et al., *OR* 2022; Waudby-Smith–Ramdas empirical-Bernstein confidence sequences) are essential if you intend to monitor your statistic during query collection and stop early — naive peeking inflates Type-I error catastrophically. Use the `confseq` Python package.

For **calibrating** the attack output itself, **isotonic regression** on a held-out shadow split outperforms Platt scaling beyond ~1k calibration points; **conformal prediction** (via `MAPIE`) gives finite-sample distribution-free coverage guarantees on the inference output, which judges relying on calibrated probabilities will reward.

## 6. Medical-domain signals: chest X-ray, dermatology, EHR

Medical AI carries an unusually rich set of demographic-shortcut signals. **Gichoya et al. (Lancet Digital Health 2022)** and **Banerjee et al. (arXiv:2107.10356, "Reading Race")** showed standard CNNs predict self-reported race from chest X-rays at AUC 0.97–0.99, and **the signal is broadband and survives extreme degradation** — cropping to 1/9 of area, downsampling to 4×4, and aggressive low-pass filtering all preserve it. BMI, breast density, scanner parameters, and obvious anatomic features do not explain the signal. Practically: any CXR model is likely to encode race as a latent shortcut whether or not the developers intended it.

**Seyyed-Kalantari et al. (Nature Medicine 2021)** demonstrated systematic underdiagnosis bias by measuring `FPR_g(No Finding) = P(Ŷ = NoFinding | Y = disease, G = g)` per subgroup; FPR is inflated for female, Black, Hispanic, age <20, and Medicaid patients, with intersectional cells (Hispanic-female) suffering the largest gaps. Bernhardt and Glocker (Nature Medicine 2022) raised the SPLIT-bias caveat that train/test prevalence shifts can confound, and **Glocker et al. (eBioMedicine 2023)** provide the audit triad: test-set resampling, multi-task probing of frozen features for `A`, and unsupervised PCA/t-SNE exploration of penultimate features stratified by demographics. **Yang et al. (Nature Medicine 2024)** prove the causal link: across radiology, dermatology, and ophthalmology, **models with the highest demographic-prediction accuracy show the largest fairness gaps**, and the "minimum-attribute-encoding" model selection criterion outperforms in-distribution fairness selection for OOD generalization (Wilcoxon p < 10⁻⁹⁰). **Larrazabal et al. (PNAS 2020)** showed that varying male:female training ratios from 0:100 to 100:0 monotonically degrades the under-represented gender's AUC by up to 0.10.

In dermatology, **Daneshjou et al. (Science Advances 2022)** released the DDI dataset, balanced across Fitzpatrick I–II vs V–VI, and demonstrated 27–36% balanced-accuracy drops for malignancy detection on dark skin (AUROC 0.56–0.67). For EHR/tabular models, the canonical audit is **Obermeyer et al. (Science 2019)**: cross-tabulating risk scores against an independent biomarker composite (HbA1c, BP, eGFR, comorbidities) by race revealed that Black patients at any score percentile were measurably sicker than white patients because the algorithm used cost as a proxy for need. Re-targeting the label would have raised Black enrollment in high-risk programs from 17.7% to 46.5%.

Concrete black-box probes that exploit these signals: (1) train a logistic-regression head on a frozen public encoder's embedding to predict `A` from probe inputs and verify that demographic signal is recoverable; (2) apply Gichoya-style frequency-spectrum ablation (LPF, HPF, BPF) and measure persistence of subgroup-disparity; (3) use Daneshjou-DDI-paired or StyleGAN-generated counterfactual pairs to compute discrimination rate `r = (1/n)Σ 1[f(x_i) ≠ f(x'_i)]`; (4) compute group-conditional reliability diagrams and ECE — `ΔECE > 0.05` is a strong indicator of training-distribution skew given the calibration impossibility theorem; (5) for EHR, perturb proxy fields (insurance, ZIP, surname) to detect implicit demographic encoding when explicit `A` is absent.

## 7. A practical Python pipeline

The audit pipeline factors into five components. The **black-box wrapper** caches and rate-limits API calls; the **shadow-model harness** trains `2k` models with controlled property variations; the **attack module** runs SNAP, KL, and LRT in parallel; the **statistical module** produces calibrated probabilities and confidence intervals; the **report module** emits per-victim probability and CI.

```python
# (1) Black-box wrapper with caching and rate limiting
import functools, hashlib, time, json, numpy as np, scipy.stats as st
from pathlib import Path

class BlackBoxAPI:
    def __init__(self, fn, rate_limit_qps=10, cache_dir="cache/"):
        self.fn = fn; self.qps = rate_limit_qps; self.last = 0
        self.cache = Path(cache_dir); self.cache.mkdir(exist_ok=True)
        self.n_queries = 0
    def __call__(self, x):
        key = hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
        f = self.cache / f"{key}.npy"
        if f.exists(): return np.load(f)
        dt = 1/self.qps - (time.time() - self.last)
        if dt > 0: time.sleep(dt)
        y = self.fn(x); self.last = time.time(); self.n_queries += 1
        np.save(f, y); return y

# (2) Shadow model harness with controlled property variation
def train_shadow_bank(arch_fn, dataset, property_fn, ratios=(0.3, 0.5),
                     n_per_ratio=8, train_epochs=10):
    bank = {r: [] for r in ratios}
    for r in ratios:
        for seed in range(n_per_ratio):
            S = subsample_with_ratio(dataset, property_fn, r, seed=seed)
            m = arch_fn(); m = train(m, S, epochs=train_epochs, seed=seed)
            bank[r].append(m)
    return bank

# (3a) KL-divergence attack (Suri et al., SaTML 2023)
def kl_attack(target_api, probes, shadow_bank):
    pT = np.array([target_api(x) for x in probes])               # n × C
    refs = {r: np.mean([np.array([m(x) for x in probes])
                        for m in shadow_bank[r]], axis=0) for r in shadow_bank}
    eps = 1e-9
    kls = {r: float((pT * (np.log(pT+eps) - np.log(p+eps))).sum())
           for r, p in refs.items()}
    return min(kls, key=kls.get), kls

# (3b) LiRA-style LRT (Carlini et al., S&P 2022; per-probe Gaussian fit)
def lira_attack(target_api, probes, shadow_bank, ratios):
    r0, r1 = ratios
    sT = np.array([target_api(x)[1] for x in probes])            # use class-1 logit
    S0 = np.array([[m(x)[1] for x in probes] for m in shadow_bank[r0]])
    S1 = np.array([[m(x)[1] for x in probes] for m in shadow_bank[r1]])
    mu0, sd0 = S0.mean(0), S0.std(0)+1e-6
    mu1, sd1 = S1.mean(0), S1.std(0)+1e-6
    ll0 = st.norm.logpdf(sT, mu0, sd0).sum()
    ll1 = st.norm.logpdf(sT, mu1, sd1).sum()
    return ll1 - ll0    # >0 → predict ratio=r1

# (4) Statistical wrappers: bootstrap CI, permutation p, BH-FDR
def bootstrap_ci(scores, B=10000, alpha=0.05):
    n = len(scores)
    means = [np.mean(np.random.choice(scores, n, replace=True)) for _ in range(B)]
    return np.mean(scores), tuple(np.quantile(means, [alpha/2, 1-alpha/2]))

def permutation_p(stat_fn, A, B_, B=10000):
    obs = stat_fn(A, B_)
    pooled = np.concatenate([A, B_]); nA = len(A)
    null = []
    for _ in range(B):
        np.random.shuffle(pooled)
        null.append(stat_fn(pooled[:nA], pooled[nA:]))
    return obs, (1 + sum(t >= obs for t in null)) / (B + 1)

def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    thresh = (np.arange(1, m+1) / m) * q
    passed = np.where(p[order] <= thresh)[0]
    k = passed.max() + 1 if len(passed) else 0
    rej = np.zeros(m, dtype=bool); rej[order[:k]] = True
    return rej

# (5) Calibration via isotonic regression + conformal coverage
from sklearn.isotonic import IsotonicRegression
def calibrate(scores_train, labels_train, scores_test):
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores_train, labels_train)
    return iso.predict(scores_test)

# (6) Reporting
def audit_report(target_api, probes, shadow_bank, ratios, calib_data):
    pred_kl, kls = kl_attack(target_api, probes, shadow_bank)
    lr = lira_attack(target_api, probes, shadow_bank, ratios)
    p_high = 1/(1+np.exp(-lr))                              # sigmoid LR
    p_calib = calibrate(*calib_data, np.array([p_high]))[0]
    ci = bootstrap_ci([p_high]*len(probes))                 # placeholder
    return {"pred_ratio_kl": pred_kl, "p_skewed": float(p_calib),
            "kls": kls, "n_queries": target_api.n_queries, "ci95": ci[1]}
```

The critical implementation detail missing from many naive attempts: **calibrate on a strictly held-out shadow split** — never reuse training queries — and report TPR at fixed FPR (0.1%, 1%) rather than only AUC, following Carlini's argument that AUC misleads at low-FPR regimes that judges care about.

## 8. The organizers' style and likely scoring

Adam Dziedzic and Franziska Boenisch co-lead the **SprintML lab** at CISPA. Their publication record over 2022–2026 is unusually consistent in methodological style. Five papers form the most likely intellectual scaffold for the Warsaw challenge: **Maini, Yaghini, Papernot — Dataset Inference** (ICLR 2021); **Dziedzic et al. — Dataset Inference for SSL** (NeurIPS 2022); **Maini, Jia, Papernot, Dziedzic — LLM Dataset Inference** (NeurIPS 2024); **Zhao, Maini, Boenisch, Dziedzic — Post-hoc Dataset Inference with Synthetic Data** (ICML 2025); and **Dubiński, Kowalczuk, Boenisch, Dziedzic — CDI: Copyrighted Data Identification in Diffusion Models** (CVPR 2025). The CDI paper is particularly informative because it achieves >99% confidence with only ~70 data points by aggregating per-sample MIA signals via a statistical hypothesis test — exactly the pattern a hackathon challenge would reward.

The lab's evaluation conventions are nearly invariant across papers: **TPR at fixed FPR (typically 1%, sometimes 0.1%)** as the primary metric — pioneered by Carlini's LiRA and adopted uniformly by SprintML; **AUC** as the secondary metric; **p-values from explicit hypothesis tests** with thresholds like `p < 0.1` and *zero false positives* on held-out independent models; **calibrated likelihood ratios** rather than hard thresholds; and **rigorous shadow-model training** (the Strong-MIA paper, Hayes et al., NeurIPS 2025, explicitly criticizes weak attacks that avoid shadow models). The **Maini–Dziedzic LLM-DI critique** of distribution-shift confounding in MIA evaluation strongly suggests that the held-out victim set will be carefully matched in distribution, penalizing teams that exploit such shortcuts.

The most likely **scoring rubric** for Warsaw, extrapolated from this evidence and the Stockholm recap language ("the best solution is the one where the reconstructed datasets most closely match the original training data"): primary metric is **mean absolute error / RMSE on the inferred property ratio** across a held-out suite of victim models; secondary metrics are **TPR@1%FPR for the binary biased-vs-balanced distinguishing task**, **Brier score / log-loss on calibrated probability outputs**, and **query efficiency** (teams using 100 queries beat teams using 10,000 with similar accuracy). A 1–2 page methodological writeup and a clean GitHub repo will likely matter for tiebreaking — this is the SprintML house style.

The probable threat-model escalations relative to Barcelona are: tighter query budget, possibly label-only access (forcing SNAP §6 / Suri-Evans 2023 §7 label-only variants), simultaneous multi-attribute properties (sex × race × age), and DP-trained or fairness-regularized victim models as decoys. The latter is essentially free to add because, as Suri & Evans (2022, 2023) and Hartmann et al. (SaTML 2023) all confirm, **DP-SGD provides little defense against distribution inference at usable ε** — DP bounds per-record influence, while property inference targets aggregate-distribution influence, which is what the model is supposed to learn. Hu et al.'s **PriSampler** (arXiv:2306.05208) and Chen & Ohrimenko's *distribution privacy* primitive (arXiv:2207.08367) explicitly motivate new defenses precisely because DP is the wrong tool.

## 9. A ready-to-execute 4-hour audit recipe

Within a 24-hour hackathon, the property-inference challenge realistically consumes ~4 hours of focused execution after problem ingestion. The following is a tight schedule.

**T+0:00–0:30 — Reconnaissance.** Probe the API with 50–100 random auxiliary medical inputs; record output format (probabilities vs labels), latency, and any rate limits. Identify the modality (CXR / dermatology / tabular EHR), the target task (binary disease classifier most likely), the candidate property (sex is the highest prior given Barcelona's explicit naming), and the architecture family. Decide whether to commit to ratio estimation (regression) or binary distinguishing (classification) based on the scoring documentation.

**T+0:30–1:30 — Shadow-model bank.** Choose a public dataset matching the modality: MIMIC-CXR or CheXpert for chest X-rays, DDI / Fitzpatrick17k for dermatology, MIMIC-IV / eICU for tabular EHR. Train **2k = 16 shadow models** (8 per candidate ratio, e.g. 0.3 and 0.5 sex skew), all with identical architecture and hyperparameters, varying only the training-set property ratio and seed. For tabular data this is 2 minutes total on a single GPU; for CXR DenseNet-121, budget 30–60 minutes by training small subsets and few epochs — the SNAP result that `k=4` shadows suffice means you do not need Ganju-scale shadow farms.

**T+1:30–2:00 — Probe construction.** Build a 1,000-query probe set partitioned as: 300 random auxiliary samples, 300 counterfactual paired probes (StyleGAN sex-swaps for CXR; Fitzpatrick re-shading or DDI-matched pairs for dermatology; insurance/ZIP swap for EHR), 200 OOD/synthetic probes, and 200 SNAP-style trigger inputs from a chosen subpopulation. If query budget is tighter, prioritize the counterfactual paired probes — they have the highest signal-to-noise ratio.

**T+2:00–3:00 — Multi-attack execution.** Run three attacks in parallel on each victim model: (i) **KL-divergence attack** (Suri et al. 2023) — train-free at attack time, just KL between victim outputs and the average shadow output per ratio; (ii) **LiRA-style LRT** with per-probe Gaussian fits, aggregating log-likelihood-ratios across probes; (iii) **Confidence-distribution test** on the SNAP trigger probes — fit Gaussians per ratio and compute the Bayes-optimal threshold. Vote / average across the three attacks and across probe partitions.

**T+3:00–3:30 — Statistics and calibration.** For each victim, compute the bootstrap 95% CI over probe sub-samples; compute a permutation p-value against the shadow null distribution; apply **Benjamini–Hochberg at q=0.05** across the victim suite. **Calibrate** the attack scores by fitting isotonic regression on a held-out shadow split (e.g., 4 of the 8 shadows per ratio reserved for calibration), and wrap the calibrated probability with a conformal-prediction interval at α=0.1 via `MAPIE`. Compute **ECE** on the calibrated probabilities to verify reliability.

**T+3:30–4:00 — Reporting.** For each victim model `M_i`, emit a JSON record with: predicted property ratio (point estimate + 95% CI), probability that the model is "biased" (calibrated, with conformal interval), per-attack votes, and number of queries used. Include a summary plot of attack accuracy vs query budget overlaid with the Hoeffding lower bound `n ≥ ln(2/α)/(2Δ²)`, and a reliability diagram of the calibrated probabilities. The deliverable matches the Boenisch–Dziedzic paper template and is what the jury will be conditioned to evaluate well.

## Key takeaways and an opinionated bet

Treat the challenge as a **dataset inference / property inference hybrid**, even though the lab's named line of work is dataset inference: the medical-bias framing is technically property inference (Ganju, Suri-Evans, SNAP) but the *scoring style* will be the Maini–Dziedzic aggregated-statistic-with-p-value template. **Do not build a meta-classifier**: SNAP-style confidence-distribution fitting and the Suri-Evans KL attack reach SOTA with `k=4` shadow models per ratio and degrade gracefully to label-only, while DeepSets meta-classifiers demand thousands of shadows that no team can train in 24 hours. **Do not skip calibration**: Brier score, isotonic regression, and conformal intervals materially improve the deliverable's reception by judges trained on the Carlini-Boenisch-Dziedzic evaluation conventions. **Do not assume DP-SGD victims are safe**: the literature is unanimous that record-level DP fails to defend distributional properties, and the organizers know this. **Do plan for label-only**: even if the API exposes probabilities in early reconnaissance, a Warsaw-vs-Barcelona escalation could remove them, and your shadow bank should already produce label-only versions of every attack.

The decisive technical edge is query efficiency, calibration, and presentation — not algorithmic novelty. Teams that know SNAP, the Suri–Evans `n_leaked` metric, the Maini-DI aggregation trick, and the BAFA active-query strategy, and who report TPR@1%FPR with conformal intervals on a clean GitHub repo, will be in serious medal contention.