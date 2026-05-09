# DUCI Hackathon Field Notes — Practical Gap-Filler for CISPA Warsaw 2026

**TL;DR**
- Train **one shadow ResNet18** on a fast schedule (≈3–5 min/model on A100/A800 with the `speedyresnet`/airbench-style recipe already integrated into `ml_privacy_meter`) and reuse it across all 9 targets; spend the remaining JURECA budget on **2–4 architecture-matched references per depth** (18/50/152), because RMIA tolerates *data-distribution* mismatch much better than *training-recipe and architecture* mismatch (ImpMIA, ICLR 2026 submission, demonstrates "even minor mismatches in normalization, architecture, optimizer, or learning-rate schedule caused severe" degradation in RMIA at low FPRs).
- Treat POPULATION as your non-member proxy but **do a sanity check**: train one extra ResNet18 on `p=0.5·MIXED ∪ rest=POPULATION_subset_A` and compute MIA AUC on `MIXED \ trained` vs. `POPULATION_subset_B` — if AUC < 0.6 your MIA is too weak and you should fall back to averaged-logit MLE on a Tong-style 21-point grid (Avg-Logit MLE is competitive at moderate budgets per Tong Table 1) rather than the per-record debiasing.
- Calibrate your 9 predictions: clamp to **[0.025, 0.975]**, and if you suspect Tong's 5%-grid was used by organizers, snap to the nearest 0.05; if not, leave continuous. Independent inference per model dominates jointly-modeled inference for this hackathon — there is **no published joint multi-target DUCI work**, and the per-record correlations Tong shows are negligible (you already know this), so don't spend hackathon time inventing it.

---

## Gap 1 — Reference-model training on Jülich, time-budgeted

**Findings**
- The fastest concrete CIFAR recipe with reproducible wall-clock numbers is **airbench (Keller Jordan, 2024)**: 94% CIFAR-10 in **3.29 s on a single A100**; 95% in 10.4 s; 96% in 46.3 s. The same architecture (`airbench96`) "directly applied to CIFAR-100 without re-tuning" beats a standard ResNet-18 baseline (Table 5 in arXiv:2404.00498). On A800 (≈80–85% of A100 throughput in FP16/TF32 for compute-bound ResNet) expect 4–6 s for the 94% recipe, ~50–60 s for the 96% recipe per model. Plan ~75 s/model for a CIFAR-100 airbench-style ResNet18-equivalent at ~70%+ accuracy.
- **`ml_privacy_meter` already integrates `speedyresnet`** (`/trainers/fast_train.py`, hlb-CIFAR10) — set `model_name: speedyresnet` in the YAML to get ~7 s/model. This is the lowest-friction path; you can train **~500 reference models in a 1 h JURECA slot** on a single A100 node, ~2000 across 4× A800 in parallel. That overshoots Tong's "42 reference models" idealized budget by ~50× — use the surplus to do **architecture-matched** references for ResNet50 and ResNet152.
- Standard non-fast recipes for CIFAR-100 + ResNet variants on A100 (corroborated across pytorch-cifar100 and DocsAid configs): SGD lr=0.1, momentum=0.9, Nesterov, weight_decay=5e-4, batch 128, cosine or step (×0.2 at epochs 60/120/160), 200 epochs to ~76–78% top-1 (ResNet18) / ~78–80% (ResNet50). Wall-clock per epoch on a single A100: ResNet18 ≈ 8–12 s; ResNet50 ≈ 20–25 s; ResNet152 ≈ 45–55 s (extrapolated from cgnorthcutt/cnn-gpu-benchmarks Titan-RTX numbers, ~1.5× faster on A100). So 50-epoch SGD reference runs cost ≈10/20/40 min on a single A100; with `--gres=gpu:4` data-parallel on dc-gpu: 3/6/12 min per model with reduced batch-size scaling.
- **Architecture-mismatch penalty (HIGH-confidence claim, source: Zarifzadeh RMIA paper §C.2):** RMIA "remains effective when pre-trained models on different datasets are used as reference models … the superiority of our attack becomes more pronounced in the presence of architecture shifts (we observe up to 3% increase in the AUC gap between our attack and others)." So architecture mismatch hurts AUC by perhaps a few percentage points but RMIA degrades less than LiRA. **Caveat (medium-confidence)**: ImpMIA (arXiv:2510.10625, Oct 2025) shows in fresh experiments that "RMIA, despite reporting state-of-the-art results in its original paper, is highly sensitive to training configurations—especially at very low false-positive rates. Even minor mismatches in normalization, architecture, optimizer, or learning-rate schedule caused severe" performance loss. The two findings are reconcilable: AUC is robust to mismatch, TPR@low-FPR is not. For DUCI you live in the **AUC regime** (debiasing uses dataset-level TPR/FPR), so you'll be OK — but that's exactly why you must pick MIA threshold β at a moderate FPR (~0.3–0.5), not at low FPR.
- **Training-recipe mismatch (medium-confidence):** No paper directly measures the MAE penalty of recipe mismatch on Tong-DUCI. The relevant dial is whether your reference models' member-vs-non-member loss-gap distribution matches the target's. If organizers used standard SGD+cosine for 50 epochs (a strong prior given they confirmed 50 epochs in the YAML default), match that. **Practical bet**: train references with **the same recipe as the privacy_meter default** (`SGD lr=0.1, wd=0, 50 epochs, batch 256`). Tong's reported MAE 0.087 → 0.034 used WRN28-2; the ResNet-34 numbers (0.053 → 0.015) suggest deeper networks actually behave *better* under DUCI — so don't panic about ResNet152.

**Sources**
- https://arxiv.org/abs/2404.00498 — airbench / 94% CIFAR-10 in 3.29 s
- https://github.com/KellerJordan/cifar10-airbench
- https://github.com/privacytrustlab/ml_privacy_meter — `speedyresnet` integration in `/trainers/fast_train.py`
- https://github.com/weiaicunzai/pytorch-cifar100 — standard CIFAR-100 ResNet recipe
- https://arxiv.org/html/2312.03262v3 — RMIA Appendix C.2 architecture-mismatch ablation
- https://arxiv.org/html/2510.10625 — ImpMIA, RMIA configuration-sensitivity warning
- https://github.com/cgnorthcutt/cnn-gpu-benchmarks — wall-clock baseline for ResNet18/50/152 on CIFAR-100
- https://apps.fz-juelich.de/jsc/hps/jureca/gpu-computing.html — JURECA dc-gpu A100 spec (4×A100-40GB/node)

**Confidence**: high on speed numbers and overall budget; medium on architecture-mismatch MAE penalty (no Tong-specific ablation found); medium on training-recipe sensitivity claim.

---

## Gap 2 — Held-out-fill-in vs POPULATION distribution mismatch

**Findings**
- The cleanest articulation of your concern is in **Cretu et al. 2025 (arXiv:2502.18986)**: there are two distinct sampling regimes — (a) members and non-members both drawn from the *same* distribution as the target's training set (your case if held-out fill-in and POPULATION are both i.i.d. from the underlying CIFAR-100 pool), vs. (b) non-members drawn from a *third* distribution. If the organizers' held-out fill-in and POPULATION are both balanced subsets of the unrestricted CIFAR-100 pool with the same class balance, this is the i.i.d. case and the standard MIA pipeline is unbiased.
- **The dominant published failure mode is *temporal/topical* distribution shift** (Duan et al. ICML 2024 "Do MIAs Work on LLMs", Meeus et al. SoK arXiv:2406.17975): a bag-of-words classifier can sometimes distinguish "members" from "non-members" near-perfectly when they're collected post-hoc from different time windows. CIFAR-100 in 2026 with hackathon-curated subsets makes this risk *low* — but **you should run their diagnostic**: train a tiny linear classifier on raw pixels/features to distinguish MIXED from POPULATION; if it gets meaningfully above 50% AUC, your two pools are not i.i.d. and your TPR/FPR estimates are biased.
- The "interfering training data" effect — model has seen `(1−p)·held-out`, which affects the *trajectory* but not membership of MIXED points — is implicitly handled by the Tong debiasing, *provided your reference models are trained on the same kind of mix*. Concretely: your reference models should also be trained on `(0.5·MIXED) ∪ (0.5·something_else)` where "something else" comes from a pool drawn the same way as POPULATION. The bias term Tong derives in Appendix E (`Corr(TPR_i − FPR_i, γ_i)/(TPR − FPR)`) is small for i.i.d. sampling. There is no published analysis isolating the "interfering data" trajectory effect for image classifiers, but the **causal-MIA framework of arXiv:2602.02819** ("MIAs from Causal Principles", 2026) names the issue ("interference between jointly included points") and provides estimators for the multi-run regime that you have access to here. It's too heavy to implement in 24 h, but worth knowing exists if a follow-up paper is your end-goal.
- **Concrete checks to run before committing (do all three in your first 2 h):**
  1. Train a 2-layer linear probe on raw flattened images: MIXED vs POPULATION — AUC should be ~0.5.
  2. Per-class count: ensure POPULATION has the same per-class distribution as MIXED (the spec says class-balanced).
  3. Train one extra reference model `θ_ref` on (0.5·MIXED) ∪ (0.5·subset_A_of_POPULATION). Compute MIA score on `MIXED \ θ_ref`'s training set vs. `subset_B_of_POPULATION`. AUC > 0.65 means you have signal; 0.55–0.65 marginal; <0.55 you have a problem and should pivot to Avg-Logit MLE.

**Sources**
- https://arxiv.org/html/2502.18986v1 — Cretu et al., heterogeneous sampling regimes for MIA
- https://arxiv.org/pdf/2406.17975 — Meeus et al., SoK on MIA non-member distribution shift
- https://arxiv.org/html/2402.07841v2 — Duan et al., MIAs on LLMs, distribution-shift confound
- https://arxiv.org/pdf/2602.02819 — MIAs from Causal Principles (formalises interference)
- https://arxiv.org/pdf/2506.06488 — MIAs for Unseen Classes / quantile-regression alternative
- https://arxiv.org/pdf/2510.19773 — TNR-of-loss as MIA-free vulnerability proxy

**Confidence**: high on the conceptual framing; high on the recommended checks; medium on whether they'll actually trigger in this hackathon (likely your two pools are clean).

---

## Gap 3 — Joint inference across the 9 targets sharing one MIXED set

**Findings**
- **No published method does joint multi-target DUCI.** I searched openreview, arXiv 2024–2026, and Tong's followup workshop paper — all DUCI work treats target models independently. Tong's own "On the possibility of pair-wise bias reduction (Second-Order Debiasing)" appendix in the DATA-FM workshop version (https://openreview.net/pdf?id=CAID1FntdA) explicitly considers *cross-record* debiasing within a single target and concludes gains are minimal because `p̂_i` are nearly independent (your Figure 4 reference). Cross-target structure is unaddressed in the literature.
- **Why it would still help in principle**: across the 9 models, the per-sample `p̂_i^{(k)}` for k=1..9 estimate the Bernoulli inclusion probability `γ_i` × indicator that target k is "easy to attack". A simple two-way decomposition `p̂_i^{(k)} ≈ a_k · b_i + c_i` (k-effect = target's MIA strength, b_i = per-sample memorability, c_i = baseline) is a low-rank model. With 9 × |X| observations you could fit it in <1 minute and produce per-target proportion estimates that share strength.
- **However**: the architecture differs across the 9 (3× R18, 3× R50, 3× R152). The "easy member" set is *architecture-dependent* (deeper models memorize different points; e.g., per Carlini et al. and ImpMIA, sample-level vulnerability has only ~0.4–0.6 rank correlation across architectures). So the practical lift from joint modelling is unlikely to beat the variance reduction of just running RMIA + Tong debiasing well per-target. **My recommendation: skip joint inference; spend the time tuning per-target.**
- The closest open-source code for "many-target shared dataset" is Maini's **PostHocDatasetInference** (sprintml/PostHocDatasetInference, ICML 2025) which uses synthetic held-outs and post-hoc calibration — designed for LLMs but the calibration framework (linear regression to align suspect/heldout score distributions) could be retrofitted. **Not recommended within 24 h.**

**Sources**
- https://openreview.net/pdf?id=CAID1FntdA — Tong DATA-FM workshop appendix on Second-Order Debiasing
- https://github.com/sprintml/PostHocDatasetInference — Zhao/Maini ICML 2025, post-hoc calibration code
- https://arxiv.org/pdf/2506.15271 — paper

**Confidence**: high that no published joint DUCI exists; medium on the analysis that joint modelling wouldn't help much here.

---

## Gap 4 — Updates to Tong DUCI since ICLR 2025; SprintML baselines

**Findings**
- **Tong et al. follow-up (DATA-FM workshop @ ICLR 2025, OpenReview CAID1FntdA)** is the only direct extension I found — it discusses pair-wise debiasing and confidence intervals but reports **no new MAE numbers that beat the ICLR'25 paper**. In the workshop version they explicitly state the second-order correction provides minimal gains.
- **SprintML's own related work** (Boenisch / Dziedzic, your hackathon PIs) is in a different lane:
  - **CDI (CVPR 2025, Dubiński et al.)** — copyrighted-data identification *for diffusion models* using GMM density on representations and a meta-classifier; binary detection, not proportion estimation. Not directly relevant for ResNet classifiers, but methodologically informative: they aggregate features (loss, gradient norm, etc.) and feed to a small classifier — this is the **flavor of solution they probably consider canonical**.
  - **Privacy attacks on IARs (Kowalczuk et al. 2025, arXiv:2502.02514)** — image-autoregressive MIAs, requires only 6 samples for DI vs 200 for DMs.
  - **PostHocDatasetInference (Zhao et al. ICML 2025)** — LLM-only, post-hoc calibration of synthetic held-out scores. The *calibration trick* is portable (regress real-vs-synthetic likelihood to remove distribution shift) but the held-out-generation pipeline is text-specific.
  - **Dataset Inference for SSL (Dziedzic NeurIPS 2022)** — you noted this; not directly applicable to supervised classifier.
- **Newer general MIA backbones worth considering as RMIA replacements:**
  - **BaVarIA (arXiv:2603.11799)** — Bayesian variance refinement of LiRA/RMIA, claims best-in-class for **low shadow-model and offline regimes** (i.e., your regime). MAE-on-DUCI not reported but TPR@low-FPR gains over RMIA are reported across 12 datasets. Recommended as a drop-in if you have time.
  - **InfoRMIA (arXiv:2510.05582)** — token-level for LLMs; not relevant.
  - **PMIA / CMIA (arXiv:2507.21412, NDSS 2026)** — proxy/cascading MIAs, target adaptive setting; not directly useful.
  - **The Tail Tells All (arXiv:2510.19773)** — proposes loss-tail-TNR as a *reference-model-free* MIA proxy. Worth knowing as a fallback if your reference models are bad — they claim to outperform low-cost RMIA.
- **No published negative result** showing a much-simpler baseline beats Tong's debiasing on a CIFAR-100 + ResNet setting. Within Tong's own Table 1, **Avg-Logit MLE with 42 reference models** is occasionally close to the debiased estimator (e.g., CIFAR-100/WRN28-2: Tong 0.0339 vs MLE 0.0617). At high reference-model count the gap is real; at low count Tong wins by 2–4×.

**Sources**
- https://openreview.net/pdf?id=CAID1FntdA — Tong workshop followup
- https://github.com/sprintml/copyrighted_data_identification — CDI (CVPR 2025)
- https://arxiv.org/abs/2502.02514 — IAR privacy attacks
- https://github.com/sprintml/PostHocDatasetInference — ICML 2025
- https://arxiv.org/pdf/2603.11799 — BaVarIA, exponential-family unification of LiRA/RMIA/BASE
- https://arxiv.org/pdf/2510.19773 — Tail-TNR, no-reference-models proxy
- https://sprintml.com/publications/ — SprintML lab publication list (master reference)

**Confidence**: high that Tong's ICLR'25 paper is still the SOTA published method for DUCI; high on the SprintML paper inventory; medium on BaVarIA being clearly better for your regime (no DUCI ablation).

---

## Gap 5 — Plugging pre-existing checkpoints into `ml_privacy_meter`

**Findings**
- The README and README-`run_duci`-related sections explicitly describe the plug-in path: "By default, the Privacy Meter checks if the experiment directory specified by the configuration file contains `models_metadata.json`, which contains the model path to be loaded. To audit trained models obtained outside the Privacy Meter, you should follow the file structure (see `<log_dir>/<models>` in the next section) and create a `models_metadata.json` file that shares the same structure as the one generated by Privacy Meter. You can also run the demo configuration file with a few epochs to generate a demo directory to start with."
- **Concrete recipe**: (1) run `run_duci.py` with the demo CIFAR-10/cifar10.yaml and `epochs: 1` to scaffold the directory; (2) replace the contents of `<log_dir>/models/model_<id>.pkl` with the organizers' checkpoints (you may need to wrap them in the right state_dict format — `torch.save(model, path)` vs `state_dict` matters; check what `models/utils.py:get_model` expects); (3) edit `models_metadata.json` so each entry maps `model_id` to the file path and the (empty) `train_split` it was supposedly trained on (you can put the full MIXED there as a placeholder, the framework only uses it to compute *reference-model* TPR/FPR, not target-model membership). Then point the YAML at this dir and run.
- **Caveat / open question**: I could not directly fetch `documentation/duci.md`. The exact JSON schema isn't documented in the snippets I retrieved — you'll want to do a fresh `cat <log_dir>/models/models_metadata.json` after a 1-epoch demo run to see the expected format.
- **No public fork or notebook found** that loads externally-trained checkpoints into `run_duci.py` end-to-end. The `demo_duci.ipynb` notebook lives in the master repo; the dev mirror at `privacytrustlab/privacy_meter_dev` provides Colab-friendly versions for `mia`/`ramia`/`duci`. No third-party reproduction of Tong's DUCI pipeline exists on GitHub as far as I can find — this codebase is the only implementation.
- **Easier alternative**: skip the framework entirely. The Tong algorithm is ~50 lines: load each target model, run a per-record RMIA call (which is about 100 lines wrapping a single forward pass on the target + each reference), threshold at β to get `m̂_i`, then compute `p̂ = (1/|X|) Σ (m̂_i − FPR̂)/(TPR̂ − FPR̂)` from reference-model-derived TPR̂/FPR̂. Given the limited time, I'd recommend **a lean from-scratch implementation** for the target-model side of the pipeline and reuse the framework only for reference-model training (where `speedyresnet` integration is a clear win).

**Sources**
- https://github.com/privacytrustlab/ml_privacy_meter — README "Auditing Trained Models" section
- https://github.com/privacytrustlab/privacy_meter_dev — Colab-friendly mirror with `demo_duci.ipynb`
- https://github.com/MantonDong/ml_privacy_meter — community fork (no DUCI-specific changes spotted)

**Confidence**: high on the general approach (the README is explicit); medium on the exact JSON schema (need to verify locally).

---

## Gap 6 — Submission strategy under MAE

**Findings**
- **Tong's evaluation grid**: `p ∈ {0.00, 0.05, 0.10, …, 1.00}` (21 values, 5% step). If organizers used this same grid (high prior given they're likely working from privacy_meter conventions), **snapping to nearest 0.05** is a near-free MAE win — your *prediction error* against any value drawn from the grid is at worst 0.025 (vs. potentially much more if you're off by uncalibrated noise). If they used continuous `p` then snapping is neutral-to-slightly-harmful (≤0.0125 expected loss).
- **Clamping**: under MAE, the optimal Bayes prediction for a single point is the **median** of the posterior. If your posterior on `p` is approximately Gaussian with std σ, your unbiased mean is also the median; clamping to `[ε, 1−ε]` (with ε ≈ 2σ̂) only helps when σ̂ pushes mass outside [0,1]. For Tong's reported MAE ≈ 0.087 with 1 reference model on CIFAR-100/WRN28-2, σ ≈ 0.10–0.13 → clamp to **[0.025, 0.975]** is sensible (loses essentially nothing if true `p` ∉ {0,1}, gains a lot if it is).
- **Shrinkage toward 0.5** (Stein-style): only beneficial under squared loss, not L1. Under MAE, the **L1-optimal estimator is the posterior median, not a shrunken mean**. Don't shrink. However, if you suspect organizers used a *uniform prior* on the 21-point grid and you have very low confidence on a particular target, regressing to the prior mean (0.5) **does help under MAE** because the average distance from 0.5 to a uniform draw on [0,1] is 0.25, vs. a random guess having expected MAE 1/3. So: if your confidence interval (Tong Eq. 8) on a target is wider than ±0.25, **predict 0.5 for that target** — better than your noisy estimate.
- **9 predictions × MAE averaged**: variance per prediction matters — don't try to be clever on individual targets. The single biggest MAE win is **never predicting 0 or 1 even when you're confident**; an over-confident wrong prediction (say true p=0.05, you predict 0) costs you 0.05/9 ≈ 0.0056 per model in the average; an over-confident wrong prediction the other way (true p=0.95, you predict 1) costs the same. But predicting 0.025 / 0.975 caps your loss per target at 0.025 in the worst case. So clamping to [0.025, 0.975] is **strictly better than clamping to {0, 0.05, …, 0.95, 1}** if you don't know whether the grid is used.
- **Public/private leaderboard split (3/9 vs 6/9 + extended hidden)**: with only 3 public-leaderboard models, the public score is extremely noisy (a single bad prediction shifts MAE by ~0.03). Don't tune to public-MAE; treat the public score as a sanity check that your pipeline runs end-to-end.

**Sources**
- https://openreview.net/pdf?id=EUSkm2sVJ6 — Tong, Section 5.1 (5%-step grid, 21 proportions, 30 trials)
- https://openreview.net/pdf?id=CAID1FntdA — workshop version, same grid, Lyapunov CLT confidence interval

**Confidence**: high on MAE-vs-MSE estimator theory; medium on the organizer-grid hypothesis (they may sample continuous `p`); the calibration recipes are robust either way.

---

## Gap 7 — Sanity-check signals before committing to debias formula

**Findings**
- Diagnostic plots / statistics, in priority order:
  1. **MIA-score histogram overlap** for `MIA(x; θ)` evaluated on `x ∈ MIXED` vs `x ∈ POPULATION`. Compute KS or Wasserstein distance. If they fully overlap, MIA has no signal.
  2. **Synthetic in/out AUC**: Train one ResNet18 on a known 50/50 split of MIXED. Run RMIA on the 50% known-in vs 50% known-out and compute AUC. **If AUC < 0.6**, you should not trust the debiasing — fall back to averaged-logit MLE. Tong's reported numbers correspond to MIAs with AUC ≈ 0.65–0.78 on CIFAR-100.
  3. **Reference-model FPR/TPR estimates**: compute over multiple reference models and check std-error. If TPR − FPR < 2× its std, your debias denominator is unstable; clip the prediction.
  4. **Per-target MIA-score mean**: simply averaging MIA scores across MIXED gives a monotone (but biased) signal that should already correlate with `p`. If you sort the 9 targets by mean MIA score and the order looks "wrong" (e.g., constant), your MIA isn't working.
- **Fallbacks if MIA AUC < 0.6**:
  - **Avg-Logit MLE** (Tong's strongest non-debiased baseline): for each candidate `q ∈ {0, 0.05, …, 1}`, fit a 1-D Gaussian `N(μ_q, σ_q²)` over the average target-class logit on MIXED across reference models trained at fraction `q`; predict `argmax_q P(target_logit | N(μ_q, σ_q²))`. Per Tong Table 1, MAE ≈ 0.06 with 42 reference models on CIFAR-100/WRN28-2.
  - **Tail-TNR proxy** (arXiv:2510.19773): no reference models needed, uses absence of high-loss outliers as risk score. Reported to outperform low-cost RMIA. Only gives you ranking, not absolute `p`, so you'd need to calibrate against POPULATION as a `p=0` anchor.
  - **Maini Dataset-Inference white-box (MinGD)**: gradient-based ℓ2 distance to decision boundary, averaged over MIXED, calibrated against POPULATION. Crude but has zero failure modes.
- **Per-record `p̂_i` ∈ [−something, 1+something]**: Tong Eq. 4 gives raw values that can fall outside [0,1]; the *aggregate* `p̂` is well-behaved by CLT, but if you see > 30% of `p̂_i` outside [−0.5, 1.5], your TPR/FPR are mis-estimated.

**Sources**
- https://openreview.net/pdf?id=EUSkm2sVJ6 — Tong, MIA Guess / MIA Score / MLE baselines (Section 5.2)
- https://arxiv.org/pdf/2510.19773 — Tail-TNR, no-reference-models alternative
- https://arxiv.org/pdf/2307.03694 — Quantile regression MIA (single-model fallback)

**Confidence**: high on the diagnostics; high that Avg-Logit MLE is the right fallback (it's Tong's own baseline).

---

## Top 3 recommended actions for next 4 hours

1. **Bootstrap the pipeline end-to-end on a single target (≤2 h).** Clone `ml_privacy_meter`, set `model_name: speedyresnet`, train **1 reference ResNet18** on a random 50% of MIXED in <60 seconds on JURECA dc-gpu. Implement Tong Algorithm 1 yourself in ~50 lines (don't fight the framework's `run_duci.py`): for one organizer checkpoint, compute `m̂_i ∈ {0,1}` per `x ∈ MIXED` via single-reference-model RMIA (a=0.3 linear approximation, β chosen at FPR ≈ 0.4); compute global TPR̂/FPR̂ from the reference; output `p̂` clamped to [0.025, 0.975]. Submit this to the public leaderboard immediately as a baseline.
   - **Stop and pivot to MLE if**: synthetic in/out AUC < 0.6 (per Gap 7 diagnostic 2).

2. **Burn JURECA budget on architecture-matched references (≈1.5 h wall-clock, parallelisable across 4×A800).** Train **8 reference ResNet18 + 4 ResNet50 + 2 ResNet152**, all on freshly sampled 50% MIXED + 50% POPULATION-disjoint splits, using either (a) `speedyresnet` if you can adapt it for ResNet50/152, or (b) standard SGD lr=0.1 wd=5e-4 cosine 50 epochs (~3/6/12 min/model on 1× A100 in dc-gpu). This puts you at Tong's ~14-reference-model regime where MAE drops from ~0.087 → ~0.04. Crucially: **train the references with the same recipe as the privacy_meter default that organizers likely used** (SGD, no weight decay, 50 epochs, batch 256), not a custom one — recipe match matters more than epoch count or augmentation.
   - **Benchmark/threshold**: if synthetic-split AUC of architecture-matched references > AUC of architecture-mismatched references by ≥0.05, commit to per-depth references for the final predictions; otherwise use ResNet18 references for everything to maximise count.

3. **Run the three Gap-2 sanity checks (≤30 min) before generating final submissions, then write submission script (≤30 min).** Specifically: (a) linear probe MIXED-vs-POPULATION raw-pixel AUC ~0.5 (else flag distribution shift); (b) per-class balance check; (c) self-trained reference attack-AUC. Final submission script: predict per-target `p̂` via Tong debiasing; clamp to [0.025, 0.975]; if Tong CLT confidence interval > ±0.25, override to 0.5; if you suspect organizers used a 5%-grid, snap to nearest 0.05 *only* on a separate experimental submission to compare. Don't tune to the 3-model public leaderboard — its signal-to-noise is too low.
   - **Threshold for action 3 override to 0.5**: σ̂_p̂ > 0.13 (i.e., 95% CI half-width > 0.255).

## Caveats

- I could not directly fetch `huggingface.co/datasets/SprintML/DUCI` or the `documentation/duci.md` file in `ml_privacy_meter` (access errors in this research environment). Specific JSON schema and the precise dataset format on HF must be verified by you locally — but the README quote in Gap 5 covers the general mechanism.
- Wall-clock numbers for ResNet50/152 on A800 (vs A100) are extrapolated from A100 benchmarks scaled by ~0.85; real numbers depend on your batch size and whether you enable `torch.compile` / TF32 / mixed precision. Run a one-epoch warmup and re-budget.
- The ImpMIA finding that "even minor mismatches in normalization … caused severe" RMIA degradation is from a paper still under ICLR review (arXiv:2510.10625, posted Oct 2025); take its claims as a yellow flag, not gospel. The original RMIA paper makes the opposite, more optimistic claim about robustness — both are in the report so you can judge.
- "Joint multi-target DUCI" is genuinely an open research direction; if you find time and your pipeline works, the 9-targets-share-one-MIXED structure is a publishable angle the SprintML PIs would likely care about. Don't pursue it during the 24 h, but flag it for post-hackathon work.
- The hackathon scoring on a hidden extended test set after submission close means **generalisation beats public leaderboard tuning by definition** — your submission strategy (clamping, no public-overfitting) should be conservative.