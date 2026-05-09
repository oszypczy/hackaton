# Deep Research Prompt — Task 1 DUCI

> Prompt gotowy do wklejenia do **Claude Deep Research**.
> Zoptymalizowany pod konkretne luki (nie powtarza tego co mamy w paperach 05/06/07 + reference impl).
> Last updated: 2026-05-09 (post-task-reveal)

---

## Prompt (kopiuj poniżej)

I'm on a 3-person team competing in a 24-hour ML security hackathon (CISPA Warsaw 2026). I need practical, actionable research to implement and tune one specific task. **Skip foundational explanations; focus on gaps in what I already know.**

### Task: Dataset Usage Cardinality Inference (DUCI)

Given **9 trained ResNet image-classifier checkpoints** (3× ResNet18 with model_id starting `0`, 3× ResNet50 starting `1`, 3× ResNet152 starting `2`), I must predict for each model a **continuous score `p ∈ [0, 1]`** representing the fraction of a "MIXED" dataset used in training. Each model was trained on `(p · MIXED) ∪ ((1-p) · held-out fill-in)` with the **total training-set size kept constant** and **classes balanced**. The MIXED set is given as `X.npy` + `Y.npy`. A separate "POPULATION" set is provided — same in-distribution data, **never seen by any of the 9 models** (use as non-member reference). The internal "held-out fill-in" used by organizers is **NOT given** and is distinct from POPULATION.

**Source**: HuggingFace `https://huggingface.co/datasets/SprintML/DUCI`. Likely CIFAR-100 (presentation said 2000 samples × 100 classes; not 100% confirmed — will verify on download).

**Metric**: Mean Absolute Error of predicted `p` vs ground truth, averaged across 9 models. Public leaderboard scored on 3/9 models (random subset); private leaderboard on the remaining 6/9 — final ranking is on the **private 6** + an **extended hidden test set** announced after submission close. **Generalization > overfitting public.**

**Compute**: 3 MacBooks M4 (MPS, no CUDA, no `bitsandbytes` 4-bit) + 1 Jülich JURECA cluster slot (`dc-gpu`, 4× A800/node, partition `training2615`, ~limited project budget shared across 800 GPUs in pool).

### What I already know (DO NOT re-summarize this; build on it)

**Primary paper — Tong et al. ICLR 2025 "How Much Of My Dataset Did You Use?" (arXiv 2502.??? or OpenReview EUSkm2sVJ6).** Key facts I have:

- Algorithm 1 / Equation 4: `p̂_i = (m̂_i − FPR) / (TPR − FPR)`, then `p̂ = (1/|X|) Σ p̂_i`. Globally estimated TPR/FPR (single value across X, NOT per-i) using ≥1 reference model trained on random 50% of MIXED.
- MIA backbone: **RMIA** (Zarifzadeh 2024). Single-reference-model variant viable. Linear approximation when N=1: `a=0.3` for the Pr(x|θ)/Pr(x) denominator.
- Reported results on CIFAR-100 with WRN28-2: max MAE ≈ 0.087 (1 ref model) → 0.034 (42 ref models). With ResNet-34 / CIFAR-100: 0.053 → 0.015.
- Reference impl exists: `github.com/privacytrustlab/ml_privacy_meter` — files `run_duci.py`, `modules/duci/module_duci.py`, `configs/duci/cifar10.yaml`, `demo_duci.ipynb`, `documentation/duci.md`. Default config: WRN28-2, SGD lr=0.1 wd=0, 50 epochs, batch 256.
- "Special sampling" (non-i.i.d. selection like EL2N coreset) hurts MAE: 0.062 → 0.109.
- Confidence intervals via Lyapunov CLT.
- Per-record correlations between p̂_i are negligible (Figure 4) — pairwise debiasing gains are minimal.

**Maini et al. ICLR 2021 (paper 06)** — original DI:
- Blind Walk (black-box): k random directions until misclassification, distance as proxy for prediction margin. 30-dim embedding (10 directions × 3 distributions: uniform/Gaussian/Laplace).
- MinGD (white-box): gradient descent to nearest decision boundary, ℓ1/ℓ2/ℓ∞ norms.
- Confidence regressor = 2-layer linear net with tanh, then Welch t-test.

**Dziedzic et al. NeurIPS 2022 (paper 07)** — DI for SSL encoders, uses GMM density on representations. **Probably not directly useful** for supervised classifier task, but methodological style of organizers (Dziedzic + Boenisch are the hackathon PIs).

### Specific gaps I need filled (RANKED by usefulness for me)

**1. Reference-model training on Jülich, time-budgeted (HIGH)**
- For CIFAR-100 + ResNet18/50/152, what's a practically tested **fast** training recipe (NOT the 200-epoch SOTA, but ~50 epochs, with augmentation) that produces well-generalized models suitable as MIA reference models? Concretely: optimizer schedule, learning rate, weight decay, data augmentation, mixed precision, expected wall-clock per epoch on A100/A800. I need to budget how many shadow models I can train in ~12h on 4× A800.
- For the MIA debiasing in paper 05, the reference model just needs to have been trained on a known random 50% of MIXED. Does it matter if its **architecture differs** from the target (e.g., I train all reference models as ResNet18 to save compute, but target includes ResNet50 / ResNet152)? Are there empirical results on **architecture-mismatched RMIA**? If yes, what MAE penalty?
- Same question for **training-recipe mismatch**: organizers' checkpoints used some training recipe we don't know exactly. How robust is RMIA + Tong debiasing when reference models use a different (but reasonable) recipe?

**2. Held-out-fill-in vs POPULATION distribution mismatch (HIGH)**
- The target models trained on `(p · MIXED) ∪ ((1-p) · held-out)`, but my non-member reference is POPULATION (a different held-out set the model never saw). If the held-out fill-in and POPULATION are both i.i.d. samples from the same underlying distribution, this should be fine — but in practice: are there documented failure modes when an MIA's "non-member proxy" is from a different sample than the actual non-members in the target's training? What checks should I run?
- More subtle: the target model has seen `(1-p) · held-out` — those samples are members of the target's training set but NOT members of MIXED. When I run MIA(x_i; θ) for x_i ∈ MIXED, the model's behavior on x_i depends on whether x_i was part of the `p` fraction. The held-out fill-in shouldn't affect this directly — but it does affect the model's overall optimization trajectory. **Are there papers analyzing this "interfering training data" effect on MIA?**

**3. Estimating `p` for ResNet18/50/152 with SAME MIXED across models — joint inference (HIGH)**
- The 9 target models all use the **same MIXED set**, just different `p` per model. Naive approach: 9 independent debiasing runs. Smarter approach: jointly infer the 9 `p` values, exploiting the shared structure (e.g., the 3 ResNet18 models give correlated information about which MIXED samples are "easy members" / "hard members"). Has anyone studied **multi-model joint DUCI**? Any open implementations?

**4. Updates to Tong DUCI since ICLR 2025 (MEDIUM)**
- Any **2025-2026 follow-up papers** to Tong DUCI? Improvements, criticisms, new MIA backbones, applications to vision-specifically?
- Any concurrent / overlapping work — e.g., from SprintML (Boenisch / Dziedzic) — using cardinality / proportion estimation? They co-organize this hackathon so methods from their group might be expected baselines.
- Any negative results — settings where Tong's method underperforms a much simpler baseline?

**5. Reference-impl orientation (MEDIUM)**
- The `ml_privacy_meter` `run_duci.py` pipeline expects to **train** target models internally. For our task we have **pretrained** target checkpoints supplied by organizers — what's the cleanest way to plug pre-existing checkpoints into the framework without retraining them? Any docs / issues / forks that did this?
- Are there **other open-source DUCI implementations** (Maini's lab repos, third-party reproductions) worth looking at? Especially anything with a notebook demonstrating the full pipeline on CIFAR-100 + ResNet18.

**6. Submission strategy under MAE (LOW-MEDIUM)**
- Public leaderboard 3/9, private 6/9, then extended hidden test set. Given a single-shot MAE metric: **what calibration tricks reduce expected MAE under uncertainty?** Specifically:
  - Should I clamp predictions to `[ε, 1-ε]` to avoid extreme errors?
  - Should I shrink toward 0.5 when confidence is low (Stein-style)?
  - Any prior on `p` distribution (uniform on [0,1]? from Tong's 5%-granularity grid?) that helps?
- 9 predictions only, MAE averaged → variance per prediction matters a lot. Any rules of thumb?

**7. Sanity-check signals before committing to debias formula (LOW)**
- Before running full debiasing, what diagnostic plots / statistics tell me whether MIA is producing usable signal at all on these checkpoints? E.g., distribution overlap between MIA scores on MIXED vs POPULATION, AUC for synthetic in/out splits, etc.
- If MIA AUC is low (<0.6) — what's the fallback that doesn't require a working MIA?

### Constraints on output

- **Cite papers + GitHub repos directly with URLs.** I will follow up by reading them.
- **No re-summarization of papers I've cited above.** Assume I will read them myself.
- Prefer **practical, time-budgeted answers** over theoretical optima. I have 24h, not 24 days.
- If a question has no satisfying answer in published literature, **say so explicitly** rather than padding.
- If you find a CIFAR-100 + ResNet recipe with concrete wall-clock numbers on A100/A800, that's worth more than 5 paragraphs of theory.

### Output format I want

For each numbered gap (1–7) above, return:
- **Findings:** 2-5 bullets with concrete claims + sources
- **Sources:** list of URLs (papers, blog posts, repos, issues)
- **Confidence:** high / medium / low — be honest about uncertain claims

Then a final section: **Top 3 recommended actions for next 4 hours** based on what you found.

---

## Notatka po researchu

Po otrzymaniu odpowiedzi z deep research:
1. zapisz znalezione URL-e do `docs/tasks/task1_duci.md` w sekcji "Resources"
2. flaguj nowe papery które warto pobrać (`references/papers/`)
3. zaktualizuj sekcję "Pułapki / open questions" w `docs/tasks/task1_duci.md` o potwierdzone/odrzucone hipotezy
