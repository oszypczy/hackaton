# Challenge D — Property Inference / Fairness Audit (OPTIONAL)

**Source:** research `06_fairness_auditing.md` + Barcelona 2025 edycja
**Sugerowana osoba:** privacy expert (alternatywa do A) lub gdy ktoś skończył wcześniej
**Czas:** 4–6h
**Status:** OPCJONALNY — bierzemy jeśli ktoś chce drugą rundę praktyki

## Po co to

Barcelona miała challenge: dany medyczny model AI (CXR / dermatology / EHR), tylko API access, zdecyduj **czy training set był demograficznie zbalansowany czy zskewowany**. Research 06 ma gotowy 4-godzinny pipeline. Bardzo duża szansa że Warsaw 2026 będzie wariantem tego.

Property inference ≠ membership inference. Pytanie tu jest globalne ("jaki % treningowych pacjentów to kobiety?"), nie indywidualne.

## Setup

### Target model (fixture, generowany przez kolegę-CUDA)
- **Modalność:** chest X-ray (CheXpert / MIMIC-CXR public datasets)
- **Architektura:** DenseNet-121, fine-tuned na 5 chorobach
- **Property:** ratio sex (male/female) w training set
- **Dwa kandydaci ratio:** balanced (0.5) vs skewed (0.3 lub 0.7)
- 10 victim modeli — 5 trenowane przy 0.3, 5 przy 0.5. Ground truth ujawnione po submisji.

### Fixture data (CUDA, ~1h)
1. CheXpert subset (publiczny, ~50k obrazów z label sex)
2. Per-ratio: 5 modeli DenseNet-121, fine-tune 10 epok, każdy seed inny
3. Plus shadow bank: 8 modeli per ratio (16 total) używane przez teamy podczas zadania
4. Public probe set: 1000 obrazów + counterfactual pairs przez StyleGAN sex-swap

```
data/D/
  victims/                  # 10 zaszyfrowanych checkpointów
  shadows/                  # 16 shadow models (do attack)
  probes/                   # 1000 chest X-rays + 500 counterfactual pairs
  api_endpoint.py           # local serving wrapper, rate-limited
```

## Zadanie + format submission

Dla każdego z 10 victims:
```json
{"victim_id": int, "predicted_ratio": float, "p_skewed": float, "ci_low": float, "ci_high": float, "n_queries": int}
```

Budget: **2000 queries per victim** (czyli max 20k łącznie).

## Scoring (SprintML eval style)

1. **MAE on predicted_ratio** (40%) — primary, mean absolute error vs ground truth
2. **TPR@FPR=1%** dla binary skewed/balanced classification (30%)
3. **Brier score na p_skewed** (20%) — calibration
4. **Query efficiency bonus** (10%) — `1 - n_queries/budget` jeśli prediction correct

Reference scores:
- Easy (random probes + threshold): MAE ≈ 0.10, TPR@1% ≈ 30%
- Solid (KL attack Suri-Evans 2023): MAE ≈ 0.05, TPR@1% ≈ 60%
- Hard (BAFA + counterfactual): MAE ≈ 0.02, TPR@1% > 80%

## Baselines

### Easy (1h)
**Suri-Evans KL attack**: dla każdego ratio b ∈ {0.3, 0.5}, średnia softmax outputów shadow modeli na probach. Dla victim, KL(p_target || p_b) → b̂ = argmin_b KL.

### Solid (3h)
**LiRA-style LRT**: per-probe Gauss fit z shadow modeli, log-likelihood ratio. Plus **counterfactual paired probes** (StyleGAN sex-swap) — pair-difference `f(x) − f(x')` redukuje variance o rzędy wielkości.

### Hard
- **SNAP poisoning** (Chaudhari et al. S&P 2023) — jeśli wolno załadować dane do training (zwykle nie wolno na hackathonie, więc skip)
- **BAFA active query selection** (Hartmann et al. 2026) — Bayesian opt na surrogate disagreement, 40× fewer queries
- **Multi-attribute**: sex × race intersection, BH FDR control

## M4 friendly?

Tak. Wszystko inference + statystyka. CheXpert subset ~5GB, mieści się.

## Co to ćwiczy

- Shadow model harness (k=4 shadows per ratio = SNAP-grade)
- KL/LRT statistical tests
- Counterfactual paired probes
- Calibration (isotonic, conformal)
- Sample complexity reasoning: `n ≈ 16·p(1−p)/Δ²`

## Recommended reading

- Suri & Evans, *Formalizing and Estimating Distribution Inference Risks* (PETS 2022) — formalny framework
- Suri, Lu, Chen, Evans, *Dissecting Distribution Inference* (SaTML 2023) — KL attack
- Chaudhari et al., **SNAP** (S&P 2023) — poisoning + logit fit
- Gichoya et al., *Reading Race* (Lancet Digital Health 2022) — co konkretnie leak'uje w CXR
- **CDI (Dubiński, Kowalczuk, Boenisch, Dziedzic, CVPR 2025)** — methodological template SprintML
