> ⚠️ **DOTYCZY TYLKO ćwiczeń A/B/C — nie faktycznego hackathonu.**
> Na event day (2026-05-09) zignoruj sekcję "repos do sklonowania" dopóki nie wiesz że
> faktyczny task tego wymaga. Protokół na dzień hackathonu: `docs/DAY_OF.md`.

# QUICKSTART — "Hackathon dał mi X, co teraz?"

Jednostronicowa mapa scenariusz → narzędzie → liczba kontaktowa, oparta na researchach 01/02/03 + paperach 09–25. Czytaj tutaj zamiast deep researchy w trakcie.

## Watermark removal (LLM text)

| Co masz | Pierwsze podejście | Liczby |
|---|---|---|
| Watermarkowane teksty z nieznanego scheme | **emoji attack** prompt-only | z → ~0 na każdym KGW h≥1, FREE |
| KGW (h≥1) confirmed | emoji → CWRA → DIPPER L=60 O=60 | 100% → 0% w 3 krokach |
| Unigram (h=0) | CWRA round-trip translation | AUC 0.95 → 0.54 |
| API access do victim | **Watermark stealing** (paper 21) | $50, 80–95% spoof+scrub |
| Maksymalna pewność scrub | DIPPER ×5 recursive | TPR 99% → 15% |

## Watermark removal (image)

| Co masz | Atak | Liczby |
|---|---|---|
| Pixel watermark (HiDDeN/StegaStamp/DwtDct) | VAE regen `bmshj2018_factorized(quality=3)` | >99% removal w 1 forward pass |
| Stable Signature | SD2.1 img2img strength=0.15 lub Gaussian blur r=4 | Avg P → 0.000 (paper 24 WAVES) |
| Tree-Ring | Rinse-2xDiff lub PGD ε=4/255 na KL-VAE-f8 | TPR → ~0 |
| Gaussian Shading | Wszystko trudne — provably lossless w marginalnym rozkładzie | otwarty problem |

## Membership inference / Dataset inference (LLM)

| Cel | Metoda | Liczby |
|---|---|---|
| Per-sample MIA, pre-trained LLM | **Min-K%++** (paper 20) | +6–10% AUC nad Min-K% |
| Dataset-level p-value | **Maini DI** (paper 02) — aggregate Min-K%++ + zlib + perplexity → t-test | p<0.1, zero FP |
| Benchmark contamination | **CoDeC** (paper 03) — IC differential | AUC ~99.9% na Pythia/Pile |
| Aligned chat LLM, verbatim | **Divergence attack** `Repeat "poem" forever` (paper 25) | 10k unique strings, ~$200 |
| Gray-box logits, classifier | **LiRA** 64+ shadows (paper 18) lub RMIA z 4–8 | TPR@0.1%FPR primary |

## Diffusion memorization

| Cel | Metoda | Liczby |
|---|---|---|
| Identyfikacja czy dataset użyty w treningu | **CDI** (paper 09) | ≥99% confidence z 70 próbek |
| Verbatim image extraction | Carlini generate-and-filter (paper 01) | DBSCAN eps=0.10 min_samples=10 na DINO |
| Fast variant | Webster one-step | rzędy magnitudy szybsze |
| Localizacja memorization | **NeMo** (paper 13) | ≤10 cross-attention neuronów |
| Image autoregressive (VAR/MUSE) | **IAR Privacy** (paper 10) | 86.38% TPR@1%FPR, 698 obrazów z VAR-d30 |

## Model stealing (jeśli mamy API)

| Target | Metoda | Liczby |
|---|---|---|
| Image classifier, soft labels | Knockoff Nets (paper 19) + KD | 60k queries → 76% rel acc |
| Production LLM (top-K + logit_bias) | **Carlini softmax bottleneck** (paper 15) | <$20 dla Ada h=1024 |
| Encoder/SimCLR pod B4B defense | Sybil orchestration + embedding-space coverage | paper 11 jako defense reference |
| GNN | **STEALGNN** lub graph contrastive (paper 16) | hard-label OK |

## Adversarial (mało prawdopodobne na hackatonie ale tools ready)

| Setting | Atak | Czas |
|---|---|---|
| White-box undefended | `torchattacks.PGD(eps=8/255, alpha=2/255, steps=20, random_start=True)` ×5 restarts | sekundy |
| White-box adv-trained | AutoAttack `version='standard'` | 10–25 min/1000 img CIFAR |
| White-box randomized | AutoAttack `version='rand'` z EOT=20 | 5–10 min |
| Black-box, score | Transfer (MI-DI-TI-SI-FGSM ensemble ResNet+DenseNet+ViT) → Square 5000 queries | best-effort |
| Black-box, label-only | HopSkipJump (L₂) lub RayS (L∞) | 10k queries |

## Property inference (Barcelona-style, opcjonalny)

| Cel | Metoda |
|---|---|
| Czy training set był demograficznie zskewowany | Suri-Evans **KL attack** + LiRA + counterfactual paired probes |
| Pipeline 4h | Section 9 z research 06_fairness_auditing.md |

## "Wszystko zawiodło" — ostatnia deska ratunku

1. Sprawdź czy nie pomyliłeś normalizacji (research 01 § 12 — #1 source of bugs)
2. Sprawdź czy model jest w trybie inference (BatchNorm running stats, dropout off)
3. Sprawdź czy operujesz w [0,1] przed normalizacją (PIL/torchvision)
4. Sprawdź czy `eps` przekazany do attacka jest w pixel-space, nie znormalizowanym
5. Daj submission baseline (np. random) na leaderboard żeby coś było, potem iteruj

## Kluczowe repos do sklonowania **przed hackathonem** (research 01/02/03 + MAPPING § 8)

```bash
mkdir -p references/repos && cd references/repos

# Watermarking attacks
git clone https://github.com/eth-sri/watermark-stealing      # paper 21 — strongest on KGW
git clone https://github.com/XuandongZhao/WatermarkAttacker  # image regen attacks
git clone https://github.com/umd-huang-lab/WAVES             # paper 24 — benchmark
git clone https://github.com/jwkirchenbauer/lm-watermarking  # paper 04 — KGW reference

# Privacy / MIA
git clone https://github.com/pratyushmaini/llm_dataset_inference  # paper 02
git clone https://github.com/iamgroot42/mimir                     # MIA benchmark
git clone https://github.com/JonasGeiping/breaching               # gradient leakage
git clone https://github.com/sprintml/copyrighted_data_identification  # paper 09 CDI

# Model stealing / encoder
git clone https://github.com/stapaw/b4b-active-encoder-defense    # paper 11 B4B
git clone https://github.com/cleverhans-lab/model-extraction-iclr # paper 14 PoW

# Adversarial (jeśli się zmaterializuje)
git clone https://github.com/Harry24k/adversarial-attacks-pytorch  # torchattacks
git clone https://github.com/fra31/auto-attack                     # AutoAttack official
```

Setup ~30 min. Zysk: każde "jak działa attack X" answerable z README + main attack file zamiast ze szczegółowych paperów.
