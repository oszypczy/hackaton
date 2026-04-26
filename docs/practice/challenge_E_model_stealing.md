# Challenge E — Model Stealing / Encoder Extraction (OPTIONAL)

**Source:** research `04_model_stealing.md` + SprintML portfolio (B4B, ADAGE, GNN extraction, encoder stealing)
**Sugerowana osoba:** każdy kto chce drugą rundę
**Czas:** 6–8h
**Status:** OPCJONALNY — wysoka szansa pojawienia się na hackathonie mimo że nie ma w paperach obowiązkowych

## Po co to

Wcześniej w sesji uznałem że "model stealing nie jest w paperach obowiązkowych więc nie będzie". **Research 04 to obala**: SprintML lab ma cały portfolio na ten temat (B4B NeurIPS 2023, encoder stealing ICML 2022, GNN extraction AAAI 2026 Oral, ADAGE 2025, Calibrated PoW ICLR 2022 Spotlight). W wywiadzie Dziedzic eksplicite mówi że "tasks are designed from our research". Encoder/GNN stealing pojawia się też w opisie poprzednich edycji.

## Setup

### Wariant: image classifier extraction (najprostszy)

- **Victim model:** ResNet-50 fine-tuned na CIFAR-100 (fixture przez kolegę-CUDA, ~1h)
- **API**: zwraca top-K=5 softmax probabilities przez lokalny FastAPI endpoint (rate-limited 5 qps)
- **Budget:** 10 000 queries
- **Cel:** wytrenuj surrogate który ma **fidelity > 0.80** na hidden test set (CIFAR-100 test split, 10k obrazów)
- **M4 OK** — surrogate to ResNet-18, training na MPS ~30 min/epoch, 10 epok wystarczy

### Wariant: encoder extraction (B4B-style, harder)

- **Victim:** SimCLR ResNet-50 (publicznie dostępny `facebook/SimCLR-rn50` lub HF equivalent)
- **API:** zwraca 2048-dim embedding per query, **z włączoną defensą B4B** (per-user transformations)
- **Cel:** wytrenuj surrogate encoder który ma **downstream linear probe accuracy > 0.85× victim** na CIFAR-10
- **M4:** pre-cached embeddings = OK; surrogate training możliwe

## Fixture data (~2h CUDA)

Wariant 1 (CIFAR-100):
1. Fine-tune ResNet-50 z `pytorch/vision` na CIFAR-100, 50 epok
2. Save checkpoint + serving wrapper z rate limit
3. Hidden test set = stratified split CIFAR-100 test

Wariant 2 (encoder):
1. SimCLR ResNet-50 z HF
2. B4B defensive wrapper (kod open-source: github.com/adam-dziedzic/B4B)
3. Public pool: ImageNet val (50k obrazów) jako "thief data"

## Zadanie + format submission

Submission = state_dict surrogate modelu w formacie safetensors.

Plus log queries: timestamp, hash, returned vector.

## Scoring (SprintML eval style)

1. **Fidelity** (60%) — `Pr_x[argmax surrogate(x) = argmax victim(x)]` na hidden test set. **Primary metric.**
2. **Adversarial transferability** (20%) — generuj FGSM adv examples na surrogate, mierz transfer rate na victim. Dobry surrogate ma >70% transfer.
3. **Query budget bonus** (15%) — `(budget - queries_used) / budget`, jeśli fidelity > threshold
4. **Task accuracy** (5%) — secondary, low weight bo fidelity ≠ accuracy

**Reference scores (CIFAR-100 wariant):**
- Easy (Knockoff random + KD): fidelity ≈ 0.55–0.65 z 10k queries
- Solid (Margin sampling + KCenter + KD + EMA): fidelity ≈ 0.75–0.82
- Hard (BADGE-style + ensemble + warm-start ImageNet): fidelity > 0.85

## Baselines

### Easy (2h)
**Knockoff random + KD**:
1. Sample 10k obrazów z ImageNet val (public pool)
2. Query victim API, otrzymaj soft labels
3. Train ResNet-18 ImageNet-pretrained z KD loss `T² · KL(softmax(z_T/T) ‖ softmax(z_S/T)) + α·CE`
4. T=4, α=0.1, AdamW lr=3e-4, 10 epok, EMA decay=0.999

### Solid (4h)
**Active learning extraction**:
1. Pierwsze 1000 queries random
2. Każdy następny batch 500: `MarginStrategy` (uncertainty) ∪ `KCenterStrategy` (diversity) na surrogate features
3. Retrain z każdym batch, warm-start z previous checkpoint
4. **Pre-cache wszystkie queries na dysku** (SQLite) — architecture changes nie burnują budgetu
5. **Zawsze submit EMA model**, nie live

### Hard
- **BADGE** (Ash et al. ICLR 2020): gradient embedding k-MEANS++. Uncertainty + diversity w jednym, hyperparameter-free.
- **CutMix na soft labels** + AMP fp16 + `torch.compile`
- **Ensemble averaging** 3 best surrogates → +1–2 fidelity points
- **Diffusion-based query synthesis** (Stealix ICML 2025): generuj queries z BigGAN/SD zamiast tylko ImageNet

### B4B-bypass (encoder wariant only, hard)
- **Sybil orchestration**: 5 fake "users" robi queries z różnymi seedami, surrogate aggreguje. B4B per-user transformations average out.
- **Embedding-space coverage minimization**: queries muszą pokryć cały latent space SimCLR — używaj k-center na CLIP features.

## SprintML submission style — krytyczne

To są reguły z paperu Carlini ICML 2024 + całego SprintML portfolio:

1. **Optimize fidelity, not task accuracy.** Submit surrogate który matches victim's errors.
2. **Pre-cache every query to SQLite.** Architecture iteration ≠ query burn.
3. **EMA model > live model.** Always.
4. **Submit early and often.** Leaderboard = diagnostyka. First submission within hour 3.
5. **Don't experiment in last hour.** Pick best EMA checkpoint, sanity check, submit, stop.

## Time-wasting traps (z research 04)

- ❌ **DFME-style joint generator training** — wymaga miliardów queries, nie zmieścisz w 10k
- ❌ **GAN-from-scratch query synthesis** — cold start
- ❌ **Hyperparameter sweep** — AdamW 3e-4, T=4, cosine, EMA 0.999 jest OK
- ❌ **AutoAttack na transfer** — FGSM ε=8/255 wystarczy
- ❌ **Confidence calibration** — useless dla fidelity score

## M4 vs CUDA

CIFAR-100 ResNet-18 na MPS to ~30 min/epoch. Realistic na M4. Ale **fixture wymaga CUDA** (fine-tune ResNet-50 + serving). Encoder wariant: surrogate training na M4 OK, fixture ResNet-50 SimCLR pre-trenowany (HF download), B4B wrapper tylko inference, CUDA niewymagany jeśli batch size mały.

## Recommended reading

W kolejności:
1. **Tramèr et al. USENIX 2016** — equation solving, theory backbone
2. **Orekondy et al. CVPR 2019** — Knockoff Nets, canonical recipe
3. **Pal et al. AAAI 2020** — ActiveThief, active learning
4. **Carlini et al. ICML 2024** — logit-bias SVD na production LLMs (najświeższy paradigm shift)
5. **Liu et al. CCS 2022** — StolenEncoder
6. **Dubiński, Pawlak, Boenisch, Trzciński, Dziedzic — B4B (NeurIPS 2023)** — defense, kod: github.com/adam-dziedzic/B4B
7. **Podhajski, Dubiński, Boenisch, Dziedzic et al. AAAI 2026 Oral** — GNN extraction, hard label, restricted budget
