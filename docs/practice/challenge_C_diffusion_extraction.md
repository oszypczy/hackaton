# Challenge C — Diffusion Memorization Discovery

**Paper:** Carlini, Hayes, Nasr, Jagielski, Sehwag, Tramèr, Balle, Ippolito, Wallace — *Extracting Training Data from Diffusion Models* (USENIX Security 2023)
**Repo (oryginalny attack):** https://github.com/ethz-spylab/diffusion_extraction (sprawdzić — może być zmiana adresu)
**Sugerowana osoba:** adversarial expert / generalista
**Czas:** 6–8h
**Typowa rola na hackathonie:** training data extraction, memorization detection

## Problem

Dostajesz pre-trained DDPM checkpoint (CIFAR-10) który **został celowo doprowadzony do memoryzacji 50 obrazów** (te obrazy były zduplikowane wielokrotnie w training secie). Plus 1000 obrazów testowych.

**Zadanie:** zranguj 1000 testowych obrazów po prawdopodobieństwie że są zmemoryzowane (czyli były w training set jako duplikaty).

To replikuje **generate-and-filter pipeline** z paperu Carliniego, ale w mini skali wykonalnej na M4.

## Setup

### Dlaczego tak małe i nie Stable Diffusion

Carlini użył full SDv1.4 + 350M training set. **My nie mamy ani GPU ani czasu**. Skalujemy w dół:

- **DDPM 32×32 na CIFAR-10**: ~36M params, sample na M4 trwa ~10–20min dla 5000 obrazów
- **50 zduplikowanych obrazów** w training set: silne memorization, łatwo złapać sygnał
- **1000 candidate images**: 50 prawdziwych duplikatów + 950 distractorów (nieduplikowane CIFAR-10 train + CIFAR-10 test)

### Fixture data (przygotowane wcześniej, nie na M4)

Dystrybuowane jako paczka ~200MB:

```
data/C/
  ddpm_cifar10_memorized.pt       # checkpoint, ~150MB
  candidates/                      # 1000 obrazów PNG 32x32
    img_0000.png ... img_0999.png
  candidates_meta.jsonl            # {"id": int, "filename": str} — bez ground truth
  ground_truth.jsonl               # ENCRYPTED (decrypt po submisji)
```

**Pre-training na Jülich** (raz, wcześniej, przeze mnie):
1. Pobierz CIFAR-10 train (50k obrazów)
2. Wybierz 50 obrazów, zduplikuj każdy ×100 → 5000 dodatkowych próbek
3. Train DDPM od zera (lub fine-tune `google/ddpm-cifar10-32`) przez ~50k kroków, lr=1e-4
4. Save checkpoint
5. Z 1000 candidates: 50 to memorized duplicates, 500 to inne CIFAR-10 train (nie-zduplikowane), 450 to CIFAR-10 test (nigdy nie widziane)

### Jeśli ja nie zdążę zbudować fixture

**Plan B**: Użyj public DDPM (`google/ddpm-cifar10-32`) bez memoryzacji + sztucznie dodaj memoryzację przez **dreambooth-style fine-tune** na 50 obrazach przez 1000 kroków na M4 (~30 min). Mniej realistic ale działa lokalnie.

## Zadanie + format submission

`submissions/C.jsonl`:
```json
{"id": 0, "memorization_score": 0.91, "rank": 1}
{"id": 1, "memorization_score": 0.88, "rank": 2}
...
```

Ranking od najbardziej prawdopodobnie zmemoryzowanego do najmniej.

## Scoring

**nDCG@50** — jak wysoko prawdziwe duplikaty są w Twoim rankingu.

Plus:
- **Recall@50** (z 50 prawdziwych ile masz w top 50): bonus dla ścisłej odpowiedzi
- **Recall@100**: jak ktoś nie złapał wszystkich top, sprawdź szerszy stride

**Reference scores:**
- Easy (pixel L2 nearest neighbor): nDCG@50 ≈ 0.40–0.55
- Solid (CLIP embedding + sample-and-match): nDCG@50 ≈ 0.70–0.85
- Hard (Carlini full pipeline z denoising trajectory): nDCG@50 > 0.90

## Baselines

### Easy (1–2h)
**Pixel-space nearest neighbor**:
1. Sample 5000 obrazów z DDPM (na M4 ~15min używając MPS)
2. Dla każdego z 1000 candidates: znajdź najbliższy sample po L2 distance pixel-wise
3. Score = 1 / (1 + min_L2_distance) — niska distance = wysoka memorization

```
from diffusers import DDPMPipeline
import torch
pipe = DDPMPipeline.from_pretrained("./data/C/ddpm_cifar10_memorized")
pipe.to("mps")
samples = []
for batch in range(50):  # 50 batchów × 100 = 5000
    out = pipe(batch_size=100, num_inference_steps=50)
    samples.extend(out.images)
```

### Solid (4h)
**CLIP-embedding similarity + DBSCAN clustering** (replikacja Carlini 2023, research 02 § 6):
1. Sample 10000 obrazów z DDPM
2. Encode wszystkie samples + 1000 candidates przez **DINOv2 ViT-S/14** lub `open_clip` ViT-B-32 (na MPS)
3. **DBSCAN clustering**: `DBSCAN(eps=0.10, min_samples=10, metric='precomputed')` na cosine distance — connected component ≥10 z diameter <0.15 = memorized prompt (Carlini Stage 3)
4. Per candidate: top-k nearest samples po cosine similarity. Score = średnia top-3 similarity.
5. Normalizuj: zmemoryzowane obrazy mają **dużo bardzo podobnych samples** w embedding space

**Dlaczego CLIP/DINO a nie pixels**: DDPM samples są noisy. Pixel L2 łapie tylko silne, perfect-match memoryzacje. CLIP/DINO łapie semantic memoryzację (ten sam obraz, lekko zniekształcony).

**Webster 2023 fast variant** (research 02): zamiast 50-step DDIM sampling, użyj **single denoising step** — wystarczy do flagowania template verbatims, orders-of-magnitude szybsze. Repo: `ryanwebster90/onestep-extraction`.

### Hard (jak masz czas — replikacja Carliniego)
**Multi-prompt generation + likelihood scoring**:
1. Dla każdego candidate, **conditioned na noisy version candidate**, generuj 100 samples z różnymi seedami
2. Carlini's metric: log-prob że candidate został zsamplowany. Aproksymowana przez:
   - Variance gradient kroku denoising (memorized images mają niższą variance — model "wie" co generować)
   - Reconstruction error przy partial diffusion (zaszumiamy 50%, denoise, sprawdzamy czy odzyskuje oryginał)
3. **Membership inference jako sub-component**: trenuj logistic regression na ~10 features: pixel-NN distance, CLIP-NN, denoising loss, sample count match

### Hard mode alternative track — CDI replikacja (high-probability hackathon target)

Carlini 2023 jest na liście obowiązkowej, ale **prawdziwy SOTA SprintML to CDI** (Dubiński, Kowalczuk, Boenisch, Dziedzic — *Copyrighted Data Identification in Diffusion Models*, **CVPR 2025**). To jest paper Boenisch+Dziedzic, dosłownie ich kierunek badań.

CDI vs Carlini:
- Carlini: generate-and-filter, wymaga sampling tysięcy obrazów
- **CDI: ≥99% confidence z tylko ~70 próbek**, agregacja per-sample MIA signals przez statystyczny hypothesis test
- CDI zwraca p-value, nie tylko ranking → **zero false positives na clean datasets**
- CDI jest aplikowalny na realne SDv1.5/SDXL, nie tylko CIFAR-10

Pipeline:
1. Wybierz subset 70 candidate images
2. Per-sample MIA features: denoising loss przy różnych t (timesteps), reconstruction error, gradient norm
3. Aggregate: t-test Welcha między rozkładem features na candidate set vs reference set z held-out images
4. Output: p-value że ten subset był w training data

Reference scores:
- Solid CDI implementacja: p < 0.05 z 70 próbek na zmemoryzowanym subsecie, p > 0.5 na clean subsecie
- Hard target: TPR@1%FPR > 90% na document-level

Plus warto przeczytać:
- **Privacy Attacks on IARs (Kowalczuk, Dubiński, Boenisch, Dziedzic, ICML 2025)** — extracts 698 training images z VAR-d30, TPR@FPR=1% = 86.38%
- **NeMo (Hintersdorf, Struppek, Kersting, Dziedzic, Boenisch, NeurIPS 2024)** — localization memoryzacji do pojedynczych cross-attention neuronów

To są dwa najświeższe SprintML papery o memoryzacji w generative models. **Bardzo wysoka szansa że Warsaw 2026 challenge oparty będzie na jednym z nich.**

## Pułapki

- **Sampling prędkość** — `num_inference_steps=50` to `~3s` per obraz na M4. 5000 obrazów = ~4h. Użyj `DDIMScheduler` z 50 krokami zamiast DDPM 1000 — to OK dla CIFAR-10.
- **MPS bugs w diffusers** — niektóre operacje (np. group_norm w fp16) mają issues na MPS. Loaduj fp32. Jeśli crash, fallback na CPU dla problematycznych warstw.
- **Embedding cache** — CLIP encode 11k obrazów to ~2min na M4. Cache na dysk (`*.npy`).
- **Distance baseline trap** — jeśli distractory to CIFAR-10 test (nigdy nie widziane), one też mają niskie pixel L2 do losowych samples bo CIFAR-10 jest mało zróżnicowane. Trzeba normalizować względem **referencyjnego rozkładu odległości** (np. distance między nieskoreloowanymi samples).
- **Diffusion path memorization** — prawdziwe Carlini paper exploituje że memorized images mają "deterministyczną" denoising trajectory. To trudne do replikacji w 8h, ale daje znaczący boost.

## Co to ćwiczy pod hackathon

- Generate-and-filter pipeline (Carlini's signature method)
- Embedding similarity jako proxy dla memorization
- Praca z `diffusers` na MPS — niespodzianki tutaj są częste, lepiej je już znać
- Łączenie kilku słabych sygnałów w ranking — to się powtarza we wszystkich privacy attacks
- Trade-off między query budget (ile sample) a accuracy

## Open question

Jeśli okaże się że na M4 nawet to za wolne (sampling 5000 obrazów > 30min), **mamy backup plan**: pre-generowane samples (paczka ~50MB) dystrybuowane razem z fixturem. Wtedy challenge to czysto **filter** stage Carliniego, bez generate. Lekka utrata realismu, ale cały sens (membership inference + similarity ranking) zostaje.
