# Task 1: Dataset Usage Cardinality Inference (DUCI)

> Źródło: **task PDF** (Downloads/task1.pdf, 3 strony, dev: Nima DindarSafa) + `docs/Info­_Session _Warsaw.pdf` slajdy 39-42
> Status: PDF ujawniony 12:00 2026-05-09 — ground truth. Presentation slides → secondary.
> Last updated: 2026-05-09 (post-PDF reveal)

## Co trzeba zrobić

Dla każdego z **9 modeli** (klasyfikatorów ResNet) podać **ciągły score `[0,1]`** mówiący *ile* danego MIXED datasetu zostało użyte do treningu.

**Inne sformułowanie:** Klasyczne membership inference pyta "czy ten **datapoint** był w treningu?". DUCI pyta "**jaki ułamek** datasetu MIXED model widział?".

## Setup zadania (PDF)

### Modele (9 sztuk, dostarczone przez organizatorów — potwierdzone PDF)
- **ResNet18:** model IDs `00, 01, 02` (id starts with `0`)
- **ResNet50:** model IDs `10, 11, 12` (id starts with `1`)
- **ResNet152:** model IDs `20, 21, 22` (rest)

### Mechanika treningu (PDF, sekcja "Challenge")
> "For each model, training uses a **different sampled fraction** of the mixed dataset, and the remaining training examples are **filled in with data from a held-out set** so that **every model is trained on the same total number of examples** and with **balanced class proportions**."

Czyli: każdy model ma identyczny `N` total + balanced classes; **jedyna zmienna to ułamek MIXED vs held-out fill-in**. To jest kluczowa konstrukcja zadania — proporcja MIXED jest "ukryta" w pełnym treningu.

### Datasety (PDF, sekcja "Dataset")
- **Mixed set** (target inference): two `.npy` files — `X.npy` (images) + `Y.npy` (labels). Rozmiar: **niepotwierdzony przez PDF** (presentation sugerowała 2000×100 klas — sprawdzić po pobraniu).
- **Population set:** "extra in-distribution dataset, sampled from the same dataset" — modele go **NIE WIDZIAŁY**. Służy jako referencja non-member.

**Uwaga:** PDF nie wymienia explicite CIFAR100 — to było tylko z presentation. Po pobraniu z HF zweryfikować rozmiar obrazów + liczbę klas.

### Źródło danych
- **HuggingFace:** `https://huggingface.co/datasets/SprintML/DUCI`
- Albo: `hackathon_setup.sh` dostarczone przez organizatorów (tworzy team folder + env per task)

## Output / submission (PDF, sekcja "Submission Format")

Plik **musi** się nazywać `submission.csv` z dokładnie tym formatem:

```csv
model_id,proportion
00,0.123
01,0.456
02,0.789
10,0.012
11,0.345
12,0.679
20,0.012
21,0.345
22,0.678
```

### Wymagania (PDF, "Requirements") — submission odrzucany jeśli nie spełnia:
- każdy `model_id` pojawia się **dokładnie raz** (9 wierszy)
- nazwy kolumn **dokładnie** `model_id` i `proportion` (case-sensitive!) — **uwaga: nazwa kolumny to `proportion`, NIE `score`**
- wartości continuous (real) ∈ [0, 1]
- brak duplikatów / brakujących wierszy / invalid entries
- max 10 MB

### Jak submitować (PDF, "Additional Resources")
- Dostarczony plik `task_template.py`
- Zamienić `YOUR_API_KEY_HERE` (zachować cudzysłowy) i `/PATH/FILE.csv`

## Metric: MAE (PDF, "Evaluation Metric")

```
MAE = (1/n) Σᵢ |GroundTruth_i − Estimation_i|
```

→ uśredniane po wszystkich modelach w danym splicie. **Niżej = lepiej.** Idealny = 0.0.

## Scoring split (PDF, "Scoring") — WAŻNE dla strategii
- **Public leaderboard:** consistent, randomly selected **3 modele** (z 9)
- **Private leaderboard (final ranking):** pozostałe **6 modeli**

→ **public 3/9 = wąski sygnał** — łatwo overfittować pod public 3. Patrz CLAUDE.md "Working principles" 🎯 — generalizacja > public score.

## Leaderboard
- URL: `http://35.192.205.84/leaderboard_page`
- Pokazuje **best result per team only** — score updateuje się tylko jeśli niższy niż zapisany.

## Bazowe metody (PDF, "Reference" — 4 papery cytowane przez organizatorów)

PDF cytuje **dokładnie 4 papery** — w tej kolejności:

1. **"How much of my dataset did you use? Quantitative data usage inference in machine learning"** ICLR 2025 — `https://openreview.net/pdf?id=EUSkm2sVJ6`
   → **THE paper o tym tasku.** Tytuł 1:1 z tematem ("how much"). Jeśli mamy go w `references/papers/`, to jest naszym primary playbook. Jeśli nie — pobrać natychmiast.
2. **"Dataset Inference: Ownership Resolution in ML"** (Maini et al.) ICLR 2021 — `arxiv.org/abs/2104.10706` (oryginalna DI; binary not cardinality)
3. **"Dataset Inference for Self-Supervised Models"** NeurIPS 2022 — `openreview.net/pdf?id=CCBJf9xJo2X`
4. **"LLM Dataset Inference: Did you train on my dataset?"** NeurIPS 2024 — `openreview.net/pdf?id=Fr9d1UMc37` (paper #02 w naszym repo)

### Implementacja (z paperów + Challenge A insights)

- **Pipeline DI (Maini)** — adaptacja do quantitative:
  1. Suspect Set = MIXED, Held-out = POPULATION; podział A/B
  2. Extract Membership Features (split A) → suspect_features_A, heldout_features_A
  3. Train scoring model: (suspect_A: label 0, heldout_A: label 1)
  4. **Quantitative twist:** zamiast binary p-value, frakcja datapointów MIXED który scoring model klasyfikuje jako member ≈ proporcja użytego MIXED
- **Min-K%++** (paper #20) — pojedyncze membership features
- **zlib ratio** — w Challenge A wygrał nad MinK++ (patrz memory `project_challenge_a_insights.md`); image-domain analog: gradient norm, loss, confidence, augmentation invariance

## Pułapki / open questions

- **Skala `proportion`:** PDF mówi "fraction of the mixed dataset that was used for training" — czyli 0.0 = "0% MIXED" / 1.0 = "100% MIXED". **Potwierdzone PDF**.
- **Held-out fill-in:** każdy model widział `(p · MIXED) + ((1-p) · held_out)` — łączna liczba przykładów stała. Held-out (≠ Population!) to wewnętrzny held-out organizatorów. **My nie mamy do niego dostępu.**
- **Population ≠ held-out:** PDF jest precyzyjny — Population to *odrębny in-distribution dataset niewidziany przez modele*, dla nas jako referencja. Held-out fill-in to coś innego, używane wewnętrznie do treningu.
- **Domain shift:** Challenge A pokazał, że minkpp można ODWRÓCIĆ przy domain shift. Tu PDF nie potwierdza CIFAR100, ale Population deklarowany jako "from the same dataset" — IID. Weryfikować empirycznie po pobraniu.
- **Architektury:** ResNet18 vs 50 vs 152 — 3 modele per arch. Test cross-arch generalizacji to MUST.
- **Public 3 / Private 6:** loss leakage — jeśli kalibrujemy progi pod 3 public modele, ryzyko regresji na 6 private.
- **No watermarking, no LLM** — klasyczny vision MIA, ale extended z binary→continuous.

## Co dostarczają organizatorzy (potwierdzone PDF)

- 9 wytrenowanych checkpointów ResNet (3× ResNet18, 3× ResNet50, 3× ResNet152)
- Mixed set (`X.npy` + `Y.npy`)
- Population set (in-distribution, niewidziany przez modele)
- `task_template.py` (submission boilerplate)
- `hackathon_setup.sh` (tworzy team folder + downloads + per-task env)

## TODO (po pobraniu danych)

- [ ] `git clone https://huggingface.co/datasets/SprintML/DUCI` lub `bash hackathon_setup.sh`
- [ ] Sprawdzić rozmiar Mixed set (X.npy shape) + liczbę klas — confirmować/zfalsifikować CIFAR100 hipotezę
- [ ] Sprawdzić architekturę ResNetów w checkpointach (input size? num_classes? CIFAR-style 3×3 conv1 czy ImageNet 7×7?)
- [ ] Zaimplementować baseline DI (Maini) z gradient/loss/confidence + Min-K%++ + zlib jako features
- [ ] Bridge binary → continuous: dla każdego modelu, frakcja Mixed datapoints klasyfikowanych jako member
- [ ] Walidować scoring model **cross-architecture** (np. trenuj na ResNet18, testuj na ResNet50/152) — generalizacja krytyczna bo Private 6 może mieć inny mix
- [ ] Pobrać paper #1 z PDF references ("How much of my dataset did you use?" ICLR 2025) jeśli nie mamy
