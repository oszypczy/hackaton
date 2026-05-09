# Task 1: Data Usage Cardinality Inference (DUCI)

> Źródło: `docs/Info­_Session _Warsaw.pdf` slajdy 39-42 (data identification)
> Status: info-session only, brak konkretnego API, danych ani modeli (czekamy na 12:00 2026-05-09)
> Last updated: 2026-05-09

## Co trzeba zrobić

Dla każdego z **9 modeli** (klasyfikatorów ResNet) podać **ciągły score `[0,1]`** mówiący *ile* danego MIXED datasetu zostało użyte do treningu.

**Inne sformułowanie:** Klasyczne membership inference pyta "czy ten **datapoint** był w treningu?". DUCI pyta "**jaki ułamek** datasetu MIXED model widział?".

## Setup zadania (slajd 40-41)

### Modele (9 sztuk, dostarczone przez organizatorów)
- **ResNet18:** model IDs `00, 01, 02`
- **ResNet50:** model IDs `10, 11, 12`
- **ResNet152:** model IDs `20, 21, 22`

Wszystkie trenowane na **MIXED Dataset**, ale każdy z innym poziomem "użycia".

### Datasety (oba na bazie CIFAR100)
- **MIXED Dataset:** 2000 obrazów × 100 klas — to jest "podejrzany" dataset, którego ułamek był użyty
- **POPULATION Dataset:** 10000 obrazów × 100 klas — *z tego samego rozkładu* (CIFAR100), ale niewidziany przez model. Punkt odniesienia / non-member set.

**Założenie:** MIXED i POPULATION są IID (założenie z paper `Maini et al. NeurIPS'24`).

## Output / submission

Dla każdego z 9 modeli zwracamy jedną liczbę `[0,1]`. Przykład ze slajdu 40:

```
Model ID:    00    01    02    10    11    12    20    21    22
DUCI Score: 0.1  0.87 0.213 0.21 0.17 0.913 0.71 0.314 0.113
```

**Format submission:** CSV (jeszcze niepotwierdzony, prawdopodobnie dwie kolumny `model_id, score`).

## Metric (slajd 42)

```
MAE_i    = |GroundTruth_i − Estimation_i|             # per model
MeanMAE  = (1/N) Σ MAE_i                              # final score, N=9
```

→ **niżej = lepiej.** Idealny score = 0.0.

## Bazowe metody (z required papers — patrz `references/papers/MAPPING_INDEX.md`)

- **Maini et al. NeurIPS'24 — LLM Dataset Inference** (slajd 31-37 to dokładnie ten paper):
  1. Stage 1: Suspect Set ↔ Held-out Set, każdy split A/B
  2. Stage 2: Extract Membership Features (per split A): suspect_features_A, heldout_features_A
  3. Stage 3: Train scoring model na (suspect_A: label 0, heldout_A: label 1)
  4. Stage 4: Statistical Test na split B → różne scores = Member, podobne scores = Non-Member
- **Min-K%++** (paper #20) — pojedyncze membership inference features
- **zlib ratio** — sprawdziło się w Challenge A (patrz memory `project_challenge_a_insights.md`)

## Pułapki / open questions

- **Skala scores:** czy 0.0 = "0% MIXED" a 1.0 = "100% MIXED"? Czy to fraction użytych obrazów? Slajd mówi "How much was used? Give a continuous score [0,1]" — interpretacja jako ułamek wydaje się oczywista, **ale potwierdzić w PDF taska 12:00**.
- **Domain shift:** Challenge A pokazał, że minkpp można ODWRÓCIĆ przy domain shift między member/non-member. Tu jednak deklarowane IID (MIXED i POPULATION oba z CIFAR100) — więc raczej nie powinno być domain shift, ale weryfikować na walidacji.
- **Architektury:** ResNet18 vs 50 vs 152 mogą wymagać różnych progów / kalibracji metryki. 3 modele per architektura → można policzyć wariancję.
- **POPULATION jako referencja** dostarczona przez organizatorów — czyli mamy non-members "za darmo". W przeciwieństwie do typowego MIA, nie musimy ich zgadywać.
- **No watermarking, no LLM** — to klasyczny vision MIA na klasyfikatorze.

## Co dostarczają organizatorzy (potwierdzone)

- 9 wytrenowanych checkpointów ResNet
- MIXED Dataset (2000 obrazów × 100 klas)
- POPULATION Dataset (10000 obrazów × 100 klas)

## Co dostarczają organizatorzy (niepotwierdzone, prawdopodobne)

- PDF z dokładną specyfikacją metryki + endpoint API
- Format submission CSV
- Loader do checkpointów (PyTorch state_dict?)
- Validation set z ground truth scores → możemy testować lokalnie

## TODO (po reveal 12:00)

- [ ] Pobrać checkpointy + datasety z HuggingFace / Jülich
- [ ] Sprawdzić architekturę ResNetów (CIFAR-100-style czy ImageNet-style head?)
- [ ] Zaimplementować baseline DI (Maini) z minkpp + zlib jako features
- [ ] Czy ground truth scores na walidacji są ujawnione? Jeśli tak → kalibracja per-model
- [ ] Walidować scoring model na nieznanych modelach (cross-architecture)
