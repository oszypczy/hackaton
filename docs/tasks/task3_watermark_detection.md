# Task 3: LLM Watermark Detection

> Źródło: **task PDF** (Downloads/Task3 LLM Watermark Detection.pdf, 3 strony, dev: Maitri Shah) + `docs/Info­_Session _Warsaw.pdf` slajdy 55-62
> Status: PDF ujawniony 12:00 2026-05-09 — ground truth. Presentation slides → secondary.
> Last updated: 2026-05-09 (post-PDF reveal)
>
> 🔥 **Krytyczna zmiana względem presentation:** PDF mówi "**robust and general detector capable of identifying the presence of watermarks regardless of the underlying embedding scheme**" + train ma 180 watermarked **"split evenly across watermark types"**. Czyli **WIELE typów watermarku** w datasecie, nie sam Kirchenbauer. Stricte Kirchenbauer-specific detector jest niewystarczający.

## Co trzeba zrobić (PDF, "Goal")

Zbudować **detektor watermarku** dla tekstu generowanego przez LLM. Dla każdego sample przypisać **ciągły score `[0, 1]`** — wyższy score = silniejsze przekonanie że sample jest watermarked.

- **Score ≥ 0.5** → watermarked (PDF jasno definiuje próg)
- **Score < 0.5** → clean

PDF, "Challenge": *"design a robust and general detector capable of identifying the presence of watermarks in the text **regardless of the underlying embedding scheme**"*. Czyli scheme jest **multi-type** i **nieujawniony** — black-box adversarial setup.

## Możliwe schematy watermarku (PDF, "References")

PDF cytuje **3 papery** sugerujące że dataset zawiera ≥3 typy watermarku:

1. **Kirchenbauer et al. 2023** — *"A Watermark for Large Language Models"* ICML 2023 — **Red-Green list** (klasyczny baseline)
2. **Liu et al. 2024** — *"A Semantic Invariant Robust Watermark for LLMs"* ICLR 2024 — semantic-invariant (odporny na paraphrasing, embedding nie token-id-based)
3. **Zhao et al. 2024** — *"Provable Robust Watermarking for AI-Generated Text"* ICLR 2024 — provable robustness (Unigram-Watermark)

Train: 180 watermark samples "split evenly across watermark types" → np. 60/60/60 jeśli 3 typy, lub 45×4 jeśli 4 typy. Walidacja sprawdzi.

## Mechanizm watermarku: Red-Green list (Kirchenbauer baseline — jeden z N typów)

**Algorytm generowania (Kirchenbauer Algorithm 1, "Hard Red List"):**
1. LLM zwraca probability vector `p^(t)` nad vocabulary
2. Zhashuj poprzedni token `s^(t-1)` jako seed RNG
3. Tym seedem dziel vocab na **green list G** i **red list R** (równe rozmiary)
4. Sample następny token `s^(t) ∈ G`, **nigdy** z R

**Detekcja:** test statystyczny — czy w tekście jest *istotnie więcej* zielonych tokenów niż przy przypadkowym wyborze (50/50)?

**Intuicja sygnału:**
- Clean text: ~50% green tokens (random)
- Watermarked text: ~70-100% green tokens (zależnie od delta) → z-score wysoki

## Dataset (PDF, "Dataset")

`Dataset.zip` zawiera 3 predefiniowane splity:

| Split | Total | Clean | Watermarked |
|---|---|---|---|
| **Train** | 360 | 180 | 180 (**split evenly across watermark types**) |
| **Validation** | 180 | 90 | 90 |
| **Test** | **2,250** | ? | ? |

- Train + Val mają labels → trenujemy/kalibrujemy detektor lokalnie
- Test split → nasze submission
- "split evenly across watermark types" w trainie → możemy **policzyć ile typów watermarku istnieje** patrząc na 180 / k samples per type

### Źródło danych
- HuggingFace: `https://huggingface.co/datasets/SprintML/llm-watermark-detection`
- `hackathon_setup.sh` tworzy team folder + downloads + per-task env

## Output / submission (PDF, "Submission Format")

Plik **musi** się nazywać `submission.csv` z formatem:

```csv
id,score
1,0.8321
2,0.0478
3,0.6214
...
```

2250 wierszy (jeden per test text).

### Wymagania (PDF "Requirements") — submission odrzucany jeśli nie spełnia:
- każdy test text pojawia się **dokładnie raz**
- nazwy kolumn **dokładnie** `id` i `score` (case-sensitive)
- score: numeric ∈ [0, 1]
- brak duplikatów / brakujących wierszy / invalid entries
- max 10 MB

### Submission tooling (PDF "Additional Resources")
- `generate_random_submission.py` — boilerplate: load dataset, generate continuous predictions, save submission
- `submission_template.py` — submisja do API (zamienić `YOUR_API_KEY_HERE` i `/PATH/FILE.csv`)

## Metric: TPR @ 1% FPR (PDF, "Evaluation Metric")

- **TPR** = TP / (TP + FN) — jak często wykrywamy watermarked
- **FPR** = FP / (FP + TN) — jak często falszywie alarmujemy na clean

**Procedura:** próg `t ∈ [0,1]` taki, że `FPR(t) = 0.01` (only 1% clean falsely flagged), raportuje TPR przy tym progu.

→ **wyżej = lepiej.** Idealny TPR = 1.0.

**Implikacja:** scores muszą być **kalibrowane jako ranking** — porządek matters, absolute value 0.5 mniej. Detector dający stabilne separujące scores wygrywa.

## Scoring split (PDF, "Scoring")
- **Public leaderboard:** 30% test texts (~675 samples)
- **Private leaderboard (final):** 70% (~1575 samples)
- "leaderboard shows the best result per team only" — score nie aktualizuje się jeśli niższy

## Bazowe metody (PDF "References" + repo)

PDF cytuje **3 papery** — sugerują **3 typy watermarku** w datasecie:

1. **Kirchenbauer et al. ICML 2023 — "A Watermark for LLMs"** (paper #04 wymagany) — Red-Green list, z-score detektor
2. **Liu et al. ICLR 2024 — "A Semantic Invariant Robust Watermark"** — semantic embedding-based zamiast token-id; odporny na paraphrasing
3. **Zhao et al. ICLR 2024 — "Provable Robust Watermarking" (Unigram-Watermark)** — provable robustness; uproszczona wersja Kirchenbauer (statyczna green list zamiast hash-based)

### Implementacja detekcji per scheme

- **Kirchenbauer (Red-Green hash-based):** odtwórz green list per token z hashu prev tokena → zlicz green hits → z-test. Wymaga znania hash function + γ.
- **Zhao (Unigram-Watermark):** **statyczna** green list (te same tokeny zawsze zielone) → łatwiejsza do uczenia black-box, bo nie potrzeba hash.
- **Liu (Semantic Invariant):** detekcja przez embedding similarity, nie token-level — **najtrudniejsze** dla naiwnego classifierra.

### Black-box features (działają na wszystkich typach)

- perplexity pod surrogate LLM (np. GPT-2, Llama-2, Mistral)
- token rank distribution
- entropy per-token (watermark zaniża entropy)
- n-gram diversity (watermark = mniejsza diversity)
- repetition rate
- zlib compression ratio

### Repo (potencjalnie pomocne)
- **WAVES** (paper #24) — image watermark adversarial robustness, dane intuicje o robust detection
- **Watermark Stealing** (paper #21) — odwrotny kierunek (atak), pokazuje co statystycznie odróżnia watermarked

## Strategia (high-level)

### Wariant A: per-scheme detectors + ensemble (rekomendowane)
PDF mocno sugeruje multi-type. Trenujemy **3 detektory** (po jednym na typ z papers):
1. Kirchenbauer detector (jeśli znamy hash → z-score; jeśli nie → token statistics)
2. Zhao Unigram detector (statyczna green list — daje się estymować na train)
3. Liu Semantic Invariant detector (embedding-based; np. cosine similarity prev/next token embedding)

Ensemble: max(detectors) lub LogisticRegression(detector_outputs).

### Wariant B: black-box klasyfikator (naiwny baseline)
- Trenuj logistic regression / mała sieć na train split (360)
- Features: perplexity (surrogate LLM), token rank distribution, entropy, n-gram diversity, repetition rate, zlib ratio
- Calibration na validation (180) → próg @ FPR=1%

### Wariant C: hybrid
- Bruteforce zgadywanie hashing scheme (SHA256(prev_token), seeded RNG, modulo) na train split → jeśli któryś daje >random separation, mamy gotowy detector
- Połącz z black-box features dla części unknown-scheme

### Krytyczna walidacja (na 180 val samples)
- TPR@1%FPR per detector → który scheme leakuje najwięcej?
- Score-by-watermark-type breakdown — sprawdzić czy któryś typ jest niewidzialny dla naszego detektora
- **Rozjazd train/val** = ryzyko regresji na private 70%

## Pułapki / open questions

- **Multi-type watermark:** detektor "Kirchenbauer-only" zfailuje na samples z innymi schemami → niedoszacowane scores na ~⅔ watermarked → niskie TPR
- **Liczba tokenów per sample:** krótkie próbki = słaby sygnał, długie = łatwe (z-score skaluje z √T). Sprawdzić długości w validation.
- **Tokenizer:** żeby liczyć zielone tokeny musimy znać tokenizer generatora. W black-box wariancie używamy własnego tokenizera surrogate'a — może rozjeżdżać.
- **Watermark strength (delta):** Kirchenbauer ma parametr siły. Niskie δ → black-box trudniej.
- **TPR@1%FPR jest brutalne:** bardzo czuły na ogon rozkładu clean. Na public 30% (~675 samples) — ~3 FP wystarczą żeby przebić 1% FPR.
- **5 min cooldown / 2 min na failed:** ograniczona ablacja na public, MUSIMY mieć stabilny lokalny benchmark
- **Train/val parametr:** clean i watermarked w train mogą pochodzić z par tych samych promptów → dataset construction matters

## Co dostarczają organizatorzy (potwierdzone PDF)

- `Dataset.zip` z 3 splitami (train 360 / val 180 / test 2250)
- Labels dla train + val
- `generate_random_submission.py` (boilerplate)
- `submission_template.py` (API submission)

## Co dostarczają organizatorzy (niepotwierdzone PDF)

- Source LLM (jaki model generuje? Llama, Mistral, GPT-2?) — PDF nic nie mówi → sprawdzić w danych
- Watermarking schemes — PDF cytuje 3 papery, ale konkretne hyperparams (γ, δ, hash) nie są ujawnione
- Czy train/val mają identyczny prompt prefix dla par clean/watermarked?

## TODO (po pobraniu danych)

- [ ] `wget https://huggingface.co/datasets/SprintML/llm-watermark-detection` (lub `bash hackathon_setup.sh`)
- [ ] Załadować train, **policzyć ile typów watermarku** (sprawdzić strukturę `watermark/` folder w dataset)
- [ ] Distribution analiza: token length, vocabulary, pewne style markers (czy 1 LLM czy wiele?)
- [ ] Implementacja per-scheme detektorów (Kirchenbauer, Zhao Unigram, Liu Semantic) — patrz papery referenced
- [ ] Black-box baseline: LogReg na perplexity + zlib + entropy
- [ ] **Walidacja per-type:** sprawdzić czy detector każdego scheme łapie WSZYSTKIE typy w val set
- [ ] Sanity: random score → TPR=1% (lower bound)
- [ ] Ensemble (max / weighted LogReg na outputach)
- [ ] Kalibracja progu na val 90+90 → public submission
