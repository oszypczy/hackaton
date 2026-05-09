# Task 3: LLM Watermark Detection

> Źródło: `docs/Info­_Session _Warsaw.pdf` slajdy 55-62 (data watermarking)
> Status: info-session only, brak konkretnego API, danych ani modeli (czekamy na 12:00 2026-05-09)
> Last updated: 2026-05-09

## Co trzeba zrobić

Zbudować **detektor watermarku** dla tekstu generowanego przez LLM. Dla każdego sample przypisać **ciągły score `[0,1]`**, gdzie:

- **Score ≥ 0.5** → tekst watermarked
- **Score < 0.5** → tekst clean

## Mechanizm watermarku: Red-Green list (slajd 57-58, Kirchenbauer et al. 2023)

**Algorytm generowania (Kirchenbauer Algorithm 1, "Hard Red List"):**
1. LLM zwraca probability vector `p^(t)` nad vocabulary
2. Zhashuj poprzedni token `s^(t-1)` jako seed RNG
3. Tym seedem dziel vocab na **green list G** i **red list R** (równe rozmiary)
4. Sample następny token `s^(t) ∈ G`, **nigdy** z R

**Detekcja:** test statystyczny — czy w tekście jest *istotnie więcej* zielonych tokenów niż przy przypadkowym wyborze (50/50)?

**Intuicja sygnału:**
- Clean text: ~50% green tokens (random)
- Watermarked text: ~70-100% green tokens (zależnie od delta) → z-score wysoki

## Dataset (slajd 62)

`dataset.zip` zawiera 3 splity:

| Split | Total | Clean | Watermarked |
|---|---|---|---|
| **Train** | 360 | 180 | 180 |
| **Validation** | 180 | 90 | 90 |
| **Test** | 2250 | ? (do oznaczenia) | ? (do oznaczenia) |

Train + Val mają labels → możemy trenować/kalibrować detektor lokalnie.
Test split → nasze submission.

## Output / submission (slajd 62)

CSV z formatem:
```
id, score
1, 0.8321
2, 0.0123
...
```

2250 wierszy.

## Metric: TPR @ 1% FPR (slajd 61)

- **TPR** = TP / (TP + FN) — true positive rate (jak często wykrywamy watermarked tekst)
- **FPR** = FP / (FP + TN) — false positive rate (jak często wskazujemy clean jako watermarked)

**Procedura:** wybiera się próg `t ∈ [0,1]` taki, że `FPR(t) = 0.01`, i raportuje TPR przy tym progu.

→ **wyżej = lepiej.** Idealny TPR = 1.0 (wszystkie watermarked wykryte przy ≤1% fałszywych alarmów na clean).

**Implikacja:** scores muszą być *kalibrowane jako ranking* — porządek matters bardziej niż absolute value 0.5. Detector dający stabilne separujące scores wygrywa.

## Bazowe metody (z required papers + repo)

- **Kirchenbauer et al. 2023 — A Watermark for LLMs** (paper #04 wymagany) — ma referencyjny detektor: liczyć z-score zielonych tokenów
- **Naiwny detektor (jeśli znamy hash):** dla każdego tokenu odtwórz green list z hashu prev tokena → zlicz green hits → z-test
- **Bez znajomości hash:** musimy uczyć detektor (klasyfikator) na train split — features: token-level statistics (entropy, perplexity, n-gram repetitions, zlib ratio, etc.)
- **WAVES** (paper #24) — image watermark adversarial robustness benchmark, daje intuicje
- **Watermark Stealing** (paper #21) — odwrotny kierunek (atak), ale pokazuje co statystycznie odróżnia watermarked

## Strategia (high-level)

### Wariant A: znamy LLM + hashing scheme (best case)
- Zaimplementować referencyjny Kirchenbauer detector (Algorithm 2 z papera #04)
- Liczyć z-score: `z = (|green_tokens| − γ·T) / sqrt(T·γ·(1−γ))` gdzie γ = 0.5
- Score = sigmoid(z) lub p-value-based

### Wariant B: nie znamy hash / scheme (likely)
- **Black-box klasyfikator:** trenuj logistic regression / małą sieć na train split
- **Features:**
  - perplexity pod surrogate LLM (np. GPT-2 / Llama-2)
  - token rank distribution
  - entropy distribution
  - n-gram diversity (watermarked = mniejsza diversity bo green list constraint)
  - repetition rate
  - zlib compression ratio
- **Ensemble:** uśrednić scores z różnych surrogates (Llama-2, Mistral, GPT-2)
- **Calibration:** użyć validation 90+90 do kalibracji progu na FPR=1%

### Wariant C: hybrid
- Bruteforce zgadywanie hashing scheme (kilku popularnych: SHA256(prev_token), seeded RNG z prev_token_id) na train split — jeśli któryś daje >random separation, mamy detector

## Pułapki / open questions

- **Liczba tokenów per sample:** krótkie próbki = słaby sygnał (wariant B), długie = łatwe (z-score skaluje z √T). Sprawdzić długości w validation.
- **Tokenizer:** żeby liczyć zielone tokeny musimy wiedzieć JAKIM tokenizerem text był generowany. W black-box wariancie używamy własnego tokenizera surrogate'a.
- **Watermark strength (delta):** Kirchenbauer ma parametr siły (`δ` = bias na green tokens). Jeśli δ jest niskie → black-box trudniej; jeśli wysokie → naiwne metody działają.
- **TPR@1%FPR jest brutalne:** bardzo czuły na ogon rozkładu clean. Nawet 5 fałszywych pozytywów na 90 clean = 5.5% FPR — to pewnie nie wystarczy.
- **Submission jako CSV** — score dla CAŁEGO test split, więc liczby submisji ograniczone (5 min cooldown).
- **Train/val ratio i potential leak:** czy clean i watermarked w train pochodzą z TEGO SAMEGO promptu? Jeśli tak, to training "z parami" daje przewagę nad treningiem na osobnych próbkach.

## Co dostarczają organizatorzy (potwierdzone)

- `dataset.zip` z 3 splitami (train/val/test)
- Labels dla train + val
- Sample submission CSV format

## Co dostarczają organizatorzy (niepotwierdzone)

- Source LLM (jaki model generuje? Llama, Mistral, GPT-2?)
- Watermarking scheme (Kirchenbauer hard? soft? hash function? γ, δ?)
- Czy train/val mają identyczny prompt prefix dla par clean/watermarked?

## TODO (po reveal 12:00)

- [ ] Pobrać `dataset.zip`, sprawdzić długości próbek (tokens count distribution)
- [ ] Sprawdzić czy clean i watermarked w train mają pary po prompcie
- [ ] Zaimplementować referencyjny Kirchenbauer z-score detector (jeśli mamy LLM)
- [ ] Black-box: trenuj klasyfikator na perplexity + n-gram features (LogReg)
- [ ] Walidacja: TPR@1%FPR na val set, kalibracja progu
- [ ] Sanity check: TPR przy random score = 1% (linia bazowa)
- [ ] Ensemble surrogate LLMs jeśli mamy compute
