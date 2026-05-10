# Task 3 — głębokie wejście w rozwiązanie `blend_kgwx_o7be` / `BEST3`

Ten dokument uzupełnia krótką wersję: [task3_prezentacja_BEST3.md](task3_prezentacja_BEST3.md). Celem jest **pełne zrozumienie**, co dokładnie liczy kod, w jakiej kolejności, i jak to się składa do końcowego CSV.

---

## 1. Co jest „wynikiem” i skąd się bierze

Końcowy plik (`task3_watermark_blend_kgwx_o7be_BEST3.csv` itd.) to **2250 wierszy** `id, score`, gdzie `score` jest w `(0,1)` (technicznie `[0.001, 0.999]` po klipach).

Powstaje w **trzech fazach**:

1. **Dwa niezależne modele scoringu** — każdy to ten sam szkielet `extract_and_train.py`, ale z innym modułem „nowych” cech: `olmo7b_entropy` albo `kgw_exact`. Wynik: dwa CSV (np. `submission_clm_o7be.csv`, `submission_clm_kgwx.csv`).
2. **`blend_csvs.py`** — łączy oba CSV w jeden wektor `score` trybem **`rank`** (średnia ważona rang + normalizacja liniowa).

Żaden z tych kroków nie robi osobnej kalibracji pod próg 0.5 — organizator i tak optymalizuje **TPR przy FPR = 1%**, czyli w praktyce **kolejność** próbek względem clean vs watermarked.

---

## 2. Dane wejściowe (`extract_and_train.py`)

### 2.1 Pliki

Z `--data-dir` (np. `llm-watermark-detection/`) czytane są:

| Plik | Zawartość |
|------|-----------|
| `train_clean.jsonl`, `train_wm.jsonl` | Teksty treningowe, etykiety 0/1 |
| `valid_clean.jsonl`, `valid_wm.jsonl` | Walidacja |
| `test.jsonl` | 2250 tekstów bez etykiety; kolumna `id` lub domyślnie `1..N` |

**Łącznie zlabelowane:** 360 train + 180 val = **540** próbek z etykietą.

### 2.2 Kolejność przetwarzania tekstów

Wszystkie teksty do feature extraction są jedną listą:

```text
[ teksty z (train∪val) w kolejności concat ], potem [ teksty testu ].
```

Indeks `0 .. n_lab-1` odpowiada zlabelowanym, `n_lab ..` — testowi. Macierz `X` jest cięta tak: `X_lab = X[:n_lab]`, `X_test = X[n_lab:]`.

### 2.3 Cache cech

- Nowy moduł (`--feature`) może być **wyciągnięty** z tekstu i zapisany do `cache-dir/features_<nazwa>.pkl` (chyba że `--no-cache`).
- „Stare” cechy baseline są **tylko ładowane z cache** — jeśli pliku braku, ta gałąź baseline jest pomijana (`None`).

Dzięki temu na klastrze nie trzeba za każdym razem liczyć np. OLMo-7B PPL dla całego baseline; ważne jest, żeby **ten sam `cache-dir`** był używany przy obu wariantach (o7be i kgwx), inaczej kolumny baseline mogłyby być niespójne między dwoma submissionami (w praktyce team trzyma jeden współdzielony cache).

---

## 3. Baseline: jakie kolumny wchodzą do `full` przed „derived”

W kodzie (`extract_and_train.py`, ok. linii 119–131) ładowane są następujące pakiety (jeśli istnieją pliki `features_<nazwa>.pkl`):

| Nazwa cache | Typowy moduł / rola (intuicja) |
|-------------|--------------------------------|
| `a` | Cechy „branch A” (np. ogólne statystyki tekstu / LM) |
| `bc` | Branch „bigram/Unigram greenlist” — **w dokumentacji zespołu podejrzewany o data leak**; na środowisku z pełnym cache może nadal być obecny |
| `d` | Branch D |
| `bino`, `bino_strong`, `bino_xl` | Binoculars / Pythia / obserwowane vs permutowane log-proby |
| `olmo_7b` | OLMo-7B: m.in. średnie log-proby, PPL po tekście |
| `multi_lm` | OLMo-1B + partnerzy — m.in. `olmo_lp_mean`, `lp_per` (GPT-2 medium) |
| `fdgpt` | DetectGPT-style / krzywizna |
| **`<new_df>`** | Kolumny z `olmo7b_entropy` **albo** `kgw_exact` |

Wszystko jest `concat` po osi kolumn, `fillna(0)`.

**Ważne:** dokładny zestaw kolumn baseline zależy od tego, co fizycznie siedzi w pickle’ach na dysku — kod nie narzuca stałej liczby kolumn, tylko wymaga obecności **konkretnych nazw**, żeby zbudować cross-LM (patrz niżej).

---

## 4. Cechy pochodne „cross-LM v1” (sztywno zakodowane nazwy)

Po zbudowaniu `full` skrypt **dokleja** kolumny z różnicami / ilorazami (tylko jeśli odpowiednie nazwy istnieją w `full`):

| Nowa kolumna | Wzór (idea) |
|----------------|-------------|
| `cross_olmo7b_vs_gpt2med_lp` | `olmo7b_lp_mean - lp_per` |
| `cross_olmo7b_vs_pythia28b_lp` | `olmo7b_lp_mean - bino_xl_lp_obs` |
| `cross_olmo7b_vs_pythia69b_lp` | `olmo7b_lp_mean - bino_xl_lp_per` |
| `cross_olmo7b_vs_olmo1b_lp` | `olmo7b_lp_mean - olmo_lp_mean` |
| `cross_pythia14b_vs_28b_lp` | `bino_strong_lp_obs - bino_xl_lp_obs` |
| `cross_olmo7b_vs_pythia_ppl_ratio` | `olmo7b_ppl / (bino_xl_ppl_obs + ε)` |

To są **te same** 6 cech, które historycznie nosiły nazwę „cross_lm v1” w trackerze zespołu. Nie są uczone osobno — to deterministyczne funkcje z kolumn baseline.

Dla `mistral_7b` w tym samym skrypcie zdefiniowano dodatkowe różnice (`cross_mistral_*`), ale pipeline **o7be / kgwx** ich nie używa, bo nie dodaje Mistrala jako `--feature`.

---

## 5. Moduł `olmo7b_entropy` (gałąź **o7be**)

**Plik:** `code/attacks/task3/features/olmo7b_entropy.py`  
**Model:** `allenai/OLMo-2-1124-7B-Instruct` (FP16 na GPU, FP32 na CPU).

### 5.1 Przepływ tensorów

1. Tekst → `tokenizer.encode(..., truncation=True, max_length=1024)` → `token_ids` długości `T`.
2. Batch `ids` kształtu `(1, T)` → forward przez `AutoModelForCausalLM` → `logits` kształtu `(1, T, V)`.
3. Dla predykcji następnego tokenu bierzemy **`logits[0, :-1]`** → macierz `(T-1, V)`. Celem (observed next token) to **`ids[0, 1:]`** → wektor długości `T-1`.

To standardowe „teacher forcing”: na pozycji `i` model widzi prefiks do tokenu `i` i przewiduje rozkład na token `i+1`.

### 5.2 Co jest liczone na każdej pozycji `t ∈ {0..T-2}`

- **Entropia** rozkładu: \( H_t = -\sum_v p_{t,v} \log p_{t,v} \) z `softmax(logits)`.
- **Rząd (rank)** faktycznego tokenu: pozycja w posortowanym malejąco `logits` (0 = argmax).
- **`lp_diff`:** `log p(y_t) - max_v log p(v)` — zawsze ≤ 0; mierzy „jak daleko od najlepszego logitu był wybrany token”.
- **`top2_diff`:** różnica log-prawdopodobieństw między pierwszym a drugim miejscem — margines „pewności”.
- **KL do rozkładu jednostajnego:** teoretycznie `log(V) - H_t` (tu `V` = rozmiar słownika z `log_probs`).

### 5.3 Agregacja do pojedynczego wektora cech (16 liczb)

Nazwy kolumn i sens:

| Klucz | Opis |
|-------|------|
| `o7be_ent_mean` / `_std` / `_p10` / `_p50` / `_p90` | Rozkład entropii w czasie (średnia, odchylenie, percentyle) |
| `o7be_rank_mean` / `_med` | Jak „wysoko” w rankingu leżą faktyczne tokeny (średnia, mediana) |
| `o7be_rank_top1_frac` | Frakcja kroków, gdzie wybrany token był top-1 |
| `o7be_rank_top10_frac` | Frakcja kroków z rank &lt; 10 |
| `o7be_rank_log_mean` | `mean(log(1+rank))` — mniej wrażliwe na ogony niż surowy rank |
| `o7be_lpvsmax_mean` / `_std` | Średnia i odchylenie „suboptymalności” vs argmax logitu |
| `o7be_top2_diff_mean` / `_p10` | Margines top-2 (średnia i dolny percentyl) |
| `o7be_kl_uniform_mean` | Średnia „odległości” rozkładu od uniform |
| `o7be_burst_low_ent` | Liczba **segmentów** co najmniej 3 kolejnych kroków z entropią **poniżej mediany** całego tekstu |

Tekst z &lt; 4 tokenami lub błąd tokenizacji → wszystkie cechy **0**.

**Intuicja watermarku:** ograniczenie losowania do podzbioru tokenów (lista zielona, semantyczny bias itd.) zwykle **zmienia kształt** pełnego rozkładu LM — nie tylko średnie log-proby (które już masz w `olmo_7b`), ale właśnie **entropię, rangi i „burst”** niskiej entropii.

---

## 6. Moduł `kgw_exact` (gałąź **kgwx**)

**Plik:** `code/attacks/task3/features/kgw_exact.py`  
**Obliczenia:** CPU; tylko **tokenizery** z HuggingFace (bez forwardu dużego LM). Dla wariantów z Llama tokenizer online może być wymagany download (`main_kgw_exact.sh` ustawia `TRANSFORMERS_OFFLINE=0`).

### 6.1 Idea statystyczna (Kirchenbauer-style)

Dla każdej pozycji `i` (z wybranym opóźnieniem `h`) definiujemy seed z **poprzednich** tokenów, potem testujemy, czy token `i` wpada do **„zielonej”** części słownika o udziale \(`\gamma`\). Formalnie przy czystym losowaniu zielonych jest \(`\gamma`\) frakcji tokenów; przy watermarkzie frakcja rośnie.

Liczymy:

\[
z = \frac{n_{\text{green}} - \gamma N}{\sqrt{N \gamma (1-\gamma)}}
\]

gdzie \(N\) = liczba pozycji z liczonego zakresu, \(n_{\text{green}}\) = trafienia do zielonej listy wg funkcji `_is_green`.

### 6.2 Funkcja `_is_green(seed_val, token_id, gamma)`

To **nie** jest pełna permutacja vocab jak w niektórych implementacjach referencyjnych — jest **szybki test haszowy** dający w przybliżeniu Bernoulli(`gamma`):

- `h = (seed_val * HASH_KEY + token_id * 2654435761) mod 2^32`
- `u = h / 2^32`
- zielony iff `u < gamma`

`HASH_KEY = 15485863` (stała w kodzie, „paper default” w komentarzu).

### 6.3 Sposób liczenia seeda dla `h` kroków wstecz

- **`h == 1`:** `seed_val = token_ids[i-1]`.
- **`h > 1`:** `seed_val` to XOR kombinacji: `token_ids[i-j] * (j+1)` dla `j = 1..h` (patrz pętla w `_zscore_fast`).

W tabeli konfiguracji używane jest **`h = 1` albo `h = 4`**.

### 6.4 Konfiguracje (7 × z-score + 3 agregaty = 10 „logicznych” grup cech)

Dla każdej trójki `(tokenizer, gamma, h)` powstaje jedna kolumna `kgwx_<tag>_g<25|50>_h<h>_z`:

| Tokenizer HF | γ | h | Przykład `tag` (ostatni segment ścieżki HF, `-`→`_`) |
|--------------|---|----|-----------------------------------|
| `NousResearch/Llama-2-7b-hf` | 0.25 | 1 | `kgwx_llama_2_7b_hf_g25_h1_z` |
| (ten sam) | 0.5 | 1 | `kgwx_llama_2_7b_hf_g50_h1_z` |
| (ten sam) | 0.25 | 4 | `kgwx_llama_2_7b_hf_g25_h4_z` |
| `allenai/OLMo-2-1124-7B-Instruct` | 0.25 / 0.5 | 1 | `kgwx_olmo_2_1124_7b_instruct_g25_h1_z` itd. |
| `gpt2` | 0.25 / 0.5 | 1 | `kgwx_gpt2_g25_h1_z` itd. |

Dokładny `tag` buduje kod: ostatni segment ścieżki HF z `/`, zamiana `-` na `_`, lower case.

Dodatkowo:

- `kgwx_max_abs_z` — max \|z\| po wszystkich konfiguracjach,
- `kgwx_sum_abs_z` — suma \|z\|,
- `kgwx_n_signif_z` — liczba konfiguracji z \|z\| > 2.

Tekst z &lt; 4 słowami (split po spacji) → zera; pusty tokenizer → pominięcie danej konfiguracji.

**Rola w ensemble:** organizatorzy mogą **nie** używać dokładnie tego hasza ani tokenizera — ale **logistyczna regresja** nad całym wektorem może wykorzystać te liczby jako **syntetyczne, skorelowane z regularnością list** cechy, nawet gdy pojedynczy z-test nie jest „oficjalnym” detektorem.

---

## 7. Uczenie: skalowanie, logreg, OOF, predykcja testu

### 7.1 Pipeline

Dla każdego foldu walidacji krzyżowej (domyślnie **5** foldów, stratyfikacja, `random_state=42`):

1. `StandardScaler()` dopasowany **tylko** do `X_train`,
2. `LogisticRegression(C=0.01, max_iter=4000, solver="lbfgs")` na `y`.

`C=0.01` to **silna regularyzacja** (priorytet mało-złożonego granicznego modelu przy wielu potencjalnie skorelowanych kolumnach).

### 7.2 OOF

Na zlabelowanych 540 próbkach zbierane są prawdopodobieństwa klasy 1 z części walidacyjnej każdego foldu → wektor `oof`. Drukowany jest **TPR@1%FPR** (`templates.eval_scaffold.tpr_at_fpr`): scores sortowane **malejąco** (wyższy score = bardziej „podejrzany” / watermarked), następnie skumulowane FP/TP vs próg tak, by FPR na clean nie przekroczył 1% w sensie przybliżonym przez ten punkt na krzywej.

**Uwaga z praktyki zespołu:** OOF na tym stosie bywa **dużo wyższy** niż publiczny leaderboard — nie należy go traktować jako prognozy absolutnego wyniku konkursowego; nadal jest przydatny do **porównywania** dwóch wersji tego samego pipeline’u.

### 7.3 Model finalny i CSV

Po CV ten sam pipeline jest **trenowany na całych 540** zlabelowanych próbkach i robi `predict_proba` na `X_test`. Wynik klipowany do `[0.001, 0.999]` i zapisywany z `id` z `test.jsonl`.

To generuje **jedno** z dwóch wejść do blendu.

---

## 8. `blend_csvs.py` — matematyka fuzji

**Domyślny tryb:** `--mode rank`.

Dla \(K\) plików CSV z tymi samymi `id` w tej samej kolejności:

1. `scores[k, n]` — score modelu `k` na próbce `n`.
2. `ranks[k, n] = rankdata(scores[k, :])[n]` — scipy `rankdata` (przy remisach — średnia rang).
3. Ważona suma rang: \(wsum_n = \frac{\sum_k w_k \cdot ranks[k,n]}{\sum_k w_k}\).
4. Min-max na wektorze `wsum` po całym teście: \((wsum - min) / (max - min + \epsilon)\), potem `clip` do `[0.001, 0.999]`.

Jeśli nie podasz wag w CLI, każdy plik ma wagę **1.0** (czysta średnia rang).

**Dlaczego to ma sens przy TPR@1%FPR:** oba modele mogą mieć **różną kalibrację** (np. jeden rozrzuca score wężej). Średnia **rang** jest invariantna względem monotonicznych transformacji per-model (z dokładnością do tie-breakingu), więc łatwiej połączyć dwa scoringi bez ręcznego dopasowywania skali.

Inne tryby w skrypcie (`median`, `geomean`, `tmean`) istniały do eksperymentów; **BEST3** opiera się na **rank**.

---

## 9. Złożoność obliczeniowa i środowisko

| Etap | Dominujący koszt |
|------|-------------------|
| Budowa cache baseline | Jednorazowo na klastrze (duże LM, I/O) |
| `olmo7b_entropy` na 540+2250 tekstów | Forward **OLMo-7B** na GPU — ciężki |
| `kgw_exact` | Tylko tokenizacja + pętla O(N) po tokenach na CPU — lżejszy |
| LogReg | Pomijalny w porównaniu z LM |
| Blend | O(K·N log N) przez rankdata — pomijalny |

Skrypty Slurm (`main_o7be.sh`, `main_kgw_exact.sh`) ustawiają `HF_HOME`, venv współdzielony zespołu itd.

---

## 10. Spójność z innymi plikami w repo

- **Krótki opis pod prezentację:** [task3_prezentacja_BEST3.md](task3_prezentacja_BEST3.md).
- **Specyfikacja zadania (metryka, format):** [task3_watermark_detection.md](task3_watermark_detection.md).
- **Historia eksperymentów (OOF vs leaderboard):** [task3_submissions_tracker.md](task3_submissions_tracker.md), [task3_handoff_session2.md](task3_handoff_session2.md).

---

## 11. Jak samemu prześledzić kod (ścieżki)

| Element | Plik |
|---------|------|
| Orchestracja treningu + baseline + derived | `code/attacks/task3/extract_and_train.py` |
| Entropie OLMo-7B | `code/attacks/task3/features/olmo7b_entropy.py` |
| Z-score KGW wielokonfiguracyjny | `code/attacks/task3/features/kgw_exact.py` |
| Rank blend | `code/attacks/task3/blend_csvs.py` |
| Metryka OOF | `templates/eval_scaffold.py` → `tpr_at_fpr` |
| Joby GPU | `code/attacks/task3/main_o7be.sh`, `main_kgw_exact.sh` |

Jeśli chcesz **odtworzyć** blend lokalnie (przy dwóch już wygenerowanych CSV):

```bash
python code/attacks/task3/blend_csvs.py \
  --csvs path/to/submission_clm_kgwx.csv:1.0 path/to/submission_clm_o7be.csv:1.0 \
  --out task3_watermark_blend_kgwx_o7be.csv \
  --mode rank
```
