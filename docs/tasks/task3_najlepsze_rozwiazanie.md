# Task 3 — dokładny opis najlepszego rozwiązania

**Plik wynikowy:** `task3_watermark_blend_kgwx_o7be_BEST3.csv` (w repozytorium jest to **ta sama zawartość** co `task3_watermark_blend_kgwx_o7be.csv` — ten sam skrót MD5; nazwa `BEST3` oznacza wybraną finalną submisję).

**Idea w jednym zdaniu:** dwa modele prawdopodobieństwa „watermarked” z identycznym szkieletem cech i uczenia, różniące się tylko **jednym blokiem** nowych kolumn (`olmo7b_entropy` vs `kgw_exact`), są połączone **średnią ważoną rang** (`blend_csvs.py --mode rank`), żeby uzyskać jeden ranking 2250 próbek testowych — zgodny z metryką **TPR @ FPR = 1%** (organizator ustala próg na czystych tekstach; decyduje **porządek** score’ów).

---

## 1. Wynik końcowy

- **Format:** `id,score` — dokładnie **2250** wierszy, `score ∈ [0.001, 0.999]` (klip po predykcji i po blendzie).
- **Źródło score’a:** najpierw `predict_proba[:,1]` z logistycznej regresji na każdej gałęzi, potem transformacja rank-blend na całym teście.

---

## 2. Przepływ obliczeń (trzy fazy)

1. **Gałąź A (`o7be`):** `extract_and_train.py --feature olmo7b_entropy` → jeden CSV (np. `submission_clm_o7be.csv`).
2. **Gałąź B (`kgwx`):** `extract_and_train.py --feature kgw_exact` → drugi CSV (np. `submission_clm_kgwx.csv`).
3. **Fuzja:** `blend_csvs.py --csvs A.csv B.csv --out ... --mode rank` (domyślnie wagi **1 : 1**).

---

## 3. Szkielet `extract_and_train.py`

### 3.1 Dane

Z `--data-dir` (dataset `llm-watermark-detection`):

| Plik | Rola |
|------|------|
| `train_clean.jsonl`, `train_wm.jsonl` | 180 + 180 zlabelowanych |
| `valid_clean.jsonl`, `valid_wm.jsonl` | 90 + 90 zlabelowanych |
| `test.jsonl` | 2250 bez etykiety |

**Razem z etykietą: 540 próbek.** Etykieta: 0 = clean, 1 = watermarked.

### 3.2 Kolejność tekstów przy ekstrakcji cech

Jedna lista: wszystkie teksty `(train ∪ val)` po kolei, potem wszystkie **testowe**. Pierwsze `n_lab = 540` wierszy macierzy cech odpowiada próbkom z etykietą; reszta — testowi.

### 3.3 Cache (`--cache-dir`)

- Moduł pod `--feature` jest liczony dla **wszystkich** 540 + 2250 tekstów (lub wczytywany z `features_<nazwa>.pkl`, o ile nie ma `--no-cache`).
- **Baseline** jest **wyłącznie** z gotowych pickle’i `features_*.pkl` — brak pliku ⇒ dana grupa kolumn nie wchodzi do modelu.

Ładowane są (jeśli istnieją), w kolejności concat:

`a`, `bc`, `d`, `bino`, `bino_strong`, `bino_xl`, `olmo_7b`, `multi_lm`, `fdgpt`, oraz **nowy** blok (`olmo7b_entropy` **lub** `kgw_exact`).

Na klastrze **oba** warianty końcowe powinny używać **tego samego** zestawu cache baseline, inaczej dwa CSV przed blendem mogłyby opierać się na różnym zestawie kolumn.

### 3.4 Sześć cech pochodnych „cross-LM v1”

Po zbudowaniu macierzy `full` skrypt **deterministycznie** dokłada kolumny (o ile istnieją wymagane nazwy):

| Nowa kolumna | Definicja |
|--------------|-----------|
| `cross_olmo7b_vs_gpt2med_lp` | `olmo7b_lp_mean - lp_per` |
| `cross_olmo7b_vs_pythia28b_lp` | `olmo7b_lp_mean - bino_xl_lp_obs` |
| `cross_olmo7b_vs_pythia69b_lp` | `olmo7b_lp_mean - bino_xl_lp_per` |
| `cross_olmo7b_vs_olmo1b_lp` | `olmo7b_lp_mean - olmo_lp_mean` |
| `cross_pythia14b_vs_28b_lp` | `bino_strong_lp_obs - bino_xl_lp_obs` |
| `cross_olmo7b_vs_pythia_ppl_ratio` | `olmo7b_ppl / (bino_xl_ppl_obs + ε)` |

Nie mają osobnych wag — wchodzą do tej samej logreg co reszta kolumn.

### 3.5 Model i wyjście

- **Pipeline (na każdym foldzie i na finałowym fit):** `StandardScaler` → `LogisticRegression(C=0.01, max_iter=4000, solver="lbfgs")`.
- **Walidacja:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — na OOF drukowany jest **TPR@1%FPR** (implementacja `tpr_at_fpr` w `templates/eval_scaffold.py`: sortowanie score malejąco, kumulacja TP/FP po czystych).
- **Finał:** fit na **całych 540** próbkach, `predict_proba` na teście, klip do `[0.001, 0.999]` → CSV gałęzi.

---

## 4. Różnica między gałęziami: `olmo7b_entropy` (o7be)

**Plik:** `code/attacks/task3/features/olmo7b_entropy.py`  
**Model:** `allenai/OLMo-2-1124-7B-Instruct` (FP16 na CUDA, FP32 na CPU).

Dla tekstu z tokenami `T` (max 1024):

1. `logits` z causal LM: kształt `(T, V)`.
2. Używane są pozycje **następnego tokenu:** `logits[:-1]` vs `ids[1:]` — dla każdego kroku \(t\) pełny rozkład \(p_{t}(\cdot)\) nad słownikiem.

**Per krok** liczone są m.in.:

- entropia \(H_t = -\sum_v p_{t,v}\log p_{t,v}\),
- **rank** faktycznego tokenu w posortowanych malejąco logitach (0 = najwyższy),
- `log p(y_t) - max_v log p(v)` (zawsze ≤ 0),
- różnica logitów top-1 vs top-2,
- `log(V) - H_t` (związane z odległością od rozkładu jednostajnego),

a następnie **agregaty po czasie** — **16 liczb** (`o7be_ent_*`, `o7be_rank_*`, `o7be_lpvsmax_*`, `o7be_top2_diff_*`, `o7be_kl_uniform_mean`, `o7be_burst_low_ent`).

`o7be_burst_low_ent` liczy, ile razy z rzędu pojawił się **segment** ≥ 3 kolejnych kroków z entropią **poniżej mediany** entropii w tym tekście.

Tekst z &lt; 4 tokenami → wszystkie cechy 0.

**Sens:** to są cechy z **pełnego rozkładu** predykcji LM w każdym kroku, nie tylko z uśrednionego log-proba (jak w cache `olmo_7b`). Watermark może zmieniać kształt rozkładu (entropia, rangi, bursty).

---

## 5. Różnica między gałęziami: `kgw_exact` (kgwx)

**Plik:** `code/attacks/task3/features/kgw_exact.py`  
**Bez forwardu LM** — tylko tokenizery HuggingFace i pętle po ID tokenów (CPU).

Dla każdej konfiguracji \((\text{tokenizer}, \gamma, h)\) liczona jest liczba pozycji, w których bieżący token jest uznany za „zielony” względem seeda z **\(h\)** poprzednich tokenów, i z tego **z-score** pod \(H_0\) losowania z prawdopodobieństwem \(\gamma\):

\[
z = \frac{n_{\text{green}} - \gamma N}{\sqrt{N \gamma (1-\gamma)}} .
\]

**„Zieloność”** tokena: losowość z hasza `(seed_val, token_id)` — nie pełna permutacja vocab; `HASH_KEY = 15485863`.

**Seed:**

- \(h=1\): poprzedni token,
- \(h=4\): XOR kombinacji `token_ids[i-j] * (j+1)` dla \(j=1..4\).

**Siedem konfiguracji z-score** (osobna kolumna każda), np. nazwy postaci `kgwx_<tag>_g25_h1_z` gdzie `tag` powstaje z ostatniego segmentu nazwy HF modelu tokenizera (np. Llama-2, OLMo-2-1124-7B-Instruct, gpt2), plus **trzy agregaty:** `kgwx_max_abs_z`, `kgwx_sum_abs_z`, `kgwx_n_signif_z` (liczba \|z\| &gt; 2).

Tekst z &lt; 4 słowami (split spacją w kodzie) → zera na wyjściu.

**Sens:** sygnały stylu **red-green list** pod wieloma tokenizacjami i \(\gamma\); logreg **łączy** je lżej lub ciężej z innymi kolumnami, nawet jeśli prawdziwy watermark konkursu nie jest bit-w-bit Kirchenbauer z tym haszem.

---

## 6. Rank-blend (`blend_csvs.py`)

Macierz score’ów ma kształt \((K, N)\): \(K\) modeli (tu 2), \(N\) próbek testowych.

1. Dla każdego \(k\) osobno: wektor \(s_{k,1},\ldots,s_{k,N}\) jest zamieniany na rangi `scipy.stats.rankdata` (remisy → średnia rang). Wynik: \(r_{k,n}\).
2. Dla każdej próbki \(n\): \(wsum_n = \sum_k w_k\, r_{k,n} / \sum_k w_k\) (domyślnie \(w_k=1\)).
3. Wektor \((wsum_1,\ldots,wsum_N)\) jest normalizowany min–max na całym teście, z \(\epsilon\) przy zerowym zakresie, potem klip do `[0.001, 0.999]`.

To stabilizuje połączenie dwóch gałęzi o **różnej kalibracji** `predict_proba`, zachowując informację potrzebną do **rankingowej** metryki ewaluacji.

---

## 7. Środowisko uruchomieniowe (referencja)

- **o7be:** GPU, offline HF jeśli modele w cache (`main_o7be.sh`).
- **kgwx:** CPU; skrypt `main_kgw_exact.sh` może włączyć **online** dla tokenizerów (np. Llama).

Lokalne odtworzenie blendu przy gotowych dwóch CSV:

```bash
python code/attacks/task3/blend_csvs.py \
  --csvs path/to/submission_clm_kgwx.csv:1.0 path/to/submission_clm_o7be.csv:1.0 \
  --out task3_watermark_blend_kgwx_o7be.csv \
  --mode rank
```

---

## 8. Mapowanie na pliki w repozytorium

| Komponent | Ścieżka |
|-----------|---------|
| Trening + baseline + derived | `code/attacks/task3/extract_and_train.py` |
| Cechy o7be | `code/attacks/task3/features/olmo7b_entropy.py` |
| Cechy kgwx | `code/attacks/task3/features/kgw_exact.py` |
| Blend | `code/attacks/task3/blend_csvs.py` |
| Job Slurm o7be | `code/attacks/task3/main_o7be.sh` |
| Job Slurm kgwx | `code/attacks/task3/main_kgw_exact.sh` |
| Metryka OOF | `templates/eval_scaffold.py` → `tpr_at_fpr` |

---

## 9. Intuicja „dlaczego to jest silne rozwiązanie ensemblowe”

- **Dwie klasy sygnału:** (i) kształt rozkładu tokenów pod **tym samym** dużym LM (OLMo-7B-Instruct), (ii) zestaw **statystyk listowych** niezależnych od forwardu tego LM.
- **Wspólny silny backbone:** cache ze stosu zespołu (binoculars, OLMo PPL, multi_lm, itd.) plus **sześć** sprawdzonych różnic między modelami (cross-LM v1).
- **Rank-merge:** dopasowany do **TPR@1%FPR**, gdzie kluczowy jest **porządek** próbek, a nie wartość bezwzględna pojedynczego score’a przed blendem.
