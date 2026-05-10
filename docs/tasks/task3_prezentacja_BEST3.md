# Task 3 — `task3_watermark_blend_kgwx_o7be_BEST3.csv`: opis rozwiązania na prezentację

Dokument dla ~2 min mówionego bloku: **co zrobiliśmy krok po kroku**, **dlaczego to ma sens przy metryce TPR@1%FPR**, oraz **krótki przegląd odrzuconych linii**. Plik `BEST3` w repozytorium jest **kopii/relabel** tej samej mieszanki co `task3_watermark_blend_kgwx_o7be.csv` (identyczny hash MD5 w kolejce submitów — ten sam wynik co blend `kgwx + o7be`).

---

## Kontekst zadania (1 zdanie na slide)

- **Cel:** dla każdego tekstu zwrócić `score ∈ [0,1]` — wyższy = bardziej „watermarked”.
- **Metryka:** **TPR przy FPR = 1%** — praktycznie **problem rankingu**: ważny jest porządek próbek, nie absolutna kalibracja „0.5”.
- **Trudność:** w danych są **różne rodzaje watermarków** (nie jeden schemat Kirchenbauera); potrzebne są sygnały **uzupełniające się**, nie jeden wąski detektor.

---

## Pipeline końcowy — kroki **dokładnie po kolei**

### Krok 1 — Wspólny szkielet uczenia (`extract_and_train.py`)

- Ładowane są **train + validation** (etykiety clean vs watermarked) oraz **test** (2250 wierszy).
- Dla każdego tekstu budowany jest **wektor cech**: cache z wcześniejszej pracy (m.in. `a`, opcjonalnie `bc`, `d`, binoculars, FDGPT, `multi_lm`, `olmo_7b`, … — zależnie od tego, co jest w `cache-dir`).
- **Dopisywane są 6 pochodnych „cross-LM v1”** (np. różnice log-prawdopodobieństw OLMo-7B vs GPT-2 medium / Pythia / OLMo-1B oraz wybrane ilorazy PPL) — to była wcześniej najsilniejsza linia zespołu („cross_lm”).
- Do tego **konkatenjowane są nowe cechy** z wybranego modułu (`--feature`): albo `olmo7b_entropy`, albo `kgw_exact`.
- **Model:** `StandardScaler` + `LogisticRegression` (domyślnie `C=0.01`, walidacja krzyżowa stratyfikowana, **OOF** z predict_proba).
- **Wyjście:** prawdopodobieństwo klasy watermarked na teście, przycięte do `[0.001, 0.999]` → **osobny CSV na gałąź** (`submission_clm_o7be.csv` vs `submission_clm_kgwx.csv`).

### Krok 2 — Gałąź **o7be** (OLMo-7B + entropie)

- Skrypt klastrowy: `code/attacks/task3/main_o7be.sh`.
- Moduł: `features/olmo7b_entropy.py` — na **OLMo-2-7B-Instruct** liczone są m.in. **średnia/rozstępy rozkładu entropii tokenów**, statystyki **rangu** prawdziwego tokenu, udział top-1/top-10 ranków, odległość logitów od maksimum, rozkład różnicy top-2, dyssymetria względem rozkładu jednostajnego, prosty sygnał „burst” niskiej entropii.
- **Intuicja:** watermark często **wyszarza** rozkład następnego tokenu (niższa entropia, inne rangi); to jest **inna fizyka** niż czysty z-test na liście zielono-czerwonej.

### Krok 3 — Gałąź **kgwx** (wielokonfiguracyjny z-test KGW)

- Skrypt: `code/attacks/task3/main_kgw_exact.sh`.
- Moduł: `features/kgw_exact.py` — dla każdego tekstu liczone są **z-score’y „green list”** w kilku konfiguracjach (różne **tokenizery**: Llama-2, OLMo-7B-Instruct, GPT-2; **γ** ∈ {0.25, 0.5}; **h** = 1 lub 4 poprzednie tokeny do seeda). Dodatkowo agregaty: max/ suma |z|, liczba |z| > 2.
- **Intuicja:** nawet jeśli **prawdziwy** watermark konkursu **nie** jest dokładnie Kirchenbauer z tym tokenizerem, **logistyczna regresja** może **wybrać** albo **łagodnie wykorzystać** te sztuczne sygnały jako proxy — bez konieczności trafienia jednej konfiguracji.

### Krok 4 — **Blend końcowy** (`blend_csvs.py`)

- Dwa CSV (wyniki kroków 2 i 3) są łączone w trybie domyślnym **`--mode rank`**.
- Dla każdego modelu baseline score → **rangi**; następnie **średnia ważona rang** (domyślnie wagi 1:1), **min–max** na całości testu, klip do `[0.001, 0.999]`.
- **Dlaczego rank:** metryka **TPR@1%FPR** ustala próg na percentylu clean — **porządek** jest wystarczający; blend rankowy **redukuje problem nieskalowalnych lub rozjeżdżających się skal** między dwoma logregami.

### Krok 5 — Plik **BEST3**

- Zapis `task3_watermark_blend_kgwx_o7be_BEST3.csv` = **ten sam blend** co `…_blend_kgwx_o7be.csv` (wygodna nazwa na „final pick” pod prezentację / submit).

---

## Dlaczego ten stack mógł wygrać na leaderboardzie (argumenty na jury)

1. **Dwa niezależne „źródła błędów”:** entropie/rangi pod dużym LM vs statystyki list zielono-czerwonej — gdy jedna część danych jest słaba dla jednego sygnału, druga może ratować ranking.
2. **Zgodność z metryką:** rank-blend jest **wyrównany z TPR@1%FPR** (decyzja = próg na frakcji fałszywych alarmów na clean).
3. **OGólność względem multi-type:** ani sama twarda detekcja KGW, ani same PPL/cross-LM nie muszą pokryć wszystkich schematów; **ensembel** to standardowe obniżenie wariancji na części testu niewidocznej publicznie.

(Uwaga metodologiczna z regulaminu hackathonu: **public leaderboard ≠ final extended test** — warto na prezentacji powiedzieć, że celowałeś w **robustny ranking**, nie w podkręcanie pod 30% public.)

---

## Krótko: inne podejścia z prób (co to było, czemu nie zostało „final blend”)

| Linia | Idea | Wynik / status (skrót) |
|--------|------|-------------------------|
| **cross_lm v1** | 6 różnic „OLMo-7B vs inne LM” na baseline cechach — historyczny **benchmark zespołu** na LB (~0.28 w trackerze) | Silny pojedynczy sygnał; dalsze eksperymenty szukały **dodatkowej różnorodności** |
| **cross_lm v2** | Zbyt wiele pochodnych cech | Rozcieńczenie sygnału, brak poprawy |
| **Same olmo7b PPL** | Sam duży LM bez bogatszych statystyk | Gorszy niż cross_lm v1 na LB |
| **LM-as-judge** | Prompty klasyfikujące tekst | Dodało OOF, często **nie uniosło** public LB w stacku |
| **Większe Bino / więcej LM** | Skalowanie modeli | Plateau — większy model ≠ lepszy sygnał pod ten konkretny mix |
| **Bezpośrednie KGW** (same z-testy, stare `branch_kgw`) | Dopasowanie do domyślnego Kirchenbauera | **Nie trafiało** w rzeczywisty schemat — **kgw_exact** traktujemy jako **cechy pod LR**, nie jako jedyny detektor |
| **Stacki 60–300+ cech + meta** | Leak-free pivot po `branch_bc` | Lepsze OOF „uczciwe”, ale **wysoka korelacja** między submissionami; blendy **median9 / w_proven** itd. — eksperymenty ensemble |
| **Paraphrase / triblend / mega** | Więcej modeli lub cech parafrazy w blendzie | W logu submisji **inne** warianty (para, mega, 4-way) — **osobne MD5**, eksploracja; **nie** wybrano jako finalnego prostego 2-way kgwx+o7be |
| **clm_mistral / clm_llama2 / judges** | Dywersyfikacja surrogate LM | Próby marginalne względem wybranego bloku |

---

## Szkic **2 minut** mówionych (timer)

**0:00–0:25 — Problem.** Multi-type watermark, wynik ciągły, ocena **TPR przy 1% fałszywych na clean** → liczy się **ranking**, nie magiczny próg 0.5.

**0:25–1:10 — Metoda.** Dwa head’y na **tym samym silnym baseline** (stare cechy + cross-LM v1): (A) **entropie i rangi tokenów z OLMo-7B**, (B) **wiele z-score’ów list zielono-czerwonej** (różne tokenizery i γ). Każdy head = logreg. **Połączenie: średnia ważona rang** → jeden CSV.

**1:10–1:50 — Dlaczego.** Uzupełniające się artefakty generacji; rank-blend zgodny z **TPR@1%FPR**; ensemble obniża ryzyko, że jeden typ watermarku „zgubi” pojedynczy detektor.

**1:50–2:00 — Uczciwość.** Testowaliśmy wiele stacków i blendów (cross-LM only, leak-free meta, paraphrase, KGW surowe) — **ten dual-head + rank** wybrałem jako najlepszą **strukturę** pod robustność i wynik na LB.

---

## Odniesienia w repo (do pokazania na slajdzie „repro”)

- `extract_and_train.py` — trening + cross-LM derivations + zapis CSV.
- `features/olmo7b_entropy.py` — gałąź o7be.
- `features/kgw_exact.py` — gałąź kgwx.
- `blend_csvs.py` — fuse dwóch CSV (`--mode rank`).
- `main_o7be.sh`, `main_kgw_exact.sh` — joby Slurm na Jülich.
