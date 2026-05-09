# Task 3 — Watermark Detection: Session Context dla nowego Claude

> Hand-off do innej instancji Claude która ma kontynuować pracę nad Task 3 nie mając historii tego czatu.
> Last updated: 2026-05-09 ~21:30 (5h przed deadline 2026-05-10 12:30)

## TL;DR — gdzie jesteśmy

- **Aktualny best**: leaderboard **0.103** TPR@1%FPR z `submission_strong_bino.csv`
- **Lider**: 0.27 (gap **2.6×**, fundamentalnie czegoś nam brak)
- **Plateau**: zwiększanie modelu Binoculars NIE pomaga dalej (XL bino gave OOF 0.6556 ale leaderboard nie wzrósł)
- **Główna hipoteza luki**: nie łapiemy KGW (Kirchenbauer) bo organizatorzy użyli innego tokenizera niż my testowaliśmy + słabo łapiemy Liu/Semantic
- **Repo**: branch `task3`, kod w `code/attacks/task3/`

---

## Task — fakty z PDF

**Ground truth source**: `docs/tasks/task3_watermark_detection.md` (PDF revealed 12:00 2026-05-09)

- **Goal**: detector watermark dla LLM-generated text, każdy sample → score `[0,1]`, wyższy = bardziej watermarked
- **3 watermark types** (PDF references):
  - **Kirchenbauer/KGW** (ICML 2023) — Red-Green list, hash(prev_token) determines green list
  - **Liu/Semantic** (ICLR 2024) — Semantic Invariant, embedding-based clustering
  - **Zhao/Unigram** (ICLR 2024) — Provable Robust, statyczna green list (ten sam zawsze zielony)
- **Dataset**:
  - Train: 360 (180 clean + 180 watermarked, "split evenly across watermark types" — 60 per type)
  - Validation: 180 (90 + 90)
  - **Test: 2250** — submission ma 2250 wierszy
- **Metric**: **TPR @ 1% FPR** — per-text ranking matters, absolute value mniej
- **Public leaderboard**: 30% test (~675), **Private**: 70% (~1575) — final scoring na 2026-05-10
- **Cooldown submit**: 5 min success / 2 min fail, score updateuje się tylko jeśli wyższy
- **API**: `POST http://35.192.205.84/submit/13-llm-watermark-detection`, klucz w `.env` (`HACKATHON_API_KEY`)

**🎯 Final scoring na EXTENDED test sets** (organizatorzy ogłosili rano 2026-05-09): jutro dokładny test set będzie inny niż public 30%. Trzeba unikać overfit do public.

---

## Dataset structure (na klastrze)

`/p/scratch/training2615/kempinski1/Czumpers/llm-watermark-detection/`:
- `train_clean.jsonl` (180), `train_wm.jsonl` (180)
- `valid_clean.jsonl` (90), `valid_wm.jsonl` (90)
- `test.jsonl` (2250)
- `Dataset.zip`, `README.md` (PUSTY!), `pyproject.toml` (zawiera `sentence-transformers` → potwierdzone Liu watermark)
- `main.py` (boilerplate, useless)

JSONL format: `{"id": int, "text": str}` — **nie ma `watermark_type` column ani metadata**. Można próbować unsupervised clustering żeby odkryć typy.

Texts wyglądają jak **outputy 7B+ LLMa** (LLaMA-2-7B, Mistral-7B, podobne). Style: explanatory Q&A, numbered lists, technical content. **NIE jest GPT-2 generated** (za dobre).

---

## Architektura kodu (`code/attacks/task3/`)

```
main.py                # orchestrator: load → extract features → OOF + final → submission
main.sh                # canonical sbatch script (running main.py)
main_<variant>.sh      # alternatywne joby z różnymi flagami
cv_utils.py            # 5-fold OOF + bootstrap CI helpers
features/
  branch_a.py          # gpt2 log-prob stats (lp_mean/std/percentiles), GLTR ranks, ngram diversity, burstiness, gzip ratio, TTR — UNIVERSAL, NO LABELS
  branch_bc.py         # UnigramGreenList (Zhao detector, learned from labels) + BigramGreenList (KGW, learned from labels) — SCHEME-SPECIFIC, OVERFITS
  branch_d.py          # sentence-transformers all-MiniLM-L6-v2: adj cosine sim, LSH KL-div — Liu/Semantic proxy
  binoculars.py        # gpt2 (observer) + gpt2-medium (performer) PPL ratio
  binoculars_strong.py # Pythia-1.4b + Pythia-2.8b
  binoculars_xl.py     # Pythia-2.8b + Pythia-6.9b
  branch_kgw.py        # Direct Kirchenbauer detection: torch.randperm(vocab_size) seeded by hash_key * prev_token. 4 configs (gpt2 g=0.25/0.5, opt-1.3b, pythia-1.4b) all hash_key=15485863
  branch_kgw_v2.py     # KGW v2: 6 hash_keys [0,1,42,100,12345,7] + h=2 multigram (mul + add seedings), gpt2 only
  fast_detectgpt.py    # NEW (gotowe ale nie odpalone) — Fast-DetectGPT z analytical curvature, Pythia-2.8b
cache/                 # features pickled (current state na klastrze):
  features_a.pkl              (390KB, 17 features)
  features_bc.pkl             (90KB, 4 features) + green_list.pkl (UnigramGreenList)
  features_bigram.pkl         (90KB, 4 features) + bigram_greenlist.pkl
  features_bino.pkl           (110KB, 5 features, gpt2/gpt2-medium)
  features_bino_strong.pkl    (110KB, 5 features, pythia-1.4b/2.8b)
  features_bino_xl.pkl        (110KB, 5 features, pythia-2.8b/6.9b)
  features_d.pkl              (90KB, 4 features)
  features_kgw.pkl            (270KB, 12 features) — wszystkie no-signal
  features_kgw_v2.pkl         (360KB, 16 features) — wszystkie no-signal
  # NIE WYEKSTRAKTOWANE jeszcze:
  features_fdgpt.pkl          (NEW — w main_fdgpt.sh, ~3 min ekstrakcji)
```

**Cluster path**: `/p/scratch/training2615/kempinski1/Czumpers/repo-multan1/code/attacks/task3/`
**Cache path**: `/p/scratch/training2615/kempinski1/Czumpers/task3/cache/`
**Submissions output**: `/p/scratch/training2615/kempinski1/Czumpers/task3/submission*.csv`

---

## Workflow lokalnie + klaster

**Branch**: `task3` (NIE main!). Edycja kodu na laptopie → `git push origin task3`.

```bash
# Na klastrze:
cd /p/scratch/training2615/kempinski1/Czumpers/repo-multan1
git pull
sbatch code/attacks/task3/main_<variant>.sh

# Pull CSV lokalnie (wybiera kanoniczne mapowanie nazw):
just pull-csv task3 submission_<variant>.csv
# → submissions/task3_watermark_<variant>.csv

# Submit:
just submit task3 submissions/task3_watermark_<variant>.csv
# Cooldown 5 min po success / 2 min po fail
```

**KRYTYCZNE**: GPU compute na Jülich `dc-gpu` partition, A800 (44GB GPU memory). Wszystkie modele są **offline cached** w `/p/scratch/training2615/kempinski1/Czumpers/.cache/hub/`:
- `gpt2`, `gpt2-medium`, `EleutherAI/pythia-1.4b/2.8b/6.9b`, `facebook/opt-1.3b`, `allenai/OLMo-2-0425-1B-Instruct`, `sentence-transformers/all-MiniLM-L6-v2`
- **NIE MA**: LLaMA-2 (gated), Mistral, Falcon, BERT-derivatives, OpenAI-detector models

Login node: `jrlogin0X.jureca`, **1TB RAM, no GPU**. Compute: `jrcXXXX`, A800.

`scripts/juelich_exec.sh "<cmd>"` runs cmd przez SSH ControlMaster (4h socket via `juelich_connect.sh`). NIE wywołuj `ssh` bezpośrednio.

---

## Wszystkie spróbowane podejścia + wyniki

### Klasyfikator: LightGBM → LogisticRegression (krytyczna decyzja!)

**Problem**: oryginalny LightGBM + IsotonicRegression dawał **bimodal collapse** — 1139 sampli przy 0.031 i 1015 przy 0.973. To kompletnie niszczy ranking dla TPR@1%FPR (ties wewnątrz klastra są nierozróżnialne).

**Root cause**: branch_bc features są SUPER discriminative na training (data leakage z labels w fittingu green list), LightGBM stops early z perfect split → outputs near 0 lub 1 → IsotonicRegression mapuje na 2 punkty.

**Fix**: Replaced with `LogisticRegression(C=0.01)` + `StandardScaler`. Sigmoid output jest naturalnie ciągły, nawet z bardzo discriminative features. Default `--classifier logreg`.

### Lista eksperymentów (chronologicznie)

| # | Wariant | Features | OOF TPR | Leaderboard | Wnioski |
|---|---|---|---|---|---|
| 1 | LightGBM Phase 1 | branch_a tylko | ~0.45 | **0.05** (historycznie) | Branch_a alone — uniwersalne ale słabe |
| 2 | LightGBM Phase 2 + iso | branch_a+bc+bino+d | ~0.95 | **niski** | Bimodal collapse, 2 klastry 0.031/0.973 |
| 3 | LogReg -bc | branch_a+bino+d (bez bc) | 0.048 | 0.05 | Bez bc brak sygnału |
| 4 | **LogReg +bc baseline** | branch_a+bc+bino+d | 0.6333 | **0.090** | bc dodaje sygnał, smooth dystrybucja |
| 5 | LogReg +bc +bigram | + BigramGreenList (KGW proxy) | 0.7037 | ≤0.09 | OOF rośnie ale leaderboard NIE → bigram overfittuje |
| 6 | LogReg +bc +strong-bino | + Pythia-1.4b/2.8b binoculars | 0.6444 | **0.103** ✅ | Pierwszy realny wzrost na leaderboard |
| 7 | LogReg +bc +kgw v1 | + 4 KGW configs (gpt2/opt/pythia, hash_key=15485863) | 0.6037 | nie submitowane | NO SIGNAL — wszystkie 4 configs noise |
| 8 | LogReg +bc +xl-bino | + Pythia-2.8b/6.9b binoculars | 0.6556 | **NIE wzrosło** powyżej 0.103 | Plateau, scaling Bino przestał pomagać |
| 9 | LogReg +bc +kgw v2 | + 6 hash_keys h=1 + 2 h=2 multigram (gpt2) | 0.6222 | nie submitowane | NO SIGNAL — żaden config nie matchuje organizatorów |
| 10 | LogReg combined (kitchen sink) | wszystko + KGW v1 + v2 (68 features) | 0.5852 | nie submitowane | KGW noise zaszkodził mimo C=0.005 |
| 11 | **Fast-DetectGPT** (TODO) | + 10 FDGPT features (Pythia-2.8b) | ? | ? | Implementowane, NIE odpalone |

### Krytyczne obserwacje

1. **OOF (0.63-0.70) NIE koreluje liniowo z leaderboard (0.09-0.10)**. Gap ~6×! To znaczy że:
   - Train+val watermark signature przepuszcza przez branch_bc (data leakage)
   - Test używa innych watermarks/seedów
   - Universal features (bino) generalizują, ale jest ich za mało

2. **Bimodal jest złe, ciągłe jest dobre**. LogReg + standardscaler + L2 regularyzacja są must-have.

3. **branch_bc działa ale słabo** (+0.04 leaderboard). Bigram NIE pomógł = za bardzo specyficzny.

4. **Skalowanie binoculars** (gpt2 → 1.4b/2.8b → 2.8b/6.9b) dało **marginalne** zyski OOF, leaderboard plateau na 0.103.

5. **Direct KGW reference** (replicating torch.randperm) z hash_keys [15485863, 0, 1, 42, 100, 12345, 7, 999] na 3 tokenizerach (gpt2, opt-1.3b, pythia-1.4b) z gammas [0.25, 0.5] i schemes [h1, h2_mul, h2_add] — **WSZYSTKIE NO SIGNAL**. Wniosek: organizatorzy NIE używali default Kirchenbauer ze znanym tokenizerem.

---

## Aktualny stan plików (lokalnie po pull)

`submissions/`:
- `task3_watermark_best.csv` — best historic (LogReg +bc, 0.090)
- `task3_watermark_strong_bino.csv` — current best (0.103)
- `task3_watermark_strong_bino_oldbest.csv` — duplicate save
- `task3_watermark_xl_bino.csv` — XL bino wynik (NIE poprawił 0.103)
- `task3_watermark_kgw_v2.csv` — KGW v2 (no signal)
- `task3_watermark_combined.csv` — kitchen sink (najgorszy OOF)

---

## Niesprobowane / TODO opcje (priorytet z notebookiem)

### 🟢 Już zaimplementowane — odpal:

**1. Fast-DetectGPT** (`main_fdgpt.sh`) — gotowe, NIE odpalone
- Per-position curvature: z = (actual log p - expected log p) / std
- Watermarked text → consistently positive z (green tokens "too predictable")
- Single forward pass na tekst, ~3 min ekstrakcji
- 10 features: mean, std, percentiles, max, min, frac_positive, frac_strong_positive
- **Hipoteza**: orthogonalny do binoculars (curvature vs PPL ratio), powinno dodać +1-3pp

### 🟡 Bardzo prawdopodobnie pomoże (do zaimplementowania):

**2. Stronger branch_a (Pythia-2.8b zamiast gpt2)** — łatwa zmiana
- Aktualne `branch_a` używa gpt2 dla lp_mean, GLTR, etc.
- Recompute z pythia-2.8b da silniejszy LM signal na tych samych metrykach
- Effort: 30 min implementacja + ~10 min ekstrakcja
- Risk: niski, expected gain: +1-3pp

**3. Multi-LM PPL fingerprinting** — średnia zmiana
- Compute PPL pod 5+ LMs (gpt2, gpt2-medium, opt-1.3b, pythia-1.4b/2.8b/6.9b) — wszystkie cached
- Find which LM gives lowest PPL → "natural" generator-LM dla tekstu
- Watermarked text będzie miał bardzo niski PPL pod właściwym LM
- Features: PPL_ratio różnych par, identity najbardziej fitującego LM
- Effort: 60 min implementacja + ~10 min ekstrakcja (już mamy bino_strong/xl features, dodać 1-2 więcej LMs)
- Risk: niski, expected gain: +2-4pp

**4. RoBERTa fine-tuned classifier** — wysoka zmiana
- Fine-tune RoBERTa-base / DeBERTa-v3-base na 540 train samples (warning: NIE w cache! musisz sprawdzić czy można pobrać offline lub skopiować z lokalnego cache laptopa)
- 5-10 epochs, output single probability
- Risk overfit (małe dane), ale pretrained encoder może znać "watermark patterns"
- Effort: 2-3h
- Risk: wysoki overfit, expected gain: niepewne (-2 do +5pp)

### 🟠 Wymaga eksperymentu (uncertain):

**5. Better Liu/Semantic detector**
- Aktualne `branch_d` ma 4 słabe features (adj cosine, LSH KL-div)
- Dodać: sentence-level embedding clustering, sequence anomalies, embedding sequence entropy
- Use stronger embedding model: `sentence-transformers/all-mpnet-base-v2` (chyba też cached, sprawdź)
- Effort: 1-2h
- Risk: medium, expected gain: +1-2pp (jeśli Liu jest 1/3 testu)

**6. Recover green list via watermark stealing**
- Jaki naukowy paper? Watermark Stealing (Jovanović et al. 2024, paper #21 w `references/papers/`)
- Iteracyjnie: oblicz aktualny green list, użyj na nieoznaczonych test → high-confidence pseudo-labels → retrain
- Może rozszerzyć branch_bc z 540 train labels na 540 + N high-confidence test
- Effort: 2-3h
- Risk: medium, gain: niepewne

**7. Train scheme classifier first → route to specialist detector**
- 3-class classifier: KGW vs Zhao vs Liu (ale brak watermark_type w datasetcie)
- Trzeba unsupervised cluster training watermarks (180 wmarked) na 3 grupy
- Następnie 3 specialiści, każdy expert per scheme
- Effort: 2-3h
- Risk: high (clustering może być błędne), gain: potencjalnie game-changing

### 🔴 Niskie priorytety / zmierzwione:

**8. KGW z multi-prefix context (h=3, h=4)** — KGW v3
- Niektóre warianty hashują 3-4 prev tokens
- Memory blowup gigantyczny, czas ekstrakcji godziny
- Mała szansa że match

**9. Multi-tokenizer KGW z LLaMA-2/Mistral**
- Te tokenizery NIE w offline cache
- Trzeba skopiować z laptopa albo TRANSFORMERS_OFFLINE=0 (wymaga internet na compute node, niepewne)
- Wysoka szansa zysku jeśli organizatorzy użyli LLaMA tokenizer (~32k vocab)

**10. Stylometric features**
- Sentence length variance, syntactic complexity, function word ratios
- Mała szansa zysku, watermarki tych metryk nie zmieniają znacząco

---

## Konkretne komendy do skopiowania

### Sprawdź status klastra
```bash
scripts/juelich_exec.sh "squeue -u \$USER --format='%.10i %.20j %.10T %.10M %.10L'"
scripts/juelich_exec.sh "ls -lt /p/scratch/training2615/kempinski1/Czumpers/task3/output/ | head -5"
```

### Zobacz output
```bash
scripts/juelich_exec.sh "cat /p/scratch/training2615/kempinski1/Czumpers/task3/output/<JOBID>.out"
```

### Odpal nowy variant
```bash
git push origin task3
scripts/juelich_exec.sh "cd /p/scratch/training2615/kempinski1/Czumpers/repo-multan1 && git pull && sbatch code/attacks/task3/main_<variant>.sh"
```

### Pull + submit
```bash
just pull-csv task3 submission_<variant>.csv
just submit task3 submissions/task3_watermark_<variant>.csv
```

### Inspekcja CSV (lokalnie)
```bash
python3 -c "
import csv, numpy as np
scores = [float(line.split(',')[1]) for line in open('submissions/task3_watermark_<variant>.csv').readlines()[1:]]
s = np.array(scores)
print(f'rows={len(s)} min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}')
print(f'unique rounded(2): {len(np.unique(np.round(s,2)))}')
for lo,hi in [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    c = ((s >= lo) & (s < hi)).sum(); print(f'  [{lo:.1f},{hi:.1f}): {c}')"
```

Healthy CSV: ~80+ unique rounded(2) values, NIE ma 2 dominujących klastrów. Min powinno być ~0.02-0.05, max ~0.93-0.97.

---

## Architectural decisions worth knowing

### Skala features
- LogReg + StandardScaler **must** — features mają drastycznie różne skale (lp_mean ~ -3, ngram_logdiv_3 ~ 0.5, bino_score ~ -1.5, kgw_z ~ 0.5)
- Bez StandardScaler LogReg nie potrafi przyznać sensownych wag

### Regularyzacja LogReg
- C=0.01 sweet spot w testach
- C=0.005 dla "kitchen sink" (więcej features → silniejsza reg)
- C=0.5 było za słabe → overfit

### Bigram greenlist overfit
- BigramGreenList bardzo dobrze fittuje training (OOF +7pp)
- Ale leaderboard NIE rośnie → unique training pattern, nie generalizuje
- Wniosek: jeśli feature dramatically improves OOF ale nie leaderboard → overfit

### Cache strategy
- Każdy feature module pisze do `features_<name>.pkl`
- `extract_cached(name, texts, fn, cache_dir, force=False)` — load if exist, extract+save else
- `--force-extract` invalidates wszystko (~30+ min recomputation, unikać)
- Multiple jobs mogą czytać te same cache files (read-only safe)
- Multiple jobs mogą pisać DIFFERENT cache files w paralleli (różne nazwy)
- Multiple jobs piszące tę samą cache: race, ostatni wygrywa (rzadkie, do uniknięcia)

---

## Last sanity checks dla nowego Claude

Po pull repo lokalnie:
```bash
git checkout task3
git pull
ls code/attacks/task3/features/  # branch_a/bc/d, binoculars/_strong/_xl, branch_kgw/_v2, fast_detectgpt
ls code/attacks/task3/main_*.sh  # main, p1, full, bigram, strong_bino, kgw, kgw_v2, xl_bino, combined, fdgpt
ls submissions/                  # task3_watermark*.csv (kilka wariantów)
tail -10 SUBMISSION_LOG.md       # historia submitów
```

W razie wątpliwości: czytaj `docs/tasks/task3_watermark_detection.md` (PDF spec) + `docs/STATUS.md` + `CLAUDE.md` (top-level instrukcje, w tym Polish/English convention).

---

## Współpraca równoległa

**Jeśli Ty (drugi Claude) pracujesz równolegle ze mną:**

1. **NIE odpalaj jobów które ja właśnie odpaliłem** — sprawdź `squeue -u $USER` przed sbatch
2. **NIE pisz do tych samych plików cache** — używaj nowej nazwy (`features_yourname.pkl`)
3. **Pracuj na osobnym branchu** jeśli potrzebujesz zmieniać wspólne pliki (main.py, cv_utils.py)
4. **Submit log**: `SUBMISSION_LOG.md` zapisuje każdy submit (md5 CSV → score) — nie konflikt
5. **Output CSV names**: użyj unikalnej końcówki `submission_<your-variant>.csv` żeby nie nadpisać moich

**Najszybsze ścieżki dla równoległej pracy** (sub-30 min implementacja):

- **Stronger branch_a (Pythia-2.8b)** — kopia `branch_a.py` z innym model_name. Cache jako `features_a_strong.pkl`. Add `--use-strong-a` flag.
- **Multi-LM PPL fingerprint** — nowy moduł `features/multi_lm.py`, oblicza min/max/mean PPL nad 4-5 LMs. Cache `features_multi_lm.pkl`.
- **Sentence embedding classifier** — embedding (all-MiniLM-L6-v2) → 384-dim wektor → LogReg z C=0.001 (silnie reg). Może działać niezależnie albo dodać 384 features do główny LogReg.

Powodzenia. Cel: closing 0.103 → 0.27 = trzeba znaleźć fundamentalnie nowy sygnał.
