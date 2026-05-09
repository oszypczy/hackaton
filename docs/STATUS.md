# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: 2026-05-09 (dzień hackathonu)

## Status snapshot
- 20 PDFs in `references/papers/` + 20/20 pre-extracted to `references/papers/txt/` (3.9 MB)
  - 2026-05-09: dodano 05/06/07 — Task 1 PDF references (Tong DUCI ICLR'25, Maini DI'21, Dziedzic DI-SSL'22)
- MAPPING router split: `MAPPING_INDEX.md` (lean, 693w) + `MAPPING.md` (rich, 3099w)
- 7/7 deep research artifacts present in `docs/deep_research/`
- `.venv` (python 3.11) z requirements.txt — działa na M4
- Repo wyczyszczony z practice challenges przed hackathonem

## Confirmed (Zoom info session 2026-05-04)

### Format hackathonu
- **3 taski** ujawniane naraz o 12:00 w sobotę, hackathon 12:30–12:30 (24h)
- Każdy team dostaje **osobny API token** do submisji
- **Live scoreboard** per task + overall ranking
- Submission: **REST API, pliki CSV**, cooldown 5 min (2 min przy failed submission)
- Top 6 teamów prezentuje rozwiązania (6 min / team = 2 min / task)
- **Prezentacja liczy się do rankingu** — jury degraduje teamy które nie rozumieją własnego rozwiązania

### Taski (potwierdzone tematycznie ze slajdów)
1. **Data identification** — LLM Dataset Inference (Maini et al.)
2. **Data memorization** — ekstrakcja danych z modeli generatywnych (Carlini et al.)
3. **Watermarking** — detekcja/atak na watermark LLM (Kirchenbauer et al.)

Zawalski (data contamination) **NIE był omówiony** w części technicznej → prawdopodobnie nie jest osobnym taskiem.

### Compute — Jülich (zweryfikowane 2026-05-08 z `kempinski1@jrlogin05.jureca`)
- **800 GPU dostępne**, partycja: **`dc-gpu`** (faktyczna nazwa slurm: lowercase z hyphen)
- Dostępne też: `dc-gpu-devel` (debug), `dc-gpu-large` (więcej GPU per node)
- Węzły: **4× A800 GPU per node**
- Projekt ID: `training2615` (PI: `herten1`, project-type C)
- Aktywacja: `jutil env activate -p training2615` — ustawia `$PROJECT=/p/project1/training2615`, `$SCRATCH=/p/scratch/training2615`
- Submit job: `sbatch main.sh` (skrypt bash z `#SBATCH --partition=dc-gpu --account=training2615`)
- Rekomendowany package manager: **UV**
- SSH host: **`jureca.fz-juelich.de`** (login nodes: `jrlogin0X.jureca`)
- MFA: TOTP — workaround: `scripts/juelich_connect.sh` (socket 4h)

### Dane i modele
- Dostarczone na start: PDF z opisem taska + dane + checkpointy modeli
- Źródła: **HuggingFace** (linki) + **Jülich** (bezpośrednio do skopiowania)
- Dozwolone zewnętrzne datasety — brak ograniczeń

### AI tools
- **W pełni dozwolone**: Claude, GPT, Copilot, Cursor, Codex i wszystko inne
- Organizatorzy oczekują że rozumiecie co robicie — weryfikowane na prezentacji

## Confirmed (Morning presentation 2026-05-09)

### 🎯 Final scoring na EXTENDED test sets
- **Przed ogłoszeniem wyników jutro (2026-05-10)** organizatorzy przepuszczą wszystkie metryki przez **rozszerzone wersje test datasetów**, których team nie widzi w trakcie hackathonu.
- **Implikacja:** maksymalizacja score'u na live scoreboard ≠ wygrana. Liczy się **ogólność rozwiązania**.
- **Praktyka:**
  - cross-walidacja na slicach (by-class, by-arch, by-prompt-length) — wariancja = ryzyko
  - unikać greedy tuningu progów/wag pod public test
  - flagować slice'y na których metoda się wyłamuje
- Pełne wytyczne i konsekwencje → patrz `CLAUDE.md` sekcja "Working principles" (bullet z 🎯).

## Cluster setup (2026-05-09 14:19)

- Owner setup zrobiony przez kempinski1: `source hackathon_setup.sh` z `TEAM_FOLDER="Czumpers"`
- Shared folder `/p/scratch/training2615/kempinski1/Czumpers/` ma ACL na 4 osoby
  (kempinski1, szypczyn1, multan1, murdzek2 — rwx) + lockdown na others
- 3 datasety pobrane:
  - `DUCI/` (Task 1)
  - `P4Ms-hackathon-vision-task/` (Task 2)
  - `llm-watermark-detection/` (Task 3)
- 3 venv'y zbudowane (uv 0.11, Python 3.12, PyTorch 2.11 + CUDA 13)
- Owner clone: `Czumpers/repo-kempinski1/` (git po SSH przez `ssh.github.com:443`)
- Teammate setup: `docs/JURECA_TEAMMATE_SETUP.md` — szypczyn1/multan1/murdzek2 do wykonania samodzielnie

## Submission (potwierdzone z templates organizatorów)

- BASE_URL: `http://35.192.205.84` (port 80)
- Endpoint: `POST /submit/<TASK_ID>` z header `X-API-Key`, multipart `file`
- TASK_IDs: `11-duci` / `27-p4ms` / `13-llm-watermark-detection`
- Klucz API w `.env` lokalnie (gitignored, var: `HACKATHON_API_KEY`)
- Submit z laptopa przez `just submit <task> <csv>` (skrypt: `scripts/submit.py`)
- Pull CSV z klastra: `just pull-csv <task>` (skrypt: `scripts/pull_csv.py`)
- Pełen flow → `docs/SUBMISSION_FLOW.md`

## Active blockers
- Faktyczne taski znane od 12:00 2026-05-09 — patrz `docs/tasks/*.md`
- Wpisanie `HACKATHON_API_KEY` do `.env` (user ma klucz, nie zapisany jeszcze)
- Setup teammate'ów (3 osoby) — instrukcja: `docs/JURECA_TEAMMATE_SETUP.md`

## Task 2 — Current State (2026-05-09 evening)

### Leaderboard standings (ostatni odczyt ~20:50)
- **task1=0.053333, task2=0.381195, task3=0.103064** (team: Czumpers)
- Task 2: #1 na leaderboardzie (potwierdzono z scraping `/leaderboard_page`)

### Score historia Task 2
| Submission | CSV | Score | Uwagi |
|---|---|---|---|
| ~17:27 | task2_shadow_old_fixed.csv | FAILED | format/walidacja |
| ~18:36 | task2_shadow_baseline_20260509_203603.csv | ~0.347 | pełne zdania vs GT raw PII |
| ~18:50 | task2_shadow_pii_only_v2_204928.csv | **0.381195** | POST-PROCESSED — regex PII extraction |
| ~19:16 | task2_shadow_hybrid_pii_only_205037.csv | brak poprawy | A+B hybrid gorzej niż baseline |

### Kluczowe odkrycie: GT format
- **Server GT = raw PII values**, np. `john.doe@example.com`, `+12312312312`, `1234 5678 9012 3456`
- **Val parquet GT = pełne zdania**, np. `"You can reach Gabriella Johnson on +13859159897."`
- Lokalna eval dawała 0.897 — mylące, bo miała pełne zdania vs pełne zdania
- Rozwiązanie: **regex post-processing** żeby wyciągnąć surowe PII z predykcji modelu

### Post-processing — extract_pii()
Plik: skrypt był inline, MUSI być zapisany jako `code/attacks/task2/shadow/extract_pii.py`:
```python
import re

def extract_pii(pred: str, pii_type: str, min_len: int = 10) -> str:
    orig = pred.strip()
    if pii_type == 'EMAIL':
        m = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', pred)
        result = m.group(0) if m else orig
    elif pii_type == 'CREDIT':
        m = re.search(r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b', pred)
        if m:
            result = m.group(1).strip()
        else:
            m = re.search(r'\b(\d{4}(?:[\s\-]\d{2,4}){2,3})\b', pred)
            result = m.group(1).strip() if m else orig
    elif pii_type == 'PHONE':
        m = re.search(r'\+\d[\d\s\-\(\)]{8,14}\d', pred)
        if m:
            result = re.sub(r'[\s\-\(\)]', '', m.group(0))
        else:
            m = re.search(r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b', pred)
            result = m.group(0) if m else orig
    else:
        result = orig
    if len(result) < min_len:
        result = orig[:100]
    return result[:100]
```

### Znane bugi / ograniczenia
1. **PHONE/CREDIT bug** w `find_conv_turn()` (`attack_shadow.py`): keyword "number" pasuje do "credit card number" → 400/1000 PHONE próbek generuje zdania CREDIT-style (długie, zawierają numer karty). Wymaga re-inference żeby naprawić.
2. **A+B hybrid nie działa**: `[REDACTED]` prefix trick powoduje garbage output dla 344 EMAIL + 743 PHONE wierszy (model halucynuje szablony danych zamiast uzupełniać PII).
3. **213 non-redacted samples**: organizer potwierdził — 213 próbek testowych ma widoczne PII (nie jest zredagowane) i **nie liczą się do finalnego score**. Traktuj jako dodatkowy validation set.

### Scoring rules (potwierdzone)
- Metryka: `1 − Normalized_Levenshtein` (match z rapidfuzz `Levenshtein.normalized_distance`)
- Public 30% / Private 70% split
- Score updateuje się tylko jeśli jest wyższy od dotychczasowego best
- API nie zwraca score w response body — sprawdzaj `/leaderboard_page`

### Best submission
- `submissions/task2_shadow_pii_only_v2_204928.csv` → server score **0.381195**
- 3000 wierszy, post-processed baseline (greedy-only, bez redacted prefix)

### Następne kroki (aby poprawić)
1. Naprawić PHONE/CREDIT bug w `find_conv_turn()` → re-inference na Jülichu
2. Sprawdzić 213 non-redacted próbek — które IDs, wyciągnąć GT, walidować extraction
3. Próbować różnych promptów (path A: kempinski1) — benchmark to 0.381195
4. Ewentualnie: fine-tune ekstrakcji na validation parquet (840 próbek GT)

### scripts/submit.py — zmodyfikowany
Dodano `_scrape_leaderboard()` która po każdym udanym submit robi GET `/leaderboard_page`
i parsuje HTML table → loguje `leaderboard task1=X task2=Y task3=Z` do SUBMISSION_LOG.md.

## Task 3 — Current State (2026-05-09 ~21:45)

### Leaderboard standings
- **task3=0.103064** (Czumpers) — lider ma **0.27** (gap 2.6×)
- 3 typy watermarku: **KGW/Kirchenbauer + Liu/Semantic + Zhao/Unigram**

### Score historia Task 3
| Submission | CSV | Score | Uwagi |
|---|---|---|---|
| ~14:42 | task3_watermark.csv | ~0.05 | LightGBM branch_a only |
| ~15:44 | task3_watermark.csv | ~0.09 | LogReg +bc baseline |
| ~15:58 | task3_watermark.csv | **0.103** | LogReg +bc +strong_bino |
| PENDING | task3_watermark_fdgpt.csv | TBD | Fast-DetectGPT — ściągnięty lokalnie |

### CSV do submitowania
- `submissions/task3_watermark_fdgpt.csv` — **gotowy lokalnie**, dystrybucja zdrowa (2250 rows, 96 unique vals, range 0.023–0.966)
- Submit: `python3 scripts/submit.py task3 submissions/task3_watermark_fdgpt.csv`

### Architektura (multan1's code, branch task3)
Klaster: `/p/scratch/training2615/kempinski1/Czumpers/repo-multan1/code/attacks/task3/`
```
main.py          # orchestrator: load → features → OOF + LogReg → submission
features/
  branch_a.py   # gpt2 log-prob stats, GLTR, ngram, burstiness, gzip, TTR
  branch_bc.py  # UnigramGreenList (Zhao) + BigramGreenList (KGW) — scheme-specific
  branch_d.py   # sentence-transformers cosine sim — Liu proxy
  binoculars_strong.py  # Pythia-1.4b + 2.8b PPL ratio
  branch_kgw_v2.py      # Direct KGW (NO SIGNAL — hash_key unknown)
  fast_detectgpt.py     # Pythia-2.8b analytical curvature — NAJNOWSZE
```

### Kluczowe wnioski multan1
1. **OOF 0.65 ≠ leaderboard 0.10** — gap 6× przez data leakage branch_bc
2. **Skalowanie Binoculars plateau** na 0.103 (GPT-2 → Pythia-6.9b nie pomaga)
3. **Direct KGW zero sygnału** — organizatorzy użyli nieznanego hash_key + tokenizera
4. **Classifier: LogReg C=0.01 + StandardScaler** — LightGBM daje bimodal collapse
5. **Healthy CSV**: 80+ unique(round(score,2)), min ~0.02, max ~0.95 (nie 2 klastry)

### Następne kroki (priorytet)
1. **Submit fdgpt CSV** — już ściągnięty, 5 min cooldown
2. **Stronger branch_a** — kopia branch_a.py z Pythia-2.8b, cache `features_a_strong.pkl`
3. **Multi-LM PPL fingerprint** — min/max/mean PPL nad 4-5 LMs, cache `features_multi_lm.pkl`
4. **LLaMA tokenizer dla KGW** — NIE w offline cache, trzeba skopiować ręcznie

### Branch workflow
- Nasz branch: `task3-murdzek2` (stworzony z origin/task3)
- Multan1's branch: `task3` — NIE commituj tam
- Klaster clone murdzek2: `/p/scratch/training2615/kempinski1/Czumpers/repo-murdzek2/`
- Pull CSV: `python3 scripts/pull_csv.py task3 submission_<name>.csv`
- Pełny kontekst: `docs/tasks/task3_session_context.md` (na branchu task3)

## Task 1 (DUCI) — extracted facts (research session 2026-05-09)

### Paper 05 (Tong 2025) — core method confirmed
- **Equation 4 (debias):** `p̂_i = (m̂_i − FPR) / (TPR − FPR)`, then `p̂ = mean(p̂_i)`. TPR/FPR estimated **globally** (one value across X), not per-i.
- **Single-reference-model is enough.** With 1 ref model on CIFAR-100/WRN28-2: max MAE ≈ 0.087. With 8 ref: 0.055. With 42 ref: 0.034.
- ResNet-34 / CIFAR-100 numbers: max MAE 0.053 (1 ref) → 0.015 (42 ref).
- MIA backbone: **RMIA** (Zarifzadeh 2024). Single-ref variant uses linear approx `a=0.3` for `Pr(x|θ)/Pr(x)` denominator.
- Per-record correlations between p̂_i are **negligible** (paper's Figure 4) — pairwise debiasing gains minimal. Stick with naive average.
- **Special sampling penalty:** non-i.i.d. selection (e.g., EL2N coreset) bumps MAE from 0.062 → 0.109. PDF says i.i.d. so we're fine (verify post-download).

### Reference implementation — DUCI repo CONFIRMED
- Repo: `github.com/privacytrustlab/ml_privacy_meter` (master branch)
- Files:
  - `run_duci.py` — entrypoint, expects to train target models internally
  - `modules/duci/module_duci.py` — DUCI class
  - `configs/duci/cifar10.yaml` — default config
  - `demo_duci.ipynb` — demo notebook
  - `documentation/duci.md` — docs
- Default config: WRN28-2, CIFAR-10, SGD lr=0.1 wd=0, 50 epochs, batch 256, RMIA, num_ref_models=1
- Pipeline: train pairs of models on complementary 50% splits → compute softmax signals on auditing dataset + population → MIA scores → debias
- **Adaptation needed:** plug pretrained organizer checkpoints as "target" instead of training from scratch (ref models still need training)

### Paper 06 (Maini DI 2021) — fallback, shadow-free
- **Blind Walk** (black-box): k random directions × {uniform, Gaussian, Laplace} → 30-dim embedding of distances-to-misclassification
- **MinGD** (white-box): gradient descent to nearest decision boundary, ℓ1/ℓ2/ℓ∞ norms
- Confidence regressor: 2-layer linear with tanh, then Welch t-test
- Tested on CIFAR-10/100/SVHN/ImageNet with WRN-28-10
- **Use as backup** if RMIA / Tong debias underperforms — does NOT require shadow models

### Paper 07 (Dziedzic SSL DI 2022) — methodologically organizers' style, BUT
- GMM density estimation on **encoder representations** (SSL-only paradigm)
- Not directly applicable to supervised classifiers (we have labels)
- **Skip for Task 1**, but note: organizers (Dziedzic+Boenisch) likely value statistical/density approaches.

### Implementation budget estimate (3 archs × ≥1 ref model)
- ResNet18 / CIFAR-100, 50ep, A800: ~30min
- ResNet50 / CIFAR-100, 50ep, A800: ~1.5h
- ResNet152 / CIFAR-100, 50ep, A800: ~3-4h
- **1 ref per arch = ~5-6h total** (sequential) on single A800; parallel on 4× A800 → ~1.5h elapsed
- Open question (in research prompt): does arch-mismatched RMIA work? If yes, 1 ResNet18 ref might cover all 9 targets.

### Deep research prompt
- Saved to `docs/tasks/task1_research_prompt.md` — gotowy do wklejenia do Claude Deep Research
