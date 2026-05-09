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
