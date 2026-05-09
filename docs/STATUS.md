# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: 2026-05-09 (dzień hackathonu)

## Status snapshot
- 25 PDFs in `references/papers/` + 25/25 pre-extracted to `references/papers/txt/` (5.6 MB)
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
