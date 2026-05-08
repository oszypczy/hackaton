# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: 2026-05-08 (po implementacji Challenge B)

## Status snapshot
- 25 PDFs in `references/papers/` + 25/25 pre-extracted to `references/papers/txt/` (5.6 MB)
- MAPPING router split: `MAPPING_INDEX.md` (lean, 693w) + `MAPPING.md` (rich, 3099w)
- 7/7 deep research artifacts present in `docs/deep_research/`
- 5 challenge specs + QUICKSTART in `docs/practice/`
- Token Optimization Plan Phase 0 + Phase 1 DONE 2026-04-26
- score_A/B/C.py gotowe
- attack_A.py (Min-K%++) + attack_B1.py + attack_B2.py gotowe; **attack_C.py brak**
- Fixture data: A + B gotowe; **C brak (wymaga GPU)**
- `.venv` (python 3.11) z requirements.txt — działa na M4

## Baselines (mock challenges, zalogowane w SUBMISSION_LOG.md)
- A: AUC=0.873 (Min-K%++ + reference model + Welch t-test)
- B1: F1=0.9588 (Kirchenbauer z-score + sliding window 100/50, threshold z>4)
- B2: score=0.8263 (ZWSP after every word; evasion 96%, BERTScore 0.864)

## Confirmed (Zoom info session 2026-05-04)

### Format hackathonu
- **3 taski** ujawniane naraz o 12:00 w sobotę, hackathon 12:30–12:30 (24h)
- Każdy team dostaje **osobny API token** do submisji
- **Live scoreboard** per task + overall ranking
- Submission: **REST API, pliki CSV**, cooldown 5 min (2 min przy failed submission)
- Top 6 teamów prezentuje rozwiązania (6 min / team = 2 min / task)
- **Prezentacja liczy się do rankingu** — jury degraduje teamy które nie rozumieją własnego rozwiązania

### Taski (potwierdzone tematycznie ze slajdów)
Trzy obszary omówione technicznie = trzy taski:
1. **Data identification** — LLM Dataset Inference (Maini et al.)
2. **Data memorization** — ekstrakcja danych z modeli generatywnych (Carlini et al.)
3. **Watermarking** — detekcja/atak na watermark LLM (Kirchenbauer et al.)

Zawalski (data contamination) **NIE był omówiony** w części technicznej → prawdopodobnie nie jest osobnym taskiem.

### Compute — Jülich
- **800 GPU dostępne** podczas hackathonu, partycja: `DCGPU`
- Węzły: **4× A800 GPU per node**
- Projekt ID: `training2615`
- Aktywacja: `jutil env activate -p training2615`
- Submit job: `sbatch main.sh` (skrypt bash z `#SBATCH --partition=DCGPU`)
- Rekomendowany package manager: **UV** (blazingly fast)
- SSH config host: `hdfml` (login node Jülicha)

### Dane i modele
- Wszystko dostarczone na start: PDF z opisem taska + dane + checkpointy modeli
- Źródła: **HuggingFace** (linki) + **Jülich** (bezpośrednio do skopiowania)
- Dozwolone zewnętrzne datasety — brak ograniczeń

### AI tools
- **W pełni dozwolone**: Claude, GPT, Copilot, Cursor, Codex i wszystko inne
- Organizatorzy oczekują że rozumiecie co robicie — to weryfikowane na prezentacji

## Active blockers
- Jülich SSH setup + test połączenia (każda osoba osobno, wymaga MFA)
- Fixture data dla challenge C (wymaga GPU — Jülich lub CUDA-teammate)
- Brak attack_C.py (Carlini lub CDI extraction)
- Każdy musi mieć UV zainstalowane (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Open questions (pozostałe)
- Kto bierze który task (A/B/C)?
- Czy Jülich access działa dla wszystkich członków zespołu?
- Dokładny format CSV per task (poznamy dopiero w sobotę o 12:00)
