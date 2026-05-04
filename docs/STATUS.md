# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: 2026-05-04

## Status snapshot
- 25 PDFs in `references/papers/` + 25/25 pre-extracted to `references/papers/txt/` (5.6 MB)
- MAPPING router split: `MAPPING_INDEX.md` (lean, 693w) + `MAPPING.md` (rich, 3099w)
- 7/7 deep research artifacts present in `docs/deep_research/` (07_hackathon_toolkit.md added 2026-04-26)
- 5 challenge specs + QUICKSTART in `docs/practice/`
- Repo has docs + reference papers + token-hygiene config; no code yet
- **Token Optimization Plan Phase 0 + Phase 1 DONE 2026-04-26.** Subagenty + slash commands + Justfile + templates + tests/smoke.py (zielony) gotowe. Per-teammate setup z `docs/SETUP.md` pending

## Active blockers
- Brak boilerplate'u dla atak-implementacji (templates są — skeleton tylko, attack code TODO)
- Brak `code/practice/score_*.py` per-challenge — Justfile referuje ale jeszcze nie istnieją
- Jülich access nieprzetestowany (https://judoor.fz-juelich.de/projects/training2615)
- Zoom info session nieogłoszony przez organizatorów
- Fixture data dla challenge B + C jeszcze nie wygenerowane (wymaga CUDA-teammate)
- `just` musi być zainstalowane na każdym laptopie zespołu (`brew install just`)

## Confirmed (Discord 2026-05-04)
- **3 challenges total** — potwierdzono na Discord przed Zoom info session
- Struktura A/B/C (jeden na osobę) jest właściwa, D/E odpada jako zakres

## Active blockers
- Brak boilerplate'u dla atak-implementacji (templates są — skeleton tylko, attack code TODO)
- score_A/B/C.py gotowe, ale brak attack_A/B/C.py
- Jülich access nieprzetestowany (https://judoor.fz-juelich.de/projects/training2615)
- Zoom info session dziś 17:00 — czekamy na: submission format, metryki, co jest dostarczone
- Fixture data dla challenge B + C jeszcze nie wygenerowane (wymaga CUDA-teammate)

## Open questions (na Zoom dziś 17:00)
- Submission format: REST API, CSV, CLI? → determinuje /submit pipeline
- Primarna metryka: TPR@1%FPR, AUC, inne?
- Co jest dostarczone na start (dane, checkpointy)?
- Czy modele można pre-cachować na Jülich przed May 9th?
- GPU spec na Jülich (VRAM, node limit)?
- Jakie 3 challengi (czy jeden z nich to Zawalski data contamination)?
- Dostęp do internetu / HuggingFace podczas hackathonu?
- Kto bierze który challenge (A/B/C)?
