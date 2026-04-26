# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: 2026-04-26

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

## Open questions
- Czy bierzemy 3 czy 5 challengeów (D + E opcjonalne)?
- Carlini vs CDI dla Challenge C?
- Kto bierze który challenge (A/B/C)?
- Kiedy mini-hackathon (sugerowane: weekend 2026-05-02/03)?
