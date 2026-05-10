# Presentation — Task 3 (BEST3)

## Pliki

| Plik | Opis |
|------|------|
| `slides.md` | Treść **3 slajdów** (format [Marp](https://marp.app/) — separatory `---`); możesz też wklejać sekcje do Google Slides / Keynote. |
| `data_flow_task3.png` | **High-level data-flow diagram (English):** shared feature layer → two heads (full-distribution OLMo-7B-Instruct vs multi-variant green-list tests) → rank fusion → CSV. No internal code symbol names; model names kept as labels. |

## BEST3 — szacunek na wykresie

W repozytarium nie ma zapisanej dokładnej wartości API dla `blend_kgwx_o7be_BEST3.csv`. Na wykresie przyjęto **0.30** jako wartość **między** ostatnim pewnym punktem zespołu `cross_lm (~0.284)` a **górną granicą** snapshotu leaderboardu (Syntax Terror ~0.40). Jeśli znasz realny wynik z tablicy — edytuj tablicę `MILESTONES` w `evolution-chart.html`.
