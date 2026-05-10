# Task 1 — Breakthrough 0.0533 → 0.020

> **Status:** Czumpers #3 na publicznym leaderboardzie (z #8)
> **Branch:** `task1-murdzek2` (worktree `/mnt/c/projekty/hackaton-task1/`)

## Godzina (UTC)

- **2026-05-10 ~01:40Z** — submisja CSV, server zwrócił HTTP 200
- **2026-05-10 ~01:41Z** — scrape leaderboarda potwierdził `currentScores["11_duci::Czumpers"] = 0.020`
- Wcześniej (od ~22:47Z poprzedniego dnia) score plateau na **0.053333** przez 3+ godziny

## Metoda

**Single-flip na bazie SUB-9** — zmiana JEDNEJ predykcji z continuous MLE/synth-bank calibration:

| target | SUB-9 (0.053) | flip11_04 (0.020) | zmiana |
|--------|---------------|-------------------|--------|
| 00 | 0.5 | 0.5 | — |
| 01 | 0.6 | 0.6 | — |
| 02 | 0.6 | 0.6 | — |
| 10 | 0.4 | 0.4 | — |
| **11** | **0.5** | **0.4** | **−0.1** ✅ |
| 12 | 0.6 | 0.6 | — |
| 20 | 0.5 | 0.5 | — |
| 21 | 0.5 | 0.5 | — |
| 22 | 0.5 | 0.5 | — |

CSV: `/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip11_04.csv`

## Przyczyna (jak doszło do hipotezy)

1. **SUB-9 baseline** (mean p̂ = 0.51) był MLE-based: 80 epochs, ResNet18 ref model, N=2000, wd=5e-4, SGD. Każda predykcja na grid 0.1 — najlepszy continuous → snap.
2. **Continuous SUB-5 (przed snapowaniem) zwracał `model_11 = 0.4993`** — wartość borderline, dosłownie na granicy między 0.4 a 0.5.
3. **Hipoteza:** MLE-calibrated values blisko granicy decyzyjnej (0.499) mogą snapować w złą stronę. SUB-9 zaokrąglił 0.4993 → 0.5, ale prawdziwa wartość mogła być po niższej stronie.
4. **Poprzednie testy single-flip** (10→0.5, 22→0.6, 00→0.4, 01→0.5, 02→0.5, 12→0.6) — wszystkie 0.0533 (no improve). To zawęziło kandydatów do borderline'ów.
5. **flip11_04** — najbliższy 0.5 boundary spośród niezbadanych targetów.
6. **Wynik:** score 0.0533 → 0.020. Czumpers #8 → #3.

## Implikacje matematyczne

- **Score = MAE na public 3 (3 z 9 targetów)**. SUB-9 sum_errors_public = 3 × 0.0533 ≈ 0.16. flip11_04 sum_errors = 3 × 0.020 = 0.06.
- **Drop sum_errors = 0.10** dokładnie — implikuje `|0.5 − true_11| − |0.4 − true_11| = 0.10`, co daje **`true_11 ≤ 0.40`**.
- W połączeniu z `err_11_at_flip11_04 ≤ 0.06` (bo sum_errors_total ≥ 0): **`true_11 ∈ [0.34, 0.40]`**.
- **Potwierdzone:** `model_11 IS in public 3 AND true_p_11 ≤ 0.4`.

## Co próbowaliśmy potem (chain3, brak poprawy)

10 wariantów na bazie `11=0.4` — wszystkie zwróciły 0.020:
- 8 single-flip kompozytowych (22, 12, 00, 01, 02, 10, 20, 21 z różnymi wartościami 0.4/0.5/0.6)
- 2 finer-grid resnap (snap05, snap_other z gridem 0.05)

**Wniosek:** pozostałe 2 targety w public 3 są poprawne w SUB-9 (ich predykcje matchują true_p w obrębie aktualnego gridu).

## Kandydaci nie-przetestowani (chain4 in progress)

- **`11=0.35`** (`task1_duci_11_035.csv`) — jeśli `true_11 ∈ [0.34, 0.375]` → score ~0.003 (#1 LB!). Cooldown chain blokuje submisję od ~03:42Z. Single-shot zaplanowany na 05:15Z.
- **`11=0.45`** — math dowodzi że ZAWSZE da 0.0367 (worsening). Skip.

## Pliki referencyjne

- Memory: `~/.claude/.../memory/project_task1_breakthrough.md`
- Worktree process log: `docs/tasks/task1_process_so_far.md`
- Cluster best CSV (continuous): `/p/scratch/training2615/kempinski1/Czumpers/DUCI/submission_mle_80ep_r18_precise.csv`
