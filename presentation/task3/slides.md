---
marp: true
theme: default
paginate: true
---

# Task 3 · Detekcja watermarku w tekście LLM

## Co oceniamy

- **Wejście:** 2250 tekstów testowych (bez etykiet).
- **Wyjście:** jeden score \([0,1]\) na tekst — wyżej = silniejsze podejrzenie watermarku.
- **Metryka organizatorów:** **TPR przy FPR = 1%** na tekstach „clean”: praktycznie **ranking** próbek, nie „magiczny” próg 0.5.

## Wyzwanie

W danych są **różne typy** watermarków (Kirchenbauer / Liu / Zhao — sens PDF); jeden wąski detektor nie wystarczy.

---


# Najlepsze rozwiązanie — logika (BEST3)

## Wspólne tło

- **~540** oznakowanych tekstów (train+val) → uczymy **głowicę** (normalizacja + klasyfikator liniowy z regularyzacją).
- **Ta sama szeroka warstwa cech** z wcześniejszej pracy: wiele ekstraktorów (GPT-2 surrogate, Binoculars × warianty, OLMo-7B/1B, krzywizna tokenowa, proxy semantyczne, opcjonalnie listy z treningu) **sklejone w jeden wektor** + **6 pochodnych cross-LM** (różnice / ilorazy między modelami).

## Dwie głowice → jeden ranking

| Gałąź | Dodatkowy sygnał |
|--------|-------------------|
| **A (o7be)** | Pełny rozkład następnego tokenu pod **OLMo-7B-Instruct**: entropie, rangi tokenów, bursty niskiej entropii. |
| **B (kgwx)** | Wiele wariantów **z-score „zielona lista”** (różne tokenizacje / \(\gamma\) / kontekst). |

**Fuzja:** na teście **nie** średnia surowych score’ów — tylko **średnia rang** obu głowic, potem rozciągnięcie do \([0.001, 0.999]\) → plik CSV pod API.

---


# Ewolucja wyniku (public leaderboard)

- Wykres: otwórz **`evolution-chart.html`** w przeglądarce.
- **Uwaga:** pierwsze słupki to **potwierdzone** mileston’y z `docs/tasks/task3_submissions_tracker.md`. Słupek **BEST3** jest **szacunkowy** (orientacyjnie między `cross_lm` a liderem tablicy z snapshotu; dokładna liczba z API nie jest zapisana w repo).

## Co powiedzieć na 30 s

„Najpierw black-box i OLMo dały skok na LB, cross-LM v1 ustabilizował ~0.28. Finalnie połączyliśmy dwie głowice — entropie rozkładu i testy listowe — **rankingowo**, żeby trafić w TPR@1%FPR.”
