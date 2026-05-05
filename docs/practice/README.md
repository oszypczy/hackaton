> ⚠️ **ĆWICZENIA PRZED HACKATONEM — NIE faktyczne zadania.**
> Challenges A/B/C to nasze własne mock-taski do nauki technik. Faktyczne zadania hackathonu
> zostaną ujawnione 2026-05-09 o 12:00 i mogą się znacząco różnić (format, metryki, dane).
> Patrz `docs/STATUS.md` po potwierdzone fakty z Zoom info session.

# Practice Challenges — przygotowanie do CISPA Hackathon (Warsaw 2026-05-09/10)

Trzy mock-challengi do samodzielnego rozwiązania w ciągu 2 tygodni przed hackathonem.
Każdy zakotwiczony w **jednym z 4 obowiązkowych paperów** które organizatorzy
(Adam Dziedzic, Franziska Boenisch, SprintML Lab) podali w mailu.

## Zasady

- Każda osoba wybiera jeden challenge i robi go solo (4–8h).
- Format CTF-style: dane wejściowe → zdefiniowane wyjście → numeryczny score.
- Trzy poziomy ambicji per challenge: **easy baseline** (1h, żeby coś było), **solid** (główny cel), **hard mode** (jak ktoś skończy wcześniej).
- Self-grading — każdy odpala scoring lokalnie. Nie ma centralnego serwera (opcja 1 z dyskusji).
- Po skończeniu: krótki debrief w zespole, co bolało, co było zaskoczeniem.

## Mapowanie challenge ↔ paper ↔ osoba

| Challenge | Paper | Sugerowana osoba |
|---|---|---|
| [A — LLM Dataset Inference](challenge_A_dataset_inference.md) | Maini et al. NeurIPS 2024 | Privacy expert |
| [B — LLM Watermark Detect & Remove](challenge_B_watermark.md) | Kirchenbauer et al. ICML 2023 | Watermark/stealing expert |
| [C — Diffusion Memorization Discovery](challenge_C_diffusion_extraction.md) | Carlini et al. USENIX 2023 | Adversarial/general expert |

## Środowisko (wspólne)

Wszyscy mamy MacBooki M4. **Brak CUDA — jest tylko MPS i MLX.**

```bash
# Setup raz na laptopie
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip

# Wspólny stack
pip install torch torchvision  # PyTorch z MPS support
pip install transformers datasets accelerate
pip install scikit-learn scipy numpy pandas matplotlib
pip install sentence-transformers  # do embedding similarity

# Per-challenge:
# A: nic dodatkowego
# B: pip install bert-score nltk
# C: pip install diffusers Pillow open-clip-torch

# (Opcjonalnie, dla LLM inference ~3x szybciej niż MPS):
pip install mlx mlx-lm
```

## Czego NIE robimy lokalnie na M4

- Trenowania DDPM/SD od zera (godziny → dni).
- Generowania watermarkowanych korpusów Llama-3-8B w fp16 (za mało pamięci).
- Trenowania shadow modeli > 1B params.

Te artefakty są **dostarczane jako fixture data** (zip do pobrania). Generowane raz na Jülich albo Colab przed praktyką — patrz sekcja "fixture data" w każdym challengu.

## Sugerowany timeline

- **Do 2026-05-01:** każdy ma działający setup, ściągnięte fixture data, easy baseline odpalony
- **2026-05-02 lub 2026-05-03:** każdy robi swój challenge w 8h (timer)
- **2026-05-04:** debrief, identyfikacja bolesnych punktów, doszlifowanie wspólnego boilerplate
- **2026-05-05 do 2026-05-08:** odpoczynek + cache modeli na Jülich
- **2026-05-09:** hackathon

## Important caveat

Te 3 challengi to **moje przewidywanie** czego organizatorzy mogą oczekiwać, oparte na 4 paperach które dali. Realne zadania będą prawdopodobnie blisko, ale **nie identyczne**. Cel praktyki to nie 1:1 powtórzenie, tylko zbudowanie odruchów: szybko czytać paper, replikować eksperyment, optymalizować pod metryke, submitować iteracyjnie.

## SprintML evaluation style — co wszystko musi mieć

Z analizy researchy 04/05/06 (Claude Research) + 12 paperów SprintML 2022–2026 wyłania się stały wzorzec ewaluacyjny używany przez Dziedzica/Boenisch:

- **Primary metric: TPR@FPR=1% (czasem 0.1%)** — pioneered przez Carlini LiRA, używane uniwersalnie. AUC tylko jako secondary.
- **p-value z explicite hypothesis testu** — threshold zwykle p < 0.1, **zero false positives** na held-out independent models.
- **Calibrated likelihood ratios** zamiast hard threshold.
- **Shadow models** to nie opcja — bez nich atak jest "weak attack" (cytat z Strong MIAs, Hayes et al. NeurIPS 2025).
- **Distribution shift jest CONFOUNDEREM**, nie sygnałem — Maini-Dziedzic LLM-DI eksplicite o tym pisze.

Każdy challenge `A/B/C` ma teraz w scoringu wymóg TPR@1%FPR + p-value, nie tylko AUC.

## Submission rules (z research 04, sekcja "24-hour playbook")

Te zasady pojawiają się w każdym papierze SprintML i w wywiadach:

1. **Optimize fidelity, NOT task accuracy.** Surrogate Acc=92% Fid=78% przegrywa z Acc=85% Fid=88% na każdym SprintML leaderboard.
2. **Pre-cache every query do dysku (SQLite/joblib).** Architecture changes nie powinny burnować budgetu.
3. **EMA model > live model** — zawsze submitować EMA shadow weights, +1–2 fidelity points.
4. **Submit early, submit often.** Leaderboard to twoja diagnostyka.
5. **Hour 0–1**: zidentyfikuj scoring function. Hour 1–3: random baseline submit. Hour 3–8: KD surrogate. Iteracje co cykl.
6. **Don't experiment in last hour.** Submit best EMA, checkpoint-5 jako backup.

## Optional challenges (do rozważenia)

Poza A/B/C podstawowymi, deep research silnie sugeruje że na hackathonie mogą pojawić się:

- **Property Inference / Fairness Audit** (Barcelona miał) — patrz `challenge_D_property_inference.md` (opcjonalny).
- **Model Stealing / Encoder Extraction** (B4B-style, SprintML core) — patrz `challenge_E_model_stealing.md` (opcjonalny).

Trzymanie się 3 challengeów dla 3 osób = OK. Ale jeśli ktoś chce drugą rundę praktyki, D/E są tutaj.
