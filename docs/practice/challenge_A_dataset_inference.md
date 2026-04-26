# Challenge A — LLM Dataset Inference

**Paper:** Maini, Jia, Papernot, Dziedzic — *LLM Dataset Inference: Did you train on my dataset?* (NeurIPS 2024)
**Repo:** https://github.com/pratyushmaini/llm_dataset_inference
**Sugerowana osoba:** privacy expert
**Czas:** 4–8h
**Typowa rola na hackathonie:** dataset inference / membership inference attack

## Problem

Dostajesz dostęp do pre-trained LLM i dwa kandydujące zbiory dokumentów. **Zdecyduj który (jeśli któryś) był w training setcie modelu**, używając tylko zapytań do modelu.

To replikuje setup z paperu Mainiego: pojedyncze MIA są zawodne, ale agregacja kilku słabych sygnałów na poziomie datasetu daje statystycznie istotny test.

## Setup

### Target model
- **Pythia-410M** (`EleutherAI/pythia-410m`) — działa na MPS w fp16, ~820MB RAM
- Pythia była trenowana na **The Pile** — to publiczne, więc znamy ground truth

### Fixture data
Dwa zbiory po 1000 dokumentów każdy:

- **Set IN:** subset The Pile (`monology/pile-uncopyrighted` na HuggingFace) — był w treningu Pythii
- **Set OUT:** subset CC-News z 2024 (`cnn_dailymail` post-cutoff albo `RealTimeData/News_2024`) — pewnie nie był

Każdy dokument: 200–500 tokenów. Dataset format: `data/A/in.jsonl`, `data/A/out.jsonl` z polami `{"id": int, "text": str}`.

**Skrypt do przygotowania fixtures** (do uruchomienia raz, dystrybuowany jako gotowy zip) — pobiera streamingiem 1000 docs z Pile, 1000 z CC-News, zapisuje do JSONL. Implementacja banalna, ~20 linii z `datasets.load_dataset(..., streaming=True)`.

### Validation set
Trzeci zbiór 200 dokumentów `data/A/val_in.jsonl` z innego splitu Pile — używany do kalibracji statystycznej (Maini wymaga reference distribution).

## Zadanie

Dla **każdego z 2000 dokumentów** wyprodukuj:
```json
{"id": int, "set": "in" | "out", "score": float, "is_member": bool}
```
Plus na poziomie **datasetu** (jak w Mainim):
```json
{"set_id": "set1" | "set2", "p_value": float, "verdict": "in_training" | "not_in_training"}
```

## Scoring

Cztery komponenty (zaktualizowane pod SprintML eval style):

1. **TPR@FPR=1%** (40% wagi) — **primary metric SprintML lab**. TPR mierzony na document-level klasyfikacji `is_member` przy progu kalibrowanym na FPR=1% na walidacji.
2. **Document-level AUC** (20% wagi) — secondary, łapie zachowanie w środku ROC.
3. **Dataset-level p-value** (30% wagi) — `-log10(p_value)` dla poprawnego dataset (clipped do 10), **kara jeśli false positive na out-of-training datasecie**. SprintML wymaga "zero FP on independent models".
4. **Calibration** (10%) — Brier score, isotonic regression na shadow split.

**Reference scores:**
- Easy baseline (loss-based MIA, single feature): TPR@1%FPR ≈ 5–15%, AUC ≈ 0.55–0.65
- Solid (Maini's full pipeline): TPR@1%FPR ≈ 20–40%, AUC ≈ 0.70–0.80, p-value < 0.01
- Hard target (LiRA-style + ensemble): TPR@1%FPR > 50%, AUC > 0.82, p-value < 0.001 (cytat ze Strong MIAs, Hayes et al. NeurIPS 2025)

## Baselines

### Easy (1h)
**Loss-based MIA (Yeom et al.)**: dla każdego dokumentu policz średnią negatywną log-likelihood per token. Dokumenty z niższą loss → bardziej prawdopodobnie member.

Pseudokod (key idea — przełącz model w tryb inference, sforuard pełen tekst z `labels=ids`, zwróć `output.loss.item()`):
```
device = "mps"
load Pythia-410m w torch.float16, .to(device), tryb inference
for each text:
    ids = tokenize(text, max_length=512)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    score[i] = loss
threshold = median(score)
is_member = score < threshold
```

### Solid (4–6h)
**Maini features** (z paperu sekcja 3):
- Per-token loss
- Perplexity (= exp(loss))
- Zlib entropy ratio: `len(zlib.compress(text)) / len(text)` — tekst high-entropy
- Lowercase ratio
- **Min-K%++** (Zhang ICLR 2025 spotlight, paper 20): standardyzowany per-token score `(log p − μ)/σ` gdzie μ, σ to mean/std rozkładu vocabulary na pozycji `i`. **Bije Min-K% o 6–10% AUC na WikiMIA.** Używaj zamiast plain Min-K%.
- Reference loss z mniejszego modelu (Pythia-160M) — różnica = "calibrated" MIA

Złóż w wektor 6-wymiarowy → train logistic regression na **shadow Pythia-160M** używając Pile-train jako positives, wikitext jako negatives. Aplikuj na docelowy Pythia-410M.

Dataset-level: t-test Welcha między rozkładami `score(set1)` a `score(val_in)`. Jeśli p < 0.05 i kierunek się zgadza → "in_training".

### Hard (jak masz czas)
- Replikuj **DC-PDD** albo **Dataset Inference v2** z paperu — selektywne łączenie tylko tych features które dają monotoniczny sygnał na walidacji
- **LiRA-style LRT** (Carlini S&P 2022): per-document Gauss fit z shadow modeli, log-likelihood ratio. To jest faktyczny SOTA, nie t-test.
- Bootstrap confidence intervals na p-value
- Defenses: spróbuj zaatakować Pythię z **DP-fine-tuningowanymi** dodatkowymi danymi (sprawdź czy MIA pada)
- Czytaj jako bonus: **Strong MIAs (Hayes et al. NeurIPS 2025)**, **Curation Leaks (Wahdany et al. ICLR 2026)**, **NIDs (Rossi et al. ICLR 2026)** — wszystko SprintML, najświeższe MIA frameworki.

## Pułapki (z mojego doświadczenia)

- **Distribution shift** — różnica AUC między Pile a CC-News może wynikać z różnicy stylu, nie z memoryzacji. Maini o tym pisze. Walidacja na **Pile-val** (ten sam rozkład co Pile-train, ale OOD) jest kluczowa.
- **Tokenization** — różne dokumenty mają różną długość po tokenizacji, normalizuj loss per-token.
- **Numerical instability na MPS** — fp16 może dać NaN w loss przy długich sekwencjach. Loaduj fp32 jeśli problem.
- **Cache** — każde query do Pythii to ~1s na M4. 2000 dokumentów × 5 features = czas. Zacache wszystko (`joblib.Memory`).

## Co to ćwiczy pod hackathon

- Membership inference w praktyce
- Statystyczny test na poziomie zbioru (nie tylko sample)
- MIA feature engineering
- Calibration vs raw AUC
- Praca z Pythią/HF (najczęstsze toolingi w privacy research)
