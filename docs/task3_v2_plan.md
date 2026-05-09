# Task 3 V2 — Plan i diagnoza

## Dlaczego V1 dał słaby wynik

Watermarki w datasecie to 3 typy (po ~60 sampli każdy):
| Typ | Mechanizm | Potrzeba do detekcji |
|---|---|---|
| Kirchenbauer (KGW) | Green list zależy od **poprzedniego tokenu** (hash prev→seed→vocab split) | Detekcja **bigramowa** |
| Zhao/Unigram | Green list **statyczna** (same tokeny zawsze zielone) | Detekcja unigramowa ✓ już działa |
| Liu/Semantic Inv. | Green list bazuje na **semantycznych grupach** tokenów | Embedding features ✓ częściowo |

**V1 gaps**:
1. `BranchBC` używa GPT-2 tokenizer → mismatch z tokenizerem watermarkowania (Kirchenbauer używał **OPT-1.3B**, Liu/Zhao używali **LLaMA-2**)
2. `BranchBC` jest **unigramowy** → nie łapie KGW, gdzie green membership zależy od poprzedniego tokenu
3. GPT-2 jako model perplexity jest za słaby

## Co zostało zaimplementowane (V2)

### 1. `features/branch_bigram.py` (NOWY)
- Uczy `P(green | prev_tok, curr_tok)` z danych treningowych
- Fallback na unigram dla niewidzianych bigram par
- Features: `bigram_mean_weight`, `bigram_pseudo_z`, `bigram_winmax_z_{50,100,200}`
- **Cel**: wykrywanie KGW/Kirchenbauer watermark

### 2. `features/branch_bc.py` — multi-tokenizer (ZAKTUALIZOWANY)
- Nowy parametr: `extra_tokenizer_names: list[str]`
- Uruchamia BranchBC z każdym tokenizerem; features prefixowane `bc2_`, `bc3_`, ...
- **Cel**: matchowanie tokenizera watermarkowania (OPT tokenizer dla Kirchenbauer)

### 3. CLI zmiany w `main.py`
```
--use-bigram                          # włącza BranchBigram
--bc-extra-tokenizers "tok1,tok2"     # dodatkowe tokenizery dla BranchBC
```

### 4. `main_v2.sh` (NOWY)
Używa na klastrze:
- `EleutherAI/pythia-2.8b` jako model perplexity (silniejszy niż GPT-2)
- `facebook/opt-1.3b` jako drugi tokenizer w BranchBC
- `--use-bigram`
- Binoculars z pythia-2.8b/6.9b zamiast gpt2/gpt2-medium

## Jak uruchomić na klastrze

```bash
# Ablacja (wybór najlepszego wariantu) — ~30 min na A800
bash code/attacks/task3/main_v2.sh ablation /path/to/Dataset.zip cuda

# Trening + infer najlepszą konfiguracją
bash code/attacks/task3/main_v2.sh train /path/to/Dataset.zip cuda
bash code/attacks/task3/main_v2.sh infer /path/to/Dataset.zip cuda

# Pull CSV + submit z laptopa
just pull-csv task3
just submit task3 submissions/task3_v2.csv
```

## Dalsze kroki jeśli V2 nadal niski

### A. Sprawdź który tokenizer matchuje (DIAGNOSTYKA)
Na zbiorze treningowym sprawdź `bc_pseudo_z` vs `bc2_pseudo_z` (OPT) — który ma wyższy TPR@1%FPR?
```bash
python -c "
# po załadowaniu danych sprawdź korelację z labelami
import pandas as pd
# ...
"
```

### B. Spróbuj LLaMA tokenizer (jeśli pobrany na klastrze)
```bash
# Dodaj do --bc-extra-tokenizers:
--bc-extra-tokenizers "facebook/opt-1.3b,NousResearch/Llama-2-7b-hf"
```

### C. Kirchenbauer key brute-force
Jeśli znamy tokenizer ale nie klucz, spróbuj popularnych kluczy z paperów:
```python
KIRCHENBAUER_CANDIDATE_KEYS = [
    "", "watermark", "test", "42", "0", "1", "2", "15485863", "23417893"
]
```
Wybierz klucz dający najwyższe z-score na train watermarked vs clean.

### D. Osobne modele per typ watermarku
Jeśli w danych jest kolumna z typem watermarku (train/val), wytrenuj 3 oddzielne detektory.

## Oczekiwana poprawa

| Komponent | V1 TPR@1%FPR (est.) | V2 delta |
|---|---|---|
| Unigram (Zhao) | ~0.7-0.9 | stabilny |
| KGW (Kirchenbauer) | ~0.2-0.4 (słaby) | +0.2-0.4 (bigram) |
| Liu Semantic | ~0.4-0.6 | +0.1 (OPT tokenizer) |
| Overall | ~0.4-0.6 | est. +0.15-0.3 |
