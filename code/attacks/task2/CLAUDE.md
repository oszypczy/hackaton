# Task 2 — working directory

> Auto-loaded by Claude Code gdy pracujesz w `code/attacks/task2/`.
> Górne CLAUDE.md w root repo dotyczy całego projektu — to jest task-local supplement.

## What this is

Multi-modal PII reconstruction attack. **2 osoby pracują niezależnie** na osobnych subbranchach. Każdy submituje własny CSV; best-of-CSVs wygrywa. **Bez merge'ów.**

## Read first

1. **`STRATEGY.md`** (w tym samym katalogu) — opis ścieżek A / B / C, anchor papers per ścieżka, common floor, decision points
2. `docs/tasks/task2_pii_extraction.md` — task spec (ground truth z PDF)
3. `references/research/task2/{task2-research-claude.md, perplexity_TASK2_research.md}` — pełne researche
4. (na żądanie) odpowiednie papery przez `references/papers/MAPPING_INDEX.md`

## Workflow — branch model

```
main                       ← shared infra (rzadko, krótkie commity)
└── task2                  ← shared task-level infra (ten branch — STRATEGY.md, CLAUDE.md)
    ├── task2-prompt       ← kempinski1, Ścieżka A
    ├── task2-shadow       ← murdzek2, Ścieżka B
    └── task2-image        ← (opcjonalnie) Ścieżka C jako fallback
```

**Reguły:**
- Edycja kodu ataku **TYLKO** na własnym subbranchu (`task2-<method>`)
- **Nigdy** direct commit na `task2` (ten branch jest tylko shared infra: STRATEGY.md, CLAUDE.md, ewentualnie README na końcu)
- **Nigdy** nie dotykasz cudzego subbranchu (ani lokalnie, ani na klastrze w `repo-<inny-user>/`)
- **Bez merge'ów między subbranchami** — finał to best-of-CSVs, nie merged code
- `CLAUDE.md` na subbranchu **dziedziczy** z `task2` w momencie odbicia. Możesz dopisywać method-specific notatki na swoim subbranchu (`## Method: prompt-injection` z prompt templates) — to nie wpływa na drugą osobę.

## Workflow — kod

**Każda ścieżka buduje swoje:**
- własny loader parquetu (`task/`, `validation_pii/`)
- własny scorer (rapidfuzz `Levenshtein.normalized_distance`)
- własny walidator CSV (3000 rzędów, length 10–100, każda `(id, pii_type)` raz)
- własny format regex / Luhn check per PII type

Tak, to jest duplikacja ~200 linii. Świadomy wybór: zero risk dryftu między ścieżkami przy no-merge.

**Sugerowana struktura subbranchu:**
```
code/attacks/task2/
├── STRATEGY.md           ← shared (z task2)
├── CLAUDE.md             ← shared (z task2)
└── <method>/             ← np. prompt/ lub shadow/
    ├── main.py           ← entrypoint
    ├── main.sh           ← sbatch script (#SBATCH --partition=dc-gpu --account=training2615)
    ├── attack.py         ← core method
    ├── loader.py         ← parquet loader
    ├── scorer.py         ← rapidfuzz wrapper
    ├── format.py         ← regex / Luhn / length validator
    └── output/           ← logi, CSVs, intermediate parquets (gitignore)
```

## Workflow — cluster

**Klaster ma wszystko** (potwierdzone 2026-05-09):

```
/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/
├── task/                          # 1000 samples × 3 questions = 3000 promptów (parquet)
├── validation_pii/                # 280 × 3 = 840 GT (parquet) — KALIBRATOR
├── target_lmm/                    # OLMo-2-1B + LLaVA-HR vision encoder, bf16, 3.6 GB
├── shadow_lmm/                    # ten sam pipeline bez PII, 3.6 GB
├── task2_standalone_codebase.zip  # architecture + dataset + finetune
├── sample_submission.{py,csv}     # template formatu
└── .venv/                         # pre-built (uv 0.11, py3.12, torch 2.11+CUDA 13)
```

**Per-user repo:** każdy ma własny clone `Czumpers/repo-$USER/`. Sync **wyłącznie** GitHub:
- edycja lokalnie na laptopie → `git push`
- na klastrze: `cd Czumpers/repo-$USER && git checkout task2-<method> && git pull`
- **NIE pushujesz z klastra**, **NIE wchodzisz do cudzego `repo-X/`**

**Komendy z laptopa do klastra:** wyłącznie `scripts/juelich_exec.sh "<cmd>"`. Wymaga: user raz na sesję `! scripts/juelich_connect.sh`.

**Pełny flow klastra:** `docs/CLUSTER_WORKFLOW.md`.

## Workflow — submisja

1. **Lokalna walidacja PRZED submit** (oszczędza cooldown):
   - 3000 rzędów dokładnie
   - każda `(id, pii_type)` raz
   - `pii_type ∈ {EMAIL, CREDIT, PHONE}`
   - `pred` length 10–100 chars (po strip)
   - bez `<|user|>`, cudzysłowów, leading/trailing whitespace
2. **Pull CSV z klastra:** `just pull-csv task2` → `submissions/task2_<method>_<timestamp>.csv`
3. **Submit z laptopa:** `just submit task2 <csv-path>`
   - POST do `http://35.192.205.84/submit/27-p4ms` z header `X-API-Key`
   - klucz w `.env` (gitignored), zmienna `HACKATHON_API_KEY`
   - cooldown 5 min na success / 2 min na fail
   - **brak feedbacku jeśli score < current best** — rate-limited info
4. **Submission scoring:** server zwraca `1 − Normalized_Levenshtein` averaged. Public 30% / private 70% split.

**Pełny flow submisji:** `docs/SUBMISSION_FLOW.md`.

## Lokalna eval — common floor

Każda ścieżka MUSI mieć ten setup zanim zacznie cokolwiek submitować:

```python
from rapidfuzz.distance import Levenshtein

def score(gt: str, pred: str) -> float:
    """Match server: 1 - dist/max(len(gt), len(pred))."""
    return 1.0 - Levenshtein.normalized_distance(gt, pred)

# Sanity:
assert abs(Levenshtein.normalized_distance("abc", "ab") - 1/3) < 1e-9
```

Eval na `validation_pii` (840 GT) → mean similarity per PII type → decyzja co wysyłać.

## Don't-touch list

- ❌ Cudzy subbranch (`task2-<inna-osoba>`)
- ❌ Cudzy `repo-<inna-osoba>/` na klastrze (ACL pozwala, konwencja zabrania)
- ❌ Direct commit na `task2` (poza wyjątkową aktualizacją STRATEGY/CLAUDE — wtedy commit message: `docs(task2): ...` i powiadom drugą osobę)
- ❌ Modyfikacja shared `Czumpers/P4Ms-hackathon-vision-task/.venv/` (zmieniasz dla 4 osób)
- ❌ `pip install` w shared envie bez konsultacji
- ❌ Submit bez lokalnej walidacji formatu (tracimy cooldown)
- ❌ Submit w ostatnich 5 minutach przed deadline

## Quick reference

```bash
# laptop
git checkout task2-<method>          # przed jakąkolwiek edycją kodu ataku
git push origin task2-<method>       # po commit

just pull-csv task2                  # ściągnij submission.csv z klastra
just submit task2 submissions/<csv>  # POST do API + log

# cluster (przez juelich_exec.sh)
cd /p/scratch/training2615/kempinski1/Czumpers/repo-$USER
git checkout task2-<method> && git pull
sbatch code/attacks/task2/<method>/main.sh
squeue -u $USER
```

## Owners

| Ścieżka | Branch | Owner |
|---|---|---|
| A — Prompt/Behavioral | `task2-prompt` | kempinski1 |
| B — White-Box Memorization Signal | `task2-shadow` | murdzek2 |
| C — Image-Side (FALLBACK) | `task2-image` | unowner'd, trigger w STRATEGY |
