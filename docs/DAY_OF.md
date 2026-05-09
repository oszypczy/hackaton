# Dzień hackathonu — protokół 2026-05-09

## 12:00 — Taski ujawnione (30 min sprint przed startem o 12:30)

### Pierwsze 5 minut — każda osoba

```
/start-task 1    # (lub 2 albo 3 — zależnie od przydziału)
```

Claude ładuje spec z `docs/tasks/taskN_*.md`, paper, metrykę i mówi co robić pierwsze. Nie czytaj nic ręcznie.

### 5–15 min — zrozum dane

Dane są na HuggingFace lub Jülichu — link w PDF taska.

**HuggingFace:**
```python
from datasets import load_dataset
ds = load_dataset("org/dataset-name")   # link z PDFa
```

**Jülich (shared storage):**
```bash
scripts/juelich_exec.sh "ls /p/project1/training2615/"
# Skopiuj na scratch:
scripts/juelich_exec.sh "cp -r /p/project1/training2615/task_X $SCRATCH/data/"
```

### 15–25 min — uruchom baseline

Każdy challenge ma `run_attack_X.py` (lub podobny). Uruchom lokalnie najpierw:

```bash
python code/attacks/run_attack_X.py --dry-run   # jeśli obsługuje
```

Lub na Jülichu jeśli model za duży:
```bash
# Edytuj templates/sbatch_gpu.sh: zmień job-name i skrypt
scripts/juelich_exec.sh "sbatch main.sh"   # wymaga potwierdzenia [y/N]
```

### 25–30 min — pierwsza submisja (nawet słaba!)

Cel: być na leaderboardzie przed 12:30. Słaby wynik jest lepszy niż brak wyniku.

```bash
python templates/submit_client.py \
    --csv submissions/task_X.csv \
    --task X \
    --endpoint <URL z PDFa> \
    --token <token z PDFa>
```

Najpierw `--dry-run` żeby sprawdzić format CSV.

---

## 12:30 — Start właściwy

### Rytm iteracji

- Submisja co ~30 min (cooldown 5 min, failowane 2 min)
- Po każdej submisji: `just baseline` — zapisuje wynik do SUBMISSION_LOG.md
- Leaderboard żyje — sprawdzaj po każdej submisji

### Token budżet — dyscyplina przez cały dzień

```bash
npx ccusage blocks --live   # zostaw w osobnym oknie
```

Przekroczyłeś 70% okna 5h → `/compact <focus>` → downshift do Haiku dla grep/klasyfikacji.

### Divide and conquer

Każda osoba siedzi na swoim tasku. Nie crossuj się przez pierwsze 8h.

Wyjątek: jeden task trudniejszy niż oczekiwano → daj znać zespołowi po 4h.

---

## Priorytety algorytmiczne (szybkie wygrane)

> Spec każdego taska: `docs/tasks/taskN_*.md`. Strategia opisana tam szczegółowo.

### Task 1: DUCI — ResNet MIA
1. Min-K%++ baseline (napisz `code/attacks/min_k_pp.py` od zera — ~50 LOC)
2. zlib ratio jako feature (z ćwiczeń: często bije minkpp)
3. Welch t-test MIXED vs POPULATION na agregowanych scorach
4. **Wnioski z ćwiczeń A (AUC 0.38 → 0.689):**
   - Sprawdź rozkład IN vs OUT ZANIM odpalisz minkpp — jeśli zlib ratio bije minkpp, wyniki minkpp mogą być odwrócone
   - Split scoring: zlib dla dataset-level t-test, mix zlib+loss dla doc-level
   - POPULATION jako non-member dostarczone przez org — nie musisz zgadywać
   - M4 MPS wystarczy dla ResNet18/50/152 inferencji

### Task 3: Watermark Detection — Kirchenbauer
1. Z-test z green-list — refimpl: `github.com/jwkirchenbauer/lm-watermarking`
2. Jeśli nie znamy hasha: klasyfikator na perplexity + zlib + n-gram diversity
3. TPR@1%FPR — kalibruj na val set (90+90 labelled)

### Task 2: PII Extraction — LMM
1. Bezpośrednie pytanie → role-play → format-template (EMAIL/CREDIT/PHONE)
2. Shadow−Target log-likelihood ratio jako confidence gate
3. Luhn check dla CREDIT, regex dla EMAIL/PHONE
4. Scrubbed image + pytanie bez patrzenia na obraz (model zna usera z TP)

---

## Prezentacja (top 6 teamów)

**Format: 6 min / team = 2 min / task**

Dla każdego taska:
- Co atakowaliśmy (1 zdanie)
- Technika (nazwa + skąd: paper numer)
- Wynik liczbowy (AUC / F1 / nDCG)
- Co by poprawiło wynik gdybyśmy mieli więcej czasu

**Jury weryfikuje rozumienie** — znaj swój atak, nie tylko wynik.

---

## Awarie

| Problem | Akcja |
|---|---|
| Jülich socket dead | `! scripts/juelich_connect.sh` → TOTP ponownie |
| Submission 429 Too Many Requests | Poczekaj 5 min (lub 2 min jeśli failed) |
| Model nie mieści się na M4 | Prześlij job na Jülich (`sbatch`) |
| Claude context full | `/compact <aktualny task>` |
| Nie wiadomo co dalej | `/grill <technika>` — 5 ranked failure modes |
