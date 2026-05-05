# Dzień hackathonu — protokół 2026-05-09

## 12:00 — Taski ujawnione (30 min sprint przed startem o 12:30)

### Pierwsze 5 minut — każda osoba

```
/start-task A    # (lub B albo C — zależnie od przydziału)
```

Claude ładuje spec, paper, metrykę i mówi co robić pierwsze. Nie czytaj nic ręcznie.

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

> **WAŻNE:** challenges A/B/C w repo to NASZE ĆWICZENIA — nie faktyczne taski hackathonu.
> Faktyczne taski poznasz o 12:00. Poniżej: techniki pasujące do potwierdzonych obszarów tematycznych.
> Dopasuj technikę do FAKTYCZNEGO zadania, nie do naszych practice specs.

### Obszar: Dataset / Membership Inference
1. Min-K%++ baseline (`code/attacks/min_k_pp.py` — nasza implementacja z ćwiczeń)
2. Reference model perplexity ratio
3. Welch t-test na poziomie datasetu
4. **Wnioski z ćwiczenia A** (AUC 0.38 → 0.689):
   - Sprawdź rozkład IN vs OUT ZANIM odpalisz minkpp — jeśli zlib ratio bije minkpp, masz domain shift i wyniki będą odwrócone
   - OUT dataset musi być z tego samego gatunku tekstu co IN (nie CNN/Wikipedia jeśli IN to web text)
   - Split scoring: zlib dla dataset-level t-test, mix zlib+loss dla doc-level
   - p-value < 0.2 = zmień OUT, nie tuninguj progu
   - M4 MPS wystarczy (Pythia-410m: ~15 min cold, <5 s z cache)

### Obszar: Watermark LLM (Kirchenbauer)
1. Z-test z green-list (kod: github.com/jwkirchenbauer/lm-watermarking)
2. Ataki: DIPPER paraphrase → sprawdź czy watermark przeżywa
3. Watermark Stealing jeśli mamy dostęp do logitów modelu
4. Wnioski z ćwiczenia B: [uzupełnij po mini-hackathonie]

### Obszar: Diffusion Memorization (Carlini / CDI)
1. Generuj dużo próbek z modelu → membership inference
2. CDI approach: 70 próbek wystarczy → confidence ≥99%
3. L2/SSIM do wykrywania duplikatów treningowych
4. Wnioski z ćwiczenia C: [uzupełnij po mini-hackathonie]

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
