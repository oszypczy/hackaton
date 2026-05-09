# Submission flow

## Endpoint (autoritative — z `task_template.py` organizatorów na klastrze)

```
POST http://35.192.205.84/submit/<TASK_ID>
Header: X-API-Key: <YOUR_API_KEY>
Body:   multipart/form-data; file=<submission.csv>
```

Response: JSON. Sukces zwraca `score` + `score_held_out` (gdy validation passed).

## Task IDs

| Task | TASK_ID | CSV header (case-sensitive) | Rows | Score direction |
|---|---|---|---|---|
| 1 — DUCI | `11-duci` | `model_id,proportion` | 9 | MAE ↓ (niżej lepiej) |
| 2 — PII | `27-p4ms` | `id,pii_type,pred` | 3000 | 1−Levenshtein ↑ |
| 3 — Watermark | `13-llm-watermark-detection` | `id,score` | 2250 ⚠ | TPR@1%FPR ↑ |

⚠ **Task 3 niespójność:** `submission_template.py` na klastrze mówi "Exactly 2400 rows"
ALE też "ids 1 to 2250". To copy-paste bug. Trzymamy się **2250** (zgodnie z PDF).
Jeśli pierwszy submit zwróci błąd "Missing N expected ids" → sprobować 2400.

## CSV format requirements (server-side validation, automatyczne odrzucenie)

### Wszystkie taski:
- Encoding: UTF-8
- Max file size: **10 MB**
- Header **dokładnie** jak w tabeli powyżej (case-sensitive)
- Brak duplikatów ID, brak braków, brak nadmiaru

### Task 1 (DUCI):
- 9 wierszy, każdy `model_id` (`00, 01, 02, 10, 11, 12, 20, 21, 22`) dokładnie raz
- `proportion`: float ∈ [0, 1]

### Task 2 (PII):
- 3000 wierszy = 1000 ids × 3 PII types
- `pii_type` ∈ {`EMAIL`, `CREDIT`, `PHONE`}
- `pred`: string długości **10–100 chars** (inclusive). Krótsze/dłuższe → odrzucone
- BEZ otaczających cudzysłowów, BEZ `<|user|>` lub innych special tokens (strip!)

### Task 3 (Watermark):
- 2250 wierszy (lub 2400 — patrz wyżej), `id` ∈ [1, N] (1-indexed!)
- `score`: float ∈ [0, 1], finite (no NaN/Inf)
- Wyższy = silniejsze przekonanie watermarked

## API key handling

**Klucz mieszka WYŁĄCZNIE w `.env` lokalnie** (gitignored):
```
HACKATHON_API_KEY=<your-team-key>
```

**Nigdy:**
- nie commitować klucza do gita
- nie kopiować klucza na klaster (submission z laptopa)
- nie wklejać klucza do logów / outputów / Slack

**Dla teammate'ów:** każdy ma swoją kopię `.env` lokalnie (pomijasz w sync). Klucz teamu jest jeden — owner przekazuje go w prywatnej wiadomości.

## Submit z laptopa (rekomendowane)

```bash
just submit <task_name> <csv_path>
# np.
just submit task1 submissions/task1_duci.csv
```

Recipe `submit` w `Justfile`:
1. Sprawdza istnienie pliku CSV i jego rozmiar
2. Ładuje `HACKATHON_API_KEY` z `.env`
3. Mapuje `<task_name>` → `<TASK_ID>` (patrz tabela)
4. POST do endpointu
5. Loguje response do `SUBMISSION_LOG.md` (data + score + hash CSV)

## Submit z klastra (alternatywa, NIE rekomendowane — klucz na shared)

Jeśli koniecznie z klastra (np. CSV za duże żeby ściągać):
```bash
# NA KLASTRZE w swoim repo (NIE shared folder):
echo "HACKATHON_API_KEY=..." > ~/.hackathon_key  # chmod 600 (private)
chmod 600 ~/.hackathon_key

# I w submission scripcie:
source ~/.hackathon_key
python submit.py
```

Klucz w `~/` jest tylko dla Twojego konta (nie shared) — bezpieczniejsze niż w `Czumpers/`.

## Pull CSV z klastra

```bash
# Generic:
scripts/juelich_exec.sh "cat /p/scratch/training2615/kempinski1/Czumpers/<task>/output/submission.csv" > submissions/<task>.csv

# Quick alias dla każdego z 3 tasków (patrz Justfile):
just pull-csv task1   # → submissions/task1_duci.csv
just pull-csv task2   # → submissions/task2_pii.csv
just pull-csv task3   # → submissions/task3_watermark.csv
```

## Cooldowns / rate limits (PDF + presentation)

- **5 min** cooldown po **successful** submission
- **2 min** cooldown po **failed** submission
- Score updateuje się **tylko jeśli wyższy** niż obecny best — **brak feedbacku jeśli niżej**
  (implikacja: nie można swobodnie ablować na public)
- Każde submission → wpis do `SUBMISSION_LOG.md` (timestamp, task, score, csv-md5)

## Public/Private split

| Task | Public (live scoreboard) | Private (final ranking) |
|---|---|---|
| 1 — DUCI | 3/9 modeli (consistent across teams) | 6/9 modeli |
| 2 — PII | 30% danych | 70% (final) |
| 3 — Watermark | 30% (~675 samples) | 70% (~1575 samples) |

⚠ **Final scoring w niedzielę 2026-05-10 idzie na EXTENDED test sets** których my nie
widzimy w trakcie. Public scoreboard ≠ wygrana. Patrz CLAUDE.md "Working principles" 🎯.

## Leaderboard

URL: `http://35.192.205.84/leaderboard_page` (port 80, HTTP)

- pokazuje **best result per team only**
- ranking overall (suma rang) + per-task

## Walidacja przed submisją (oszczędź submisji)

Przed `just submit`:
```bash
# 1. Format check (lokalnie):
python -c "
import pandas as pd
df = pd.read_csv('submissions/task1_duci.csv')
assert list(df.columns) == ['model_id','proportion']
assert len(df) == 9
assert df.proportion.between(0,1).all()
print('OK')
"

# 2. (opcjonalnie) `just eval` jeśli mamy lokalny smoke test
```

Każde failed submission = 2 min cooldown + nic z scoreboardu. Walidacja lokalna = zero kosztu.
