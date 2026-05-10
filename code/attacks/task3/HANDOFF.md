# Task 3 — Handoff murdzek2 (2026-05-09 ~21:45)

> Po /clear zaczynaj TUTAJ. Godziny do deadline: ~15h (12:30 2026-05-10).

## Pierwsze 5 minut po /clear

```bash
# 1. Sprawdź branch
git branch  # powinno być task3-murdzek2

# 2. Submit gotowy CSV (czeka lokalnie)
python3 scripts/submit.py task3 submissions/task3_watermark_fdgpt.csv
# → auto-scrape leaderboard po 5s → wynik w SUBMISSION_LOG.md

# 3. Sprawdź klaster
scripts/juelich_exec.sh "squeue -u \$USER --format='%.10i %.20j %.10T %.10M %.10L'"

# 4. Czytaj pełny kontekst multan1 (300+ linii, cały stan task3):
git show origin/task3:docs/tasks/task3_session_context.md | head -200
```

## Kontekst task3

**Best leaderboard:** task3=**0.103** TPR@1%FPR — lider: **0.27** (gap 2.6×)

**3 typy watermarku** (PDF references, "split evenly" w train):
- **KGW/Kirchenbauer** — hash(prev_token) → green list
- **Liu/Semantic** — embedding-based, invariant na paraphrasing
- **Zhao/Unigram** — statyczna green list, provably robust

**Metryka:** TPR@1%FPR — ranking ważniejszy niż wartość bezwzględna. Public 30%, Private 70%.

## Stan CSVs

| Plik lokalny | Opis | Status |
|---|---|---|
| `submissions/task3_watermark_fdgpt.csv` | Fast-DetectGPT (Pythia-2.8b) | **DO SUBMITU** |

Cluster CSVs (`/p/scratch/training2615/kempinski1/Czumpers/task3/`):
- `submission_fdgpt.csv` — najnowszy (21:29), już ściągnięty lokalnie
- `submission_strong_bino.csv` — current best 0.103
- `submission_xl_bino.csv` — nie poprawił 0.103

## Pull CSV z klastra

```bash
# Ogólna forma (po naprawie pull_csv.py):
python3 scripts/pull_csv.py task3 submission_<name>.csv
# → zapisuje do submissions/task3_watermark_<name>.csv

# Manualna forma (zawsze działa):
scripts/juelich_exec.sh "cat /p/scratch/training2615/kempinski1/Czumpers/task3/submission_<name>.csv" > submissions/task3_watermark_<name>.csv
```

## Submit i leaderboard

```bash
python3 scripts/submit.py task3 submissions/task3_watermark_<name>.csv
# Auto-scrape /leaderboard_page po sukcesie → log do SUBMISSION_LOG.md
# Cooldown: 5 min success / 2 min fail
# Score updateuje się TYLKO jeśli wyższy od best
```

## Architektura kodu (multan1's, branch task3)

```
/p/scratch/training2615/kempinski1/Czumpers/repo-multan1/code/attacks/task3/
├── main.py                # orchestrator
├── main_<variant>.sh      # sbatch scripts
├── features/
│   ├── branch_a.py        # gpt2 log-prob, GLTR, ngram, gzip, TTR
│   ├── branch_bc.py       # UnigramGreenList (Zhao) + BigramGreenList (KGW)
│   ├── branch_d.py        # sentence-transformers cosine — Liu proxy
│   ├── binoculars_strong.py  # Pythia-1.4b + 2.8b PPL ratio
│   ├── branch_kgw_v2.py   # Direct KGW — NO SIGNAL (hash_key nieznany)
│   └── fast_detectgpt.py  # Pythia-2.8b curvature — NAJNOWSZE
└── cache/                 # features pickled (na klastrze)
```

## Kluczowe wnioski (żeby nie powtarzać błędów)

1. **OOF 0.65 ≠ leaderboard 0.10** — gap 6× przez data leakage w branch_bc
2. **Skalowanie Binoculars plateau** — GPT-2 → Pythia-6.9b, brak poprawy powyżej 0.103
3. **Direct KGW = zero sygnału** — 8 hash_keys, 3 tokenizery — organizatorzy NIE użyli żadnego
4. **Classifier: LogReg C=0.01 + StandardScaler** — LightGBM daje bimodal collapse
5. **Healthy CSV**: 80+ unique(round(score,2)), min~0.02, max~0.95

## Następne eksperymenty (priorytet)

### Gotowe — odpal teraz
1. **Fast-DetectGPT** — `submissions/task3_watermark_fdgpt.csv` ściągnięty, submituj

### Do implementacji (sub-30 min każde)
2. **Stronger branch_a** — kopia branch_a.py z Pythia-2.8b zamiast gpt2. Cache `features_a_strong.pkl`. Gain: +1-3pp
3. **Multi-LM PPL fingerprint** — min/max/mean PPL pod 4-5 LMs. Cache `features_multi_lm.pkl`. Gain: +2-4pp

### Wymaga czasu (2-3h)
4. **Lepszy Liu detector** — sprawdź czy `all-mpnet-base-v2` w offline cache
5. **LLaMA tokenizer dla KGW** — NIE w cache, trzeba ręcznie skopiować z laptopa

## Infrastruktura klastra

```
/p/scratch/training2615/kempinski1/Czumpers/
├── task3/                   # submission CSVs + cache/
├── repo-multan1/            # multan1's clone (NIE wchodź!)
├── repo-murdzek2/           # nasz clone (utwórz jeśli nie ma)
└── .cache/hub/              # HF offline: gpt2, gpt2-medium, pythia-1.4b/2.8b/6.9b, opt-1.3b
                             # NIE MA: LLaMA-2, Mistral (gated)
```

## Branch workflow

```bash
git checkout task3-murdzek2  # nasz branch
git pull origin task3-murdzek2

# Edytuj kod → commit → push
git push origin task3-murdzek2

# Na klastrze:
scripts/juelich_exec.sh "cd /p/scratch/training2615/kempinski1/Czumpers/repo-murdzek2 && git checkout task3-murdzek2 && git pull"
# Uruchom job:
scripts/juelich_exec.sh "cd /p/scratch/training2615/kempinski1/Czumpers/repo-murdzek2 && sbatch code/attacks/task3/main_<variant>.sh"
```

**NIGDY** nie commituj na `task3` (multan1's branch). Nie wchodź do `repo-multan1/`.
