# Task 3 — Handoff dla następnej sesji (murdzek2)

> Napisany: 2026-05-09 ~21:45. Nowy Claude zaczyna tutaj.

## Stan natychmiastowy

- **Best leaderboard**: 0.103 TPR@1%FPR (multan1, `submission_strong_bino.csv`)
- **Lider**: 0.27 — gap 2.6×
- **Do submitowania teraz**: `submissions/task3_watermark_fdgpt.csv` (już ściągnięty lokalnie)
  - 2250 wierszy, zdrowa dystrybucja (96 unique vals, range 0.023–0.966)
  - Fast-DetectGPT (Pythia-2.8b) odpalony przez multan1 o 21:29
  - **Wrzuć najpierw**: `python3 scripts/submit.py task3 submissions/task3_watermark_fdgpt.csv`

## Branch model

- multan1 pracuje na branchu `task3` — **nie commituj tam**
- Nasz branch: utwórz `task3-murdzek2` z `task3`:
  ```bash
  git fetch origin task3
  git checkout -b task3-murdzek2 origin/task3
  ```
- Pull CSV z klastra: `scripts/juelich_exec.sh "cat /p/scratch/training2615/kempinski1/Czumpers/task3/submission_<name>.csv" > submissions/task3_watermark_<name>.csv`

## Pełny kontekst task3

**Czytaj koniecznie**: `git show origin/task3:docs/tasks/task3_session_context.md`
(multan1 napisał szczegółowy handoff — 300+ linii, wszystko tam)

## submit.py — leaderboard scraping

Już działa dla task3. Po `python3 scripts/submit.py task3 <csv>`:
- Auto-scrape `/leaderboard_page` po 5s
- Log do `SUBMISSION_LOG.md`: `leaderboard task1=X task2=Y task3=Z`

## pull_csv.py — poprawki do zrobienia

Aktualnie `pull_csv.py` hardkoduje `submission.csv`. Powinno obsługiwać dowolną nazwę.
Patch: dodaj opcjonalny argument `<remote_filename>`:
```python
# argv[2] = remote filename (default: "submission.csv")
remote_file = argv[2] if len(argv) > 2 else "submission.csv"
remote = f"{CLUSTER_BASE}/task3/{remote_file}"
```

## Co dalej (z context multan1)

Priorytety:
1. **Submit fdgpt CSV** — już gotowy, 5 min cooldown
2. **Stronger branch_a (Pythia-2.8b)** — copy branch_a.py, zmień model_name, cache jako `features_a_strong.pkl`
3. **Multi-LM PPL fingerprint** — nowy moduł, 4-5 LMs (gpt2/gpt2-medium/pythia-1.4b/2.8b/opt-1.3b), cache `features_multi_lm.pkl`
4. **Retry KGW z LLaMA tokenizer** — NIE w offline cache; potrzeba skopiowania tokenizer files

## Infrastruktura na klastrze

```
/p/scratch/training2615/kempinski1/Czumpers/
├── task3/                              # submission CSVs, cache/
│   ├── submission_fdgpt.csv            # NEWEST, gotowy do submitu
│   ├── submission_strong_bino.csv      # current best (0.103)
│   └── cache/                          # features pickled
├── repo-multan1/                       # multan1's clone (nie wchodź)
└── .cache/hub/                         # HF models offline cache
    # dostępne: gpt2, gpt2-medium, pythia-1.4b/2.8b/6.9b, opt-1.3b, OLMo-2-1B
    # NIE MA: LLaMA-2, Mistral (gated)

# Nasz clone:
/p/scratch/training2615/kempinski1/Czumpers/repo-murdzek2/
# (jeśli nie istnieje, utwórz: git clone git@github.com:oszypczy/hackaton.git repo-murdzek2)
```

## Scoring

- Metric: TPR@1%FPR — ranking matters, nie absolute score
- Public 30% / Private 70% — nie overfituj do public
- Cooldown: 5 min po success, 2 min po fail
- Score updateuje się tylko jeśli wyższy od best
