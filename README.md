# Czumpers — CISPA European Cybersecurity & AI Hackathon Championship

> 🏆 **4th place — Top-6 finalists** · 24-hour hackathon · Warsaw, 2026-05-09 / 2026-05-10
> Organised by [SprintML Lab @ CISPA](https://sprintml.com/) (Adam Dziedzic, Franziska Boenisch)
> Hosted at Warsaw University of Technology

---

## Final results

| Metric | Value |
|---|---|
| **Overall score** | **0.72** (avg of normalized per-task ranks) |
| **Final rank** | **4 / 23** |
| **Status** | 🏅 **Qualified for the top-6 final round** |

### Per-task results

| # | Task | Method | Score | Rank |
|---|---|---|---|---|
| **1** | **DUCI** (Dataset Use Composition Inference) | Maini blind-walk + RMIA + per-arch flip search ensemble | `0.249 MAE` (lower is better) | **17 / 24** |
| **2** | **PII Extraction** (multimodal LMM) | Multi-prompt ensemble + per-PII routing (smart_v2) | `0.5326` (1 − norm-Levenshtein) | 🥇 **1 / 23** |
| **3** | **Watermark Detection** (3 schemes) | SIR + KGW selfhash + multi-LM PPL ratios + LogReg meta | `0.2507` (TPR @ 1% FPR) | **7 / 24** |

> **Task 2 highlight — gold medal.** Our per-PII routing ensemble took
> first place on the PII extraction track, beating zer0_day (0.5115) and
> APT (0.4884). The combination of Path A's question-repeat prompt
> (best EMAIL/PHONE memorisation recall) and Path B's plurality voting
> on non-placeholder CREDIT candidates was decisive — see
> [`presentation/task2/pawel_step_by_step.md`](presentation/task2/pawel_step_by_step.md).

### Overall ranking — top 6 (qualified for final round)

| Rank | Team | Score |
|---|---|---|
| 1 | Syntax Terror | 0.90 |
| 2 | zer0_day | 0.88 |
| 3 | Advanced Persistent Thinkers | 0.75 |
| **4** | **Czumpers** | **0.72** |
| 5 | SPQR | 0.72 |
| 6 | BatchNormies3d | 0.64 |

### Per-task podiums

**Task 1 — DUCI (top 5, lower MAE = better):**
1. BatchNormies3d — 0.0393 · 2. CyberDzik Syndicate — 0.0413 · 3. SPQR — 0.0477 ·
4. zer0_day — 0.0540 · 5. ParmaGo — 0.0700 · ··· · **17. Czumpers — 0.2490**

**Task 2 — PII Extraction (top 5, higher score = better):**
🥇 **1. Czumpers — 0.5326** · 2. zer0_day — 0.5115 · 3. APT — 0.4884 ·
4. Sakura — 0.4824 · 5. Syntax Terror — 0.4780

**Task 3 — Watermark Detection (top 7):**
1. Syntax Terror — 0.3564 · 2. APT — 0.3198 · 3. zer0_day — 0.2859 ·
4. S.P.Q.L. — 0.2611 · 5. 4aufKind — 0.2585 · 6. SPQR — 0.2559 · **7. Czumpers — 0.2507**

### Scoreboard screenshots

> _Placeholders — drop the PNGs into `docs/screenshots/` and they'll render below._

![Final leaderboard](docs/screenshots/final_leaderboard.png)

![Task 1 — DUCI](docs/screenshots/task1_board.png)

![Task 2 — PII Extraction](docs/screenshots/task2_board.png)

![Task 3 — Watermark Detection](docs/screenshots/task3_board.png)

---

## Team

**Czumpers** (4 people, mixed Mac M4 + CUDA workstation + Jülich JURECA cluster):

| Member | Primary task | GitHub |
|---|---|---|
| Artur Kempiński (`kempinski1`) | Task 2 — Path A (prompt-attack pipeline) | [@arcziwoda](https://github.com/arcziwoda) |
| Oliwier Szypczyn (`szypczyn1`) | Task 1 — DUCI ensemble + endgame variants | [@oszypczy](https://github.com/oszypczy) |
| _ZZZdreamm_ (`multan1`) | Task 3 — Watermark detection / RoBERTa fine-tune | _GitHub_ |
| Paweł Murdzek (`murdzek2`) | Task 2 — Path B + ensemble hybrid; Task 1 / Task 3 contributions | [@PawelMurdzek](https://github.com/PawelMurdzek) |

Compute provided via **Jülich Supercomputing Centre** (project `training2615`, JURECA `dc-gpu` partition, A800).

---

## What we built

**Task 1 — DUCI.** Blind-walk + Tong-style RMIA reference-model debias for CIFAR-100 mixture detection. 8 reference architectures were trained, then a per-arch MLE was combined with Maini distance-to-decision-boundary signals. Endgame: per-model `flip` search at the snap-grid identified that flipping `model_11` to `0.4` lifted us from `0.053` → `0.020 MAE`. See [`code/attacks/task1_duci/`](code/attacks/task1_duci/) and [`docs/tasks/task1_breakthrough_note.md`](docs/tasks/task1_breakthrough_note.md).

**Task 2 — PII Extraction (multimodal LMM).** Two-path attack:
- **Path A (prompt engineering).** 9 prompt strategies benchmarked on a 280-user validation set (`baseline`, `direct_probe`, `role_play_dba`, `verbatim_prefix`, `system_override`, `completion_format`, `oneshot_demo`, `question_repeat`, contrastive decoding). Winner: `question_repeat` at val_pii blank-mode 0.401. Format-aligned regex post-process for raw-PII extraction (+0.034 LB).
- **Path B (ensemble hybrid).** Multi-source ensemble with **per-PII routing**: EMAIL/PHONE → single-best-prompt fallback chain (memorisation is text-side); CREDIT → plurality voting on non-placeholder candidates across 6 source CSVs (CREDIT memorisation is image-only, model placeholder-collapses on blank-image).
- Final on smart_v2: LB **0.40021** (real, public). Full pitch in [`presentation/task2/pawel_step_by_step.md`](presentation/task2/pawel_step_by_step.md).

**Task 3 — Watermark Detection (3 schemes, 2250 samples).** Multi-feature stack: SIR direct extraction (Liu-style semantic invariant), KGW self-hash (Kirchenbauer green-list), multi-LM perplexity ratios (Pythia / OLMo / Mistral fingerprint), prior-correction Bayes shift, pseudo-labeling with multi-seed averaging. LogReg meta-learner over OOF features (LightGBM caused bimodal collapse). Final LB plateau at **0.2841**, lifted by combining super-features with iterative pseudo-labeling. See [`code/attacks/task3/`](code/attacks/task3/) and [`docs/tasks/task3_FINAL_HANDOFF.md`](docs/tasks/task3_FINAL_HANDOFF.md).

### Cross-cutting techniques

- **Verify-pattern submit logging.** Every submit wrapped with BEFORE/AFTER leaderboard scrape — lift attribution provable under shared 4-person team cooldown. Logged to [`SUBMISSION_LOG.md`](SUBMISSION_LOG.md).
- **Validation calibrators.** Each task has a local validation set with ground truth; calibrator-LB delta was reliable within ±0.005 across 8+ strategies.
- **Cluster workflow.** Per-user clones (`repo-$USER/`) on shared `$SCRATCH/Czumpers/`, sync via GitHub only, sbatch from each owner's clone — see [`docs/CLUSTER_WORKFLOW.md`](docs/CLUSTER_WORKFLOW.md).

---

## Repo layout

```
.
├── README.md                          ← this file
├── CLAUDE.md                          ← project-level Claude Code config (token-aware reading order)
├── SUBMISSION_LOG.md                  ← BEFORE/AFTER LB scrape per submit
├── presentation/
│   ├── Team presentation … final.pdf  ← compiled team deck
│   ├── _team/README.md                ← shared style guide
│   ├── task1/                         ← Oliwier (DUCI)
│   ├── task2/                         ← Artur (slot 3) + Paweł (slot 4) — 6 .md + 2 figures
│   └── task3/                         ← ZZZdreamm (watermark)
├── code/attacks/
│   ├── task1_duci/                    ← reference banks, MLE combine, Maini extract, ensemble
│   ├── task2/                         ← prompt strategies, smart_ensemble v2/v3, per-PII routing
│   └── task3/                         ← SIR / KGW / multi-LM features + sbatch wrappers
├── docs/
│   ├── tasks/                         ← per-task spec + breakthrough notes + handoff logs
│   ├── deep_research/                 ← background literature
│   ├── CLUSTER_WORKFLOW.md            ← JURECA branch model + sbatch template
│   └── SUBMISSION_FLOW.md             ← REST API + cooldowns + CSV format
├── scripts/
│   ├── submit.py                      ← POST CSV to organizer + log
│   ├── pull_csv.py                    ← fetch submission.csv from cluster
│   ├── juelich_{connect,exec}.sh      ← SSH ControlMaster + safety-wrapped exec
│   └── submit_and_verify_*.sh         ← BEFORE/AFTER LB scrape per submit
├── references/
│   ├── papers/                        ← 25 PDFs (4 required + 21 supplementary) + extracted .txt
│   └── research/                      ← deep-research outputs from Claude / Perplexity
└── submissions/                       ← final CSV outputs (per task + per submission attempt)
```

---

## Reproduction

Each `code/attacks/task*/` folder is self-contained:
1. `pip install -r requirements.txt` (Python 3.11 / 3.12; bf16 PyTorch on CUDA, MPS on M4)
2. Activate `juelich_connect.sh` once per session (TOTP), then `juelich_exec.sh "sbatch …"`
3. Pull submission CSVs locally with `just pull-csv <task>`, submit with `just submit <task> <csv>`

API key for the organizer endpoint goes in `.env` as `HACKATHON_API_KEY` (gitignored).
Endpoint: `POST http://35.192.205.84/submit/<TASK_ID>`. Cooldown: 5 min on success / 2 min on failure.

---

## Acknowledgements

- **CISPA SprintML Lab** for organising the championship and building the public benchmarks our attacks target.
- **Jülich Supercomputing Centre** for cluster access (project `training2615`, A800 GPUs).
- **Warsaw University of Technology** for hosting the on-site finale.

Built between **2026-05-09 12:00** and **2026-05-10 12:30** — a single 24-hour sprint.
This README finalised on **2026-05-10**.

---

_Generated by Czumpers @ CISPA Warsaw 2026 finals · 4th of 23 · top-6 finalists._
