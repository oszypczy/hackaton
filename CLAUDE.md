# CISPA European Cybersecurity & AI Hackathon Championship — Warsaw

## Context
3-person team at a 24h hackathon (**2026-05-09/10**) at Warsaw University of Technology.
Organized by SprintML Lab (CISPA) — Adam Dziedzic & Franziska Boenisch.
AI tools are explicitly allowed during the competition.

## Reading order for Claude (token-aware)

When user asks about state, next steps, papers, or attack techniques — read in this order, stop as soon as you have enough:

1. **`TODO.md`** (root) — current state, what's blocked, what's next
2. **`docs/tasks/task<N>_*.md`** — autoritatywne taski (revealed 12:00 2026-05-09)
3. **`references/papers/MAPPING_INDEX.md`** (~700 words) — lean router for paper corpus. One line per paper: number, title, "use for". For grep terms + key sections jump to the matching entry in `MAPPING.md` (rich, ~3.1k words; load on demand).

Only if 1–3 don't have the answer:
- Open `docs/deep_research/0N_*.md` (**30–55 KB each, ~7–14k tokens** — heavy, justify before loading)
- Read `references/papers/txt/NN_*.txt` (extracted text — preferred over PDF)
- Raw `references/papers/NN_*.pdf` only if `.txt` lost something critical (rare)

## Retrieval rules

For ANY question about the paper corpus, follow this order strictly:
1. Read `references/papers/MAPPING_INDEX.md` first (lean router — picks ONE paper, ~700 words)
2. Read the relevant entry in `references/papers/MAPPING.md` (rich — grep terms + key sections per paper, ~3.1k words; do NOT load whole file, jump to the entry)
3. Use Grep on `references/papers/txt/*.txt` for the key terms from MAPPING
4. Read `references/papers/txt/NN_*.txt` with `offset`/`limit` for surgical reads
5. Read raw PDF only if `.txt` extraction lost something critical (rare; flag it)

Hard rules:
- Never read >2 papers per turn (research #2: top retrieval failure on cross-paper synthesis)
- Never read full PDF when `.txt` exists (60k → 25k tokens saving)
- Read tool caps at 25k tokens/file — use `offset`/`limit` (txt) or `pages: "1-15"` (PDF) for big files

Current status: see @docs/STATUS.md

## Confirmed tasks (revealed 2026-05-09 12:00)

| Task | File | TASK_ID | Summary |
|---|---|---|---|
| **1 — DUCI** | `docs/tasks/task1_duci.md` | `11-duci` | 9 ResNet models, predict fraction of MIXED (CIFAR100) used in training; MAE ↓ |
| **2 — PII Extraction** | `docs/tasks/task2_pii_extraction.md` | `27-p4ms` | Extract EMAIL/CREDIT/PHONE from multimodal LMM via scrubbed images; 1−NormLevenshtein ↑ |
| **3 — Watermark Detection** | `docs/tasks/task3_watermark_detection.md` | `13-llm-watermark-detection` | Multi-type watermark (Kirchenbauer/Liu/Zhao); 2250 test, score [0,1]; TPR@1%FPR ↑ |

⚠ **Task 3 known gotcha:** `submission_template.py` na klastrze mówi "Exactly 2400 rows" + "ids 1 to 2250" — sprzeczność. Trzymamy się **2250** (zgodnie z PDF). Jeśli pierwszy submit zwróci "Missing N expected ids" — sprobować 2400. Patrz `docs/SUBMISSION_FLOW.md`.

Pełny flow submisji + endpoint + cooldowns → `docs/SUBMISSION_FLOW.md`.
Cluster workflow + sbatch template + ACL rules → `docs/CLUSTER_WORKFLOW.md`.

## Source separation (important)

| Source | What | Authority |
|---|---|---|
| `docs/01_email_*.txt`, `02_email_*.txt`, `references/papers/01–04` | Organizer mails + 4 required papers | **Ground truth** |
| `docs/tasks/*.md` | Task specs (revealed at 12:00) | **Ground truth** |
| `docs/deep_research/0N_*.md`, papers `05–25` | Claude Research output + our extrapolations | **Educated guesses** |

## Environment
**Most team members on MacBook M4** (MPS / MLX, no CUDA). **One teammate has a CUDA GPU**.

Implications for M4:
- `bitsandbytes` 4-bit / some `torchattacks` ops do not work
- MLX-LM is ~3× faster than MPS for LLM inference
- Models > 1B params: prefer MLX 4-bit; for training-from-scratch use the CUDA teammate or Jülich

GPU access during competition: Jülich Supercomputer (https://judoor.fz-juelich.de/projects/training2615).

### Executing commands on Jülich from Claude

Use `scripts/juelich_exec.sh "<cmd>"` — **never** call `ssh` directly.

**Prerequisite (user must do once per session):**
```bash
! scripts/juelich_connect.sh   # prompts for TOTP; socket lives 4h
```

**Then Claude can call freely:**
```bash
scripts/juelich_exec.sh "squeue -u $JUELICH_USER"
scripts/juelich_exec.sh "cat ~/results.txt"
scripts/juelich_exec.sh "python train.py --epochs 5"
```

**⚠ Shared state — 3 teammates pracują równolegle na tym samym `$PROJECT` / `$SCRATCH` (`training2615`).** Każda komenda `juelich_exec` może klobnąć cudzą pracę (checkpointy, datasety, joby). PRZED każdym wywołaniem oceń blast radius:
- czy operacja modyfikuje pliki/foldery w `$PROJECT` lub `$SCRATCH`?
- czy nadpisuje istniejące dane (przekierowanie `>`, `cp`, `mv`, `tar -x` bez katalogu)?
- czy ubija jobs/procesy które mogą być teammate'a?

Jeśli odpowiedź "tak" lub "może" → **najpierw zapytaj usera**, nawet jeśli `juelich_exec.sh` nie blokuje komendy. Bezpieczeństwo skryptu pokrywa najgorsze przypadki, ale nie zna kontekstu (czyje pliki, czyje joby).

**Safety:** wrapper blokuje destrukcyjne wzorce (`rm`, `dd`, `git reset --hard`, etc.) i pyta y/N o `sbatch` / `scancel <id>` / `| bash`. Read-only (`ls`, `cat`, `squeue`, `python`) idą bez prompta. Dla `cp`/`mv`/redirektów do `$SCRATCH`, `pip install` w shared envie, modyfikacji cudzych folderów — **zapytaj usera** mimo że wrapper nie blokuje. Pełna lista: `head -120 scripts/juelich_exec.sh`.

**If socket is dead** (>4h since connect or machine rebooted): tell user to run `! scripts/juelich_connect.sh` again. Do not attempt to restart it yourself.

**Config lives in `.juelich.local`** (gitignored). Each teammate has their own file with `JUELICH_USER`, `JUELICH_KEY`, `JUELICH_HOST`.

## Cluster workflow (TLDR — szczegóły w `docs/CLUSTER_WORKFLOW.md`)

**Stan:** owner setup zrobiony 2026-05-09 14:19. Shared team folder
`/p/scratch/training2615/kempinski1/Czumpers/` zawiera 3 datasety, 3 venv'y, ACL na 4 osoby
(kempinski1 + szypczyn1 + multan1 + murdzek2).

**Per-user repo (KRYTYCZNE):** każdy teammate ma **własny git clone** w `Czumpers/repo-<jego-username>/`.
Sync **wyłącznie** przez GitHub: edycja + `git push` LOKALNIE z laptopa → `git pull` na klastrze
w SWOIM `repo-$USER/`. **NIE wchodzisz do cudzego** `repo-X/`. **NIE pushujesz z klastra.**

**Branch model (KRYTYCZNE):**
- `main` — shared infra (docs, scripts, Justfile, requirements)
- `task1` / `task2` / `task3` — kod taska (po jednym branchu na task; owner jeden)
- Edycja kodu taska → **TYLKO** na branchu `taskN`, nigdy direct na `main`
- Edycja shared infra → na `main` (rzadko, krótkie commity)
- Squash merge `taskN` → `main` **dopiero na koniec hackathonu** (przed prezentacją)

**Per-task ownership:** każdy teammate bierze 1 task na własność. Workflow:
1. `git checkout task<N>` lokalnie, edytuj `code/attacks/<task>/main.py`, `git commit`, `git push`
2. Na klastrze: `cd Czumpers/repo-$USER && git checkout task<N> && git pull && sbatch code/attacks/<task>/main.sh`
3. CSV → laptop: `just pull-csv <task>`; submit z laptopa: `just submit <task> <csv>`

**Cluster rules (z `Hackathon_Setup.md` § 7):**
- ❌ NIE zmieniaj ACL ręcznie na shared folders / NIE usuwaj shared folders
- ❌ NIE odtwarzaj `Czumpers/<task>/.venv/` (są pre-built)
- ❌ NIE rób `pip install` w shared envie bez konsultacji z teamem (zmieniasz dla 4 osób)
- ✅ Modyfikacje w **swoim** `~/` (`~/.bashrc`, `~/.ssh/`) są OK
- ✅ Praca w `Czumpers/repo-$USER/` (Twój clone)
- ✅ Logi do `Czumpers/<task>/output/`
- ⚠ NIE edytuj cudzego `repo-<inny-user>/` (ACL pozwala, ale konwencja zabrania)

**Submission flow (TLDR — szczegóły w `docs/SUBMISSION_FLOW.md`):**
- Klucz API w `.env` lokalnie (gitignored), nie na klastrze
- Submit z laptopa przez `just submit <task> <csv>` (REST POST do `http://35.192.205.84/submit/<TASK_ID>`)
- 5 min cooldown success / 2 min fail; score updateuje się tylko jeśli wyższy
- Walidacja CSV lokalnie PRZED submit (oszczędność cooldownu)

## Quick commands

```bash
just eval           # smoke test (<30s)
just baseline       # ostatnie 5 linii SUBMISSION_LOG.md
just submit <task> <csv>     # POST submission do API + log
just pull-csv <task>         # ściągnij submission.csv z klastra do submissions/
```

## Repo structure
```
CLAUDE.md                                    # this file
TODO.md                                      # ACTIVE — hackathon tracker
SUBMISSION_LOG.md                            # one-liner per successful /submit
requirements.txt
Justfile                                     # eval / baseline / submit / pull-csv / extract-papers
.claudeignore                                # PDFs, fixtures, lockfiles
.claude/
  settings.json                              # token-hygiene defaults
  agents/                                    # paper-grep, pytorch-debug, code-reviewer
  commands/                                  # /submit, /grill, /eval, /baseline
templates/                                   # pytorch_train_loop, hf_dataset_loader, eval_scaffold
tests/
  smoke.py                                   # `just eval` runs this; <30s, exits 0/1
code/
  attacks/                                   # attack implementations (add task-specific files here)
data/                                        # task data (populated from HF/Jülich)
submissions/                                 # submission files
docs/
  STATUS.md                                  # volatile project state (out of CLAUDE.md prefix)
  CLUSTER_WORKFLOW.md                        # JURECA workflow, branch model, sbatch template
  SUBMISSION_FLOW.md                         # endpoint, CSV format, API key, cooldowns
  JURECA_TEAMMATE_SETUP.md                   # setup guide for non-owner teammates
  FAQ.md                                     # team Q&A — append on every recurring question
  LEARNINGS.md                               # session insights — append on every /compact
  DAY_OF.md                                  # day-of checklist and playbook
  01_email_invitation_papers.txt             # ORGANIZER mail #1
  02_email_registration_confirmed.txt        # ORGANIZER mail #2
  zoom_transcript.txt                        # Zoom info session 2026-05-04 transcript
  Info­_Session _Warsaw.pdf                  # organizer slides
  tasks/
    task1_duci.md                            # Task 1 spec + strategy
    task2_pii_extraction.md                  # Task 2 spec + strategy
    task3_watermark_detection.md             # Task 3 spec + strategy
  deep_research/
    02_model_inversion.md                    # background for Task 2
    03_watermarking.md                       # background for Task 3
    07_hackathon_toolkit.md                  # tooling reference
    08_tooling_audit.md                      # M4/Jülich compatibility audit
scripts/
  extract_papers.sh                          # pdftotext → references/papers/txt/
  juelich_connect.sh                         # sets up ControlMaster socket (TOTP required)
  juelich_exec.sh                            # runs commands on Jülich via socket
  submit.py                                  # POST CSV to organizer API + log
  pull_csv.py                                # fetch submission.csv from cluster
references/
  papers/                                    # 19 PDFs (01-04 required + 05-25 supplementary)
    MAPPING_INDEX.md                         # ⚠ ALWAYS READ FIRST (lean router, ~700 words)
    MAPPING.md                               # rich per-paper entries (~3.1k words; jump to entry only)
    txt/                                     # ⚠ READ ONLY THE .txt EXTRACTION (NN_*.txt)
    NN_*.pdf                                 # raw PDFs — DO NOT READ; use txt/ instead
```

## Working principles
- **Task specs live in `docs/tasks/`** — read the relevant `taskN_*.md` before proposing an approach.
- **Deep research artifacts are heavy.** Don't auto-load `docs/deep_research/*` unless MAPPING didn't answer. State explicitly when you're about to load one and why.
- Submission: REST API, CSV files, 5-min cooldown (2-min on failure), team API token provided at start.
- Data and model checkpoints: provided via HuggingFace links + Jülich.
- **🎯 Final scoring runs on EXTENDED test sets (announced morning 2026-05-09).** Przed ogłoszeniem wyników jutro organizatorzy przepuszczą wszystkie metryki przez ROZSZERZONE wersje test datasetów których my nie widzimy. **Maksymalizacja score'u na live scoreboard ≠ wygrana.** Konsekwencje:
  - **Generalizacja > overfitting do public test set.** Nie tuningujemy hiperparametrów aż do wyciśnięcia ostatniego 0.001. Jeśli model robi 0.85 na public ale działa solidnie na różnych slicach walidacji → lepszy niż 0.92 z pikiem na public ale wąskim distribution.
  - **Walidacja krzyżowa.** Dla każdego taska podziel dostępne dane na cross-val splits (np. by-class, by-architecture, by-prompt-length), sprawdź czy score jest stabilny. Wariancja score'u na slicach ≈ ryzyko regresji na extended test set.
  - **Unikamy "magic numbers" dopasowanych do public.** Jeśli próg/wagę dobrałeś greedy na public test → zostaw też wersję z prostszym (mniej skalibrowanym) progiem na backup.
  - **Sprawdzaj czy metoda jest "ogólna":** czy działa na wszystkich 9 modelach Task 1 (różne ResNety)? Na wszystkich 3 PII typach Task 2? Na różnych długościach tekstu Task 3? Jeśli któryś slice się wyłamuje → flag.
  - **Submisje informacyjne, nie ostateczne.** Live scoreboard pokazuje trend, ale jutrzejszy ranking będzie inny. Submitujemy żeby uczyć się o systemie/danych, nie żeby gonić leaderboard.

## Output rules
- No preamble. No "Great question," no "I apologize," no "Here's the implementation."
- Do not restate the user's question
- Diff-only edits — never echo full files unless explicitly asked
- Bullets > prose. Numbers > adjectives.
- No summary at the end of tool sequences

## Language
User communicates in Polish. Respond in Polish unless code/technical context requires English. Code identifiers, paper titles, library names — keep original. Comments in code: terse English by default.
