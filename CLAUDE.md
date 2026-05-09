# CISPA European Cybersecurity & AI Hackathon Championship — Warsaw

## Context
3-person team at a 24h hackathon (**2026-05-09/10**) at Warsaw University of Technology.
Organized by SprintML Lab (CISPA) — Adam Dziedzic & Franziska Boenisch.
AI tools are explicitly allowed during the competition.

## Reading order for Claude (token-aware)

When user asks about state, next steps, papers, or attack techniques — read in this order, stop as soon as you have enough:

1. **`TODO.md`** (root) — current state, what's blocked, what's next
2. **`references/papers/MAPPING_INDEX.md`** (~700 words) — lean router. One line per paper: number, title, "use for". For grep terms + key sections jump to the matching entry in `MAPPING.md` (rich, ~3.1k words; load on demand).

Only if 1–2 don't have the answer:
- Open `docs/deep_research/0N_*.md` (**30–55 KB each, ~7–14k tokens** — heavy, justify before loading)
- Open a paper PDF (`references/papers/NN_*.pdf`, **0.8–22 MB**, often 30k+ tokens — last resort)

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

## Source separation (important)

| Source | What | Authority |
|---|---|---|
| `docs/01_email_*.txt`, `02_email_*.txt`, `references/papers/01–04` | Organizer mails + 4 required papers | **Ground truth** |
| `docs/ARCHIVE/hackathon_preparation_pre_zoom.md` | Pre-Zoom speculation — **ARCHIVED** | **Ignore** |
| `docs/deep_research/0N_*.md`, papers `05–25` | Claude Research output + our extrapolations | **Educated guesses** |

## Strong signal from required papers
All 4 required papers are about **privacy + LLM watermarking**:
1. Carlini et al. — extracting training data from diffusion models
2. Maini et al. — LLM dataset inference
3. Zawalski et al. — data contamination detection in LLMs (NeurIPS Workshop 2025, very recent)
4. Kirchenbauer et al. — LLM watermarking

Confirmed task themes (Zoom 2026-05-04): Data Identification + Data Memorization + Watermarking.

**SprintML portfolio note:** model stealing (B4B NeurIPS 2023, ADAGE 2025, GNN extraction AAAI 2026 Oral) is high-probability even though not in required papers. See `references/papers/MAPPING.md` sekcja 6.

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

**Safety rules baked into `juelich_exec.sh`:**
- `rm`, `rmdir`, `dd`, `mkfs`, `git reset --hard`, `scancel` (no ID), `truncate`, `chown` → **BLOCKED** (exit 2). Use `--force` only if user explicitly asks.
- `sbatch`, `scancel <id>`, `| bash` → **asks [y/N]** — always show user what will run and wait for their confirmation before calling.
- Read-only ops (`ls`, `cat`, `squeue`, `sinfo`, `python`, `uv`) → free, no prompt needed.

**Niezablokowane, ale wymagają ostrożności (Claude pyta usera zanim odpali):**
- `cp`, `mv`, `tar -x`, redirekty `> file` / `>> file` w `$PROJECT` / `$SCRATCH`
- `git push`, `git checkout -- .`, `git stash drop`
- `pip install` / `uv pip install` w shared envie (zamiast tego user-local venv)
- `sbatch` z dużą liczbą GPU lub długim time-limit (zjada budżet projektu dla całego teamu)
- modyfikacja katalogów z prefixem teammate'a (np. `kempinski1/`, jeśli ty jesteś innym userem)

**If socket is dead** (>4h since connect or machine rebooted): tell user to run `! scripts/juelich_connect.sh` again. Do not attempt to restart it yourself.

**Config lives in `.juelich.local`** (gitignored). Each teammate has their own file with `JUELICH_USER`, `JUELICH_KEY`, `JUELICH_HOST`.

## Quick commands

```bash
just eval           # smoke test (<30s)
just baseline       # ostatnie 5 linii SUBMISSION_LOG.md
```

## Repo structure
```
CLAUDE.md                                    # this file
TODO.md                                      # ACTIVE — hackathon tracker
SUBMISSION_LOG.md                            # one-liner per successful /submit
requirements.txt
Justfile                                     # eval / baseline / extract-papers
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
data/                                        # task data (will be populated from HF/Jülich at 12:00)
submissions/                                 # submission files
docs/
  STATUS.md                                  # volatile project state (out of CLAUDE.md prefix)
  SETUP.md                                   # per-teammate setup checklist
  FAQ.md                                     # team Q&A — append on every recurring question
  LEARNINGS.md                               # session insights — append on every /compact
  DAY_OF.md                                  # day-of checklist and playbook
  claude_token_playbook.md                   # research #1 — Claude Code token economics
  claude_retrieval_strategy.md               # research #2 — RAG vs no-RAG verdict (recommends no-RAG)
  01_email_invitation_papers.txt             # ORGANIZER mail #1
  02_email_registration_confirmed.txt        # ORGANIZER mail #2
  zoom_transcript.txt                        # Zoom info session 2026-05-04 transcript
  ARCHIVE/hackathon_preparation_pre_zoom.md  # ARCHIVED — pre-Zoom speculation
  deep_research/
    01_adversarial_attacks.md
    02_model_inversion.md
    03_watermarking.md
    04_model_stealing.md
    05_image_attribution.md
    06_fairness_auditing.md
    07_hackathon_toolkit.md
scripts/
  extract_papers.sh                          # pdftotext → references/papers/txt/
  juelich_connect.sh                         # sets up ControlMaster socket (TOTP required)
  juelich_exec.sh                            # runs commands on Jülich via socket
references/
  papers/                                    # 25 PDFs total
    MAPPING_INDEX.md                         # lean router — READ FIRST (~700 words)
    MAPPING.md                               # rich per-paper entries (grep terms + sections, ~3.1k words)
    txt/                                     # pre-extracted .txt from PDFs (5.6 MB)
    01–04 required (organizers' email)
    05–08 supplementary surveys
    09–19 hidden papers (SprintML 2022–2026)
    20–25 competition-ready tools (Min-K%++, Watermark Stealing, DIPPER,
          Recursive Paraphrasing, WAVES, ChatGPT divergence)
```

## Working principles
- **Do not invent challenge details** — actual tasks revealed 2026-05-09 at 12:00.
- **Deep research artifacts are heavy.** Don't auto-load `docs/deep_research/*` unless MAPPING didn't answer. State explicitly when you're about to load one and why.
- Submission: REST API, CSV files, 5-min cooldown (2-min on failure), team API token provided at start.
- Data and model checkpoints: provided via HuggingFace links + Jülich at task reveal.
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

## When to update CLAUDE.md / TODO.md

Trigger an update of these files when:
- Task details revealed at 12:00 → add task specs, scoring format, API endpoint
- Jülich access tested OK / fails → STATUS.md
- New paper added to `references/papers/` → MAPPING.md + CLAUDE.md repo structure paper count

Tip: during a session, press `#` to ask Claude to incorporate a learning into CLAUDE.md.

## Language
User communicates in Polish. Respond in Polish unless code/technical context requires English. Code identifiers, paper titles, library names — keep original. Comments in code: terse English by default.
