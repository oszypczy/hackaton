# CISPA European Cybersecurity & AI Hackathon Championship — Warsaw

## Context
3-person team preparing for a 24h hackathon (**2026-05-09/10**) at Warsaw University of Technology.
Organized by SprintML Lab (CISPA) — Adam Dziedzic & Franziska Boenisch.
AI tools are explicitly allowed during the competition.

## Reading order for Claude (token-aware)

**Pre-event setup work** — if user is asking you to set up the repo, optimize token usage, configure Claude Code, build subagents/slash commands, or pre-extract papers: read `docs/TOKEN_OPTIMIZATION_PLAN.md` first. It's a self-contained handoff with all decisions, exact commands, and rationale. Stop when done; you don't need the rest of this list.

When user asks about state, next steps, papers, or attack techniques — read in this order, stop as soon as you have enough:

1. **`TODO.md`** (root, ~5 KB) — current state of prep, what's blocked, what's next
2. **`references/papers/MAPPING_INDEX.md`** (~700 words) — lean router. One line per paper: number, title, "use for". For grep terms + key sections jump to the matching entry in `MAPPING.md` (rich, ~3.1k words; load on demand).
3. **`docs/practice/QUICKSTART.md`** (~3 KB) — "scenario → tool → numbers" cheat sheet; covers all 7 attack categories (text/image watermark, MIA, diffusion mem, model stealing, adversarial, property inference)
4. **`docs/practice/challenge_X_*.md`** (~5–7 KB each) — full spec for a specific mock challenge
5. **`docs/practice/README.md`** (~5 KB) — overview, role mapping, SprintML eval style

Only if 1–5 don't have the answer:
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
| `docs/ARCHIVE/hackathon_preparation_pre_zoom.md` | Autorska analiza poprzednich edycji — **ARCHIVED, superseded by STATUS.md** | **Ignore** |
| `docs/deep_research/0N_*.md`, `docs/practice/`, papers `05–25` | Claude Research output + our extrapolations | **Educated guesses** |

Do not mix these in recommendations. Official emails are thin: no challenge count, format, scoring server, or API spec given. Zoom info session announced but not yet scheduled.

## Strong signal from required papers
All 4 required papers are about **privacy + LLM watermarking**:
1. Carlini et al. — extracting training data from diffusion models
2. Maini et al. — LLM dataset inference
3. Zawalski et al. — data contamination detection in LLMs (NeurIPS Workshop 2025, very recent)
4. Kirchenbauer et al. — LLM watermarking

Likely challenge directions: training data extraction, dataset/membership inference, data contamination detection, watermark detect/remove.

**Caveat (deep research correction):** initial analysis said "ZERO on model stealing → unlikely". This was wrong — SprintML lab's portfolio (B4B NeurIPS 2023, ADAGE 2025, GNN extraction AAAI 2026 Oral, Calibrated PoW ICLR 2022 Spotlight) makes model stealing a high-probability target despite absence from the required-papers list. Organizer interview cytuje "tasks designed from our research". Patrz `references/papers/MAPPING.md` sekcja 6 dla SprintML author signatures.

## Environment
**Most team members on MacBook M4** (MPS / MLX, no CUDA). **One teammate has a CUDA GPU** — use them for tasks that don't fit on M4.

Implications for M4:
- `bitsandbytes` 4-bit / some `torchattacks` ops do not work
- MLX-LM is ~3× faster than MPS for LLM inference
- Models > 1B params: prefer MLX 4-bit; for training-from-scratch use the CUDA teammate or Jülich

Use the CUDA teammate for:
- Pre-generating fixture data for practice challenges B (Llama-3-8B watermarked corpus) and C (DDPM CIFAR-10 with forced memorization)
- Any training-from-scratch run
- Anything where `bitsandbytes` / standard CUDA-only stack matters

GPU access during competition: Jülich Supercomputer (https://judoor.fz-juelich.de/projects/training2615) — register early.

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

**Safety rules baked into `juelich_exec.sh`:**
- `rm`, `rmdir`, `dd`, `mkfs`, `git reset --hard`, `scancel` (no ID), `truncate`, `chown` → **BLOCKED** (exit 2). Use `--force` only if user explicitly asks.
- `sbatch`, `scancel <id>`, `| bash` → **asks [y/N]** — always show user what will run and wait for their confirmation before calling.
- Read-only ops (`ls`, `cat`, `squeue`, `sinfo`, `python`, `uv`) → free, no prompt needed.

**If socket is dead** (>4h since connect or machine rebooted): tell user to run `! scripts/juelich_connect.sh` again. Do not attempt to restart it yourself.

**Config lives in `.juelich.local`** (gitignored). Each teammate has their own file with `JUELICH_USER`, `JUELICH_KEY`, `JUELICH_HOST`.

## Quick commands

```bash
just eval           # smoke test (<30s)
just score A        # scoruje attack_A.py przeciwko data/A/
just score B        # j.w. dla B
just score C        # j.w. dla C
just gen-B          # generuje fixture data B (wymaga CUDA)
just gen-C          # generuje fixture data C (wymaga CUDA)
just gen-B-dry      # dry-run bez GPU
just baseline       # ostatnie 5 linii SUBMISSION_LOG.md
```

## Repo structure
```
CLAUDE.md                                    # this file
TODO.md                                      # ACTIVE — overall hackathon prep tracker
SUBMISSION_LOG.md                            # one-liner per successful /submit
requirements.txt
Justfile                                     # eval / score / submit / baseline / extract-papers
.claudeignore                                # PDFs, fixtures, lockfiles
.claude/
  settings.json                              # token-hygiene defaults
  agents/                                    # paper-grep, pytorch-debug, code-reviewer
  commands/                                  # /submit, /grill, /eval, /baseline
templates/                                   # pytorch_train_loop, hf_dataset_loader, eval_scaffold
tests/
  smoke.py                                   # `just eval` runs this; <30s, exits 0/1
code/
  practice/
    score_A.py                               # AUC + p-value dla challenge A
    score_B.py                               # F1 (B1) + BERTScore+z-score (B2)
    score_C.py                               # nDCG@50, Recall@50/100
data/
  A/                                         # fixture data gotowe (2026-04-28)
  B/                                         # fixture data gotowe (LFS, 200 texts + 50 removal)
  C/                                         # fixture data PENDING (wymaga CUDA)
docs/
  TOKEN_OPTIMIZATION_PLAN.md                 # pre-event setup playbook (read for token/setup work)
  STATUS.md                                  # volatile project state (out of CLAUDE.md prefix)
  SETUP.md                                   # per-teammate setup checklist
  FAQ.md                                     # team Q&A — append on every recurring question
  LEARNINGS.md                               # session insights — append on every /compact
  claude_token_playbook.md                   # research #1 — Claude Code token economics
  claude_retrieval_strategy.md               # research #2 — RAG vs no-RAG verdict (recommends no-RAG)
  01_email_invitation_papers.txt             # ORGANIZER mail #1
  02_email_registration_confirmed.txt        # ORGANIZER mail #2
  ARCHIVE/hackathon_preparation_pre_zoom.md  # ARCHIVED — pre-Zoom speculation, superseded by STATUS.md
  deep_research/
    deep_research_prompts.md                 # 7 research prompts
    01_adversarial_attacks.md                # Claude Research result for prompt 1
    02_model_inversion.md                    # Claude Research result for prompt 2
    03_watermarking.md                       # Claude Research result for prompt 3
    04_model_stealing.md                     # Claude Research result for prompt 4
    05_image_attribution.md                  # Claude Research result for prompt 5
    06_fairness_auditing.md                  # Claude Research result for prompt 6
    07_hackathon_toolkit.md                  # Claude Research result for prompt 7
  practice/                                  # mock challenges (3 + 2 optional) + quickstart
    README.md                                # overview, mapping, SprintML eval style
    QUICKSTART.md                            # "if hackathon throws X, do Y" 1-page lookup
    challenge_A_dataset_inference.md         # paper 02 (Maini et al.) + 20 (Min-K%++)
    challenge_B_watermark.md                 # paper 04 (Kirchenbauer) + 21/22/23/24 attacks
    challenge_C_diffusion_extraction.md      # papers 01 + 09 (Carlini + CDI hard mode)
    challenge_D_property_inference.md        # OPTIONAL — Barcelona-style fairness audit
    challenge_E_model_stealing.md            # OPTIONAL — B4B / CIFAR-100 extraction
references/
  papers/                                    # 25 PDFs total
    MAPPING_INDEX.md                         # lean router — READ FIRST (~700 words)
    MAPPING.md                               # rich per-paper entries (grep terms + sections, ~3.1k words)
    txt/                                     # pre-extracted .txt from PDFs (Phase 0 step 4)
    01–04 required (organizers' email)
    05–08 supplementary surveys
    09–19 hidden papers (SprintML 2022–2026, fetched 2026-04-26)
    20–25 competition-ready tools (Min-K%++, Watermark Stealing, DIPPER,
          Recursive Paraphrasing, WAVES, ChatGPT divergence) added 2026-04-26
          after pull researchy 01/02/03
scripts/
  extract_papers.sh                          # pdftotext → references/papers/txt/
  generate_A_fixtures.py
  generate_B_fixtures.py                     # wymaga CUDA + HF_TOKEN (Llama-3-8B gated)
  generate_C_fixtures.py                     # wymaga CUDA
  requirements_B.txt
  requirements_C.txt
```
Note: `references/repos/` is referenced in older notes but does **not exist** yet. MAPPING.md sekcja 8 ma listę repos do sklonowania (CDI, IAR Privacy, B4B, NeMo, PoW, Maini, Kirchenbauer).

## Working principles
- **Do not invent challenge details** the organizers haven't published. The Zoom info session is the next official source.
- **KRYTYCZNE: Challenges A/B/C to wyłącznie nasze ćwiczenia** wygenerowane przez Claude do nauki technik. NIE są faktycznymi zadaniami hackathonu. Faktyczne taski poznamy 2026-05-09 o 12:00 i mogą się znacząco różnić. Kod i wnioski z A/B/C to materiał referencyjny — nie gotowe rozwiązania.
- **Practice = paper replication**, not generic ML security. Each `docs/practice/challenge_*.md` is anchored in a specific paper.
- **Fixture data for challenge C must be pre-generated on a non-M4 GPU** (Jülich / Colab T4 / CUDA-teammate). M4s cannot train DDPM from scratch. Challenge B fixtures are already in LFS (`git lfs pull`).
- **Challenges D and E are OPTIONAL** (Property Inference, Model Stealing). Treat A/B/C as primary; only suggest D/E when team explicitly considers a second round of practice.
- **Deep research artifacts are heavy.** Don't auto-load `docs/deep_research/*` unless 1–4 above didn't answer. State explicitly when you're about to load one and why.

## Output rules
- No preamble. No "Great question," no "I apologize," no "Here's the implementation."
- Do not restate the user's question
- Diff-only edits — never echo full files unless explicitly asked
- Bullets > prose. Numbers > adjectives.
- No summary at the end of tool sequences

## When to update CLAUDE.md / TODO.md

Trigger an update of these files when:
- Zoom info session content drops (challenge count, format, scoring server URL → CLAUDE.md "Status" + new TODO items)
- Jülich access tested OK / fails → TODO.md status flag
- Fixture data generated by CUDA-teammate → TODO.md sekcja 3 checkboxes
- New paper added to `references/papers/` → MAPPING.md + CLAUDE.md repo structure paper count
- Challenge spec materially changes → README.md mapping in `docs/practice/`

Tip: during a session, press `#` to ask Claude to incorporate a learning into CLAUDE.md.

## Language
User communicates in Polish. Respond in Polish unless code/technical context requires English. Code identifiers, paper titles, library names — keep original. Comments in code: terse English by default.
