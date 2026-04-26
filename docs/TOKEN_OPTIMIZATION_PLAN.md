# Token Optimization Plan — Pre-hackathon Setup

> **Self-contained handoff document.** If you are a fresh Claude session or a teammate
> picking this up, read this entire file first, then execute Phase 0. Everything you
> need is here — you do not need to read the conversation that produced it.

**Created:** 2026-04-26.
**Hackathon:** 2026-05-09/10 (CISPA Warsaw, Warsaw University of Technology).
**Authority:** This plan supersedes `TODO.md` sekcja 7 ("Synteza materiałów pod minimalny token footprint"), which was waiting for these research results.

---

## TL;DR

Goal: cut Claude Code token burn by 30–50% during the 24h event without sacrificing capability.

Three big moves (ranked by ROI):
1. **Pre-extract all 25 PDFs to `.txt`** — `pdftotext -layout`, ~10 min runtime, cuts ~60k→~25k tokens per paper read.
2. **Refactor `references/papers/MAPPING.md` from summary to ROUTER** — every paper gets `.txt` path, token estimate, key sections by name + grep terms (NOT page numbers — see Decisions), cross-refs.
3. **Slim `CLAUDE.md` to be cache-stable** — kill the volatile `## Status (snapshot ...)` section (cache-killer), add `Output rules` and `Retrieval rules`.

Plus: `.claude/settings.json` defaults, `.claudeignore`, three subagents, four slash commands, `tests/smoke.py`, `Justfile`, account verification, sleep rotation.

**SKIP:** RAG, FAISS, vector DB. Break-even >100 cached queries; hackathon won't reach it. Anthropic prompt caching at $0.30/M reads on Sonnet 4.6 has destroyed RAG's economic moat for corpora <500 pages.

**Stretch only:** Qdrant + `voyage-code-3` for semantic code search across PyTorch repos — activate ONLY if ripgrep fails on >10 code-search queries in the first 4h of the event.

---

## Why this plan exists

Source documents (read for full context, NOT required to execute the plan):
- `docs/claude_token_playbook.md` (32 KB) — research #1: Claude Code token economics for the 24h event (plan limits, model routing, prompt caching mechanics, subagent economics, anti-patterns)
- `docs/claude_retrieval_strategy.md` (162 lines) — research #2: RAG vs no-RAG verdict for our 19/25-paper corpus (recommends MAPPING.md router + native Read with `pages`/`offset`)

Convergent findings between both researches:
- Pre-extract PDFs is the single highest-leverage move
- MAPPING.md as router (not summary) is the right interface
- Cache hygiene is non-negotiable — one timestamp in CLAUDE.md invalidates 90% of the ephemeral discount on every turn
- Sonnet 4.6 default; Opus 4.7 only ≤3 specific moments; Haiku 4.5 for grep/classification subagents
- Skip RAG; corpus fits Anthropic's "<200k tokens, just include it" regime per-paper (380k total split across 19 papers = router + per-paper read pattern)

---

## Decisions made (with rationale)

| Decision | Rationale |
|---|---|
| Skip RAG/FAISS/vector DB | Break-even >100 cached queries; hackathon does <100. Engineering cost: 6–13h vs 1h for MAPPING router. Anthropic's own guidance: "<200k tokens, just include it." |
| Hybrid: MAPPING router + .txt extracts (TODO.md sekcja 7 opcja "d") | Best of (a) richer index + (b) fast grep on text, no infrastructure overhead |
| **No page numbers in MAPPING.md** | We read `.txt` files. Read tool's `pages` parameter works ONLY for PDFs. For `.txt`: Grep finds key terms → Read with `offset`/`limit`. Page numbers add maintenance cost without lookup value. Section names + grep terms are sufficient. |
| Sonnet 4.6 = default for ~80% of work | Within 1.2pp of Opus 4.7 on SWE-bench Verified, beats it on OSWorld and GDPval-AA, 60% cheaper, ~5× less subscription quota burn |
| Opus 4.7 only 3× total over 24h | Each Opus session burns ~5× quota equivalent. Reserve for: initial plan-mode (h1–2), lateral brainstorm when stuck (h10–14), final ablation interpretation (h20). |
| Haiku 4.5 for grep/classification subagents | 1/5 cost of Sonnet, 91 tok/s, matches old Sonnet 4 on SWE-bench |
| `MAX_THINKING_TOKENS=10000` (not default 31999) | Empirical (Claude Code Camp): medium effort matches high effort on most coding tasks (1051 vs 1049 output tokens) but 60s vs 20s. Default 32K is waste. |
| Person 1 (Pro plan) — never use Opus | 44K/5h Pro budget; one Opus prompt blocks the entire window |
| Extra Usage enabled on one Max account ($50–100 cap) | Insurance for final hours / weekly cap hit |
| `ttl: "1h"` explicit on every API script we write | Claude Code default TTL silently moved 1h→5min around March 2026 (issue #46829). API scripts MUST set explicit `cache_control: {"type": "ephemeral", "ttl": "1h"}`. |
| Qdrant + voyage-code-3 = stretch goal only | Activate only if code search across PyTorch repos exceeds 10× in first 4h. Setup is 15 min, but unnecessary by default. |

---

## Cache killers — what NOT to put in CLAUDE.md prefix

Cache hierarchy: `tools → system → CLAUDE.md → conversation`. Any change at level N invalidates levels N+1 onward (the 90% read discount is lost on the next turn).

DO NOT put in CLAUDE.md (or any file referenced via `@` in CLAUDE.md):
- Timestamps (e.g., `## Status (snapshot 2026-04-26)`)
- Anything that updates between sessions (counters, daily state)
- Anything dated, versioned, incrementing

DO NOT do mid-session:
- Switch model (`/model opus` ↔ `/model sonnet`) — per-model KV caches don't share
- Add/remove an MCP server
- Toggle web search
- Change thinking budget
- Add/remove an image to/from context

Mitigation: emit volatile content via `<system-reminder>` in user messages or tool results — those get cached as message history without invalidating the prefix.

---

## Read tool gotchas (April 2026)

- **25k token hard cap per file read.** Big PDFs in our corpus that trip this without `pages`: CDI (21 MB), IAR (17 MB), recursive_paraphrase (12 MB), NeMo (10 MB), WAVES (9.4 MB), strong_MIA (9 MB), Carlini diffusion (8.7 MB), watermarks_provably_removable (8.4 MB). After Phase 0 step 4 they're all `.txt` and this stops mattering for paper reads.
- **20-page hard cap per call** (PDFs only).
- **Grep does NOT search inside PDFs.** Pre-extracting to `.txt` is what makes Grep useful for the corpus.
- **PDF via vision path = ~60k tokens for a 30-page paper.** Same content as `.txt` from `pdftotext` ≈ 25k tokens. Direct ~2.4× saving per read; with caching the first read amortizes across all subsequent.

---

## ACTION PLAN

### Phase 0 — token-hygiene baseline (do FIRST, ~2h total)

#### Step 1 — `.claude/settings.json` (~5 min)

Create directory and file:

```bash
mkdir -p /Users/arturkempinski/hackaton/.claude
```

Write `/Users/arturkempinski/hackaton/.claude/settings.json`:

```jsonc
{
  "model": "sonnet",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70",
    "USE_BUILTIN_RIPGREP": "0",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1"
  }
}
```

Rationale per setting:
- `model: sonnet` — default for ~80% of work; explicit `/model opus` only for 3 hard moments
- `MAX_THINKING_TOKENS=10000` — 60–80% reduction in thinking-token spend, no measurable quality loss on coding
- `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70` — compact at 70% of context (default 83%), more headroom
- `USE_BUILTIN_RIPGREP=0` — uses system `rg`, 5–10× faster (verify it's installed: `which rg`)
- `DISABLE_NON_ESSENTIAL_MODEL_CALLS=1` — kills background Claude calls

DO NOT add `CLAUDE_CODE_SUBAGENT_MODEL=haiku` globally — routes the planner to Haiku and compounds errors. Pin Haiku per-subagent in agent definitions instead.

#### Step 2 — `.claudeignore` (~2 min)

Write `/Users/arturkempinski/hackaton/.claudeignore`:

```
references/papers/*.pdf
**/__pycache__/
.venv/
data/
*.ckpt
*.safetensors
*.pt
*.bin
node_modules/
dist/
*.lock
```

Rationale: stops Claude from rescanning PDFs after we have `.txt`, and from looking at fixture data / model checkpoints / lock files.

#### Step 3 — slim CLAUDE.md (~15 min)

User has explicitly approved these changes (decision recorded 2026-04-26).

3a. **Cut the entire `## Status (snapshot 2026-04-26)` section** from `CLAUDE.md` (currently ~lines 19–26 in the version dated 2026-04-26). It's volatile content = cache killer on every status update.

3b. **Move that content to a new file** `/Users/arturkempinski/hackaton/docs/STATUS.md`. Use this template (copy current content from `CLAUDE.md` before deleting):

```markdown
# Project status (volatile)

> NOT loaded into CLAUDE.md prefix. Edit freely; cache stays warm.
> Last updated: <date>

## Status snapshot
- <bullet>

## Active blockers
- <bullet>

## Open questions
- <bullet>
```

3c. **In `CLAUDE.md`**, replace the deleted section with one line:
```
Current status: see @docs/STATUS.md
```

3d. **Add an `## Output rules` section** to `CLAUDE.md` (after "Working principles"):

```markdown
## Output rules
- No preamble. No "Great question," no "I apologize," no "Here's the implementation."
- Do not restate the user's question
- Diff-only edits — never echo full files unless explicitly asked
- Bullets > prose. Numbers > adjectives.
- No summary at the end of tool sequences
```

Source: research #1 cites a 60% output-token reduction from these rules.

3e. **Add an `## Retrieval rules` section** to `CLAUDE.md` (right after `## Reading order for Claude`):

```markdown
## Retrieval rules

For ANY question about the paper corpus, follow this order strictly:
1. Read `references/papers/MAPPING.md` first (router — points to specific files and sections)
2. Use Grep on `references/papers/txt/*.txt` for key terms identified in MAPPING
3. Read `references/papers/txt/NN_*.txt` with `offset`/`limit` for surgical reads
4. Read raw PDF only if `.txt` extraction lost something critical (rare; flag it)

Hard rules:
- Never read >2 papers per turn (research #2: top retrieval failure on cross-paper synthesis)
- Never read full PDF when `.txt` exists (60k → 25k tokens saving)
- Read tool caps at 25k tokens/file — use `offset`/`limit` (txt) or `pages: "1-15"` (PDF) for big files
```

3f. **Add new files to the `## Repo structure` listing** in CLAUDE.md:

```
docs/
  TOKEN_OPTIMIZATION_PLAN.md     # this plan — read first if doing setup work
  claude_token_playbook.md       # research #1: token economics
  claude_retrieval_strategy.md   # research #2: RAG vs no-RAG verdict
  STATUS.md                      # volatile project state (out of CLAUDE.md prefix)
references/papers/txt/           # pre-extracted from PDFs (Phase 0 step 4)
scripts/                         # extract_papers.sh and helpers (Phase 0 step 4)
.claude/
  settings.json                  # token-hygiene defaults
  agents/                        # paper-grep, pytorch-debug, code-reviewer (Phase 1)
  commands/                      # /submit, /grill, /eval, /baseline (Phase 1)
.claudeignore                    # PDFs, fixtures, lockfiles
```

3g. **Verify**: `wc -l /Users/arturkempinski/hackaton/CLAUDE.md` should still return ≤200 lines after the changes.

#### Step 4 — pre-extract PDFs (~10 min runtime)

Prerequisite (each laptop in the team): `brew install poppler ripgrep` (poppler ships `pdftotext`).

Create `/Users/arturkempinski/hackaton/scripts/extract_papers.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p references/papers/txt
ls references/papers/*.pdf | xargs -P 8 -I {} sh -c '
  out="references/papers/txt/$(basename "{}" .pdf).txt"
  if pdftotext -layout "{}" "$out" 2>/dev/null; then
    echo "OK: $out"
  else
    echo "FAIL: {}" >&2
  fi
'
echo "---"
echo "Extracted: $(ls references/papers/txt/*.txt 2>/dev/null | wc -l) / $(ls references/papers/*.pdf | wc -l) PDFs"
echo "Total size:"
du -sh references/papers/txt
```

Make executable and run:
```bash
chmod +x /Users/arturkempinski/hackaton/scripts/extract_papers.sh
bash /Users/arturkempinski/hackaton/scripts/extract_papers.sh
```

Expected output: 25 `.txt` files in `references/papers/txt/`, each 50–500 KB. Total ~5–10 MB.

Failure modes & recovery:
- **Encrypted PDF** → `pdftotext` errors. Fix: `qpdf --decrypt in.pdf decrypted.pdf` then re-run.
- **Garbled output** (heavy math/figures, OCR'd scans) → fallback to `pymupdf4llm` for that specific paper:
  ```bash
  uv pip install pymupdf4llm
  python -c "import pymupdf4llm; print(pymupdf4llm.to_markdown('references/papers/NN_xxx.pdf'))" \
    > references/papers/txt/NN_xxx.txt
  ```
- **Marker-pdf for SOTA quality** if a critical paper extracts badly — see `docs/claude_retrieval_strategy.md` §"PDF extraction" (slower, ~30–60 min on M4 for whole corpus, only worth it if `pdftotext` loses critical equations).

#### Step 5 — refactor MAPPING.md to router format (~1–2h)

Current `references/papers/MAPPING.md` (245 lines, ~12 KB) is in summary form: "core idea / key result / use for". The refactor adds: `.txt` file path, token estimate, key sections by **name + grep terms** (NOT page numbers — see Decisions table), cross-refs.

5a. **Per-paper template** (replace existing entries with this format):

```markdown
### NN — Author Year, *Title* (Venue) [SprintML?]
- File: `txt/NN_filename.txt` | Tokens: ~XXk | arXiv: YYYY.NNNNN | Repo: github.com/...
- **Use for:** <1-line query routing — "what kind of question goes here">
- **Key sections** (grep these terms in `txt/NN_filename.txt`):
  - "Section name 1" — terms: `unique_term_1`, `unique_term_2`
  - "Section name 2" — terms: `another_term`
- **Core idea:** <1 sentence — keep>
- **Key result:** <1 number/claim — keep>
- **Cross-refs:** extends/uses NN
```

5b. **Helper for extracting section headings from `.txt`** (manual review still required):

```bash
# Heuristic: find numbered section headings
for f in references/papers/txt/*.txt; do
  echo "=== $(basename $f) ==="
  grep -nE '^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]+[A-Z][A-Za-z]' "$f" | head -15
  echo
done > /tmp/section_headings.txt
less /tmp/section_headings.txt
```

5c. **Strategy**:
- Keep the existing "Quick-attack lookup" section (sekcja 5b in current MAPPING.md) — it's already in router form
- Refactor per-paper entries (sections 1–4) with the new template
- Target: ≤2k tokens for the whole index file (research #2 recommendation)
- If the file grows past 2.5k tokens, split into:
  - `MAPPING_INDEX.md` (lean, ≤1k tokens — file path + 1-line "use for" per paper)
  - `MAPPING.md` (current rich form — load on demand)
- **Honesty check**: best done by someone who has actually skimmed the papers, OR by reading the post-Phase-0 `.txt` files to extract real section names. Do NOT have Claude alone fabricate section names — verify against the `.txt`.

5d. **Verification**:
- Every paper entry has a `txt/` path
- "Key sections" lists ≤3 entries per paper, each with ≥1 unique grep term
- "Use for" lines route a query in one sentence
- `wc -w references/papers/MAPPING.md` < 2000 words (≈ 2.7k tokens)

---

### Phase 1 — before mini-hackathon 2026-05-02/03 (~half day)

#### Step 6 — three subagents (`.claude/agents/*.md`)

```bash
mkdir -p /Users/arturkempinski/hackaton/.claude/agents
```

`/Users/arturkempinski/hackaton/.claude/agents/paper-grep.md`:
```markdown
---
name: paper-grep
description: Search the ML security paper corpus (`references/papers/txt/`) for terms. Returns ≤5 hits as `<file>:<line> — <quote>`.
model: haiku
tools: Grep, Read, Glob
---

You search `references/papers/txt/*.txt` for terms relevant to the user's query.

Rules:
- Use Grep first. Only Read files to capture context around hits.
- Return at most 5 hits, formatted: `txt/NN_*.txt:<line> — "<short quote>"`
- Do NOT summarize, interpret, or explain — just locate.
- Never read more than 2 files in full.
- If 0 hits with the literal term, try ≤2 paraphrases (e.g., "MIA" → "membership inference"); then stop.
```

`.claude/agents/pytorch-debug.md`:
```markdown
---
name: pytorch-debug
description: Triage a PyTorch stack trace or runtime error. Returns root cause hypothesis + 3 candidate fixes. Does NOT write code.
model: sonnet
tools: Read, Grep, Bash
---

Triage PyTorch errors. Output format:
- Root cause hypothesis (1 sentence)
- Top 3 candidate fixes ranked by likelihood (1 line each)
- Specific `<file>:<line>` to inspect

Do NOT write code. Do NOT explain PyTorch basics. Assume an expert user.
```

`.claude/agents/code-reviewer.md`:
```markdown
---
name: code-reviewer
description: Read-only code review for ML attack/defense implementations. Flags bugs, edge cases, perf — no fixes.
model: sonnet
tools: Read, Grep, Glob
---

Review code. Flag:
- Bugs (logic errors, off-by-one, dtype/device mismatches, missed `.eval()`/`torch.no_grad()`)
- Missed edge cases (empty inputs, NaN/Inf, batch=1, single-class)
- Perf issues (Python loops where vectorization is possible; redundant `.cpu()`/`.numpy()`)

Format: `<file>:<line> — <severity:H/M/L> — <issue>`. No fixes, no rewrites — pointers only.
```

#### Step 7 — slash commands (`.claude/commands/*.md`)

```bash
mkdir -p /Users/arturkempinski/hackaton/.claude/commands
```

`/Users/arturkempinski/hackaton/.claude/commands/submit.md`:
```markdown
---
description: Run eval, submit if score > current best
---
!just eval
Read SUBMISSION_LOG.md for current best.
If new score is better, run `just submit` and append "<timestamp> <score> <method>" to SUBMISSION_LOG.md.
Otherwise print comparison and stop.
```

`.claude/commands/grill.md`:
```markdown
---
description: 5 ranked failure modes of $ARGUMENTS, no fixes
---
Read $ARGUMENTS.
List 5 ways this code can break, ranked by likelihood. Do NOT propose fixes. Do NOT explain.
```

`.claude/commands/eval.md`:
```markdown
---
description: Run smoke test + score current attack
---
!just eval
```

`.claude/commands/baseline.md`:
```markdown
---
description: Print last 5 lines of SUBMISSION_LOG.md
---
!tail -5 SUBMISSION_LOG.md
```

#### Step 8 — supporting docs

`/Users/arturkempinski/hackaton/docs/FAQ.md` — append-only, human-curated:
```markdown
# FAQ — internal team Q&A

> Stop the same Claude question being asked three times. Append every time someone learns
> something the team should share. Format: `Q: <question>?` then `A: <answer> — <file:line> if relevant`.

Q: <example>?
A: <example> — see `<path>:<line>`.
```

`/Users/arturkempinski/hackaton/docs/LEARNINGS.md` — dump after every `/compact`:
```markdown
# LEARNINGS — what we discovered, what worked, what didn't

> Append after every `/compact` or significant insight. Cheap reference for the next session.

## <YYYY-MM-DD HH:MM> <topic>
- <bullet>
```

#### Step 9 — templates + smoke tests + Justfile

```bash
mkdir -p /Users/arturkempinski/hackaton/templates /Users/arturkempinski/hackaton/tests
```

`templates/` should contain (one Python file each — write actual content during this step, the names are placeholders):
- `pytorch_train_loop.py` — generic PyTorch training loop (DDPM / Llama-LoRA / encoder skeleton)
- `hf_dataset_loader.py` — HuggingFace streaming loader for Pile / News_2024 / CIFAR-10
- `eval_scaffold.py` — load submission JSONL + ground truth + print one score (AUC / F1 / nDCG@50)

`tests/smoke.py` — runs every attack against tiny fixtures, exits 0/1, must complete in <30s:
- Loads 100 docs (not 1000)
- Runs Min-K%++ / Kirchenbauer detect / similarity-match diffusion extraction
- Asserts each returns a valid score, no NaN

`/Users/arturkempinski/hackaton/Justfile`:
```
default:
    @just --list

eval:
    python tests/smoke.py

score CHALLENGE:
    python code/practice/score_{{CHALLENGE}}.py

submit:
    @echo "Submission packaging — TBD per challenge"

baseline:
    @tail -5 SUBMISSION_LOG.md

extract-papers:
    bash scripts/extract_papers.sh
```

#### Step 10 — account verification (per teammate)

Each teammate runs:
```bash
# 1. Verify subscription (NOT API key) is active
echo "${ANTHROPIC_API_KEY:-EMPTY}"
# Expected: EMPTY. If a key is set, Claude Code uses API billing and the subscription does NOTHING.
# Fix:
unset ANTHROPIC_API_KEY
# Also remove from ~/.zshrc / ~/.bashrc / ~/.config/fish/config.fish if persisted
claude logout && claude login   # Pick the Pro/Max account explicitly

# 2. Install live monitoring
npm install -g ccusage  # or use: npx ccusage blocks --live

# 3. Verify model defaults (after Phase 0 step 1)
claude --version
# In a session: /model  (should show: sonnet)
```

Plan check before event:
- P1 (Pro plan, 44K/5h) — never use Opus. Glue/submission/paper-grep work only.
- P2 (Max 5x, 88K/5h) — heavy ML track A.
- P3 (Max 5x, 88K/5h) — heavy ML track B. **Has Extra Usage enabled** (Settings → Usage → Extra Usage → enable, $50–100 monthly cap) as final-hours insurance.

#### Step 11 — explicit `ttl: "1h"` on every API script we write

For ANY script we write that calls the Anthropic API directly (scoring servers, batch eval, fixture generation, embedding pipelines), set the cache TTL explicitly:

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    system=[{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }],
    messages=[...]
)
```

Why: Claude Code default TTL silently moved 1h→5min around March 2026 (issue #46829). Without explicit `ttl: "1h"`, every iteration on a long-running script pays cache-write rate (1.25× input). With it: 90% discount holds for the full hour, so a 50-iteration eval loop costs ~10% of what it would otherwise.

For interactive Claude Code sessions: there is no API-level control; rely on session warmth and avoid cache-killers (see "Cache killers" section above).

---

## How to use the subagents and slash commands

This section is for any teammate (or fresh Claude session) who needs to know what's available and when to reach for it.

### Subagents — what / when / why

Subagents live in `.claude/agents/` and are spawned automatically by the main Claude when it judges the work fits. You can also force one with explicit phrasing.

| Agent | Use when | Don't use when | Example trigger |
|---|---|---|---|
| `paper-grep` | Searching the corpus for a term across many papers; want hits out of main context | You already know which 1 paper has the answer (just Read it) | "Use paper-grep to find every place LiRA is defined across our corpus" |
| `pytorch-debug` | You have a stack trace, don't know the cause, want a 3-candidate triage | The error is one line of obvious typo; just fix it | "Hand this stack trace to pytorch-debug" |
| `code-reviewer` | Just finished a non-trivial implementation, want a fresh-eyes pass before committing | One-line edit; no review needed | "Have code-reviewer audit src/attacks/dataset_inference.py" |

Subagent vs main-context decision rule (research #1, ranked by burn cost):
- Spawn ONLY if expecting >10 file reads OR >15K tokens of verbose output to keep out of main context OR 3+ truly parallel branches
- BELOW those thresholds: ask main Claude to do parallel tool calls in one turn instead (no 20K subagent bootstrap cost)
- "I need parallel subagents" is most often "I need Claude to issue 4 Greps in one turn" — prefer the latter

Cleanup: idle subagents still consume tokens. After their work is done, don't keep them around.

### Slash commands — what / when / why

Slash commands live in `.claude/commands/`. Type the slash command as the first thing in your message (or Claude Code's native ones — those are built in).

#### Custom commands (in `.claude/commands/`)

| Command | Use when | What it does |
|---|---|---|
| `/eval` | After any code change to an attack/defense | Runs `just eval` (smoke test + score) and prints result |
| `/baseline` | "What's our current best score?" | Tails `SUBMISSION_LOG.md` |
| `/submit` | When `/eval` shows your score beat the previous best | Verifies vs baseline, runs `just submit`, appends to log |
| `/grill <file>` | Early review of code you wrote — "what could break here?" | Lists 5 ranked failure modes, no fixes proposed |

#### Built-in Claude Code commands you should actually use

| Command | Use when | Why |
|---|---|---|
| `/clear` | Switching to an unrelated task | Drops history; cheaper than `/compact`. Prevents context drift between unrelated subtasks. |
| `/compact <focus>` | Long task running, context filling up, want to keep going on the same thread | Compacts but preserves the focus area. Reuses cached prefix — cheap relative to its value. |
| `/btw` | Quick aside that shouldn't pollute history | Side question, no context impact |
| `/usage` | Check budget burn so far in current session | Self-diagnostic; pair with `npx ccusage blocks --live` for window-level view |
| `/context` | Curious what's eating context window | Shows what's loaded. Use to find bloat. |
| `/effort high` | Hard debug or novel attack design | Bumps thinking budget for next turn |
| `/effort medium` | Default work | Reset to medium thinking (matches our `MAX_THINKING_TOKENS=10000`) |
| `/model opusplan` | Starting a hard architectural plan | Opus plans, Sonnet executes — high-ROI pattern (research #1) |
| `/model sonnet` | After any `opus*` use | Reset to default. Mid-session model switches lose KV cache, so do this between tasks, not within one. |
| `/plan` (Shift+Tab cycle) | Any change touching >2–3 files | Plan mode is read-only, ~80% fewer tokens than execution. Skip for typo fixes. |
| `/rewind` | Made a wrong-direction edit | Roll back to a checkpoint without re-explaining |
| `/mcp` | Audit which MCP servers are active | Each adds ~150 tok × N tools. Disable unused. |

### Standard event flow (cheat sheet)

```
Start of a new task:
  /clear
  describe the task in one prompt with @file references (not paste)

Iterate:
  edit code → /eval → improve → /eval → ...

Score improved:
  /submit

Stuck on an error:
  spawn pytorch-debug for the trace
  OR /effort high + ask Claude to brainstorm

Code feels risky:
  /grill <file>
  OR spawn code-reviewer

Window 70% full (per `ccusage`):
  /compact <focus>
  + downshift next subtask to Sonnet (or Haiku for grep/classification)

Switching to unrelated task:
  /clear
```

### Two-strikes rule

After correcting Claude twice on the same issue: `/clear` and rewrite the prompt with what you learned. Cheaper than fighting context drift for 10 turns.

---

### Phase 2 — optional / stretch (only if needed during the event)

#### 2a. Qdrant + voyage-code-3 (only if code search after 4h)

Trigger: in the first 4h of the hackathon, you've grepped PyTorch code repos >10 times and ripgrep is failing on semantic queries (e.g., "find all places that implement a LiRA-style shadow-model attack" — literal grep won't find it).

Setup (~15 min):
```bash
# 1. Vector DB on M4 (multi-arch ARM image)
mkdir -p ~/qdrant_storage
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# 2. Python deps
uv pip install qdrant-client voyageai

# 3. Voyage API key (first 200M tokens free; covers ~50M code corpus)
export VOYAGE_API_KEY=...

# 4. Index code (full script in docs/claude_retrieval_strategy.md §"Setup recipe Option B")
python scripts/index_code.py

# 5. Wire MCP server to Claude Code
cat > ~/.claude/mcp.json <<'JSON'
{ "mcpServers": {
    "qdrant": { "command": "uvx", "args": ["mcp-server-qdrant"],
      "env": { "QDRANT_URL": "http://localhost:6333",
               "COLLECTION_NAME": "code" } } } }
JSON
```

After indexing: create a Qdrant snapshot for 30s recovery on corruption. Commit `qdrant_storage/` periodically to git.

DO NOT use Chroma's `PersistentClient` from three laptops on a shared folder (HNSW cache-thrashing, lock corruption).

#### 2b. Cheat-sheet compression of deep_research artifacts

`docs/deep_research/04_model_stealing.md` (55 KB), `05_image_attribution.md` (56 KB), `06_fairness_auditing.md` (31 KB) — each one full read = 7–14k tokens.

Only worth compressing to 5–10 KB cheat-sheets if Phase 0 retrieval (MAPPING.md → grep `.txt` → read section) doesn't answer 80% of paper questions. Decide post-mini-hackathon (2026-05-02/03 debrief).

---

### Phase 3 — operational rules during the 24h event

#### Live monitoring

Pin in tmux pane on every laptop:
```bash
npx ccusage blocks --live
```

Team rule: when anyone crosses 70% of their 5h window:
1. `/compact <focus>` to reclaim context space
2. Downshift the next subtask to Sonnet (or Haiku if it's grep/classification)
3. Post a one-liner in the team chat so others know

#### Worktrees (parallel Claude sessions on same repo)

```bash
git worktree add ../hack-A -b person2/extraction
git worktree add ../hack-B -b person3/watermark
# Each Claude session: cd ../hack-A && claude
# Or with v2.1.50+: claude --worktree A
```

Avoid: two Claudes editing the same file. Worktree disjointness = parallelism wins. Same-file = 2× burn for one result.

#### Sleep rotation (24h)

- h0–8: all 3 awake (iteration peak; window resets ~h5 give a fresh budget)
- h8–14: P2 sleeps (P1 + P3 work)
- h14–20: P3 sleeps (P1 + P2 work)
- h20–24: all 3 awake (final crunch)

**Critical: never both Max 5x users sleep simultaneously** — they're the team's token capital.

#### When NOT to use Claude at all

- Boilerplate from `templates/` — copy-paste, don't ask Claude
- Patterns you've written 3+ times before (e.g., JBDA extraction loop) — type it
- Renames / refactors — `ast-grep` (`brew install ast-grep`)
- Submission packaging — `just submit`, no Claude turn
- Reading papers — that's pre-event work, not in-event work

Claude is for novel design and integration, not for typing.

---

## Verification checklist

After Phase 0, a fresh Claude session should be able to confirm:

- [ ] `ls /Users/arturkempinski/hackaton/.claude/` shows `settings.json`
- [ ] `cat /Users/arturkempinski/hackaton/.claudeignore` covers `*.pdf`, `__pycache__`, `.venv`, `data/`
- [ ] `ls /Users/arturkempinski/hackaton/references/papers/txt/ | wc -l` returns 25
- [ ] `wc -l /Users/arturkempinski/hackaton/CLAUDE.md` ≤200 lines, NO `Status (snapshot ...)` section
- [ ] `wc -w /Users/arturkempinski/hackaton/references/papers/MAPPING.md` < 2000 words, every paper entry has `txt/` path + Key sections
- [ ] Ask Claude "What does CDI's statistical test do?" → it should read MAPPING.md first, then grep + offset-Read on `txt/09_cdi_dubinski2025.txt`, NOT the PDF
- [ ] `npx ccusage blocks --live` shows a live budget meter
- [ ] On every laptop: `echo $ANTHROPIC_API_KEY` returns empty

After Phase 1:
- [ ] `ls /Users/arturkempinski/hackaton/.claude/agents/ | wc -l` returns 3
- [ ] `ls /Users/arturkempinski/hackaton/.claude/commands/ | wc -l` returns 4
- [ ] `just eval` runs `tests/smoke.py` in <30s
- [ ] At least one teammate has Extra Usage enabled with $50–100 cap
- [ ] `docs/FAQ.md` and `docs/LEARNINGS.md` exist as skeletons

---

## Anti-patterns — DO NOT do (ranked by burn cost)

1. **Letting Opus auto-spawn subagents for small tasks.** 50K tokens for 3K of work; 5–10× waste documented. Default to Sonnet; spawn subagents only for >10-file reads or 3+ parallel branches.
2. **Never running `/clear` or `/compact`.** Every retry resends entire history + system + tools; one prompt can hit 50K–300K tokens.
3. **Using Opus where Sonnet suffices.** Switching default cuts ~50% overnight.
4. **Vague queries** ("improve this codebase") that trigger 15–20 file reads = 100K+ tokens. Be specific or use plan mode.
5. **Loading whole PDFs after we have `.txt`.** Same content as `.txt` is ~40% the tokens.
6. **Pasting file content instead of `@file` references.** `@path` lets Claude read selectively.
7. **Re-explaining context** instead of `claude --resume` + named sessions + CLAUDE.md persistence.
8. **Asking Claude to echo whole files back.** Re-emits as output (5× input price). Always: "Give changes as a unified diff."
9. **Multi-turn elaboration spirals.** Batch related questions in one prompt.
10. **Default thinking budget (32K).** Cap at 10K (already in `settings.json`).
11. **Skipping plan mode** for changes touching >2–3 files. Wrong-direction implementations are the most expensive failure mode.
12. **Bloated CLAUDE.md** (>200 lines → Claude starts ignoring parts of it).
13. **MCP server bloat.** Each tool def adds ~150 tok × N tools. Run `/context` to audit; `/mcp` to disable unused.
14. **Tool-result accumulation.** Every tool result is permanent context until `/compact`. Reading a 5000-line log sticks for the rest of the session — use `head` / `tail` / `grep` on the bash side first.
15. **Letting `ANTHROPIC_API_KEY` leak into the shell.** Claude Code silently uses API billing instead of subscription, your Pro/Max plan does nothing, you're paying twice.
16. **Spawning subagents instead of parallel tool calls in main context.** Subagent bootstrap = ~20K tokens. Most "I need parallel" is really "I need 4 Greps in one turn."
17. **Touching anything in `CLAUDE.md` mid-session.** Cache invalidation cascades through all subsequent turns.

---

## Open questions / external decisions to defer

These are not blockers for executing this plan, but track them:

- **Zoom info session content** — when announced, may shift challenge focus and trigger CLAUDE.md / TODO.md update. See `docs/01_email_*.txt` and `docs/02_email_*.txt`.
- **Jülich GPU access** — register early at https://judoor.fz-juelich.de/projects/training2615; test before event. Status flag in `docs/STATUS.md`.
- **Fixture data generation for Challenges B and C** — needs CUDA teammate (~6h work). Coordinate before mini-hackathon 2026-05-02. Details in `TODO.md` sekcja 3.
- **Number of challenges to attempt (3 vs 5)** — D and E are optional. See `TODO.md` sekcja 1.
- **Carlini vs CDI for Challenge C** — see `TODO.md` sekcja 1, last bullet on Challenge C choice.

---

## References

| File | What | Size | Authority |
|---|---|---|---|
| `docs/claude_token_playbook.md` | Research #1: full token economics + Claude Code mechanics | 32 KB | Educated guess (Claude Research) |
| `docs/claude_retrieval_strategy.md` | Research #2: RAG vs no-RAG verdict for our corpus | 162 lines | Educated guess (Claude Research) |
| `CLAUDE.md` | Main project instructions — read for project overview | ~150 lines | Project source of truth |
| `TODO.md` | Hackathon prep tracker; sekcja 7 was waiting for these decisions | ~120 lines | Project source of truth |
| `references/papers/MAPPING.md` | Paper lookup — REFACTORED in Phase 0 step 5 | 245→<2000w | Project source of truth |

---

## Handoff note

If you (next Claude session, next teammate) are picking this up cold:

1. **You just read this file. Good. Continue.**
2. Skim `CLAUDE.md` for project overview (~5 min)
3. Skim `TODO.md` for hackathon prep state (~3 min)
4. Execute Phase 0 (steps 1–5) — it's mechanical, ~2h total. Steps 1, 2, 4 are independent and can run in parallel.
5. Verify with the Phase 0 checklist before moving to Phase 1.
6. Schedule Phase 1 work before 2026-05-02 (mini-hackathon weekend).

If anything in this plan is wrong against current code state on a fresh read (file moved, tool changed, settings format updated), trust the current code over this document. This was written 2026-04-26.

If you need full context on WHY a decision was made: open the relevant research file (`docs/claude_token_playbook.md` or `docs/claude_retrieval_strategy.md`). They're long but exhaustive.

End of plan.
