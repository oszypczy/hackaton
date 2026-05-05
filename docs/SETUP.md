# Per-teammate setup checklist

> Each person on the team runs through this **before** the mini-hackathon (2026-05-02/03)
> and again on **2026-05-08** (day-before sanity check).

## 1. System tools

```bash
# macOS — required
brew install poppler ripgrep just
curl -LsSf https://astral.sh/uv/install.sh | sh   # lub: brew install uv

# verify
which pdftotext rg just uv  # all four should print paths
pdftotext -v 2>&1 | head -1  # 26.x
rg --version | head -1       # 15.x
uv --version                 # 0.9.x
```

If on Linux: `apt install poppler-utils ripgrep` and grab `just` from <https://just.systems/man/en/chapter_4.html>.

## 2. Pre-extract papers (one-time)

After cloning the repo:

```bash
bash scripts/extract_papers.sh
# expect: "Extracted: 25 / 25 PDFs"
```

If 0 / N or any FAIL: see `docs/TOKEN_OPTIMIZATION_PLAN.md` Phase 0 step 4 for `pymupdf4llm` fallback.

## 3. Claude subscription, NOT API key

```bash
# CHECK — must print EMPTY
echo "${ANTHROPIC_API_KEY:-EMPTY}"

# If a key is set, Claude Code uses API billing and your subscription does NOTHING:
unset ANTHROPIC_API_KEY
# also remove from ~/.zshrc / ~/.bashrc / ~/.config/fish/config.fish if persisted
claude logout && claude login   # pick the Pro/Max account explicitly
```

Then in a Claude Code session:

```
/model              # should show: sonnet
/usage              # smoke-check budget reads
```

## 4. Claude Code per-user config (NOT versioned)

These live in `~/.claude/settings.json` and are NOT in the repo — each teammate applies them locally.

### 4a. MCP servers (project-scoped, auto-loaded from `.mcp.json`)

After `git pull` and restart of Claude Code, verify:

```
/mcp
# expect 5 entries: github, hf, paper-search, playwright, duckdb
```

First-use prereqs per server:

| Server | Prereq | First-call cost |
|---|---|---|
| `github` | `.env` with `GITHUB_PERSONAL_ACCESS_TOKEN` | none |
| `hf` | free HuggingFace account | opens browser → OAuth login |
| `paper-search` | `uv` (section 1) | `uvx` pulls package, ~30s |
| `playwright` | none | downloads ~250 MB Chromium — **on home Wi-Fi, BEFORE event** |
| `duckdb` | `uv` (section 1) | creates `./submissions.duckdb` in cwd |

Smoke check `playwright` before event day: ask Claude "via playwright open https://huggingface.co and screenshot the page" — first run pulls Chromium, subsequent runs reuse cache.

### 4b. Token-discipline flips (`~/.claude/settings.json`)

```json
{
  "alwaysThinkingEnabled": false,
  "enabledPlugins": {
    "coderabbit@claude-plugins-official": false
  }
}
```

- `alwaysThinkingEnabled: false` — thinking tokens billed as output ($15/MTok Sonnet); silent-thinking turn ≈ $0.12. Opt-in via `/think` for known-hard subtasks (math, derivations).
- `coderabbit: false` — disabled during the 24h race (review turns cost tokens with no submission impact). Re-enable after.

## 5. Live budget monitoring

Pin in a tmux pane on event day:

```bash
npx ccusage blocks --live
# or: npm install -g ccusage
```

When you cross 70% of your 5h window:
1. `/compact <focus>` to reclaim context
2. Downshift the next subtask to Sonnet (or Haiku for grep/classification)
3. Post a one-liner in team chat

## 6. Plan tier per person (decide BEFORE event)

| Person | Plan | 5h budget | Role |
|---|---|---|---|
| P1 | Pro | 44K | Glue / submission / paper-grep work only — **never use Opus** |
| P2 | Max 5x | 88K | Heavy ML track A |
| P3 | Max 5x | 88K | Heavy ML track B. **Enable Extra Usage** ($50–100 cap) as final-hours insurance |

Extra Usage: Claude Code → Settings → Usage → Extra Usage → enable monthly cap.

## 7. Smoke test

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

just eval
# expect 6 PASS in [metrics], 3 SKIP in [attacks], "OK", exit 0
```

If `just` not found → install per step 1. If `uv` not found → install per step 1. If `numpy` missing → `uv pip install numpy datasets transformers`.

## 8. Sleep rotation (during 24h event)

- h0–8: all 3 awake (iteration peak; window resets ~h5)
- h8–14: P2 sleeps (P1 + P3 work)
- h14–20: P3 sleeps (P1 + P2 work)
- h20–24: all 3 awake (final crunch)

**Critical: never both Max 5x users sleep simultaneously** — they're the team's token capital.

## 9. Worktrees for parallel Claude sessions

```bash
git worktree add ../hack-A -b person2/extraction
git worktree add ../hack-B -b person3/watermark
# in each: cd ../hack-A && claude
```

Avoid two Claudes editing the same file. Worktree disjointness = parallel wins; same file = 2× burn.

## 10. Jülich SSH setup (each teammate, one-time)

This allows Claude to execute GPU jobs on JURECA without TOTP interruptions.

### 10a. Generate SSH key

```bash
ssh-keygen -t ed25519 -C "your_juelich_email@example.com" -f ~/.ssh/juelich_ed25519
```

### 10b. Register key on JuDOOR

1. Log in at https://judoor.fz-juelich.de
2. **Profile → SSH Public Keys → Add**
3. Paste contents of `~/.ssh/juelich_ed25519.pub`
4. Wait ~5 min for propagation

### 10c. Create your local config

```bash
cp .juelich.local.example .juelich.local
```

Edit `.juelich.local` — fill in your Jülich username and key path:

```
JUELICH_USER=your_juelich_nick    # e.g. kowalski1
JUELICH_KEY=~/.ssh/juelich_ed25519
JUELICH_HOST=jureca.fz-juelich.de
```

This file is gitignored — never commit it.

### 10d. Test connection

```bash
./scripts/juelich_connect.sh
# Prompts for TOTP (6-digit code from your authenticator app)
# Prints: "Connected to jureca.fz-juelich.de as <user>"
```

If you haven't set up MFA yet: https://judoor.fz-juelich.de → Profile → 2FA.

### 10e. Verify Claude can reach the cluster

```bash
scripts/juelich_exec.sh "hostname && squeue -u $USER"
```

Should print `jrloginXX.jureca` and your (empty) job queue.

### 10f. On event day — connect at session start

```bash
! scripts/juelich_connect.sh   # type in Claude Code prompt; TOTP once, socket lives 4h
```

After that Claude handles Jülich commands automatically (read-only free; `sbatch` asks confirmation).

---

## 11. When NOT to use Claude

- Boilerplate from `templates/` — copy-paste, don't ask
- Patterns you've written 3+ times before — type it
- Renames / refactors — `ast-grep` (`brew install ast-grep`)
- Submission packaging — `just submit`, no Claude turn
- Reading papers — pre-event work, not in-event
