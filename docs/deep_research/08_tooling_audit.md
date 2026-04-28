# CISPA Hackathon — Claude Code Loadout (24 h, Warsaw, May 9–10 2026)

Your existing stack already covers code review, paper grep, and PyTorch triage. The gaps are: live external data (HF/arXiv), a webform-driven submission path, a queryable score history, ML-security recipe automation, and aggressive token discipline for a 24 h continuous session. Recommendations below assume Sonnet 4.6 default, Haiku 4.5 for subagents, and offline-tolerant configs.

## TABLE 1 — Winning ROI (top 12)

| # | Name (type) | Install / pointer | Hackathon scenario | Failure mode prevented | M4 caveat |
|---|---|---|---|---|---|
| 1 | **HuggingFace MCP** (official, MCP) | `claude mcp add hf -t http "https://huggingface.co/mcp?login"` — github.com/huggingface/hf-mcp-server | Hour 1: pull dataset/model cards for `wikitext`, `the_pile_dedup`, watermarked-LLM checkpoints; hour 8: locate Kirchenbauer green-list reference impl; hour 14: fetch diffusion checkpoint card to plan extraction attack | Stale model IDs, wrong tokenizer, copy-pasting cards into chat | Remote — needs Wi-Fi; cache cards to `references/hf_cache/` first hour |
| 2 | **paper-search-mcp** (MCP) | `claude mcp add paper-search -- uvx paper-search-mcp` — github.com/openags/paper-search-mcp | Hour 2: arXiv + Semantic Scholar + IACR ePrint search for any paper not in your 25-PDF corpus (e.g., new dataset-inference baselines, CRYPTO 2025 watermark removals) | Wasting tokens on web summaries; missing the SoTA baseline | Pure Python; PDF download flaky offline — pre-fetch likely papers |
| 3 | **Playwright MCP** (MCP) | `claude mcp add playwright npx @playwright/mcp@latest` — github.com/microsoft/playwright-mcp | Hour 22: scoring server is a Streamlit/Gradio webform — Claude submits CSV via accessibility tree, captures the leaderboard screenshot for `/baseline` | Late-night manual submission errors, missed deadline | First run downloads ~250 MB Chromium — do this on home Wi-Fi |
| 4 | **Jupyter MCP** (Datalayer, MCP) | `claude mcp add jupyter --scope user --env JUPYTER_URL=http://localhost:8888 --env JUPYTER_TOKEN=X -- uvx jupyter-mcp-server@latest` — github.com/datalayer/jupyter-mcp-server | Hour 6: iterative attack development — Claude edits cells, runs on live MPS kernel, reads stack traces and plot outputs in a self-correcting loop | Re-paying token cost to re-load tensors; built-in `NotebookEdit` produces messy diffs (per ReviewNB) | MPS works; pin `jupyterlab==4.4.1`, `datalayer_pycrdt==0.12.17` in fresh venv (collides with other pycrdt) |
| 5 | **DuckDB MCP** (motherduck, MCP) | `claude mcp add --scope user duckdb -- uvx mcp-server-motherduck --db-path ./submissions.duckdb --read-write` — github.com/motherduckdb/mcp-server-motherduck | Hour 12: "top 3 attempts on Challenge B by TPR@1%FPR with seed and config hash"; replaces ad-hoc grep over `SUBMISSION_LOG.md` | Picking wrong baseline; double-submitting an inferior config | None (pure Python) |
| 6 | **Custom skill `ml-security-attacks`** (Skill) | Hand-roll under `.claude/skills/ml-security-attacks/SKILL.md` with subsections per challenge (diffusion-extraction, dataset/membership inference, watermark detect+remove, model stealing, property inference). Pattern: anthropics/skills SKILL.md + scripts/ | Hour 4: skill auto-loads when prompt mentions "watermark" — injects ~400-token recipe with reference algos and your `pytorch_train_loop` template | Re-deriving Carlini/Kirchenbauer math under time pressure | None — no off-the-shelf ML-security skill exists; mukul975/Anthropic-Cybersecurity-Skills is SOC/MITRE, wrong domain |
| 7 | **Subagent `submission-packager`** (Haiku) | `.claude/agents/submission-packager.md` — validates eval-server schema, runs `just eval`, computes CSV md5, appends to SUBMISSION_LOG, returns only `{score, delta, hash}` | Hour 23: parallel packaging across 3 challenges without polluting main context | Schema rejection on submit; lost score history | None |
| 8 | **Subagent `token-budget-watcher`** (Haiku) | `.claude/agents/token-budget-watcher.md` — shells `bunx ccusage blocks --json`, compares to per-block budget, returns `OK/WARN/STOP` | Every 30 min, autonomously triggered via Stop hook | Unnoticed Opus runaway burning the daily budget at hour 18 | None — read-only |
| 9 | **Subagent `attack-template-fetcher`** (Haiku) | `.claude/agents/attack-template-fetcher.md`, allowlists `hf_*`, `arxiv_*`, `paper_search_*` MCP tools only | Hour 3: "fetch reference impl + minimal eval harness for MIA on language models" — returns a 200-line scaffold without main agent ever loading the paper | 30 k tokens of irrelevant repo browsing in main context | None |
| 10 | **PreCompact hook → LEARNINGS.md + git** (Hook) | `.claude/settings.json`: `"PreCompact": [{"matcher":"auto","hooks":[{"type":"command","command":".claude/hooks/save_learnings.sh"}]}]` — script appends recent assistant turns to `LEARNINGS.md`, commits. Pattern: github.com/anthropics/claude-code/issues/15923, yuanchang.org PreCompact handover | Hour 16: 65% threshold fires — failure modes from `/grill` are preserved as bullet points; new context starts smaller | "Compaction memory loss" — Claude forgets the bug it just diagnosed | None |
| 11 | **Stop hook → git push SUBMISSION_LOG** (Hook) | `"Stop":[{"hooks":[{"type":"command","command":"git add -A SUBMISSION_LOG.md LEARNINGS.md && git commit -m \"auto $(date -Iseconds)\" && git push","async":true}]}]` — docs: code.claude.com/docs/en/hooks | Continuous backup so teammate 2 on the M4 next to you can `git pull` and resume | Lost progress on laptop crash; teammate working on stale state | `async:true` so it never blocks |
| 12 | **PostToolUse Read-trimmer (`pith`)** (Hook bundle) | `bash <(curl -s https://raw.githubusercontent.com/abhisekjha/pith/main/install.sh)` — github.com/abhisekjha/pith | Compresses every Read/Bash/Grep return before it hits context (claimed 88–91 % on large files / `npm install` style spam) | Auto-compact firing 2× per challenge | Aggressive — keep `LEAN` mode (50 % fill), avoid `ULTRA`; review on PRs to `code-reviewer` |

Notes on coverage of required slots: (a) #1, (b) #2, (c) #4, (d) #3, (e) #5, (g) #7-9, (h) #6, (i) #10-12. Slot (f) — vector store: confirmed no-go (see DO NOT INSTALL).

## TABLE 2 — Token Savers (top 8)

| # | Mechanism | Effect & source | One-liner | Risk |
|---|---|---|---|---|
| 1 | **`alwaysThinkingEnabled: false`** + opt-in `/think` for hard subtasks | Thinking tokens billed as output ($15/MTok Sonnet); MAX_THINKING_TOKENS=8000 ⇒ up to ~$0.12 per silently triggered thinking turn. Source: platform.claude.com/docs/en/about-claude/pricing (extended thinking = output-priced) | `"alwaysThinkingEnabled": false` in `~/.claude/settings.json` | Forgetting `/think` on a property-inference math step → weaker baseline |
| 2 | **Keep default = Sonnet 4.6; never default to Opus 4.7 1M** | Per Anthropic pricing page: *"Opus 4.7 uses a new tokenizer compared to previous models … may use up to 35 % more tokens for the same fixed text."* Combined with 5× input price, every Opus turn ≈ 6.7× a Sonnet turn. 1 M context expires under 5 min cache TTL → re-write at 1.25× input price | `/model sonnet` (default); `/model opus` only for the 1–2 hardest derivations | Slightly weaker on extreme long-horizon reasoning vs Opus |
| 3 | **`ccusage statusline` with visual burn rate + context %** | Live $/hr + 5 h block remaining + context %. Source: ccusage.com/guide/statusline | `"statusLine":{"type":"command","command":"bunx ccusage statusline --visual-burn-rate emoji --context-low-threshold 50 --context-medium-threshold 70"}` in settings.json | Negligible; one extra subprocess per tick |
| 4 | **`pith` PostToolUse compression** | Hook README: large file Read 1,800 → 210 tokens (−88 %); `npm install` 940 → 80 (−91 %). github.com/abhisekjha/pith | (see Table 1 #12) | Over-compression of code Claude needs verbatim; mitigated by `/pith mode LEAN` |
| 5 | **Haiku-only subagents for grep / fetch / pack** | Haiku 4.5 = $1/$5 vs Sonnet $3/$15 — 3× cheaper, and subagent context never re-enters main thread. Source: claude.com/blog subagents docs, github.com/anthropics/claude-code arXiv overview "AgentTool" | In every `.claude/agents/*.md` frontmatter set `model: haiku` and minimal tool allowlist | Haiku weaker at multi-step planning — keep agents single-purpose |
| 6 | **MAX_MCP_OUTPUT_TOKENS cap + force MCP summaries** | Default warns at 10 k; setting it lower forces tool authors' summarization paths. Source: code.claude.com/docs/en/mcp | `export MAX_MCP_OUTPUT_TOKENS=6000` in shell rc | A truncated HF model card may miss a license note — verify on submit |
| 7 | **1 h cache TTL for stable system context** (CLAUDE.md, MAPPING_INDEX, skill SKILL.md) | 5 min default expires during paper-reading pauses → full re-write at 1.25× base. 1 h write is 2× base but read is 0.1× → break-even after **1 cache hit beyond 5 min**. Source: platform.claude.com/docs/en/build-with-claude/prompt-caching | Set request header `anthropic-beta: extended-cache-ttl-2025-04-11` and `cache_control: {"type":"ephemeral","ttl":"1h"}` on the system block (note: Claude Code does not always expose this; if not, structure CLAUDE.md to be re-touched every 4 min via SessionStart hook). See also github.com/anthropics/claude-code/issues/46829 on TTL drift | 2× upfront write cost wasted if session ends within 30 min |
| 8 | **`pyright-lsp` configured to errors-only, not hints/info** | LSPs flood context with `unused-import` hints; restrict to errors keeps the signal | In pyright config: `"typeCheckingMode": "basic"`, `"reportMissingImports":"error"`, all hint-class rules `"none"` | May miss real type issues — `code-reviewer` subagent compensates on PR |

## DO NOT INSTALL (low-ROI for this threat model)

- **`obra/superpowers` plugin** — github.com/obra/superpowers. Bootstraps a mandatory brainstorm → plan → TDD red-green-refactor flow with a SessionStart-injected EXTREMELY_IMPORTANT prompt and 14 always-loaded skills. Excellent for week-long projects, **wrong tool for a 24 h ranked race**: forces test-first ceremony Claude can't skip, and the bootstrap prompt + skill manifests cost ~3–5 k tokens every session. Keep disabled.
- **`superpowers-chrome`** — duplicate of Microsoft's Playwright MCP, less mature; pick #3 instead.
- **lancedb / qdrant / chroma MCP** — your prior break-even analysis (>100 cached queries) stands. With 25 PDFs + lean MAPPING router, ripgrep wins; vector-DB MCP adds ~2 k token tool-schema cost per session and embedding latency.
- **`e2b-mcp-server`** — requires paid E2B API key (constraint forbids); Datalayer Jupyter MCP (#4) covers the same loop locally.
- **`mukul975/Anthropic-Cybersecurity-Skills`** — 754 SOC / MITRE ATT&CK skills. Wrong domain (network forensics, not ML extraction/MIA), and a 754-entry skill manifest blows context just on discovery.
- **Opus 4.7 1 M-context as default model** — see Table 2 #2; reserve for explicit hard subtasks.
- **Bulk-install marketplaces** (e.g. `ando-marketplace`'s 87 plugins, `claudemarketplaces` aggregators) — every enabled plugin's tool schemas inflate startup context. Install only the seven items above.
- **Heavy linter/LSP plugins beyond `pyright-lsp`** (rust-analyzer, eslint-LSP, etc.) — your stack is Python; extra LSPs flood diagnostics.
- **`coderabbit` PR-review plugin during the race** — already installed; *disable* until post-hackathon; it adds review turns that cost tokens with no submission impact during the 24 h.

### Reproducibility (per-teammate, ≤30 min)

```bash
# 1. MCPs
claude mcp add hf -t http "https://huggingface.co/mcp?login"
claude mcp add paper-search -- uvx paper-search-mcp
claude mcp add playwright -- npx -y @playwright/mcp@latest
claude mcp add jupyter --scope user --env JUPYTER_URL=http://localhost:8888 \
  --env JUPYTER_TOKEN=$JT -- uvx jupyter-mcp-server@latest
claude mcp add duckdb --scope user -- uvx mcp-server-motherduck \
  --db-path ./submissions.duckdb --read-write
# 2. Hooks + statusline
bash <(curl -s https://raw.githubusercontent.com/abhisekjha/pith/main/install.sh)
# 3. Token savers in ~/.claude/settings.json (alwaysThinkingEnabled, statusLine, MAX_MCP_OUTPUT_TOKENS)
# 4. Pull team .claude/{agents,skills,hooks} from project repo (already version-controlled)
```

Pre-flight on home Wi-Fi: warm Chromium, `uvx` caches, HF model cards, and the 25 PDFs into `references/`. Venue Wi-Fi outage then degrades you only on (1) and (2).