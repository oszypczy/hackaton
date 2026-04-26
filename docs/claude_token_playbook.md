# Burning less Claude at CISPA: a 24h cost playbook

**Bottom line.** With one Pro and two Max 5x seats, your team has roughly **440K Claude-Code tokens per 5-hour window collectively** (Pro ~44K + 2×Max 5x ~88K), and a **weekly cap on top** that can bite mid-event if anyone arrived hot. Default everything to Sonnet 4.6, route paper-grep and file-walking work to Haiku 4.5 subagents, save Opus 4.7 for ≤3 plan-mode moments, and treat prompt caching like a production SLO — a single tool-list change or timestamped CLAUDE.md edit invalidates 90% of your ephemeral discount. Get the inner verification loop (`just eval`) and 19 PDFs pre-extracted to text **before May 9**: those two moves alone typically cut 30–50% of hackathon Claude burn versus a cold start. The rest of this report quantifies each lever.

A note on April 2026 reality: Anthropic does not publish hard token quotas. All "X tokens per 5h" numbers below are community-measured (Verdent, TokenMix, Faros, Portkey), reproducible via `npx ccusage`, and consistent with the API price ratios — but treat them as planning grade, not contractual. Several April 2026 GitHub issues (#51222, #52472, #52921) document weekly-counter accounting bugs; assume ±20% on weekly numbers.

## 1. Plan limits as of April 2026

The pricing tiers are **Pro $20/mo (or $17 annual)**, **Max 5x $100/mo**, **Max 20x $200/mo**. All three give access to **Opus 4.7, Sonnet 4.6, and Haiku 4.5**. Anthropic's only official quota wording is relative: Max 5x is "5× Pro per session," Max 20x is "20× Pro." Community-measured 5-hour caps land at **~44K tokens / ~45 short messages / 10–40 Claude Code prompts on Pro**, **~88K tokens / ~225 messages on Max 5x**, and **~220K tokens / ~900 messages on Max 20x**. Variance with demand is real — peak hours (5–11am PT, 1–7pm GMT) tighten effective limits by ~7%.

**Weekly caps went live August 28, 2025** and persist in 2026. Pro has one weekly cap across all models (~40–80 Sonnet-hours/week). Max 5x has two weekly caps: an all-models cap (~140–280 Sonnet-hours) and a separate Opus-only cap (~15–35 hours). Max 20x: ~240–480 Sonnet-hours and ~24–40 Opus-hours. **Crucially, Claude Code, claude.ai, and Claude Desktop share the same bucket** — there is no separate Claude Code quota. The single most consequential mechanic for your team: **Opus burns roughly 5× faster than Sonnet against your subscription quota**, mirroring the API price ratio of $5/$25 vs $3/$15 (it's actually 1.67× per token but Opus also runs longer thinking traces, producing the ~5× operational ratio Anthropic itself cites). Opus 4.7 also ships a new tokenizer that produces **up to 35% more tokens for the same code text** — so the same prompt on Opus 4.7 vs Opus 4.6 can quietly cost more.

**At the cap**, you get a hard block with the reset time displayed. New chats do not reset; only the rolling 5h window does. **Extra Usage** (pay-as-you-go API rates) is now available even on Pro since late 2025 — enable it in Settings → Usage with a monthly cap as overflow insurance for the final hours. Hitting the weekly cap is not recoverable inside the event window; only switching to API key billing or upgrading tier rescues you.

## 2. Prompt caching mechanics, and how to actually hit cache

Caching is the single biggest free lever in Claude Code. Two TTL options exist: **5 minutes (default ephemeral)** and **1 hour (extended)**. Pricing relative to base input: **5-min cache write 1.25×, 1-hour write 2.0×, cache read 0.10×** (a 90% discount). Writes pay back after one read on 5-min TTL or two reads on 1-hour TTL. Output tokens are unaffected. **A flagged regression**: dev.to and GitHub issue #46829 report Anthropic silently downgraded Claude Code's default TTL from 1h back to 5m around March 6, 2026, which raised observed cache-creation costs 20–32% on long sessions. If you control the API call, set `"ttl": "1h"` explicitly.

Minimum cacheable prefix: **4,096 tokens for Opus 4.7 and Haiku 4.5; 2,048 tokens for Sonnet 4.6**. Below threshold, caching is silently skipped. Maximum **4 explicit `cache_control` breakpoints per request**; Claude Code's automatic cache uses one of those slots.

**Claude Code's cache hierarchy is `tools → system → CLAUDE.md → conversation`.** Any change at a level invalidates everything after. This produces a brutally specific list of cache-killers: timestamps in CLAUDE.md, a date string in your system prompt, switching `/model` mid-session (per-model KV caches don't share), adding/removing an MCP server, toggling web search, adding an image, changing thinking budget mid-conversation, and on Swift/Go any tool whose JSON keys re-serialize in different order. **Mitigation**: per Thariq (Anthropic Claude Code engineer): never edit prefix content to inject fresh data — emit it via `<system-reminder>` in the next user message or tool result. Tool results and file reads *do* get cached automatically as part of message history, which is why long sessions stay affordable; only the marginal turn pays write rate.

**CLAUDE.md structure to maximize hits.** Static content first, volatile content nowhere in the prefix. Target 60–200 lines (Karpathy's viral template benchmark; Anthropic warns "bloated CLAUDE.md causes Claude to ignore your instructions"). Five sections: project overview + stack versions, directory map, exact build/test/lint commands in code fences, conventions that *differ from defaults* with explicit anti-patterns, and quirks/gotchas. Move volatile per-task knowledge to **Skills** (`.claude/skills/<name>/SKILL.md`) which only load metadata into context until invoked, and put your 19-paper index in `docs/PAPERS.md` referenced via `@docs/PAPERS.md` rather than inlined.

## 3. Model routing and pricing in April 2026

Verified pricing per million tokens:

| Model | Input | 5-min cache write | Cache read | Output |
|---|---|---|---|---|
| **Opus 4.7** | $5.00 | $6.25 | $0.50 | $25.00 |
| **Sonnet 4.6** | $3.00 | $3.75 | $0.30 | $15.00 |
| **Haiku 4.5** | $1.00 | $1.25 | $0.10 | $5.00 |

Batch API stacks 50% off both directions. **The intuition "Opus is 5× Sonnet" is wrong for current sticker pricing — Opus is 1.67× Sonnet per token, but Opus burns subscription quota ~5× faster because it also reasons longer and produces longer outputs.** Sonnet:Haiku is 3:1, Opus:Haiku is 5:1. Both Opus 4.7 and Sonnet 4.6 now include **1M-token context at standard pricing** (the >200K premium tier is gone), though context-rot above ~200K is real and Anthropic still recommends not exceeding it.

**Quality deltas that matter for ML security work.** Opus 4.7 leads SWE-bench Verified at 87.6% vs Sonnet 4.6's 79.6% — an 8-point gap that materializes on multi-file refactors and long agentic loops, but disappears on isolated code edits. On OSWorld and GDPval-AA, **Sonnet 4.6 actually beats Opus 4.6** at half the cost — most coding work has no quality reason to use Opus. GPQA Diamond is saturated at the frontier (Opus 94.2%, Sonnet ~83%) — Opus's reasoning edge shows up most clearly on hard novel debugging, multi-file architecture, and **adversarial design** (red-teaming, attack composition, vulnerability hunting). Haiku 4.5 reaches 73.3% on SWE-bench, matching previous-gen Sonnet 4, at 91 tokens/sec and 1/5 the cost — its sweet spot is classification, tagging, document chunking, file searching, and acting as a worker subagent under a Sonnet orchestrator.

**Decision rule for your hackathon**: Sonnet 4.6 is the default for ~80% of work (PyTorch iteration, debugging, eval harness, refactors). Opus 4.7 only for: initial attack-design plan in plan mode (h1–2), the "we're stuck" lateral brainstorm (h10–14), and final ablation interpretation (h20). Haiku 4.5 for paper grep, file walking, log filtering, and any subagent that doesn't need to write code. Use **`/model opusplan`** so Opus plans and Sonnet executes — one of the highest-ROI Claude Code patterns.

## 4. Context engineering: the budget you actually have

Mental model for one Max 5x person: **88K tokens per 5h window × ~5 windows in 24h ≈ 440K tokens of comfortable budget**, before weekly caps and Opus multipliers. A single careless prompt with full file paste, no cache, and extended thinking can eat 50K–300K tokens — meaning **one bad prompt can vaporize a window**. Pro at 44K/5h is one or two careless prompts away from a hard block.

Tactics that meaningfully extend the budget:

The **"just-in-time retrieval" pattern** from Anthropic's own context engineering guide (Sept 2025): store lightweight identifiers (file paths, IDs, summaries) and let Claude fetch via `Read`/`Glob`/`Grep` tools rather than dumping corpora. Concretely, a `docs/PAPERS.md` with one-line summaries plus paths converts O(corpus) tokens per session into O(index + retrieved subset) — typically a 10–50× reduction.

**`@file` references vs paste-into-prompt**: `@path/to/file` injects the entire file plus its CLAUDE.md cascade; bare references let Claude read selectively via the Read tool, often 80% cheaper. Paste 20–30 lines of a stack trace, never the full log. For PDFs above ~10MB Claude Code errors out anyway (issues #15054, #9789); pre-extract to text via `pdftotext -layout` or `pymupdf`, save section-by-section .md files, and use page-range syntax `@paper.pdf:1-5` for surgical reads.

**`/compact` reuses your cached prefix** so it's cheap relative to its value, but reserve a compaction buffer — auto-compact triggers around 83% of context. Set `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70` to compact earlier with more headroom. Use `/clear` between unrelated tasks (cheaper than `/compact`), `/compact <focus>` to continue a long task with explicit preservation, and `/btw` for quick asides that don't pollute history. The **two-strikes rule**: after correcting Claude twice on the same issue, `/clear` and rewrite with what you learned — cheaper than fighting context drift for ten turns.

## 5. Subagent economics: when they save and when they bleed

**Bootstrap cost: ~20,000 tokens per subagent invocation.** This combines a fresh system prompt, full tool schema reload (24 built-in tools at ~5–8K tokens), CLAUDE.md re-injection, and environment block. Subagents do **not** inherit the parent's loaded context — they cold-start. Anthropic's own docs say agent teams use ~7× tokens of a single-agent session; some measurements show 15× for full Agent Teams.

**Net rule of thumb**: spawn a subagent only if you expect it to (a) read >10 files, OR (b) keep >15K tokens of verbose output out of main context, OR (c) be one of 3+ truly parallel branches. Below those thresholds, do the work inline. Multiple horror stories quantify the downside: a `/typescript-checks` workflow with 49 parallel subagents burned ~887K tokens/min and an estimated $8K–$15K in 2.5 hours; a financial-services team's 23 unattended subagents consumed $47K over 3 days; a single Pro user hit their 5-hour cap in 15 minutes with 5 auto-spawned subagents. Issue #27645 documents Opus auto-spawning subagents to fix 3–5 test files at a time when direct edits would have been **5–10× cheaper**.

**Critical alternative: parallel tool calls in the main context.** The leaked Claude Code system prompt explicitly tells Claude to batch independent tool calls in one turn. Same token count as serial calls, **no 20K bootstrap**, just no isolation. Many "I need parallel subagents" instincts are really "I need Claude to issue 4 Greps in one turn." Prefer this unless you specifically want context isolation.

For your team, the right subagents to ship in `.claude/agents/`: a **paper-grep** agent on Haiku, read-only, that searches `papers/txt/` and returns at most 5 hits as `<file>:<line> — <quote>`; a **pytorch-debug** agent on Sonnet for stack-trace triage; a **code-reviewer** read-only on Sonnet. Background subagents (Ctrl+B) cost the same per token — the gain is wall-clock parallelism, not money. **Idle teammates still consume tokens** — clean up agent teams when work is done.

## 6. Extended thinking budgets

Thinking tokens bill as **output tokens** at the model's standard output rate. There is no separate "thinking" pricing. Opus 4.7 returns only a summarized reasoning trace but charges for the full underlying count, often 3–10× the visible summary on hard problems. As of January 2026, **`ultrathink` is officially deprecated** — extended thinking is auto-enabled on supported models, with a default budget of ~31,999 tokens. Canonical control is now `/effort low|medium|high|xhigh|max` and the `MAX_THINKING_TOKENS` env var. Opus 4.7 also rejects manual `budget_tokens` (HTTP 400) — only adaptive thinking is supported.

The Claude Code natural-language keywords still work as intent signals: `think` ≈ 4K, `think hard` / `megathink` ≈ 10K, `think harder` / `ultrathink` ≈ 32K. Empirical Claude Code Camp measurements on Sonnet 4.6 found that **medium effort matches high effort on most coding tasks** (1,051 vs 1,049 output tokens) but takes 60s vs 20s — 3.5× slower for zero quality gain. Per-task cost on Opus: basic ~$0.06, `think` ~$0.10, `think hard` ~$0.25, `ultrathink` ~$0.48.

**Where thinking pays off**: hard debugging (race conditions, multi-file logic), novel attack design and red-teaming, architecture planning, math-heavy reasoning, stuck-in-loop recovery. **Where it's waste or actively harmful**: boilerplate, syntax fixes, file searches, formatting, pattern-matching tasks (research shows extended thinking hurts performance up to 36% on intuitive tasks). For your event, **set `MAX_THINKING_TOKENS=10000` in `~/.claude/settings.json`** as the default — community reports cite 60–80% spend reductions when combined with model and autocompact tweaks.

## 7. Claude Code vs Anthropic API direct

**Claude Code is included in Pro/Max and uses subscription quota** — but only if you log in via subscription. **Critical gotcha**: if `ANTHROPIC_API_KEY` is set in your shell, Claude Code authenticates via API billing and your subscription does nothing. Run `claude logout && claude login` and pick the Pro/Max account. The Anthropic API direct is a separate workspace pay-as-you-go account on platform.claude.com, not included in Pro/Max.

Field-measured crossover: under 50M tokens/month, raw API beats Pro; 50–200M tokens/month, Max 5x breaks even; 200M+, Max 20x saves hundreds to thousands monthly. The savings come almost entirely from **cache reads being free under subscription** (90%+ of heavy CC sessions are cache reads at $0.50/MTok on API).

**Drop to API for**: bulk classification of model outputs (e.g., classifying 1000 attack-success scores — Haiku 4.5 batch at $0.50/$2.50 ≈ $37 per 10,000 turns), repetitive evaluation across thousands of inputs, paper indexing pipelines (use Voyage embeddings, not Claude), fixture generation, and any deterministic-loop workflow where Claude Code's interactive overhead is pure tax. The **Batch API gives 50% off and stacks with caching** for combined ~95% savings vs sync — completes in <1h typically, max 24h, up to 100K requests per batch.

SDK options: `anthropic` (Python/TS client SDK, low-level, build your own tool loop, ideal for batch jobs and evals), and `claude-agent-sdk` (formerly `claude-code-sdk`, packages Claude Code's engine including all 24 tools and subagents as an importable Python/TS library). For your "classify 1000 outputs" workflow, write a 30-line Python script using the client SDK with `batch_create`, hit Haiku 4.5, and pay ~$1–5 of API credits instead of burning subscription quota.

**One April 2026 caveat**: as of April 4, third-party agentic tools (Cursor, OpenClaw) are blocked from using subscription quota — they must use API billing. Only Claude Code CLI, claude.ai, Claude Desktop, and Claude Cowork count.

## 8. Team coordination patterns for 3 people

The single highest-leverage anti-burn move is a **shared `docs/FAQ.md`** that prevents the same Claude question from being asked three times. Append-only, human-curated, format `Q: how do we load the diffusion model? A: see attacks/diffusion/loader.py — HF AutoPipeline OOMs at 24GB; use SDPipeline with torch_dtype=float16`. Pair it with `docs/LEARNINGS.md` (dump after every `/compact`) and `docs/PAPERS.md` (one-line summary + path per PDF).

**Shared CLAUDE.md committed to git** is the Boris Cherny pattern — Anthropic's own Claude Code team commits theirs and contributes multiple times a week, adding entries every time they see Claude do something incorrectly. All three of you `git pull` before every new session. Keep it under 200 lines, with `@docs/FAQ.md` and `@docs/LEARNINGS.md` references for progressive disclosure.

**Use git worktrees, not multiple checkouts**, when running parallel Claude sessions on the same repo. `git worktree add ../hack-extraction -b person2/extraction` lets two Claudes work on disjoint branches without stomping each other; `claude --worktree extraction` (v2.1.50+) auto-creates the worktree. The pattern from Anthropic's own HackTheBox red team: "*part of why we could achieve such speed is that we had multiple versions of Claude running at the same time tackling different challenges.*"

**Role assignment for your team:**
- **Person 1 (Pro, 44K/5h)**: Lightest workload — boilerplate, glue code, paper grep, submission scripts, eval harness changes. Acts as the "Claude operator" for shared-code changes since Pro budget is small. Do not let Person 1 use Opus.
- **Person 2 (Max 5x)**: Heavy ML track A — extraction or dataset inference attacks.
- **Person 3 (Max 5x)**: Heavy ML track B — watermarking or diffusion memorization.

Parallel querying wins when tracks are file-disjoint. It loses (and burns 2× tokens for one result) when both edit the same file. **Pin `npx ccusage blocks --live` in a tmux pane on each laptop** — when anyone crosses 70% of their 5h budget, the team rule is: `/compact`, switch the next subtask to Sonnet or Haiku.

## 9. Window management across 24 hours

The 5-hour window is **rolling, anchored to your first prompt of a fresh bucket** — usage gradually ages out rather than instantly resetting. The weekly window is also rolling, 7 days from your first prompt of that cycle. The Settings → Usage UI shows two countdown bars on Max plans: "all models" and (per the original Aug 2025 announcement) Opus-only weekly, despite the help-center page saying "Sonnet only" — treat the Opus-cap interpretation as canonical based on the original Anthropic spokesperson statement and the dashboard reality.

**Stagger sessions** so all three people don't peak at once. Suggested rhythm:
- **Hours 0–8 (all 3 awake)**: Iteration peak. Window resets ~h5 give a fresh budget.
- **Hours 8–14**: P2 sleeps, P1 + P3 work. P3's Max 5x covers ML; P1 does glue/submission monitoring.
- **Hours 14–20**: P3 sleeps, P1 + P2 work. P2's fresh window covers the next iteration burst.
- **Hours 20–24**: All 3 awake for final crunch.

Critical: never let both Max 5x users sleep simultaneously — they're your token capital.

**Recovery**: hitting the 5h cap means waiting for the displayed reset (starting a new chat does nothing). Hitting the weekly cap means switching to API key billing or upgrading tier — short waits don't help. **Enable Extra Usage** in Settings → Usage on at least one Max account before the event, with a $50–100 cap, as overflow insurance for the final hours.

**When to fall back to GPT-5.4 or Gemini 3.1 Pro**: if all three Claude accounts are weekly-capped (rare), or for the specific tasks where they outperform — Gemini 3.1 Pro leads BrowseComp (web research) and GPT-5.4 Pro leads pure browsing. For the ML security workload here, neither beats Opus 4.7 / Sonnet 4.6 on coding or attack design — they are emergency fallbacks, not first choices.

## 10. Hacks that actually win in 2026

**Pre-event preparation, ranked by impact:**

The single highest-leverage move is **pre-extracting all 19 PDFs to text before May 9**. `pdftotext -layout` to `papers/txt/`, hand-write `docs/PAPERS.md` with cite-key + year + key claim + path. This replaces "Claude reads 22MB PDF (10–50K tokens of garbled OCR)" with "Claude greps text or hits an embedding index (200–2000 tokens)" — a 10–50× context reduction on every paper lookup. Bonus: install `moinulmoin/ai-grep` (local Jina Code embeddings, zero API cost) or build a 30-line FAISS index with `sentence-transformers/all-MiniLM-L6-v2`. Beacon plugin's `PreToolUse: Grep` hook redirects every grep to semantic search — reported 70% token reduction and 64% fewer iterations.

**Make Claude terse via CLAUDE.md output rules.** A reproducible 60% output-token reduction (drona23/claude-token-efficient benchmarks) comes from explicit rules: no preamble, no restating the question, no "Great question," no "I apologize," no summary at the end of tool sequences, code-only outputs without "Here's the implementation:" framing, diff-only output for file edits, line numbers not "around the top." These reinforce Claude Code's already-terse default system prompt.

**Settings.json defaults that cut spend.** Community-reported 60–80% reduction:
```jsonc
// ~/.claude/settings.json
{
  "model": "sonnet",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    "USE_BUILTIN_RIPGREP": "0"
  }
}
```
`USE_BUILTIN_RIPGREP=0` makes Claude use your system `rg`, 5–10× faster. Caveat: `CLAUDE_CODE_SUBAGENT_MODEL=haiku` cuts subagent cost dramatically but also routes the planner to Haiku, which can compound errors — leave it on the default unless your subagents are pure search.

**Slash commands that pre-compute context** (Boris Cherny's pattern):
```markdown
# .claude/commands/submit.md
---
description: Run eval, submit if score > current best
---
!just eval
Read SUBMISSION_LOG.md for current best.
If new score is better, run `just submit` and append to SUBMISSION_LOG.md.
Otherwise print comparison and stop.
```
Inline `!command` runs at command invocation, injecting result without a Claude turn. Ship `/submit`, `/grill <file>` (5 failure modes ranked, no fixes proposed), `/baseline`, and `/eval` on day zero.

**Verification feedback loops.** Boris's #1 quality lever: "*give Claude a way to verify its work — it will 2-3× the quality.*" Build `tests/smoke.py` that loads tiny fixtures and runs every attack in <30s, and a `just eval` that returns one score. Make these the things Claude runs after every change.

**Local tools on PATH**: ripgrep, ast-grep (structural search — "find all `nn.Module` subclasses without `forward`"), fd, uv (10–100× faster pip), just (task runner). These prevent Claude from flailing with bash.

**Plan mode** (Shift+Tab cycle, or `/plan`) uses ~80% fewer tokens than execution — read-only tools only. Use it for any change touching >2–3 files. Skip for typo fixes.

**Repository templates worth forking the night before**: `affaan-m/everything-claude-code` (Anthropic hackathon winner — strip aggressively for your scope), `VoltAgent/awesome-claude-code-subagents` (cherry-pick `data-scientist`, `debugger`, `code-reviewer`), `transilienceai/communitytools` (security focus, three-agent coordinator/executor/validator pattern that maps to your team).

## 11. The ranked list of token-burn anti-patterns

1. **Letting Opus auto-spawn subagents for small tasks** — 50K tokens for 3K of work; 5–10× waste documented in issue #27645. Default to Sonnet; spawn subagents only for >10-file reads or 3+ parallel branches.
2. **Never running `/clear` or `/compact`** — every retry resends entire history + system + tools; one prompt can hit 50K–300K tokens. Use `/clear` between unrelated tasks; `/compact <focus>` to continue.
3. **Using Opus where Sonnet suffices** — switching default cuts ~50% overnight (Sonnet 4.6 is within 1.2pp of Opus on SWE-bench Verified, beats it on OSWorld and GDPval-AA).
4. **Vague queries** ("improve this codebase") that trigger 15–20 file reads = 100K+ tokens. A DEV Community study of 42 agent runs found 70% of tokens were waste (irrelevant exploration). Be specific or use plan mode.
5. **Loading whole PDFs** instead of pre-extracted sections. 22MB PDFs already error in Claude Code at 12.9MB+ (#15054). Always pre-extract.
6. **Pasting file content instead of `@file` references** — pasted content stays full for the session; `@path` lets Claude read selectively. 80% reduction on 400-line files.
7. **Re-explaining from scratch / pasting same content multiple times** — use `claude --resume`, named sessions (`/rename`), and CLAUDE.md persistence so you never re-explain.
8. **Asking Claude to echo whole files back** — re-emits as output tokens (5× input price). Always ask for unified diffs: "Give changes as a diff, not the full file."
9. **Multi-turn elaboration spirals** — batch related questions into one prompt. Cache discount only helps within the 5-min ephemeral window.
10. **Unbounded grep on huge dirs without `.claudeignore`** — `node_modules/`, `dist/`, `*.lock`, `__pycache__` re-scanned per session. Add the file.
11. **MCP server bloat** — each tool def adds ~150 tokens × N tools. Heavy MCP loadout drops usable context to ~70K. Run `/context` to see what's eating space; `/mcp` to disable unused servers.
12. **Bloated CLAUDE.md** — keep <200 lines. After 80, Claude starts ignoring parts of it.
13. **Tool-result accumulation** — every tool result is permanent context. A 5,000-line log read sticks for the rest of the session. Use a PreToolUse hook to filter (grep ERROR before Claude sees output).
14. **Default thinking budget** — `MAX_THINKING_TOKENS=10000` is the single biggest non-model lever, 60–80% reduction.
15. **Skipping plan mode** — wrong-direction implementations are the most expensive failure mode. Plan mode is ~80% cheaper.

## 12. The 24-hour budget per Max 5x user

Anchor: **~88K tokens / 5h × 5 windows ≈ 440K tokens of comfortable budget over 24h**, before weekly cap and Opus multipliers. With Opus + extended thinking, one careless window burns the whole day. Default to Sonnet, Opus only in plan mode for hard moments, Haiku for subagents.

| Phase | Hours | Token target | Model mix | Work |
|---|---|---|---|---|
| Setup | 0–1 | ~30K (7%) | Sonnet | env, paper indexing already done, smoke tests, CLAUDE.md sync |
| Baseline | 1–3 | ~60K (14%) | Sonnet + Haiku subagents | trivial baseline attack, eval harness, **first submission by h3** |
| Iteration | 3–18 | ~250K (57%) | Sonnet primary, Opus 1–2× | actual attack development, hyperparam search, multiple submissions |
| Refinement | 18–22 | ~70K (16%) | Sonnet | tuning, ablations, edge cases |
| Final submit | 22–24 | ~30K (7%) | Sonnet | final submission, write-up, demo |

Use Opus + extended thinking only three times: hour 1–2 plan mode for attack design, hour 10–14 lateral brainstorm when stuck, hour 20 ablation interpretation. ~30K Opus total = ~150K Sonnet-equivalent. More is waste.

**When NOT to use Claude at all**: copy-paste boilerplate from a `templates/` dir (PyTorch training loop, HF dataset loader); known patterns from memory (you've written JBDA extraction 3× — type it); reading papers (do it pre-event); short stack-trace bugs with obvious fixes; renames and refactors (use ast-grep); submission packaging (`just submit` once, run 50× without Claude). **Claude is for novel design and integration, not for typing.**

## Conclusion: where the savings actually live

Three buckets dominate cost. **Cache hygiene** (don't bust the prefix) sets your floor — once you're paying $0.30/MTok cache reads instead of $3.00 base, sessions become 10× cheaper. **Model routing** (Sonnet default, Opus rare, Haiku for subagents) sets your slope — getting this wrong overruns budget linearly. **Context engineering** (pre-extracted PDFs, terse output rules, plan mode, `/clear` discipline) sets your ceiling — without it, one bad prompt vaporizes a 5h window.

The non-obvious insight from the April 2026 data: **Sonnet 4.6 has closed enough of the gap to Opus 4.7 that defaulting to Opus is now actively wrong for ~80% of coding work**. Sonnet beats Opus on OSWorld and GDPval-AA, sits within 1.2 points on SWE-bench Verified, costs 60% less per token, and burns ~5× less subscription quota. Opus 4.7's gains concentrate on long-horizon agentic loops (Cognition reports "hours of coherent work"), multi-file refactors, novel attack design, and dense-screenshot vision — exactly the high-leverage moments where you should pay for it consciously, not the routine debugging where Claude Code defaults will silently route you there.

The other under-discussed mechanic: **the new Opus 4.7 tokenizer expands code by up to 35%**. Same prompt, same sticker price, more tokens billed. Measure before migrating heavy workloads.

For your specific event, the pre-event prep checklist is doing more work than any in-event optimization: pre-extracted PDFs, embeddings index, templates dir, smoke tests, slash commands, subagents, shared CLAUDE.md committed to git. A team that walks in on May 9 with this scaffolding has 30–50% more effective Claude budget than one that doesn't, and the difference compounds with every hour.

---

# One-page cheat sheet

**Plan budgets (community-measured 5h windows):** Pro 44K • Max 5x 88K • Max 20x 220K. Weekly caps exist on top — check Settings → Usage 24h before. Opus burns ~5× faster than Sonnet on quota.

**Model defaults:** Sonnet 4.6 = default. Haiku 4.5 = subagents, paper grep, classification. Opus 4.7 = plan mode + 2 hard moments only. Use `/model opusplan`.

**Pricing $/MTok (in / cache-write 5m / cache-read / out):** Opus 4.7 5/6.25/0.50/25 • Sonnet 4.6 3/3.75/0.30/15 • Haiku 4.5 1/1.25/0.10/5.

**Settings.json:**
```jsonc
{ "model": "sonnet",
  "env": { "MAX_THINKING_TOKENS": "10000",
           "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70",
           "USE_BUILTIN_RIPGREP": "0",
           "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1" }}
```

**CLAUDE.md output rules:** No preamble. No restating questions. No "I apologize." Diff-only edits. Bullet > prose. Numbers > adjectives. Run smoke test after every change. <200 lines total.

**Cache killers (avoid):** timestamps in prefix • mid-session model switch • adding/removing MCP server • toggling web search • thinking-budget change mid-conversation • images added/removed.

**Subagent rule:** spawn only if >10 files OR >15K tokens of noise to isolate OR 3+ parallel branches. Otherwise use parallel tool calls in main context (no 20K bootstrap).

**Pre-event (mandatory before May 9):** `pdftotext` all 19 PDFs to `papers/txt/` • write `docs/PAPERS.md` index • build `docs/FAQ.md` + `LEARNINGS.md` skeletons • install ripgrep/ast-grep/fd/uv/just • `templates/` dir with PyTorch loop + HF loader + eval scaffold • `tests/smoke.py` <30s • slash commands `/submit /grill /eval` • subagents `paper-grep`(haiku) `pytorch-debug`(sonnet) `code-reviewer`(sonnet,RO) • git remote for shared CLAUDE.md • Extra Usage enabled on one Max account.

**Slash commands:** `/clear` between tasks • `/compact <focus>` to continue long task • `/btw` for asides • `/rewind` checkpoint • `/usage` budget check • `/context` see what fills context • `/effort medium` (default) `/effort high` (hard debug) • `/model opusplan`.

**Roles:** P1 (Pro) = glue, submission, paper grep, no Opus ever. P2 (Max 5x) = extraction/inference. P3 (Max 5x) = watermark/diffusion. Worktrees per person. `npx ccusage blocks --live` in tmux on every laptop. At 70% of any window: `/compact` and downshift model.

**Sleep rotation:** h0–8 all awake • h8–14 P2 sleeps • h14–20 P3 sleeps • h20–24 all awake. Never both Max 5x users asleep simultaneously.

**24h budget per Max 5x user:** Setup 30K (h0–1) • Baseline 60K + first submit by h3 • Iteration 250K (h3–18) • Refinement 70K (h18–22) • Final 30K (h22–24). Total ~440K.

**Drop to API for:** 1000-output classification (Haiku batch ~$1–5) • paper embedding pipeline • repetitive eval suite • fixture generation. Use Batch API (50% off, stacks with caching).

**When to skip Claude entirely:** boilerplate from templates • patterns you've written before • short stack traces • renames (ast-grep) • submission packaging (`just submit`).

**Emergency:** weekly cap hit → API key on overflow account, or upgrade tier mid-event. 5h cap hit → downshift to other teammate, wait for reset.