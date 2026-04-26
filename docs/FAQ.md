# FAQ — internal team Q&A

> Stop the same Claude question being asked three times. Append every time someone learns
> something the team should share. Format: `Q: <question>?` then `A: <answer> — <file:line> if relevant`.

Q: How do we read a paper without burning tokens?
A: First `MAPPING_INDEX.md`, then the paper's entry in `MAPPING.md`, then Grep `references/papers/txt/NN_*.txt` for terms, then offset-Read. Never load the PDF when `.txt` exists.

Q: Why is `.claude/settings.json` set to model: sonnet?
A: Plan decision (`docs/TOKEN_OPTIMIZATION_PLAN.md` decisions table). Sonnet 4.6 is within 1.2pp of Opus on SWE-bench, beats it on OSWorld, 60% cheaper, ~5× less subscription quota burn. Use `/model opus` only for hard plan-mode (h1–2), lateral brainstorm (h10–14), final ablation interpretation (h20).

Q: What if `pdftotext` produces garbled output for a paper?
A: Fallback to `pymupdf4llm` for that single file. See `docs/TOKEN_OPTIMIZATION_PLAN.md` Phase 0 step 4 failure modes.
