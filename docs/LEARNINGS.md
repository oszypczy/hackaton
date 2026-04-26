# LEARNINGS — what we discovered, what worked, what didn't

> Append after every `/compact` or significant insight. Cheap reference for the next session.

## 2026-04-26 22:40 Token Optimization Plan Phase 0
- BSD `xargs` on macOS chokes on inline `sh -c` with multi-line script ("command line cannot be assembled, too long"). Fix: extract logic to a function, `export -f`, then call via `bash -c 'fn "$@"' _ {}`. Saved in `scripts/extract_papers.sh`.
- 25 PDFs / 122 MB → 25 .txt / 5.6 MB. ~22× storage reduction; ~2.4× token reduction at read time per plan.
- Splitting MAPPING.md (3099 words = ~4.1k tokens) into INDEX (693 words) + rich MAPPING beats single trimmed file. Two-stage routing keeps default-load cost low while preserving rich grep-term content.
- Cache hierarchy lesson: any `Status (snapshot YYYY-MM-DD)` block in CLAUDE.md invalidates 90% read discount on every status update. Volatile content goes to `docs/STATUS.md` and CLAUDE.md keeps just `@docs/STATUS.md` reference.
