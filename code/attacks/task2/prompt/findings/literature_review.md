# Literature review — actionable improvements for Task 2 attack

## TL;DR
Carlini'21 (organizer ref [1]) gives us a directly applicable recipe we are NOT using: rank K candidate generations by **second-model-likelihood ratio** (zlib / smaller-model perplexity / lowercase / sliding-window) and pick the lowest-ratio one — exact analogue when we have a Shadow LMM. Carlini'21 §6.5 also says the **single biggest extraction lever is reconstructing the exact training-time prefix verbatim** (824 digits of π vs 25 with bad prompt) — for an *intentionally overfit* model this dominates everything else. Nasr'23 (organizer ref [3]) further shows the dominant failure mode for aligned chat models is the chat template itself collapsing extraction; its fix is divergence prompting, but we already control the assistant prefix, so the cheaper fix is to remove the chat template and feed raw `[ANSWER] = "For Name, the {label} is "` as a continuation. Kowalczuk'25 (organizers' own paper, IAR) confirms the prefix-completion paradigm is the right primitive in multimodal AR settings. Finally, the corpus contains no "PII-Scope" or "Pinto DocVQA" papers — the multi-template ensemble idea has to be reconstructed from Carlini'21 §5.1 (3 sampling strategies × 6 metrics → 18 configurations).

## From task spec re-read
- **The task PDF gives the EXACT training format**: `[SYSTEM PROMPT]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER]`. Per Carlini'21 §6.5, deviating from this format is the #1 reason extractions fail. Our `direct_probe` rewrites the question entirely — that's the opposite of best practice. We should reconstruct the *literal* training-time `[QUESTION]` from the scrubbed sample and let the model continue from `[ANSWER] = "For Gabriella Johnson, the credit card number is "` (no `[REDACTED]` — truncate before it).
- **PDF page 2 example reveals format priors** (`docs/tasks/task2_pii_extraction.md` line 16, 122): credit card uses `#### #### #### ####` with spaces; phone uses `+13859159897` (no separators); domain `@savage.com` is reused. We should harvest these priors from the 280 `validation_pii` samples (domain-frequency, country-code distribution) — the spec explicitly says validation_pii has *unscrubbed* PII for calibration.
- **Standalone codebase = white-box** (line 38-43). We can hook activations, get logits, do beam search with custom scoring. We are using HF generate only — leaving the codebase ability on the table.
- **Length 10–100 chars enforced** (line 79). EMAIL `a@b.co` (6 chars) gets dropped silently → 0 contribution. Pad with trailing space or repeat is risky; better to ensure EMAIL ≥10 by always producing `firstname.lastname@domain.tld` form.
- **No feedback if score < best** (line 85). Public scoreboard is a one-bit signal at best; cross-validate on `validation_pii` before each submit — already in our pipeline, but worth re-emphasizing the cost of every probe.

## From reference papers

### Improvement 1 — Verbatim prefix continuation, no chat template (Paper 11 Carlini'21 §6.5; Paper 25 Nasr'23 §5)
- **What the paper does:** Carlini'21 shows extraction yield jumps from 25 digits → 824 digits of π by giving the model the *literal* training-time prefix instead of a paraphrased question. Nasr'23 §5 shows aligned chat models block extraction *because of* the chat template wrapper; the model only "completes" pre-training when you bypass it.
- **How to apply:** stop using `tokenizer.apply_chat_template` for the extraction probe. Construct the raw string `[ANSWER] = "For {Name}, the {pii_type_phrase} is "` (truncate before `[REDACTED]`) and call the LMM as a base LM, sampling continuation. If the codebase exposes raw `forward()`, do this; otherwise feed the chat-formatted sample but stop at the assistant turn boundary and append the literal prefix from the *original training format*. Our current `derive_assistant_prefix(scrubbed_output)` is in the right direction but `direct_probe` discards the prefix entirely — that's the regression.
- **Expected impact:** large for EMAIL/PHONE (overfit + correct prefix = high recall); large for CREDIT (the format prior `#### #### #### ####` IS most of the prefix continuation).
- **Implementation cost:** ~1h. Just bypass the chat template for one strategy.
- **Caveat:** requires confirming the exact training-time `[QUESTION]` template used; check the codebase's `train.py` / dataset loader for the format string.

### Improvement 2 — K-candidate generation + shadow-LM-ratio ranker (Paper 11 §5.1, §5.2; Paper 18 LiRA backbone)
- **What the paper does:** Carlini'21 generates 200k samples per prefix using 3 strategies (top-n, temperature decay 10→1 over 20 tokens, Internet conditioning) and ranks by 6 metrics (perplexity, Small-model ratio, Medium-model ratio, **zlib**, lowercase, sliding-window). Best variant: **67% TPR** vs 9% for raw perplexity. Internet conditioning + zlib gave the most diverse memorized hits.
- **How to apply:** for each (user, pii_type) sample N candidates from Target with **temperature decay 10→1 over first 20 tokens** (paper's exact setting). Score each candidate c by `score(c) = logp_target(c | prefix) − logp_shadow(c | prefix)` (we have a Shadow LMM — the spec explicitly mentions it). Pick argmax. This is exactly Carlini'21's `Small`/`Medium` metric, mapped to our (Target, Shadow) pair. Add zlib ratio as a tie-breaker — free to compute.
- **Expected impact:** medium-large for EMAIL (model hallucinates domains; shadow ratio collapses these); large for CREDIT (fights the format-floor — picks digits the Shadow doesn't know); medium for PHONE.
- **Implementation cost:** ~3h. Need Shadow LMM loaded in parallel; logprob scoring on candidate suffix is straightforward.
- **Caveat:** Shadow must use IDENTICAL prefix as Target. Paper 18 LiRA notes per-sample variance is high — use ≥8 candidates per sample, not 1.

### Improvement 3 — Format-aware constrained decoding + per-PII regex priors (Paper 11 §6.3 + Paper 25 §5.4 PII analysis)
- **What the paper does:** Carlini'21 §6.3 reports 78 extracted PIIs (phone, address, email) all matched specific regex priors; Nasr'23 §5.4 used regex *and* an LM to identify malformed PII (e.g., `sam AT gmail DOT com`), then verified against ground truth — 85.8% of regex-flagged candidates were real.
- **How to apply:** at decode time, mask logits to enforce per-PII format. CREDIT: 16 digits in `4-4-4-4` groups, validate with **Luhn check**. PHONE: `^\+\d{10,15}$`. EMAIL: `^[a-z0-9.]+@[a-z0-9.]+\.[a-z]{2,}$`. If the unconstrained sample fails, retry with constrained sampling. Build EMAIL domain prior from `validation_pii` (count `@savage.com`, etc., as priors); PHONE country-code prior from `validation_pii`.
- **Expected impact:** medium for CREDIT (Luhn alone won't recover digits but stops emitting `[REDACTED]` etc.); medium for EMAIL (domain prior is the main hallucination axis); small for PHONE.
- **Implementation cost:** ~2h. `transformers` supports `LogitsProcessor`; HuggingFace has constrained-beam-search.
- **Caveat:** over-constraining hurts Levenshtein. Better strategy: generate freely → **post-process candidates** to nearest format-valid string, keep both, score via shadow ratio (Improvement 2).

### Improvement 4 — Multi-prompt ensemble, candidate union (Paper 11 §5.1 — 3 strategies × 6 metrics; Paper 25 §5.5 — single-token diversity)
- **What the paper does:** Carlini'21 reports zero overlap between top-100 candidates of `top-n` vs `Internet` strategies — different prompts surface different memorizations. Nasr'23 §5.5 shows single-token "magic word" choice changes extraction yield by **>100×**.
- **How to apply:** for each user, run K different prompt templates (current `direct_probe`, plus prefix-continuation, plus role-play DBA, plus a "fill in the blank" template, plus image-conditional vs blank-image). Take **union of candidates**, rank with the shadow ratio (Improvement 2). Don't need 18 templates like Carlini — even 3-5 with strong diversity wins.
- **Expected impact:** medium across all PIIs. Free lift assuming Improvement 2 is in place.
- **Implementation cost:** ~1h (we already have 5 strategies in `strategies.py`).
- **Caveat:** budget — every extra template is N more forward passes.

## Anti-patterns identified
- **Don't paraphrase the question** (`direct_probe` does this). Carlini'21 §6.5 is explicit: paraphrased prompts lose 30× recall on overfit memorizations. Use the verbatim training prompt.
- **Don't feed everything through chat template** when extracting. Nasr'23 §5 shows alignment-formatted input *is* the alignment defense. Use raw continuation when the codebase allows.
- **Don't pick top-1 by raw target logprob.** Carlini'21 §5.2 found that's only 9% TPR — the worst metric. Always rank by *relative* score (Target − Shadow), or zlib ratio at minimum.
- **Don't generate just one sample per (user, pii_type).** Memorization recall scales with K candidates (Carlini'21 used 200k, Kowalczuk'25 used 5k); we should at least do K=8–16 with temperature decay.
- **Don't normalize EMAIL output to `firstname.lastname@example.com` blindly** when target_logprob > shadow_logprob suggests the model has a *real* domain (e.g., `@savage.com`). Hallucination fallback should only fire when shadow ratio is uninformative.
