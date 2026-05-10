# Hybrid & alternative methods for Task 2

## TL;DR
Three highest-ROI bets: (1) **per-PII strategy mux** — direct_probe for EMAIL/PHONE, keep baseline-prefix for CREDIT to preserve the 4-4-4-4 training format the model copies (cheap +0.02-0.04, free of risk); (2) **K-shot consensus + own-logprob filter on target** — generate K=8 candidates per row, pick the medoid Levenshtein-wise OR fall back to a name/format prior when mean per-token loss is high (memorization gate); (3) **caption / meta-cue injection from the visible left-side panel of `task/` images** — Pinto-style: inject OCR'd date/place/handle into the prompt as additional anchor for memorized association. The biggest single unexplored lever is CREDIT, currently a hard floor; nothing prompt-side cracks it without either logprob-ranked candidates or visual fingerprint matching.

## Proposals (ranked by expected ROI)

### 1. Per-PII strategy mux + format-prior on low-confidence rows  [+0.02 to +0.04]
- Concept: direct_probe is best for EMAIL/PHONE, but for CREDIT it dumps 14-digit garbage instead of the trained 4-4-4-4 format. Model copies training format only with the assistant-prefix. Mux: route each PII type to its empirically best builder, then post-process every prediction through a format-prior fallback (name-based EMAIL `firstname.lastname@example.com`, format-only CREDIT `0000 0000 0000 0000`, format-only PHONE `+10000000000`) when output fails regex.
- Sketch: in `multi_eval.py`-style loop, score each (strategy, pii_type) cell on val_pii blank; freeze winning cell. At predict time, branch on `pii_type`. Add a regex gate: if pred fails the strict regex for that pii_type, replace with name+format prior.
- Per-PII: EMAIL +0.005 (already saturated), PHONE +0.005, CREDIT +0.02 (16-digit format vs current 14-digit halucynacja gives +1 char align on Levenshtein), OVERALL +0.01-0.02.
- Cost: ~30 lines, 0 GPU-min (reuses existing v2 generations + post-process).
- Risk: minimal. Worst case = identical to current.
- Verifiable on val_pii blank? **Y** (already have the data; just reroute).

### 2. K-shot consensus + own-logprob filter (memorization gate)  [+0.03 to +0.05]
- Concept: greedy is brittle when the model's confidence is low. Sample K=8 with T=0.5–0.8, then (a) Levenshtein-medoid → pick the candidate closest in mean-similarity to the others (modes cluster around the memorized answer; halucynacje scatter), and (b) compute mean per-token NLL on each candidate and use it as a memorization confidence. Below a threshold (calibrated on val_pii blank), fall back to the format-prior. Replaces "always trust greedy" with "trust only if candidates agree AND likelihood is high".
- Sketch: in `attack.py`, add `generate_k_candidates(n=8, do_sample=True, temperature=0.6, top_p=0.95)`; reuse codebase per-token loss recipe from `inference_example.py`; pick `argmax mean(pairwise_sim)` among candidates; if `mean_nll > τ_pii` swap pred for prior. τ_pii fitted on val_pii blank.
- Per-PII: EMAIL +0.02 (medoid eats trailing-period / TitleCase noise), PHONE +0.02 (medoid eats missing-`+` / wrong-suffix noise), CREDIT +0.005 (still mostly halucynacja but at least floored), OVERALL +0.02-0.04.
- Cost: ~80 lines (sampler + medoid + NLL hook), ~50 GPU-min for K=8 over 3000 rows on A800.
- Risk: sampling on overfitted model may flatten signal vs greedy; mitigate with temperature sweep on val_pii. Levenshtein-medoid from 8 noisy hallucinations can drift further from GT than greedy if no real memory exists.
- Verifiable on val_pii blank? **Y** (calibrate τ and T there).

### 3. Caption / meta-cue extraction from `task/` image  [+0.02 to +0.05]
- Concept: PDF explicitly mentions visible meta-cues (date, place, place-of-birth) on the scrubbed image — "September 21 2014, Nagoya, Japan, West Lisaburgh". Image was the memorization key (Phase 5: blank-image collapses 0.96→0.31). The Name + visible meta-text is what the model used to index this user during training. Currently we throw all of that away by sending only the question. Inject these tokens into the prompt as context.
- Sketch: one-time OCR pass over `task/` images using pytesseract (already used in scrub PoC) → extract left-panel fields (date, location, handle, like-count, post number). Build prompt: `"User profile: Name=<name>, posted on <date> from <place>, handle <handle>. What was the <pii_type> for this user during training?"`. Feed to direct_probe template.
- Per-PII: EMAIL +0.04 (handle / username often correlates), PHONE +0.02 (location → area code consistency check), CREDIT +0.005 (no semantic anchor; CC random vs user identity).
- Cost: ~120 lines (OCR + field extraction regex + prompt builder), ~10 GPU-min OCR for 1000 images, ~14 min predict.
- Risk: OCR noise on stylized fonts; over-specific prompts may push model out-of-distribution from training format → drop in lift. **Caveat:** val_pii blank-mode does NOT contain the meta-cues (image is blank), so calibration is only partial — must run a single anchor submit to confirm. Use scrubbed val_pii image (preserves left panel) for closer validation.
- Verifiable on val_pii blank? **Partial** (no meta-cues in blank). Use scrubbed val_pii (Phase 7 setup) instead.

### 4. Two-stage: image → user_id → PII  [+0.01 to +0.03 IF stage 1 works]
- Concept: model was trained on user_ids; if asked, may emit them from image. Stage 1: "What is the user id for this user?" Stage 2: "For user_id=<X>, what was the <pii_type>?" Bootstraps memory by getting the model to commit to its internal user index before answering.
- Sketch: 2 forward passes per row. Stage-1 prompt structurally identical to a question type the model saw if user_id was ever in trained answers (PDF doesn't guarantee this). If model emits gibberish → skip stage 2, fall back to v2 direct_probe.
- Per-PII: EMAIL +0.02 if user_id is memorized, PHONE +0.02, CREDIT +0.01.
- Cost: ~40 lines, ~28 min predict (2× forwards).
- Risk: model probably never trained to emit user_id (nothing in NOTES suggests it). Likely no-op or worse. **Test stage 1 on val_pii FIRST**: if mean similarity of emitted user_id to known val_pii user_id is < 0.3, abort.
- Verifiable on val_pii blank? **Y** (we know all val_pii user_ids).

### 5. Image swap from val_pii (template fingerprint test)  [+0.00 to +0.05, high variance]
- Concept: layouts in task/ have 4-6 variants; same layouts exist in val_pii. Hypothesis: model may have memorized PII keyed not on the unique user image but on the **template** + name. Swap task/ image with the closest val_pii image (matched by panel layout / color / position via cheap CLIP-embedding nearest neighbor) and probe with task/ user's name. If model outputs val_pii user's PII → image is a unique key (no benefit). If it outputs something different but coherent for task/ user's name → name dominates over image and we have a cleaner signal channel.
- Sketch: cluster all val_pii + task/ images by CLIP embedding; for each task/ row pick the nearest val_pii image of same layout cluster; swap and run direct_probe with task/ name. Compare vs no-swap baseline on val_pii (where we know GT).
- Per-PII: unclear. If template-keyed memory exists → +0.03 across all types; if image-instance-keyed → 0 or negative.
- Cost: ~80 lines (CLIP encode + KNN + swap), ~30 GPU-min total.
- Risk: high. Image was the dominant signal (Phase 5 blank ablation). Swapping may collapse score; could be net-negative. **Strictly contingency**, not P0.
- Verifiable on val_pii blank? **Indirect** — must validate on val_pii self-swap (swap user A's image with user B's similar-layout image, score against user A's GT).

### 6. Constrained decoding (CFG / regex grammar)  [+0.005 to +0.02]
- Concept: guarantee output passes per-PII regex by constraining the logits distribution at decode time. Removes wrong-mode failures (EMAIL with no `@`, PHONE without leading `+`, 14-digit CC). `transformers-cfg` LogitsProcessor, MPS-safe, drop-in.
- Sketch: 3 grammars — EMAIL: `[a-z0-9.]+@[a-z0-9.]+\.[a-z]{2,4}`, PHONE: `\+[1-9][0-9]{6,14}`, CC: `[0-9]{4} [0-9]{4} [0-9]{4} [0-9]{4}`. Use direct_probe prompt + grammar. Luhn check post-hoc, NOT in grammar (deadlock risk).
- Per-PII: EMAIL +0.005 (post-process already covers most format errors), PHONE +0.01, CREDIT +0.01 (forces 16-digit grouped format model trained on, even when memory is weak).
- Cost: ~50 lines, ~3 GPU-min overhead total. `transformers-cfg` not yet on shared venv → request install OR write minimal LogitsProcessor (~30 lines).
- Risk: constraining decode on a low-memory token may push model off-trajectory and hurt EMAIL recall (rare chars like `_` problematic). Run as additive A/B vs unconstrained on val_pii blank.
- Verifiable on val_pii blank? **Y** (drop-in test on existing pipeline).

### 7. Cross-PII multi-turn consistency  [+0.005 to +0.015]
- Concept: 3 PII types per user often share signals. EMAIL `crystal.serrano@...` and PHONE `+1505...` should agree on the same user identity. Run as a single 3-turn dialog so each turn conditions on previous answers. If turn 1 (EMAIL) emits `firstname.lastname` matching the visible name, turn 2 PHONE area code is more likely to be the user's true area code (stable per user in training).
- Sketch: 3 sequential generations in same chat session, append previous answer to context, then ask next pii_type. Reuse direct_probe wording.
- Per-PII: EMAIL flat (turn 1 has no extra context), PHONE +0.01 (anchored on EMAIL's user identity), CREDIT +0.005.
- Cost: ~30 lines, ~14 min predict (no extra GPU).
- Risk: model might drift / hallucinate confidently in turn 2/3 building on a wrong turn 1 → cascading errors.
- Verifiable on val_pii blank? **Y**.

## DON'T try
- **Chat-divergence ("repeat the word X forever")** — model is not RLHF-aligned, NOTES already plans 1 ablation and close. Don't expand scope.
- **System-prompt override / privilege escalation** — multi-eval Phase 7 showed `system_override` HURTS by -0.014 OVERALL. Model doesn't respect prompt hierarchy (not aligned). Dead end.
- **Local fine-tune / Janus-style amplification** — too expensive in 24h, requires retraining the 3.6 GB target. Ruled out in STRATEGY already; reaffirmed.
- **Local pytesseract scrub mode as calibrator** — Phase 7 showed scrubbed val_pii is WORSE than blank val_pii for ranking strategies. Stick with blank.
- **Pure greedy beam=8 on direct_probe** without sampling — beam search on overfitted model collapses to greedy modes, gives no diversity for medoid. K-shot needs sampling (T>0).

## Top single recommendation
Per-PII strategy mux (#1) + K-shot consensus with own-logprob gate (#2): cheap, additive, both verifiable on val_pii blank, expected combined OVERALL lift ~0.04-0.06 (target ≥0.42 on task/). Caption/meta-cue injection (#3) is the highest-upside single experiment but requires a real submit to confirm because val_pii blank doesn't carry the meta-cues — schedule it after #1+#2 land.
