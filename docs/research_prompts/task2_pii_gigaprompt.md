# Task 2 PII Extraction — deep-research gigaprompt

> **Co to jest:** prompt do wklejenia w Claude Research / Perplexity Deep Research / Gemini Deep Research.
> **Cel:** raport `.md` (~4–6k słów) który uzupełnia *gap* w naszych lokalnych zasobach, NIE re-deriva tego co już mamy w `references/papers/` + `docs/deep_research/02_model_inversion.md`.
> **Hard constraint:** dane głównie z 2025–2026, foundational tylko explicite oznaczone.
>
> **Confidentiality note:** paraphrase'owany task spec (nie literal cytat z PDF organizatorów). Setup ma być rozpoznawalny dla researchera ale nie identyfikuje hackathonu. Jeśli regulamin restryktywny — zapytaj orgów przed paste.
>
> **Zalecana kolejność:** Claude Research (najlepsza synteza paperów) → Perplexity (najświeższy web/GitHub) → Gemini DR (broad landscape). Diff outputów = większa wartość niż jeden tool.

---

## GIGAPROMPT (start of paste)

```
# Role
You are a senior AI security research analyst. Your audience is a 3-person team
at a 24-hour hackathon (Europe, May 2026) attacking a privacy task on a
multimodal language model. They have ~22 hours from when they read your
report. They are technical (PyTorch / HF transformers fluent) and time-pressed.

# Mission
Produce a comprehensive, action-ready Markdown report (target 4000–6000 words,
hard cap 8000) that **fills the gap** in their existing knowledge about
PII reconstruction attacks on multimodal LMs. Do not re-derive material they
already own (listed below). Focus on what's NEW, what's MULTIMODAL-specific,
and what's PRACTICAL under hackathon constraints.

# Hard constraints (non-negotiable)
- **Recency**: Prioritize work published in **2025 or 2026**. Today is
  2026-05-09. Foundational pre-2025 work (Carlini USENIX'21, Carlini USENIX'23,
  Nasr 2023, LiRA S&P'22, Min-K%++ ICLR'25 spotlight) MUST be marked
  `[FOUNDATIONAL — already covered in our corpus]` and skipped unless a 2025+
  paper directly extends them.
- **Cite or omit**: Every external claim → URL/DOI + month-year. Distinguish
  peer-reviewed (venue + arXiv ID) from preprints from blog posts from
  GitHub READMEs.
- **Anti-fabrication**: Do NOT invent paper titles, author names, GitHub URLs,
  library APIs, or numerical results. If a fact cannot be verified from a real
  source you can cite, write `[NOT VERIFIED]` and skip rather than guess.
  If a code snippet cannot be confirmed against a real repo's API, present it
  as pseudocode with the comment `# pseudocode — adapt to actual API`.
- **Honest uncertainty**: When literature is silent on a question, write
  `No published evaluation found for this specific scenario.` Do NOT extrapolate.
- **Resource flag**: If a technique requires >24 GPU-hours of A100/H100 time,
  proprietary data, or training-from-scratch, label `RESOURCE-HEAVY` and
  suggest a cheap proxy.
- **Apple Silicon / CUDA flag**: For each tool/library you cite, state whether
  it works on MPS / MLX or is CUDA-only. The team is primarily on Apple M4
  laptops; one member has CUDA + access to an HPC cluster.
- **Output language**: English. Pure Markdown (no LaTeX outside inline `$...$`,
  no HTML). Code blocks tagged with language.

# Task we are attacking (paraphrased)
A multimodal language model (LMM) has been INTENTIONALLY OVERFIT on a small
visual question-answering (VQA) dataset. Each sample contains a fictitious
user with three pieces of personally identifying information: an email, a
credit-card number, and a phone number. Memorization is amplified on purpose.

The team has white-box access to:
- Weights of the **target LMM** (overfit on PII).
- Weights of a **shadow LMM** (same training pipeline, no PII data).
- The **full standalone codebase** — architecture, inference, training pipeline.
- A **scrubbed evaluation set** of 1000 samples × 3 questions = 3000 queries.
  Each query has the format
  `[SYSTEM_PROMPT]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER]`
  where `[ANSWER]` contains `[REDACTED]` instead of the real PII, and
  `[IMAGE]` is the original training image with PII regions WIPED.
  `[QUESTION]` references the user by name.
- A **reference set** of 280 samples × 3 questions where the original
  PII is still visible — used for local validation only.

The team must produce a CSV `id, pii_type, pred` (3000 rows). `pii_type ∈
{EMAIL, CREDIT, PHONE}`. `pred` length must be 10–100 chars inclusive
(out-of-range = rejected). Score = `1 − Normalized_Levenshtein(GT, Pred)`,
averaged across rows. 30% public / 70% private split (final ranking on the
private 70%, which is held back). Submission cooldown is 5 min (2 min on
failure); no feedback if a submission scores below current best — submissions
are rate-limited information.

The architecture is NOT stated; it is likely a LLaVA-class or Qwen-VL-class
model with a small LM backbone (≤7B), but the codebase will reveal the truth.
Treat both as plausible.

# What the team already owns — DO NOT RE-DERIVE
The team has local PDFs and a synthesized hackathon playbook covering the
following — treat them as the SOLVED baseline and report only what is NEW
or SUPERIOR relative to them:

1. Carlini, Tramèr, Wallace et al., *Extracting Training Data from Large
   Language Models*, USENIX Security 2021 (arXiv:2012.07805). Six MIA features
   (perplexity, small/XL ratio, zlib ratio, lowercase ratio, window-PPL,
   reference-model LR), >600 verbatim extractions from GPT-2 XL.
   **Section 6.5 — context-dependency:** the same string can be reproduced
   only when the EXACT training-time prefix is supplied; "pi is 3.14159"
   yields 30× more memorized digits than "3.14159" alone. Implication:
   reconstruct the dialogue format verbatim up to the `[REDACTED]` boundary.
   **Section 7:** 33 insertions of a string sufficient for full memorization
   in 1.5B GPT-2; on an INTENTIONALLY OVERFIT model the threshold collapses
   to k≈1 (every PII appears once and is memorized). Naive extraction
   should already work; instrument validation_pii first to confirm.
   `[FOUNDATIONAL — PDF + .txt in corpus as paper 11 (renumbered from 05 during 2026-05-09 merge to avoid collision with teammates' 05_a/05_duci); full code in our playbook]`
2. Carlini et al., *Extracting Training Data from Diffusion Models*, USENIX
   Security 2023 (arXiv:2301.13188). Generate-and-cluster pipeline, ≥109
   verbatim Stable Diffusion extractions.
   `[FOUNDATIONAL — already in corpus]`
3. Nasr, Carlini, Hayase, Jagielski et al., *Scalable Extraction of Training
   Data from (Production) Language Models / ChatGPT divergence attack*, 2023
   (arXiv:2311.17035). Divergence prompt collapses alignment, ~150× baseline
   leak rate, ~$200 → 10k unique strings INCLUDING PII.
   `[FOUNDATIONAL — already in corpus, including code]`
4. Carlini et al., *Stealing Part of a Production Language Model*, ICML 2024
   Best Paper (arXiv:2403.06634). Logit-bias + SVD recovers hidden dim and
   final-layer projection.
5. Carlini, Chien, Nasr et al., *MIA From First Principles* (LiRA), S&P 2022
   (arXiv:2112.03570). Per-sample Gaussian likelihood ratio, the universal
   shadow-model methodology.
6. Maini, Jia, Papernot, Dziedzic, *LLM Dataset Inference*, NeurIPS 2024
   (arXiv:2406.06443). Aggregate weak per-sample MIA features via Welch t-test.
7. Zhang et al., *Min-K%++*, ICLR 2025 spotlight (arXiv:2404.02936).
   Standardized per-token log-prob, 6.2–10.5% AUC over Min-K% on WikiMIA.
8. Hintersdorf, Struppek, Boenisch, Kersting et al., **NeMo: Localizing
   Memorization in Diffusion Models**, NeurIPS 2024 (arXiv:2406.02366).
   Cross-attention value-matrix memorization neurons.
9. Kowalczuk, Dubiński, Boenisch, Dziedzic, *Privacy Attacks on Image
   Autoregressive Models*, ICML 2025 (arXiv:2502.02514). Organizers' own lab;
   86.38% TPR@1%FPR on VAR-d30; dataset inference with 6 samples.
10. Hayes, Boenisch, Dziedzic, Cooper et al., *Strong MIAs on LLMs*,
    NeurIPS 2025 (arXiv:2505.18773). Reference-model scaling for LiRA on LLMs.
11. Zhang et al., *CoDeC: Detecting Data Contamination via In-Context Learning*,
    NeurIPS Workshop 2025 (arXiv:2510.27055).

The team also has a synthesized 8000-word "ML privacy playbook" with PyTorch
code for: Fredrikson MI, PPA / IF-GMI, perplexity-zlib ranking, Min-K%++,
LiRA online-Gaussian-fit, Maini DI aggregator, Inverting Gradients (DLG/IG),
Carlini DM generate-and-cluster, CoDeC, DP-SGD via Opacus.

DO NOT re-explain any of these. If you mention them, mention them in one
sentence as a known anchor and immediately pivot to what's NEW.

# What is GENUINELY MISSING (this is what you research)

The team's gap, in priority order:

1. **Vision-language / LMM-specific extraction in 2025+.** All the local
   work is on LLMs (text-only) or diffusion models (image-only). Nothing
   directly addresses an autoregressive LM that takes IMAGE + TEXT and
   memorizes (image, name) → PII triples. What is the 2025–2026 literature
   on memorization, MIA, and extraction in LLaVA / Qwen-VL / IDEFICS /
   MiniCPM-V / InternVL / Pixtral / Llama 3.2-Vision / Phi-4-Multimodal?
2. **PII reconstruction with format priors.** The PII is structured
   (16-digit credit card with Luhn, E.164 phone, RFC-5321 email). What
   2025+ work combines extraction with constrained / format-aware decoding?
   Which library is the current default (`outlines`, `lmql`, `guidance`,
   `transformers-cfg`, `vllm` grammars)? What gain does format-aware
   decoding give over free-form sampling on memorized PII?
3. **Scrubbed-image-as-key.** The image was the model's training key for
   that user; PII regions are now wiped, but the overall image structure
   remains. Does the residual image content still index the memorized PII?
   Any 2025+ paper on multimodal MIA where the image is partially redacted?
4. **Overfit-targeted attacks.** Most extraction literature targets
   production models trained one epoch on huge corpora. Our target was
   intentionally overfit on a small dataset — memorization is extreme.
   What attacks are specifically tuned for this regime? Easy-mode baselines?
   Predictable failure modes when memorization is "too strong"?
5. **Recent SprintML / CISPA work** by Adam Dziedzic, Franziska Boenisch,
   Jan Dubiński, Antoni Kowalczuk, Bartłomiej Marek, Marek Kowalski,
   Pratyush Maini, and collaborators. Prioritize 2025–2026. The hackathon
   task is designed by their lab; their published threat model is the
   closest oracle for what the attack is "supposed to look like."
6. **Submission strategy under rate limits.** With 5-min cooldown and no
   feedback below current best, submissions are stochastic-bandit signals.
   Any 2024–2025 hackathon write-up / blog with empirical scheduling advice?
7. **Public/private leaderboard divergence.** Final scoring uses an extended
   private set; methods over-tuned to public regress. Any 2025+ work or
   ML competition retrospective that quantifies this drop and identifies
   robust vs. fragile method classes?

# Research Sections (THE REPORT)

For every technique below, return EXACTLY this template:
- **Name** + 1-line summary
- **Source(s)**: full citation (authors, title, venue, year, arXiv ID, URL)
- **Threat model**: white/grey/black-box; data needed; cost
- **Reported metrics on closest analog** (dataset, extraction rate or
  similarity score) — numerical
- **Mapping to our setup**: directly applicable / requires adaptation /
  not applicable, with one-sentence reason
- **Concrete recipe**: step list or code/pseudocode block (≤25 lines).
  Mark `# pseudocode` if the API isn't verified.
- **Validation hook**: how to test on our 280-sample reference set in
  <30 minutes, end-to-end
- **Compatibility**: Apple MPS / MLX / CUDA-only / heavy
- **Red flags**: compute, special access, ethics, license

## Section A — Multimodal LMM extraction (2025+)
Search the 2025–2026 literature for: training-data extraction, memorization,
MIA, and PII leakage in **vision-language models specifically**. Cover
both autoregressive LMMs (LLaVA, Qwen-VL, IDEFICS, InternVL, MiniCPM-V,
Pixtral, Llama 3.2-Vision, Phi-4-Multimodal) and image-autoregressive models
(VAR, RAR, MAR — only as analog, since IAR is already in the team's corpus).

Specifically resolve:
- A.1 Where does memorization live in an LMM? Vision encoder, projection
  MLP, cross-attention, LM backbone, LM head? Cite per-paper localization
  results.
- A.2 What attacks have been demonstrated on LMMs in 2025+ (with reported
  metrics)? Aim for ≥6 entries.
- A.3 How does the *image* act as a memorization key? If the image is
  partially scrubbed but not replaced, what fraction of the memorization
  signal survives?
- A.4 Has anyone updated Nasr-style chat-divergence for MULTIMODAL chat
  models in 2025+?

## Section B — Format-aware decoding for PII (2025+)
Resolve:
- B.1 Library landscape 2025+: `outlines`, `lmql`, `guidance`,
  `transformers-cfg`, `vllm` grammars, `xgrammar`. Verify each repo is
  active in 2025/2026. State which integrates with HF transformers
  multimodal pipelines (LlavaForConditionalGeneration etc.) without monkey
  patching.
- B.2 Empirical gain of constrained decoding on memorized PII (any 2025
  paper or blog with numbers).
- B.3 Risk: when does constrained decoding HURT? E.g. when the model
  remembers the PII but in slightly different format → constraint
  forces wrong digits. Cite or flag as `[NOT VERIFIED]`.
- B.4 Concrete recipes: a 10–20 line snippet for each of the three PII
  types using whichever library you recommend. Mark pseudocode if the
  API isn't verified.
- B.5 Levenshtein-aware reranking: any 2024+ work on selecting predictions
  to minimize Levenshtein expectation under uncertainty?

## Section C — Overfit-targeted extraction
Resolve:
- C.1 Has anyone published "easy-mode" baselines on intentionally overfit
  models in 2024–2026 (e.g., Tramèr SPY-Lab fine-tune extraction follow-ups,
  small-dataset memorization studies)?
- C.2 At what point does memorization stop being extractable by greedy
  decoding and need search? Empirical threshold?
- C.3 What heuristics flag "this model definitely memorized X" given
  white-box access + 280 ground-truth samples? Perplexity gap, gradient
  signature, attention concentration?
- C.4 Specific failure modes when the model is OVER-overfit (e.g., can't
  generalize the prompt structure, only emits PII verbatim from training
  prompt format).

## Section D — Scrubbed-image conditioning
Resolve:
- D.1 Any 2025+ paper that empirically measures the marginal contribution
  of image vs question-only on multimodal extraction?
- D.2 Counterfactual image studies (replace image with mean-image,
  random-noise, blank, semantically-matched-but-different) — known
  results?
- D.3 Practical recipe for ablating image contribution on the team's
  280 reference samples in <1 hour.

## Section E — Recent SprintML / CISPA work (2025+)
Search exhaustively (Google Scholar, arXiv, OpenReview, lab websites) for
2025–2026 work by: Adam Dziedzic, Franziska Boenisch, Jan Dubiński,
Antoni Kowalczuk, Bartłomiej Marek, Marek Kowalski, Pratyush Maini,
Igor Shilov, Yves-Alexandre de Montjoye, Florian Tramèr (collaborates),
and the SprintML lab (https://sprintml.com). For each new paper:
- One-paragraph summary
- Direct relevance to our task (high / medium / low) with reason
- Whether the paper publishes code

## Section F — Tooling, code, prompt libraries (2025+, verified)
- F.1 Open-source extraction codebases active in 2025/2026. Verify each
  GitHub URL exists and last commit ≥2025-01-01. Examples to investigate
  (verify before quoting): `iamgroot42/mimir`, `pratyushmaini/llm_dataset_inference`,
  `JonasGeiping/breaching`, `eth-sri/watermark-stealing`,
  `sprintml/privacy_attacks_against_iars`, `ml-research/localizing_memorization_in_diffusion_models`,
  `ffhibnese/Model-Inversion-Attack-ToolBox`, garak, promptfoo. Add others
  found during search. Report: stars, last-commit, attacks implemented,
  CUDA / MPS / MLX support.
- F.2 LMM-specific extraction repos in 2025+: anything beyond pure-LLM
  tooling that handles multimodal pipelines.
- F.3 Inference utilities for batched LMM generation with custom hooks
  in 2025/2026: `transformers`, `vllm`, `lmdeploy`, `sglang`, `TensorRT-LLM`,
  Apple `mlx-vlm`. Compare batched-throughput and ease of attaching
  attention hooks. State which run on Apple M4.
- F.4 Levenshtein library that matches the standard "Normalized Levenshtein"
  definition `dist/max(len(a), len(b))`: `python-Levenshtein` vs
  `rapidfuzz` vs custom — confirm the formula.

## Section G — Open uncertainties / decisions for the team
This section is NOT for you to answer with literature — it FRAMES decisions
the team must take after they look at the data and codebase. For each:
- **Question**
- **Why it matters** (impact on score, 1 sentence)
- **What evidence resolves it** (specific check, runtime <30 min)
- **Default if no time** (best guess based on lit)
- **Priority** P0 / P1 / P2

Include exactly the following 8:
1. Which prompt template extracts the most PII: naive `[REDACTED]`-fill,
   role-play, full-dialogue replay, assistant-prefix manipulation, or
   chain-of-thought "complete the user's record" framing?
2. Image conditioning ON vs OFF vs blank-replacement vs noise-replacement —
   what helps?
3. Sampling: greedy vs beam(b∈{4,8,16}) vs T=0.7 + N samples + best-by-
   likelihood vs constrained decoding.
4. Shadow model gate: when to use `logp_target − logp_shadow` and at
   what threshold? Cost: 2× inference.
5. Format-aware decoding ON vs OFF — when does constraint help, when does
   it actively hurt?
6. White-box hooks (attention probing, gradient ascent on PII tokens,
   embedding inversion of `[REDACTED]`) — invest engineering hours, or
   stick to prompt-only?
7. Submission scheduling under 5-min cooldown (~12 submissions/hour
   maximum; no feedback below current best). How to allocate over 22 hours?
8. Public-vs-private leaderboard generalization — which method classes
   are typically robust, which over-fit to public?

## Section H — Suggested timeline (action-oriented)
Four tiers. For each, list ≤5 concrete tasks with rough time-box and
decision criterion:
- **First 30 min** — environment + data + first baseline (sanity submission)
- **First 2 hours** — naive extraction, calibrate on `validation_pii`
  (280-sample reference set), first real submission
- **Hours 2–8** — best-technique scaling + format-aware decoding +
  shadow comparison
- **Hours 8–22** — refinement, white-box attempts only if time, generalization
  stress-test, final submission
- **Contingency** — if score plateaus near 0.4, what's the +0.1 next move?
- **Contingency** — if score is already very high (~0.9), what's the
  generalization-protection move?

# Output structure (mandatory skeleton)

The final report MUST follow this skeleton exactly:

```
# PII Extraction from Multimodal Models — Hackathon Research Briefing

## TL;DR (≤150 words, top 3 actions for the first 2 hours)
## Section A — Multimodal LMM extraction (2025+)
### A.1 Where memorization lives
### A.2 [Technique 1]
### A.3 [Technique 2]
... (one subsection per technique; aim for 6–10)
## Section B — Format-aware decoding for PII
## Section C — Overfit-targeted extraction
## Section D — Scrubbed-image conditioning
## Section E — Recent SprintML / CISPA work (2025+)
## Section F — Tooling, code, libraries (verified 2025+)
## Section G — Open uncertainties / decisions
| # | Question | Priority | Why it matters | Evidence | Default |
## Section H — Suggested timeline
## Appendix 1 — Reading list, tiered (MUST / SHOULD / MAY); ≤25 entries
## Appendix 2 — Risk register (5–10 risks, with mitigation)
## Appendix 3 — Glossary (≤10 acronyms)
```

# Tone & style
Concise. Numbers over adjectives. Code over prose. No filler ("It is
important to note that…", "In conclusion…"). Tables when comparing
>3 items. A sleep-deprived engineer at 03:00 must be able to act on
each section in <5 min of skim-reading.

# Final checklist before you start
- [ ] Recency hard-locked to 2025–2026
- [ ] Every external claim cited; otherwise omitted
- [ ] No fabricated paper titles, author names, GitHub URLs, library APIs
- [ ] Distinguish "verified" from "speculative"
- [ ] Resource constraints flagged; Apple Silicon vs CUDA noted
- [ ] Skipped material the team already owns (Section "What the team
      already owns")
- [ ] Section G frames questions, doesn't answer them
- [ ] Final report in pure Markdown
- [ ] Length 4000–6000 words, hard cap 8000

The team prints this report at 13:00 and acts on it through the night.
Be ruthlessly useful.
```

## GIGAPROMPT (end of paste)

---

## Po użyciu (merge strategy)

1. Wklej w 3 toole. Czas: ~15 min Claude Research, ~10 min Perplexity, ~20 min Gemini DR.
2. Każde narzędzie zwróci osobny raport `.md`.
3. Merguj ręcznie do **`docs/research_prompts/task2_research_brief.md`** — bierz najlepszą sekcję z każdego źródła:
   - Claude Research: Sections A, B, C (głębia + cytowania)
   - Perplexity: Section F (tooling, świeże GitHub repos), Section E (SprintML latest)
   - Gemini DR: Section A.2-A.3 multimodal landscape, Section H timeline
4. Sprawdź anty-fabrykację: dla każdego cytowanego repo zrób `WebFetch` na URL, dla każdego paperu spr. arXiv ID. Jeśli nie żyje → wykreśl.
5. Section G (open uncertainties) = bezpośredni input do `TODO.md` jako P0/P1 zadania.
6. Section H timeline → strawić do `docs/tasks/task2_pii_extraction.md` "Strategia" sekcja jako konkretne kroki.

## Decision Log (multi-agent review)

| Decyzja | Co zmieniono | Źródło uwagi |
|---|---|---|
| Verbatim task → paraphrase | Confidentiality risk pasting organizer PDF do 3rd-party tools | Constraint Guardian |
| Dodać "what team already owns" listę 11 paperów | Avoid re-deriving co już mamy w korpusie + dr/02 | Skeptic — czytaliśmy MAPPING.md |
| Dodać SprintML/CISPA jako Section E | Task autor = paper autor (Marek + Kowalczuk) — bezpośredni signal | Skeptic |
| Dodać "intentionally overfit" jako Section C | Hackathon-specific signal, nie standardowy threat model | Skeptic |
| 12→8 pytań w Section G | Dilution na zbyt długiej liście | Skeptic |
| Dodać anti-fabrication rule | Tooly halucynują GitHub URLs i API | Skeptic |
| Dodać Section H timeline (4 tiers) | Action-oriented, sleep-deprived team o 03:00 | User Advocate |
| Tiered reading list MUST/SHOULD/MAY | Cognitive load triage | User Advocate |
| TL;DR ≤150 słów, top-3 actions | Same | User Advocate |
| Apple MPS / MLX flags wszędzie | Team na M4 primarily | Constraint Guardian |
| Mention 5-min cooldown + no-feedback-below-best | Submission strategy as research question | Constraint Guardian |
| Public/private leaderboard divergence | CLAUDE.md priority — extended test set | Constraint Guardian |

## Disposition: **APPROVED**

Rationale:
- 4 focus areas pokryte (attack tech 2025+ / tooling / multimodal+defenses / open uncertainties).
- Hard recency lock 2025+, foundational pre-2025 explicite oznaczone.
- Anti-fabrication zakodowane (NOT VERIFIED / pseudocode markers).
- Apple MPS vs CUDA flagi wymagane.
- Action-oriented (Section H timeline + Section G P0/P1/P2).
- Tool-agnostic ale z tool-specific tipami w wrapperze.
- Lista 11 lokalnie posiadanych paperów (z dr/02) — research nie traci czasu na re-derivation.
- Section E gold mine: SprintML lab = task authors.

**Status: ready to fire.**
