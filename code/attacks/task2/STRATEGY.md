# Task 2 — Attack Strategy

> Owners: **kempinski1** + **murdzek2** (2 osoby, niezależne ścieżki, no-merge).
> Source-of-truth dla zadania: `docs/tasks/task2_pii_extraction.md` (PDF).
> Researche bazowe: `references/research/task2/{task2-research-claude.md, perplexity_TASK2_research.md}`.
> Last updated: 2026-05-09.

## TL;DR

- **2 ortogonalne ścieżki ataku, każda osoba bierze jedną.** Ścieżka A = prompt/behavioral (kempinski1). Ścieżka B = white-box memorization signal (murdzek2). Ścieżka C = image-side, **fallback only**.
- **Każdy submituje własny CSV. Best-of-CSVs wygrywa.** Bez merge'ów. Bez wspólnego kodu (każda ścieżka buduje swój loader/scorer/walidator).
- **Lokalna kalibracja na `validation_pii` (280 sampli × 3 PII = 840 GT)** przed jakąkolwiek submisją. Wszystko mierzymy `1 − rapidfuzz.distance.Levenshtein.normalized_distance`.
- **Greedy + exact training-format prompt + assistant-prefix priming** to common-floor baseline każdej ścieżki. Pierwszy submit z tego.

## Common floor (każda ścieżka MUSI to mieć przed jakimkolwiek eksperymentem)

1. **Reconstruct exact training prompt template** z `task2_standalone_codebase.zip` → `src/lmms/dataset/general_vqa_dataset.py`, `src/lmms/dataset/task_dataset.py`. Format z PDF: `[SYSTEM_PROMPT]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER]`. Wrong template = #1 failure mode dla overfitted modeli.
2. **Validation_pii (280 × 3 = 840 GT)** jako kalibrator. Wszystkie decyzje (prompt template, decoding mode, threshold) przechodzą przez ten set ZANIM submitujemy.
3. **Greedy decode + assistant-prefix priming** jako baseline (np. `"The credit card number for {name} is "`).
4. **Submission format (PDF — DOKŁADNE):** CSV `id,pii_type,pred`, **3000 wierszy**, każda `(id, pii_type)` raz, `pii_type ∈ {EMAIL,CREDIT,PHONE}`, `pred` length **10–100 chars**, bez `<|user|>` / cudzysłowów / whitespace okołoznakowego. Min-len edge case: padować short EMAIL do 10 chars.
5. **Lokalny scorer:** `from rapidfuzz.distance import Levenshtein; score = 1 - Levenshtein.normalized_distance(gt, pred)`. **Sanity check:** `Levenshtein.normalized_distance("abc","ab") == 1/3` (= dist/max(len)).
6. **Pierwszy submit:** common-floor greedy + format walidacja. Anchor leaderboard. Cooldown 5 min na success, 2 min na failure.

## Ścieżka A — Prompt / Behavioral Extraction

**Owner:** kempinski1. **Branch:** `task2-prompt` (odbity od `task2`).

**Hipoteza:** model jest intentionally overfitted, więc klucz do PII to **właściwy prompt + format-aware decoding**. Sygnał wyciągamy behaviorally (forward-only, bez patrzenia w aktywacje).

### Anchor papers

- **Pinto et al. ICML 2024** — *Extracting Training Data from Document-Based VQA Models* — arXiv:2407.08707. **Najbliższy publikowany analog naszego setupu** (blacked-out region + original question). Direct recipe match.
- **Nakka et al. PII-Scope** — *PII-Scope: Comprehensive Study on PII Extraction Attacks in LLMs* — arXiv:2410.06704 (v2 May 2025). Multi-query attack daje **5× boost** vs. single-query. Templates: naive / assistant-prefix / role-play / dialog-replay / CoT recall.
- **Carlini et al. USENIX'21** — paper #11 w `references/papers/txt/11_*.txt`. Six MIA features, context-dependency Sec 6.5, insertion-frequency threshold Sec 7.
- **Nasr et al. 2023** — paper #25 w `references/papers/txt/25_*.txt`. Chat-divergence attack — **prawdopodobnie nie aplikowalne** (model nie jest RLHF-aligned), ale 1 ablation warto zrobić.

### Workflow

1. **Multi-prompt ablation:** 5 templates × 50 sampli z `validation_pii` per PII type → wybierz best per `(template, pii_type)`. Templates do testu (PII-Scope §4):
   - T1 naive: `[QUESTION]`
   - T2 assistant-prefix: `[QUESTION] The {pii_type} is `
   - T3 role-play: `You are a data assistant. {QUESTION}`
   - T4 full-dialogue replay: name appearing twice
   - T5 CoT recall: `Let's recall the user's record. {pii_type}: `
2. **Format-aware constrained decoding** dla CC i PHONE przez `transformers-cfg` (drop-in `LogitsProcessor`, MPS-safe). Luźny regex:
   - CC: `\d{13,19}` z optional spaces/hyphens, **Luhn check post-hoc** (NIE w grammatyce — deadlock risk)
   - PHONE: `\+?[1-9]\d{6,14}` (E.164-lite)
   - EMAIL: regex post-hoc walidacja, **bez constraint** w grammatyce (rare chars typu `_` mogą być w GT)
3. **Image ablation** (jednorazowo, 280 sampli × 5 treatments na klastrze):
   - orig / blank-white / mean-pixel / random-noise / swap-with-other-user / text-only-no-image
   - Jeśli text-only ≥ 0.9× orig → drop image (2× szybszy inference)
   - Swap test: jeśli model emituje PII użytkownika A nawet z obrazem B → image nie jest kluczem (zgodnie z Wen NeurIPS 2025 cross-modal asymmetry)
4. **K=8 candidates beam/sampling → medoid:** kandydat z najwyższym mean similarity do innych. Działa lepiej niż argmax-perplexity dla overfit (ref: research_claude §B.5). `rapidfuzz` dla pairwise.
5. **Per-PII-type tuning:** greedy dla CC/PHONE (deterministic format), beam-8 dla EMAIL (większa zmienność).

### Stack

`transformers` + `transformers-cfg` + `rapidfuzz` + `outlines` (opcjonalnie). Prototyp na M4 (`mlx-vlm` dla speed na małej próbce, OLMo-2-1B w bf16 ≈ 2 GB), full run **na A800 klastra** (`dc-gpu`).

### Wyjście

CSV `submission_prompt.csv` w formacie z PDF.

## Ścieżka B — White-Box Memorization Signal

**Owner:** murdzek2. **Branch:** `task2-shadow` (odbity od `task2`).

**Hipoteza:** mamy white-box (target + shadow LMM, kod modelu, hooks). Sygnał memoryzacji wyciągamy z **logp delta** i **layer activations** — nie polegamy tylko na tym co model zwraca w tekście.

### Anchor papers

- **Kowalczuk, Dubiński, Boenisch, Dziedzic. ICML 2025** — *Privacy Attacks on Image AutoRegressive Models* — arXiv:2502.02514. **Autor zadania = autor papera.** Per-token loss MIA na autoregressive models, sygnatura wprost mapowana na LM head naszego LMM. Paper #10 w `references/papers/txt/10_*.txt`. Code: https://github.com/sprintml/privacy_attacks_against_iars.
- **Li et al. NeurIPS 2024** — *Membership Inference Attacks against Large VLMs* — arXiv:2411.02902. `logp_target − logp_ref` na LLaVA, MIA target-based attacks. Code: https://github.com/LIONS-EPFL/VL-MIA.
- **Nguyen et al. ICLR 2025** — *DocMIA: Document-Level MIA* — arXiv:2502.03692. Multi-question per-document loss aggregation (3 PII per user → razem score'ujemy zaufanie do usera). Code: https://github.com/khanhnguyen21006/mia_docvqa.
- **"Watch Out Your Album" ICML 2025** — arXiv:2503.01208. Layer-wise probing classifier na intermediate activations dla VQA inadvertent memorization. Code: https://github.com/illusionhi/ProbingPrivacy.
- **Hintersdorf et al. NeMo NeurIPS 2024** — paper #13 w `references/papers/txt/13_*.txt`. Memorization neuron localization (DM, ale technika hook-on-attention ta sama).
- **Hayes et al. NeurIPS 2025** — arXiv:2505.18773. Strong MIAs na LLMs — uzasadnia że **single shadow model wystarczy** dla meaningful Δ (bez full LiRA sweep 4000+ models).

### Workflow

1. **Per-token Δ** = `logp_target(ans|prompt) − logp_shadow(ans|prompt)` na `validation_pii`. Threshold: **fit logistic regression** features=[Δ, length, format_match, name_token_attn] → target=correct/wrong na 840 GT. Cross-val 5-fold.
2. **Layer activation probe** ("Watch Out Your Album" recipe):
   - hook na ostatnich 3-5 layers `UnifiedModel.layers[-5:]` (architektura z `target_lmm/model.txt`)
   - ekstrahuj `[name_token]` activation lub `[REDACTED]` slot activation
   - fit `sklearn.LogisticRegression` na 280 ref samples → predict `is_high_confidence_memorized`
   - low-confidence rows → padding format-valid placeholder (zamiast gambling); high-confidence → trust greedy
3. **Candidate ranking via Δ:** dla każdego (id, pii_type) wygeneruj K=8 kandydatów (sampling T=0.7), score by Δ, **pick argmax**. Lub fallback: medoid by pairwise similarity (jak w A).
4. **Per-document aggregation** (DocMIA-style): score 3 PII per user razem. Jeśli user "lights up" memorization (high mean Δ across 3 typów) → wszystkie 3 predykcje z większym confidence.
5. **Opcjonalnie (P2, jeśli zostanie czas):** gradient ascent na embedding tokenu `[REDACTED]` dla low-Δ samples. CUDA-only (MPS autograd niestabilne dla VLM). Drogi — ostatnia rzecz do robienia.

### Stack

`transformers` + custom `forward_hook` + `sklearn` + `rapidfuzz`. **Wymaga klastra** — gradient ops + 2× model load (target 3.6 GB + shadow 3.6 GB) > 7 GB VRAM. M4 tylko do edycji kodu i prototypu na 5 sampli.

### Krytyczne sanity checks

- **Tokenizer match target↔shadow:** assert `tokenizer.vocab_size`, `pad_token_id`, `eos_token_id`, `chat_template` identyczne. Jeśli nie → Δ jest invalid. (Ryzyko R5 z perplexity research.)
- **Token boundary memorization:** beam może produkować różne tokenizacje tego samego stringu — Levenshtein traktuje je equal, więc OK.
- **Shadow distribution shift:** jeśli shadow trenowany na innej dystrybucji niż target (poza brakiem PII) → Δ szumi. Treat jako **rerank signal, nie hard gate** (zgodnie z OpenLVLM-MIA caveat arXiv:2510.16295).

### Wyjście

CSV `submission_shadow.csv` w formacie z PDF.

## Ścieżka C — Image-Side Inversion (FALLBACK)

**Status:** **kontyngencja, nie equal billing.** Bierzemy tylko jeśli A albo B się wywróci wcześnie i jedna z osób ma luz.

**Hipoteza (słaba):** scrubbed images mają residual content (tło, layout, meta-cues typu data/miejsce z PDF przykładu — "September 21 2014 at 04:33 AM, Nagoya, Japan"), z którego można odzyskać PII independent of LM memorization.

### Anchor papers

- **Wu et al. ICIMIA** — *Image Corruption-Inspired MIA against LVLMs* — arXiv:2506.12340. White-box vision embedding similarity przed/po corruption.
- **"Watch Out Your Album" ICML 2025** — arXiv:2503.01208. Task-irrelevant image content (watermarks, background) jest enkodowane w intermediate LM layers — sugeruje że scrubbing nie jest perfekcyjny.

### Hipotetyczny workflow

1. OCR / super-resolution na scrubbedowanych regionach obrazka — może resztki PII są channel-decodable
2. Region inpainting reverse: spróbuj odzyskać co było pod redaction box
3. Image swap test: jeśli model emituje PII zależnie od obrazka → image jest kluczem; wtedy image conditioning matters

### Dlaczego fallback, nie equal billing

- **Wen NeurIPS 2025** (arXiv:2506.05198): cross-modal recall asymetric — text-conditioning dominates
- **"Captured by Captions" ICLR 2025**: text encoder dominates over image encoder w CLIP memorization
- **Pinto ICML 2024**: extraction works z blackoutem PII region — image jest głównie conditioning, nie keyem
- Inwestycja 12h na C = wysokie ryzyko, niski expected value
- **Trigger to switch to C:** jeśli A LUB B daje similarity < 0.4 na `validation_pii` po godzinie 4 → diagnoza, ewentualne odbicie na C zamiast wciskania głębiej

## Co NIE jest na liście (i dlaczego)

- **Pure black-box (KCMP NeurIPS 2025, DP-MIA RIGEL ICLR 2026 review)** — mamy white-box, głupio go nie używać. KCMP idea (image perturbation sensitivity) **wchodzi jako sub-feature do B**.
- **Janus-style fine-tune amplification** (arXiv:2310.15469) — za drogi na 24h, niepewny zysk.
- **Recursive paraphrasing / DIPPER** — nie ma sensu dla extraction taska (to defense against watermarks, paper #22 w naszym repo, dla Task 3).
- **Chat-divergence (Nasr'23 "repeat the word X forever")** — model nie jest RLHF-aligned, prawie na pewno nie zadziała. **Robimy 1 ablation (50 sampli) i zamykamy temat.**
- **Inverting Gradients / DLG** — federated learning setup, nie nasz. Gradient-based methods zostają jako P2 w B.

## Decision points

| Godzina | Decyzja | Trigger |
|---|---|---|
| H+1 | First submit common-floor (greedy + format) | każda ścieżka osobno |
| H+4 | Czy `validation_pii` similarity > 0.5 per PII type? | jeśli nie → debug template przed eskalacją |
| H+8 | Czy A i B są na > 0.7? | jeśli nie i jedna jest blocked → wskocz na C tą jedną osobą |
| H+16 | Freeze metody, tylko refinement | przestajemy zmieniać core attack, tylko hyperparam |
| H+20 | Final submit | nigdy w ostatnich 5 minutach przed deadline |

## References — files in this repo

- `docs/tasks/task2_pii_extraction.md` — task spec (ground truth)
- `references/research/task2/task2-research-claude.md` — research #1 (systematic, Pinto-anchored)
- `references/research/task2/perplexity_TASK2_research.md` — research #2 (KCMP/DocMIA-anchored)
- `references/papers/MAPPING_INDEX.md` — paper router
- `references/papers/txt/10_*.txt` — Kowalczuk IAR (Path B anchor, by task author)
- `references/papers/txt/11_*.txt` — Carlini'21 LLM extraction (Path A anchor)
- `references/papers/txt/13_*.txt` — NeMo memorization neurons (Path B sub-anchor)
- `references/papers/txt/25_*.txt` — Nasr'23 ChatGPT divergence (Path A ablation only)
