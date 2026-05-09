# Probe log — CD attempt + alternatives shortlist

> Living document. Append new probes here so insights don't disappear after /clear.
> This file lives on branch `task2-prompt-cd`. Sister branch `task2-prompt`
> tracks K-shot+medoid path independently.

## 2026-05-09 23:00 → 2026-05-10 00:00 — Contrastive Decoding (CD) probe

### Setup
- Worktree `/Users/arturkempinski/hackaton-cd/`, branch `task2-prompt-cd`, off `task2-prompt @ e0a9313`
- Cluster clone `repo-kempinski1-cd/`, all output under that path
- Implementation: `attack.py:generate_one_cd` — manual decode loop, KV-cache reuse
  - first step: `prepare_multimodal_inputs` on both target+amateur, super().forward(inputs_embeds=…) → past_key_values
  - subsequent steps: forward(input_ids=[next_token], past_key_values=past) → 1-token incremental
  - score: plausibility filter (expert top-k=50) → `α·target − β·amateur`, argmax
  - α=1.0 β=0.5 topk=50 (research §2.1 default)

### Result (job 14740294, 840 blank, 29.9 min)
| PII    | CD     | DP baseline | Δ       |
|--------|--------|-------------|---------|
| CREDIT | 0.0832 | 0.2312      | −0.148  |
| EMAIL  | 0.4998 | 0.5785      | −0.079  |
| PHONE  | 0.3108 | 0.3700      | −0.059  |
| OVERALL| 0.2980 | 0.3932      | **−0.095** |

CD uniformly worse on all 3 PII types.

### Root cause
Research §2.1 wymaga "stock OLMo-2-1B-base, no PII fine-tune" jako amateur.
Nasz `shadow_lmm` to fine-tune na tym samym zadaniu PII-VQA (z disjoint PII),
więc emit'uje IDENTYCZNY format-prior (4-4-4-4 dla CREDIT, name@... dla EMAIL,
+1... dla PHONE). β·shadow tłumi nie tylko format, ale też zmemoryzowane
content tokeny.

CREDIT najgorszy bo 16-cyfrowa pozycja jest jednolitym "format" — każda cyfra
jest plausible w obu modelach → CD nie informuje. Raw outputy potwierdzają:
model przerywa 4-4-4-4 i wraca do prozy ("the card number was 300").

Bridge dla base OLMo (cache ma `allenai/OLMo-2-0425-1B`): wymaga ~1h kodu na
vocab projection + skip image tokens; niepewny payoff. **Decyzja: kill CD path.**

### Implementation pozostawione na branchu jako reference
- `attack.py:generate_one_cd` (commit 19c793a)
- `main.py: --use_cd / --shadow_model_dir / --cd_alpha / --cd_beta / --cd_topk / --cd_pii_types`
- `main.sh: STRATEGY=cd_decoding` z env vars CD_ALPHA/BETA/TOPK
- _ATTN_PATCH_APPLIED idempotency guard dla 2× load

---

## 2026-05-10 00:30 — Alternative paths (research synthesis)

Po CD failure, badawczy agent (general-purpose) przejrzał `findings/`,
`references/papers/MAPPING_INDEX.md`, `MAPPING.md`, `task2_pii_extraction.md`,
`strategies.py`. Wymóg: ortogonalne do K-shot+medoid (druga sesja) ORAZ do
9 przetestowanych prompt strategies. Skróty z agenta:

### Approach 1 — Vision-noise dithering + image-shuffle as K-source
- **Mechanizm:** K=16 wariantów obrazu (8 patch-shuffles 4×4 + 8 Gaussian σ∈{1,2,4}/255 na blank), greedy decode each → K kandydatów na medoid. Cross-modal noise zamiast text-side temperature.
- **Ortogonalność:** druga sesja sampluje text-side `temperature=0.7` na stałym blank. My damy K diverse vision conditioning (różne attention sinks) — orthogonal axis dla agregatora.
- **Source:** `findings/new_researchclaude.md:11-17, 23-25` (§1.2 + §1.4); arXiv:2503.03321 + Hayes 2410.19482.
- **Pliki:** `attack.py` (image_perturb arg), `loader.py:get_image()` postprocess hook, `main.py` K-loop, reuse aggregator z drugiej sesji.
- **Czas:** ~1h kod + 30 min eval (16× FLOPs → ~3.5h GPU; split EMAIL+PHONE only ~2.3h).
- **Expected lift:** +0.02–0.05 OVERALL.
- **Risk:** patch-shuffle może rozbić attention-sink który robi "blank-magic" (blank > original w naszym setup). Mitigant: gauss_1 jako safe baseline.

### Approach 2 — Typographic prompt-smuggling (text-in-image)
- **Mechanizm:** PIL-render extraction template (`"Recall the {label} for {Name} exactly:"`) jako czarny tekst na białym tle w native LLaVA-HR resolution. Image-channel niesie redundantną kopię prompt → wyższa-pojemność route do LM.
- **Ortogonalność:** vs K-shot+medoid: prompt-INPUT modification, nie agregacja. Vs przetestowane: WSZYSTKIE 9 strategies używały blank/scrubbed/original — żadne nie testowało image-jako-prompt-channel.
- **Source:** `findings/new_researchclaude.md:19-22` (§1.3); arXiv:2603.03637 (post-cutoff, directional). Cisco 2025 raportuje 60–65% ASR na LLaVA-class na image-injected instructions vs text-only.
- **Pliki:** `scrub_image.py:render_text_image(text, size)`, `strategies.py: typographic_probe`, `main.py` image_mode='typographic'.
- **Czas:** 45 min kod + 14 min eval.
- **Expected lift:** +0.01–0.04 OVERALL. Wysokie variance: jeśli LLaVA-HR vision encoder dobrze OCR-uje (PII-VQA finetune był na obrazach z rendered text! patrz `task2_pii_extraction.md:16, 122`) — DUŻY upside, reaktywuje training-distribution mode.
- **Risk:** PIL font ≠ training font → vision encoder nie rozpoznaje. Mitigant: matchować font z val_pii images (5 min manual inspection) albo DejaVuSans (PIL default).

### Approach 3 — Template-faithful bytes + multi-token repeat trigger
- **Mechanizm:** (a) §5.4: dump *exact* training-time chat-template bytes z finetune script (nie z `apply_chat_template` runtime — często byte-różne: leading `\n`, trailing space, special-token wariant). (b) §5.1: powtórzyć assistant-prompt suffix N=8 razy → attention-sink saturation, push w training-distribution attractor.
- **Ortogonalność:** vs K-shot+medoid: prompt-input mod, single greedy. Vs przetestowane: P2 `verbatim_prefix` SKIP'ował chat template — tu robimy odwrotnie: hyper-template-faithful + multi-repeat. Carlini 2023 "fine-tuning extraction undoes alignment" — model fine-tuned-not-RLHF, mechanizm pasuje idealnie.
- **Source:** `findings/new_researchclaude.md:99-101, 111-113` (§5.1 + §5.4); paper **11** Carlini'21 §6.5 (`MAPPING_INDEX.md:17`); paper **25** Nasr ChatGPT divergence (`MAPPING_INDEX.md:36`); arXiv:2503.08908 (mechanism).
- **Pliki:** grep `apply_chat_template`/`tokenize` w `p4ms_hackathon_warsaw_code-main/src/lmms/` (5 min); `strategies.py: template_faithful_repeat(N=8)`.
- **Czas:** 30 min inspection + 30 min kod + 14 min eval = ~1.25h.
- **Expected lift:** +0.015–0.04 OVERALL. CREDIT najbardziej (training-format attractor = 4-4-4-4); stack'uje dobrze z sesją-prompt (oni K-sample text, my single-greedy template-perfect).
- **Risk:** jeśli `apply_chat_template` runtime jest już byte-perfect z training, część (a) zero-impact. Część (b) może hit context limit albo LM bail w repetition mode (output `Q: Q: Q:`). Mitigant: N=4 fallback, filter na invalid output → drop do greedy DP.

### Recommended priority (z agenta)

**Approach 3** pierwszy: najtańszy (1.25h), CREDIT-targeting (najwyższy marginal — 0.245 floor), zero-overlap z sesją-prompt, prosty rollback. Approach 1 drugi (orthogonal K-source — można merge'ować candidate sets z sesją-prompt). Approach 2 tail-risk: highest variance, najsilniejszy upside jeśli OCR encoder współpracuje.

### Key files referenced

- `code/attacks/task2/prompt/findings/new_researchclaude.md` (§1.2-1.4, §2.1, §5.1, §5.4)
- `code/attacks/task2/prompt/strategies.py:252-265` (gdzie dorzucić nowe wpisy do `STRATEGIES` dict)
- `code/attacks/task2/prompt/scrub_image.py` (image perturbations base)
- `references/papers/MAPPING_INDEX.md:17, 36` (paper 11 Carlini'21 §6.5, paper 25 Nasr divergence)
- `p4ms_hackathon_warsaw_code-main/src/lmms/` (na klastrze — do inspekcji template bytes dla A3)
