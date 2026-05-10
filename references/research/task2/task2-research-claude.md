# PII Extraction from Multimodal Models — Hackathon Research Briefing

> Compiled 2026-05-09. Targets a 22-hour hackathon team with white-box weights of a target LMM (overfit on PII) and a shadow LMM. The team already owns a playbook covering Carlini'21/'23, Nasr'23, Carlini'24 ICML, LiRA, Maini DI, Min-K%++, NeMo, Kowalczuk IAR, Hayes-Boenisch'25, CoDeC, plus PyTorch implementations of Fredrikson MI, PPA/IF-GMI, perplexity-zlib, LiRA, Inverting Gradients, DM generate-and-cluster, and Opacus DP-SGD. Foundational pre-2025 work is referenced as anchor only. Pivot is to **2025–2026 multimodal-specific** material.

## TL;DR (top 3 actions for the first 2 hours)

1. **Establish a greedy white-box baseline immediately**: load both target and shadow LMM in `transformers`, batch the 280-sample reference set, run greedy generation with the original `[QUESTION]` referencing the user by name plus an *assistant-prefix* like `"The {pii_type} of {name} is "`. Compute Levenshtein on the reference set with `rapidfuzz.distance.Levenshtein.normalized_similarity`. Submit *one* CSV inside the first hour to anchor the leaderboard (cooldown is 5 min, so cheap submissions are fine).
2. **Add format-aware decoding ASAP** using `outlines` (works with HF `LlavaForConditionalGeneration` and Qwen2-VL via the `outlines.models.transformers_vision` adapter on MPS) or `xgrammar` regex on a CUDA box. Constrain EMAIL to RFC-5321-lite, PHONE to E.164 `\+?[1-9]\d{6,14}`, and CREDIT to a Luhn-validated 13–19 digit regex with optional spaces.
3. **Run the shadow-vs-target log-likelihood gate over candidates**: for each generated answer, compute `Δ = logp_target(ans|prompt) − logp_shadow(ans|prompt)`. Use Δ to rerank top-K beam/sample candidates — this is the cleanest signal that the *target* memorized a specific string.

Calibrate everything on the 280-sample reference set before sinking time into white-box gradient methods.

---

## Section A — Multimodal LMM extraction (2025+)

### A.1 Where memorization lives in an LMM

The LMM stack has four touchable surfaces — vision encoder (ViT/CLIP/SigLIP), projection MLP (linear or Q-Former), cross-attention/concat into the LM, and the LM backbone with its head. 2025+ literature converges on the following:

| Component | Reported memorization role | Source |
|---|---|---|
| Vision encoder (frozen) | Almost no PII memorization in instruction-tuned LMMs; encoder is usually frozen during instruction tuning. | Li et al., *MIA against Large VLLMs*, NeurIPS 2024 (arXiv:2411.02902) — finds image-MIA AUC on MiniGPT-4/LLaMA-Adapter close to 0.5 because only the projector is updated. |
| Projection MLP | Mild memorization signal; projector-only updates yield weak MIA but enough for image-side membership. | Li et al. NeurIPS 2024 (arXiv:2411.02902); Wu et al., ICIMIA, arXiv:2506.12340 (Jun 2025) — embedding-similarity attack on the projected image embedding. |
| LM backbone | Where the bulk of PII memorization lives. Pinto et al. ICML 2024 (arXiv:2407.08707) show that DocVQA models reconstruct the redacted answer from the question alone for 0.1–8% of training samples (k=1 repetitions). LLaVA-1.5 (full LLM fine-tune) is far more vulnerable than MiniGPT-4 (projector-only). | Li et al. NeurIPS 2024; Pinto et al. ICML 2024. |
| LM head | LM-head logits are the key MIA signal in white-box; first-token-extraction (TTE) attacks rely solely on the first generated token's softmax. | Abad-Rocamora et al. summarised in arXiv:2602.00688 (CP-Fuse defense, May 2025+). |
| Cross-attention to image tokens | In LLaVA-style concat models, blocking image-token attention in layers 15–24 has minimal impact on generation (Neo et al., as summarized in *Mech Interpretability Meets VLMs*, ICLR 2025 Blogposts, https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/). Suggests the LM has already absorbed image content by mid-stack. |

**Implication for our setup**: PII almost certainly lives in the LM-backbone weights (the team is told the model is "overfit"). Image is likely a weak conditioning key after redaction. Prioritize text-side white-box probes over image-side gradient inversion. See Section D.

### A.2 Attack entries on LMMs in 2025+

#### A.2.1 Pinto et al. — Extracting Training Data from Document-Based VQA Models
- **Source**: Pinto, Rauschmayr, Tramèr, Torr, Tombari. *Extracting Training Data from Document-Based VQA Models*. ICML 2024 (arXiv:2407.08707, Jul 2024). Borderline-2024 but methodology is the closest published analog to our hackathon task.
- **Threat model**: White- or black-box; attacker has the document with the answer region wiped (BLACK BOX rectangle) and asks the original training question. Mirrors our setup almost exactly.
- **Reported metrics**: On Donut/Pix2Struct/PaLI-3 fine-tuned on DocVQA, fraction of extractable PII ranges roughly 0.1% (PaLI-3) to single-digit % (Donut); Donut memorizes secrets repeated only once.
- **Mapping**: **Directly applicable** — this is essentially our hackathon's threat model. Their "blacked-out region + original question" is our "[IMAGE] with PII regions WIPED + [QUESTION] referencing user by name."
- **Concrete recipe**:

```python
# pseudocode — mirrors Pinto et al. §3
def attack_sample(model, processor, image_redacted, question, name, pii_type):
    prompts = [
        f"{question}",                                              # naive
        f"{question} The {pii_type.lower()} is ",                   # assistant-prefix
        f"Recall {name}'s {pii_type.lower()} from the document: ",  # role-play
    ]
    cands = []
    for p in prompts:
        ids = processor(images=image_redacted, text=p, return_tensors="pt")
        out = model.generate(**ids, do_sample=False, max_new_tokens=64)
        cands.append(processor.decode(out[0], skip_special_tokens=True))
    return cands
```

- **Validation hook**: Run on 280-sample reference set; compute `Levenshtein.normalized_similarity` against ground truth. Pinto reports that the simplest blacked-out-image baseline already extracts ≥1% — if you see ≥0.4 average similarity on reference you are well above noise.
- **Compatibility**: HF transformers; works on MPS for ≤7B at batch 1. CUDA recommended for batched.
- **Red flags**: None — this paper is the direct anchor.

#### A.2.2 Li et al. — VL-MIA (Membership Inference Attacks against Large Vision-Language Models)
- **Source**: Li, Wu, Chen, Tonin, Abad-Rocamora, Cevher. *Membership Inference Attacks against Large Vision-Language Models*. NeurIPS 2024 (arXiv:2411.02902). Code: https://github.com/LIONS-EPFL/VL-MIA. Borderline-2024.
- **Threat model**: White-box logits. MIA on instruction-tuning data (LLaVA-1.5, MiniGPT-4, LLaMA-Adapter v2). Datasets VL-MIA/Flickr (real non-members) and VL-MIA/DALL-E (counterfactual non-members generated from member captions).
- **Reported metrics**: On LLaVA-1.5 instruction-tuning detection, target-based attacks (loss, Min-K%, Min-K%++) outperform target-free; on description text length 32, AUC up to ~0.7 (paper Table 3, exact numbers per setting).
- **Mapping**: **Requires adaptation** — VL-MIA tells us *which* samples are members; our hackathon already knows membership. Use VL-MIA's ranking signal (`logp_target − logp_ref`) as a *gate* on candidate predictions: when target loss on a candidate answer is much lower than shadow loss, accept it.
- **Recipe**: Use `mimir`-style reference attack with the shadow LMM in place of the typical reference LM.
- **Validation**: Compute target-vs-shadow Δ on the 280-sample reference set; check if higher Δ on the *correct* PII string vs. distractors.
- **Compatibility**: Code is PyTorch; LLaVA-1.5 official repo. CUDA-only baseline; MPS works for ≤7B if you avoid 8-bit kernels.
- **Red flags**: Li et al. observed that LLaMA-Adapter and MiniGPT-4 (less LM tuning) are far less leaky than LLaVA-1.5 — informative for *expected upper bound* of our extraction.

#### A.2.3 Hu, Li, Liu et al. — MIAs Against Vision-Language Models (USENIX Security 2025)
- **Source**: Hu, Li, Liu, Zhang, Qin, Ren, Chen. USENIX Security 2025 (arXiv:2501.18624, Jan 2025).
- **Threat model**: Black-box; query LMM with set of samples and probe sensitivity to *temperature*. Targets instruction-tuning data.
- **Reported metrics**: Their set-based, temperature-sensitivity attack outperforms prior MIA on multiple LMMs; precise AUC numbers per model in the paper (we will not fabricate exact figures).
- **Mapping**: **Requires adaptation** — primarily a black-box MIA. We have white-box, so we should use their core insight (temperature-sensitivity differs between members and non-members) as a *secondary* feature for reranking.
- **Recipe**: Generate same prompt with T∈{0.0, 0.7, 1.2}; if the top-1 answer is *stable* across temperatures, that is correlated with memorization.
- **Compatibility**: Pure HF generation. MPS-friendly.

#### A.2.4 ICIMIA — Image Corruption-Inspired MIA against Large VLMs
- **Source**: Wu, Lin, Zhang et al. (Penn State). arXiv:2506.12340 (Jun 2025).
- **Threat model**: Both white-box (image embedding cosine before/after corruption) and black-box (output text embedding stability under image corruption).
- **Reported metrics**: Improves on VL-MIA on VL-MIA/Flickr-2K; exact AUC/TPR@5%FPR in their Tables 5–6 (don't paraphrase).
- **Mapping**: **Requires adaptation** for our scrubbed-image case — the image is *already* corrupted (PII region wiped). Use the *complementary* signal: corrupt the image *further* (Gaussian blur, motion blur) and check answer stability. If the model still produces the same PII string, that PII is being read from text-side memorization, not from any residual image signal.
- **Validation**: Compare answer between (i) image with PII wiped, (ii) image with everything but the user's face/name region wiped, (iii) blank image. Do this on 280 reference samples (Section D).

#### A.2.5 DocMIA — Document-Level Membership Inference (ICLR 2025)
- **Source**: Nguyen, Kerkouche, Fritz, Karatzas. ICLR 2025 (arXiv:2502.03692; OpenReview gNxvs5pUdu). Code: https://github.com/khanhnguyen21006/mia_docvqa.
- **Threat model**: White-box and black-box; aggregates multiple Q-A pair losses per document. Does *not* require auxiliary data — designed for the case where the attacker only has the model.
- **Reported metrics**: Outperforms baseline MIA across DocVQA models (VT5, Donut, Pix2Struct) and PFL-DocVQA; per-model TPR@1%FPR in Tables 3–4 (do not paraphrase).
- **Mapping**: **Directly applicable to membership detection but our task is reconstruction, not detection.** Use the per-document loss aggregation idea: for each user, you have 3 question types — aggregate the (target_loss − shadow_loss) across all 3 to get a confidence score. If the user is highly memorized in *general*, your candidate predictions for them are more trustworthy. Use this as a gate to skip submitting low-confidence rows or pad them.
- **Compatibility**: PyTorch; CUDA-tested. MPS untested.

#### A.2.6 Quantifying Cross-Modality Memorization (NeurIPS 2025)
- **Source**: Wen, Liu, Chen, Lyu et al. *Quantifying Cross-Modality Memorization in Vision-Language Models*. NeurIPS 2025 poster (arXiv:2506.05198, Jun 2025).
- **Threat model**: Memorization audit; trains Gemma-3-4B on synthetic personas and probes recall in the *other* modality.
- **Reported metrics (paraphrased)**: Cross-modal recall is consistently *lower* than same-modal recall; larger models (4B → 12B → 27B) improve same-modal recall but the *cross-modal transfer slope is roughly unchanged*.
- **Mapping**: **Diagnostic only** — tells us that if PII is memorized from text-side training (questions referencing the user by name), it is best extracted by text-side prompts; the image is mostly conditioning. Argues for "image off / blank image" ablations. See Section D.

#### A.2.7 OpenLVLM-MIA (WACV 2026)
- **Source**: Miyamoto, Fan, Kido, Matsumoto, Yamana. WACV 2026 (arXiv:2510.16295, Oct 2025). Code: https://github.com/yamanalab/openlvlm-mia.
- **Threat model**: Controlled benchmark of 6,000 images with carefully balanced member/non-member distributions across vision-encoder pretraining, projector pretraining, and instruction tuning.
- **Reported metrics**: SOTA MIA methods drop to AUROC 0.407–0.527 — essentially chance — once the dataset distribution shift is controlled.
- **Mapping**: **Cautionary**. Most published MIA wins on LMMs are partly distributional artifacts. Our setting is very different: we have *known* members and the model is *intentionally overfit*. Memorization here should be near-saturated, so simple greedy extraction will likely beat any clever MIA on this overfit regime. Don't waste time on calibrated MIA tricks; spend on extraction quality.

#### A.2.8 PII-Scope (multi-query PII extraction in LLMs)
- **Source**: Nakka, Frikha, Mendes, Jiang, Zhou. *PII-Scope: A Comprehensive Study on Training Data PII Extraction Attacks in LLMs*. arXiv:2410.06704 (v2 May 2025).
- **Threat model**: Black-box LLM (GPT-J 6B, Pythia 6.9B), pretrained and fine-tuned on PII (email, phone). Multi-query attack budget.
- **Reported metrics**: With multi-query adversarial prompting, PII extraction rate increases up to **5×** vs. single-query; fine-tuned models leak more than pretrained.
- **Mapping**: **Directly applicable**. Our model is fine-tuned-and-overfit on PII — the regime where leakage is highest. Use the multi-query strategy: per (user, pii_type) issue several diverse prompts, then rerank.
- **Recipe**: Prompts to try (PII-Scope §4): (1) plain `[QUESTION]`, (2) `[QUESTION] Answer: `, (3) role-play "You are a data assistant. {QUESTION}", (4) full dialogue replay with the user's name appearing twice, (5) chain-of-thought "Let's recall the user's record. Email: ".
- **Compatibility**: HF transformers; MPS-friendly.

#### A.2.9 Janus-style amplification through additional fine-tuning (anchor: arXiv:2310.15469, "Janus")
A 2025+ trend: small additional fine-tuning of an aligned model on a few PII pairs unlocks large additional leakage. Sun et al. (DynamoFL, arXiv:2307.16382) and follow-ups reported similar effects in 2024. Not directly applicable: we have only 22h and re-fine-tuning the LMM (RESOURCE-HEAVY) is unlikely to be worth it on M4 / one CUDA box. **Skip unless you find time.**

#### A.2.10 First-Token Extraction (TTE) and CP-Fuse adaptive analysis
- **Source**: Abad-Rocamora et al., as referenced in arXiv:2602.00688 (provably protecting fine-tuned LLMs from TDE). The TTE primitive is described in 2025 work.
- **Threat model**: White-box; predict only the *first* token from the model's output distribution; use selective classification to abstain.
- **Mapping**: Useful as a feature, not the main attack — the first token of an EMAIL/PHONE/CREDIT is informative (digit, letter, sign). Combine with constrained decoding from Section B.

### A.3 Image as memorization key — fraction of signal surviving partial scrubbing

No published evaluation found that directly measures the *marginal contribution of image* on extraction in models intentionally overfit on tiny PII VQA datasets with PII regions wiped. Closest evidence:

- Pinto et al. ICML 2024 (arXiv:2407.08707) report extraction *with* the answer region blacked out and the rest of the document visible. Memorization persists — 0.1–8% across models — meaning the surviving image content (layout + non-PII text) plus question-text is enough.
- Wen et al. NeurIPS 2025 (arXiv:2506.05198) suggest cross-modal transfer is asymmetric: training-modality recall dominates.
- Hu et al. USENIX 2025 (arXiv:2501.18624) note set-based and temperature-based attacks succeed black-box, implying image is largely conditioning.

**Working hypothesis (verify on 280-sample reference)**: image off / blank / noise will not dramatically reduce extraction rate compared to scrubbed-image; the LM has memorized the user-name → PII map text-side. Plan ablation in Section D.3.

### A.4 Multimodal chat-divergence updates of Nasr 2023

The Nasr et al. 2023 ChatGPT divergence attack (`repeat the word X forever`) has 2025 follow-ups, but: (i) most apply to *aligned production chatbots*, not to a small overfit research LMM; (ii) the LMM in the hackathon is unlikely to be RLHF-aligned. The Dropbox follow-up (https://dropbox.tech/machine-learning/bye-bye-bye-evolution-of-repeated-token-attacks-on-chatgpt-models) shows the technique still works on production OpenAI models with multi-token strings (Jan 2024). For multimodal divergence attacks, no 2025+ paper specifically extends Nasr to LMMs as our anchor; the closest 2025 PII-extraction line in chat models is a "ChatBug" template-mismatch literature (arXiv:2406.12935) that exploits chat-template handling. **Don't burn time on divergence on our overfit non-aligned LMM** — it is unlikely to yield more than direct prompt extraction.

---

## Section B — Format-aware decoding for PII

### B.1 Library landscape (verified May 2026)

| Library | HF transformers (text-only) | HF LMM (LlavaForConditionalGeneration etc.) | vLLM backend | MPS / MLX | License |
|---|---|---|---|---|---|
| `outlines` (https://github.com/dottxt-ai/outlines) | Yes — first-class | Yes — `outlines.models.transformers_vision` adapter; HF Cookbook tutorial uses SmolVLM (https://huggingface.co/learn/cookbook/en/structured_generation_vision_language_models) | Yes | Yes (via transformers MPS); also ports to MLX through `outlines-core` Rust | Apache-2.0 |
| `xgrammar` (https://github.com/mlc-ai/xgrammar) | Via vLLM/SGLang/TensorRT-LLM/MLC-LLM; HF integration in progress (see https://blog.mlc.ai/2024/11/22/) | Through MLC-LLM and vLLM (vLLM has VLM support but XGrammar+VLM is bleeding-edge) | **Default** in vLLM (https://docs.vllm.ai/en/latest/features/structured_outputs/) | Yes via MLC-LLM Metal backend; via vllm-mlx (https://github.com/waybarrios/vllm-mlx) it composes with MLX | Apache-2.0 |
| `lmql` | Yes | Limited multimodal | Yes | Partial | Apache-2.0 |
| `guidance` | Yes | Image inputs supported for select backends | Via `vllm` guidance backend | Partial | MIT |
| `transformers-cfg` (https://github.com/epfl-dlab/transformers-CFG) | Yes — drop-in `LogitsProcessor` | Works for text branch of any HF causal-LM-style LMM (LLaVA, Qwen-VL) — set `logits_processor=[GrammarConstrainedLogitsProcessor(...)]` in `model.generate()` | No | Yes (pure-Python logits processor) | MIT |
| vLLM `guided_regex`/`guided_grammar` | Via vLLM API | Limited multimodal coverage | Native | Indirectly via `vllm-mlx` | Apache-2.0 |

**Recommendation for our task**: On Apple M4 use `transformers-cfg` (no monkey-patching, drop-in `LogitsProcessor`, MPS-safe) or `outlines` (broader integrations). On the CUDA box use `vllm` with `xgrammar` regex backend for batch throughput.

### B.2 Empirical gain of constrained decoding on memorized PII

No published 2025+ paper provides head-to-head numbers of constrained-vs-unconstrained decoding specifically on *memorized PII reconstruction* in LMMs. The community evidence is indirect:

- HF Cookbook "Structured Generation from Images or Documents Using Vision Language Models" (https://huggingface.co/learn/cookbook/en/structured_generation_vision_language_models) shows reliable JSON extraction from documents with SmolVLM + outlines, demonstrating the integration works.
- vLLM structured outputs documentation (https://docs.vllm.ai/en/latest/features/structured_outputs/) shows email regex constrained decoding as a built-in example.

**Working assumption**: when the model has memorized the format (the redacted EMAIL/CREDIT/PHONE strings in training had a regular format), constrained decoding only forbids invalid tokens — net effect is positive when the model wants to wander, neutral when it doesn't. See B.3 for failure modes.

### B.3 When constrained decoding hurts

Failure modes documented in the structured-generation literature (XGrammar paper arXiv:2411.15100, BentoML write-up https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction):

1. **Tokenizer mismatch**: PII strings are often tokenized as multi-token sequences that span letter+digit boundaries (`@`, `+`, `-`, hyphens in CC). A grammar that constrains to a regex over *characters* must be paired with a tokenizer-aware compiler (`outlines` and `xgrammar` do this; naive regex masking does not).
2. **Greedy under regex hurts when the *only valid* next token is wrong**: if the memorized email is `john_42@x.com` but your regex disallows `_`, the model is forced to a wrong path.
3. **Length under-/over-shoot**: if your CC regex requires exactly 16 digits but the memorized number had 15 (Amex) you reject the truth. Use `\d{13,19}` not `\d{16}`.
4. **Luhn check applied at the *grammar* level can deadlock**: implementing Luhn inside a CFG is messy. Better: generate unconstrained, then re-rank by Luhn.

### B.4 Snippets per PII type

```python
# pseudocode — adapt to actual API
# transformers-cfg + LLaVA / Qwen-VL text branch (works on MPS)

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

device = "mps" if torch.backends.mps.is_available() else "cuda"
proc  = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)

EMAIL_GBNF = r"""
root        ::= local "@" domain
local       ::= [A-Za-z0-9._%+-]{1,40}
domain      ::= label ("." label){1,3}
label       ::= [A-Za-z0-9-]{1,30}
"""

PHONE_GBNF = r"""
root        ::= "+"? digits
digits      ::= [0-9]{7,15}
"""

CREDIT_GBNF = r"""
root        ::= [0-9]{13,19}
"""   # post-filter with Luhn

def decode(image, question, gbnf, max_new=64):
    grammar  = IncrementalGrammarConstraint(gbnf, "root", proc.tokenizer)
    lp       = GrammarConstrainedLogitsProcessor(grammar)
    inputs   = proc(images=image, text=question, return_tensors="pt").to(device)
    out      = model.generate(**inputs, max_new_tokens=max_new,
                              do_sample=False, logits_processor=[lp])
    return proc.batch_decode(out, skip_special_tokens=True)[0]

def luhn_ok(num):
    s = [int(d) for d in num if d.isdigit()]
    chk = 0
    for i, d in enumerate(reversed(s)):
        chk += d if i % 2 == 0 else (d*2 - 9 if d*2 > 9 else d*2)
    return chk % 10 == 0
```

For CC: generate K candidates with sampling, keep only Luhn-valid, then pick the lowest target-LM loss. For EMAIL: validate with `email_validator` (Python lib, MIT) post-decode. For PHONE: optionally `phonenumbers` (E.164) but our score is Levenshtein, so a slightly malformed string is still partial credit.

### B.5 Levenshtein-aware reranking (2024+)

Score is `1 − Normalized_Levenshtein(GT, Pred)` averaged. Levenshtein is *robust to short edits*. Implication for reranking:

1. Generate K candidates per (user, pii_type) with beam=8 or sampling T=0.7 N=8.
2. Compute pairwise Levenshtein among candidates with `rapidfuzz.distance.Levenshtein.normalized_similarity` (`distance / max(len(a), len(b))` matches our scoring norm; verify with the host's evaluator).
3. Pick the **medoid** (candidate with highest mean similarity to others). If the model has memorized the answer, beams collapse around it; medoid extraction beats argmax-perplexity in moderate-leak regimes.

`rapidfuzz` (MIT, https://github.com/rapidfuzz/RapidFuzz) is the modern fast implementation. `python-Levenshtein` is GPLv2 — avoid in MIT/Apache stacks. Both expose `Levenshtein.ratio` but `rapidfuzz.distance.Levenshtein.normalized_distance(s1,s2)` uses `distance/max(len1+len2 weights)`; `rapidfuzz.distance.Levenshtein.normalized_similarity` is `1 - normalized_distance`. **Confirm the host uses `dist/max(len(a),len(b))`**: this is the *Indel*-distance normalization of `Levenshtein.normalized_similarity` only when weights are uniform `(1,1,1)`. Test on a few known pairs before trusting.

---

## Section C — Overfit-targeted extraction

### C.1 Easy-mode baselines on intentionally overfit models, 2024–2026

The Pinto et al. ICML 2024 (arXiv:2407.08707) experiment with k=1 secrets in DocVQA *is* an "easy-mode" baseline — they find one occurrence in training is sufficient. PII-Scope (arXiv:2410.06704, 2024–2025) and Memorization in Fine-Tuned LLMs (arXiv:2507.21009, 2024) both show that overfit fine-tuning yields near-100% memorization on the fine-tune set when sampling matches the training prompt format. On overfit models the right baseline is:

1. Greedy decode with the **exact training prompt template** — for our hackathon, `[SYSTEM_PROMPT]<|user|>[IMAGE_REDACTED][QUESTION]<|assistant|>`. The team has the codebase, so reading the training format off the dataloader is a 10-minute exercise.
2. Append a **token-level prefix that matches the answer head** the model saw during training. If the training data showed `Email: <addr>`, prepend `Email: ` to the assistant turn.

### C.2 Greedy vs search threshold

Heuristic for over-overfit models (no published threshold table for our exact setup):

- If reference Levenshtein-similarity with **greedy** ≥ 0.85, stop. Search will not help much; pick greedy and spend remaining time on reliability (ensembling target+shadow).
- If 0.5 ≤ greedy < 0.85, beam search (b=8 or 16). Beam helps for partially memorized strings.
- If greedy < 0.5, switch to multi-prompt + sampling + Levenshtein medoid (B.5).
- Constrained decoding always on for CC and PHONE (cheap).

### C.3 White-box heuristics for "this model memorized X" given 280 ground-truth

Use the 280-sample reference set as a labeled "members" set. For each sample compute features:

- `loss_target(answer | prompt)` — main signal.
- `loss_shadow(answer | prompt)` — calibration baseline.
- `Δ = loss_shadow − loss_target` (higher = more memorized by target).
- Min-K%++ score (already in the team's toolbox).
- Generation match: greedy candidate Levenshtein-similarity to GT.

Fit a small logistic regression on the 280 samples to predict "high-confidence memorized." On the 3000-row evaluation set, use the predicted confidence to (a) decide whether to submit greedy or beam, (b) rerank candidates, (c) potentially abstain — but **note**: our scoring rejects pred outside 10–100 chars; you cannot abstain, only submit a low-quality prediction. Default for low-confidence rows: pad with a plausible format-compliant fake (for EMAIL: `unknown@example.com`; for PHONE: `+0000000000`; for CREDIT: a Luhn-valid string of correct length). This caps your worst-case Levenshtein at maybe 0.3 instead of 0 (rejected).

### C.4 Failure modes when over-overfit

- **Hallucinating a *different* training PII**. With 1000 users and overfitting, the model can confuse user A's email for user B's email when the names are similar. Mitigate with constrained decoding and multi-prompt + medoid.
- **Token boundary memorization**: if the tokenizer split `+49 175 ...` into rare tokens, beam search may produce alternative tokenizations of the same number — these are equal under Levenshtein. Don't worry.
- **Catastrophic prompt sensitivity**: Mireshghallah et al. and follow-ups note that overfit models often only emit the secret when the prompt *exactly* matches training. Reproduce the training formatter from the codebase; do not re-invent it.

---

## Section D — Scrubbed-image conditioning

### D.1 Marginal contribution of image vs. question-only

No paper directly evaluates "image-with-PII-wiped vs. blank image" on instruction-tuned LMMs intentionally overfit. The closest signals:

- Pinto et al. ICML 2024 show extraction works *with* the image (rest of doc visible). Their ablation does not isolate "image off entirely" since DocVQA needs the doc.
- Wen et al. NeurIPS 2025 (arXiv:2506.05198) show cross-modal recall is asymmetric — text-conditioning often suffices when the model was trained on text+image mapping to a name.
- Hu et al. USENIX 2025 (arXiv:2501.18624) succeed black-box without image manipulation, implying the text branch carries most of the signal.

**Plan: measure on our 280-sample reference (D.3).**

### D.2 Counterfactual image studies

Treatments to ablate: (i) original scrubbed image; (ii) mean-pixel image (per-channel mean over the training image stats); (iii) Gaussian noise image at the model's expected resolution; (iv) blank/white image; (v) semantically-matched image (a different user's scrubbed image, paired with the wrong user-name in `[QUESTION]`); (vi) pure no-image text-only forward (drop image tokens or set the image-projection output to zero).

The (v) treatment is the killer: if we feed user A's question with user B's scrubbed image, and the LMM still emits user A's PII, the image is *not* the key. If it switches to user B's PII, the image is dominant.

### D.3 Recipe for ablating image contribution on 280 samples in <1 hour

```python
# pseudocode
import numpy as np
from rapidfuzz.distance import Levenshtein

treatments = {
  "orig": lambda img: img,
  "mean": lambda img: np.full_like(img, fill_value=img.mean(axis=(0,1), keepdims=True).astype(img.dtype)),
  "noise": lambda img: np.random.RandomState(0).randint(0, 255, img.shape, dtype=np.uint8),
  "blank": lambda img: np.full_like(img, 255),
  "swap": lambda img, other_img: other_img,  # paired
}

results = {t: [] for t in treatments}
for sample in ref_280:
    for t, fn in treatments.items():
        img = fn(sample.image) if t != "swap" else fn(sample.image, swap_partner.image)
        pred = greedy_decode(img, sample.question, model)
        results[t].append(Levenshtein.normalized_similarity(sample.gt, pred))

for t, scores in results.items():
    print(t, np.mean(scores))
```

If `mean`/`noise`/`blank` score ≥ 0.9× `orig`, drop the image entirely on the test set: simpler and cheaper. With ~7B model on MPS, 280×3 questions × 5 treatments ≈ 4200 forward passes; at 1–2 tok/s on M4 Max for ≤7B with image encoding it is tight in 1h. Use the CUDA box.

---

## Section E — Recent SprintML / CISPA work (2025+)

The hackathon task pattern (multi-task hackathon, code in `sprintml/` GitHub) matches CISPA/SprintML's running competition format (Stockholm Feb 2026 hackathon https://github.com/sprintml/hackathonStockholm; Paris hackathon repo also referenced). Highly likely the hackathon authors come from this research line — read their public papers for *exactly* the attack family expected.

**SprintML / CISPA papers 2025–2026 directly relevant:**

1. **Kowalczuk, Dubiński, Boenisch, Dziedzic. *Privacy Attacks on Image AutoRegressive Models*. ICML 2025 (arXiv:2502.02514).** TPR@FPR=1% of 86.38% (v1; later v4 shows 94.57% on largest IARs). Reconstructs 698 of training images from VAR-d30. Directly relevant for the *autoregressive*-style memorization signal (LMM LM-head is autoregressive). Code: https://github.com/sprintml/privacy_attacks_against_iars (last commit ≥2025; ICML pmlr-v267).
   - **Direct relevance: high.** Use their per-token loss-based MIA score as a candidate ranker.
   - Code published: yes, Apache-2.0.
2. **Hayes, Shumailov, Choquette-Choo, Jagielski, Kaissis, Nasr, Ghalebikesabi, Annamalai, Mireshghallah, Shilov, Meeus, de Montjoye, Lee, Boenisch, Dziedzic, Cooper. *Exploring the Limits of Strong MIAs on Large Language Models*. NeurIPS 2025 (arXiv:2505.18773).** Trains 4000+ reference models. **Direct relevance: medium** — full LiRA on LMMs is RESOURCE-HEAVY; team already owns LiRA. Useful for understanding why a single shadow model can give a meaningful Δ even without a full LiRA fit.
3. **Rossi, Marek, Boenisch, Dziedzic. *Privacy Auditing for LLMs with Natural Identifiers*. ICLR 2026 (OpenReview jp4XlcpRIW; OpenReview doaAUf9Pi7 also).** Argues structured random strings (hashes, SSH keys, **wallet addresses**) act as natural canaries and as same-distribution held-out data. **Direct relevance: medium.** Our PII (emails, CCs, phones) *are* natural identifiers — apply their canary-style memorization audit using same-format synthesized strings to estimate the model's prior on each format. Use the resulting prior as a Bayesian rerank weight. Code: https://github.com/sprintml/NIDs_for_Privacy_and_Data_Audits (updated Apr 2026).
4. **Marek, Rossi, Hanke, Wang, Backes, Boenisch, Dziedzic. *Auditing Empirical Privacy Protection of Private LLM Adaptations*. ICLR 2026.** Shows DP-LoRA leaks more when adaptation data is similar to pretraining distribution. Direct relevance: low for our task; useful framing only.
5. **Zhao, Maini, Boenisch, Dziedzic. *Unlocking Post-hoc Dataset Inference with Synthetic Data*. ICML 2025 (arXiv referenced in OpenReview a5Kgv47d2e).** Code: https://github.com/sprintml/PostHocDatasetInference. Direct relevance: low — DI predicts dataset-level membership; we already have member labels.
6. **Dubiński, Kowalczuk, Boenisch, Dziedzic. *CDI: Copyrighted Data Identification in Diffusion Models*. CVPR 2025 (arXiv:2411.12858).** Direct relevance: low for our LMM task.
7. **Hu et al., USENIX Security 2025 (arXiv:2501.18624).** Yang Zhang co-author affiliated to CISPA. Already covered in A.2.3.

**General pattern**: SprintML's 2025+ playbook on autoregressive models is "score candidates by `loss_target − loss_ref` per token, optionally with Min-K%++ aggregation." Build that pipeline first. The key 2025+ result that survives OpenLVLM-MIA's distribution-shift critique: **on intentionally-overfit fine-tuning data, even simple loss-based attacks achieve near-perfect TPR.** The hackathon's "intentionally overfit" framing matches the regime where these attacks dominate.

---

## Section F — Tooling, code, libraries (verified 2025+)

### F.1 Open-source extraction codebases active 2025+

| Repo | Last commit | Stars (approx) | Attacks covered | CUDA / MPS / MLX |
|---|---|---|---|---|
| `iamgroot42/mimir` (https://github.com/iamgroot42/mimir) | 2025 (Duan et al., COLM 2024 reference impl + later updates) | ~130 | Loss, Reference, zlib, neighborhood, Min-K%, Min-K%++ — **text-only LMs**. | CUDA primary; MPS untested but should run for inference-only attacks. |
| `sprintml/privacy_attacks_against_iars` (https://github.com/sprintml/privacy_attacks_against_iars) | ICML 2025 release | ~7 | MIA + DI + reconstruction for image autoregressive models. | CUDA. |
| `JonasGeiping/breaching` (https://github.com/JonasGeiping/breaching) | 2023 (older); still extended via forks (e.g. Geminio HKU-TASR/Geminio ICCV 2025). | ~300 | Federated learning gradient inversion (DLG, IG, Decepticons, Robbing the Fed). **Not directly applicable** — our setting isn't FL. | CUDA. |
| `eth-sri/watermark-stealing` | [NOT VERIFIED — could not locate active 2025+ commits in search results]. Skip. | – | – | – |
| `sprintml/PostHocDatasetInference` | 2025 (ICML) | ~5 | LLM dataset inference with synthetic held-out. | CUDA. |
| `khanhnguyen21006/mia_docvqa` (https://github.com/khanhnguyen21006/mia_docvqa) | 2025 (ICLR DocMIA release) | – | White-box and black-box MIA on DocVQA models (VT5, Donut, Pix2Struct). | CUDA. |
| `LIONS-EPFL/VL-MIA` (https://github.com/LIONS-EPFL/VL-MIA) | NeurIPS 2024 release | – | MIA on LLaVA-1.5, MiniGPT-4, LLaMA-Adapter v2. | CUDA. |
| `yamanalab/openlvlm-mia` (https://github.com/yamanalab/openlvlm-mia) | 2025 (WACV 2026 release) | – | Controlled MIA benchmark, 10 baseline methods reimplemented. | CUDA. |
| `safr-ai-lab/pandora-llm` | 2024 | – | Supervised white-box LLM MIA + extraction. | CUDA. |
| `albertsun1/gpt3-pii-attacks` | 2023 | – | OpenAI fine-tuning API PII leakage. Mostly black-box. | API-only. |
| `ml-research/localizing_memorization_in_diffusion_models` (NeMo) | 2024 (NeurIPS) | – | Localizes memorization in diffusion. Not directly applicable but inspires per-attention-head probing. | CUDA. |
| `ffhibnese/Model-Inversion-Attack-ToolBox` | [NOT VERIFIED for 2025+ activity — see Awesome-MI list at https://github.com/AndrewZhou924/Awesome-model-inversion-attack instead]. | – | – | – |
| `garak` (NVIDIA) | Active 2025 | many | LLM red-teaming probes incl. data leakage; mostly black-box prompt patterns. | API-friendly. |
| `promptfoo` | Active 2025 | many | Eval/red-team for prompts. Useful for CI of attack prompts. | API-friendly. |

### F.2 LMM-specific extraction repos beyond LLM tooling

`khanhnguyen21006/mia_docvqa`, `LIONS-EPFL/VL-MIA`, `yamanalab/openlvlm-mia` are the three explicitly LMM-focused. None ships an LMM-extraction recipe; they all do MIA. For extraction, the closest is **the Pinto et al. ICML 2024 protocol**, which (as of search) does not have a public canonical repo — implement directly.

### F.3 Inference utilities for batched LMM generation

| Tool | Batched throughput | Attention hooks | Apple M4 |
|---|---|---|---|
| `transformers` (HF) | Modest (eager / SDPA) | Native: `output_attentions=True`, `register_forward_hook` | MPS supported for most LMMs; LLaVA-1.5/Qwen2-VL OK; FlashAttention not on MPS |
| `vllm` | High (paged KV, continuous batching) | Hooks limited (forward-only); customizing logits via `LogitsProcessor` works | `vllm-mlx` (https://github.com/waybarrios/vllm-mlx) provides MLX backend; up to 525 tok/s text on M4 Max per author's benchmarks, with 28× speedup on repeated images via prefix cache |
| `lmdeploy` | High | Limited custom hooks | CUDA-only |
| `sglang` | High | Limited custom hooks | CUDA-only |
| `TensorRT-LLM` | Very high | None practical | CUDA-only (NVIDIA) |
| `mlx-vlm` (https://github.com/Blaizzy/mlx-vlm) | M4-native | Hooks via raw MLX | **MLX, M4 native — Qwen3-VL, Gemma 3, LLaVA, DeepSeek-OCR-2, Pixtral all listed** per Pebblous 2026 write-up (https://blog.pebblous.ai/blog/mlx-vlm-physical-ai-edge/en/) |

**Recommendation**: One M4 in `mlx-vlm` for fast iteration on prompts; the team's CUDA box runs `vllm` + `xgrammar` for the 3000-row final batched generation.

### F.4 Levenshtein library

Use `rapidfuzz` (MIT). Specifically:

```python
from rapidfuzz.distance import Levenshtein
score = 1 - Levenshtein.normalized_distance(gt, pred)  # in [0,1]
```

`Levenshtein.normalized_distance` is `distance / max` per `rapidfuzz/RapidFuzz` docs (https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html). **Verify** the host's exact normalization on a few synthetic pairs of differing lengths. If host uses `dist/(len1+len2)` (Indel-style), use `Levenshtein.ratio` from `rapidfuzz/Levenshtein` C-extension.

---

## Section G — Open uncertainties / decisions (frame, don't answer)

| # | Question | Priority | Why it matters | Evidence to resolve (≤30 min) | Default if no time |
|---|---|---|---|---|---|
| 1 | Best prompt template (naive `[REDACTED]`-fill, role-play, full-dialogue replay, assistant-prefix manipulation, CoT "complete the user's record") | P0 | PII-Scope reports up to 5× variance across prompts on LLMs | Run all five templates over 50 reference samples; pick highest mean Levenshtein-similarity | Use the *exact training formatter* read from the codebase + assistant-prefix `"The {pii_type} of {name} is "` |
| 2 | Image conditioning ON vs OFF vs blank vs noise vs swap | P0 | If image is irrelevant we save compute; swap-image test reveals whether image keys the model | Section D.3 — 280 samples × 5 image treatments on CUDA box | Keep original scrubbed image; only ablate if reference time permits |
| 3 | Sampling: greedy vs beam(b∈{4,8,16}) vs T=0.7+N samples vs constrained | P1 | Beam helps partial memorization; constrained reduces Levenshtein loss on mis-formatted PII | Compare greedy vs beam-8 vs T=0.7 N=8 medoid on 50 reference samples | Greedy + constrained for CC/PHONE; beam-8 for EMAIL |
| 4 | Shadow model gate threshold on `logp_target − logp_shadow` | P1 | Prevents over-confident wrong submissions; mirrors VL-MIA / DocMIA logic | Compute Δ on 280 reference samples; pick threshold that maximizes mean similarity (or fit logistic regression) | Threshold = median of correct-prediction Δs |
| 5 | Format-aware decoding ON vs OFF | P0 | Cheap win for CC/PHONE; can hurt for EMAIL with rare characters | Run greedy vs `transformers-cfg`-constrained on 50 reference samples per pii_type | ON for CC and PHONE; OFF for EMAIL |
| 6 | White-box hooks (attention probing, gradient ascent on PII, embedding inversion of `[REDACTED]`) vs prompt-only | P2 | Gradient methods are slow but can recover when prompt-only fails | Try gradient ascent on 5 reference samples; check if it lifts low-confidence predictions | Skip; spend time on prompt + constrained + rerank |
| 7 | Submission scheduling under 5-min cooldown over 22 h | P0 | Max ≈264 submissions in 22h (12/h × 22). Use most for full-3000 evaluations; occasional partial-set diagnostic submits to confirm scoring assumptions | Plan a submission cadence: every 30 min in early phase, every 2h in late phase | Submit 1 baseline immediately; thereafter submit only after >0.02 expected gain on reference |
| 8 | Public-vs-private leaderboard generalization | P1 | 30/70 split with no feedback below current best — easy to overfit to public | Evaluate consistency between two halves of the 280-sample reference; if variance high, freeze submissions earlier | Submit the *less aggressive* of two top candidates as final; do not chase last 0.01 on public |

---

## Section H — Suggested timeline (4 tiers + 2 contingencies)

### First 30 min — env + data + sanity submission
- One person: clone codebase, load both target and shadow LMM in `transformers` on the CUDA box; load also via `mlx-vlm` on M4. Confirm both produce sensible greedy output for 1 reference sample.
- One person: parse the 280-sample reference set and the 3000-query eval set into a uniform `(id, image_path, question, pii_type, [gt])` dataframe.
- One person: implement Levenshtein scoring with `rapidfuzz.distance.Levenshtein.normalized_distance`, verify against a hand-computed example (`dist=2, len_a=10, len_b=11 → norm=2/11`).
- Submit a **dummy CSV** with format-valid filler (`unknown@example.com`, `+0000000000`, `4111111111111111`) to confirm submission pipeline + length validation. (Levenshtein floor will be ~0.1–0.2 — sets a baseline.)

### First 2 h — naive extraction, calibrate, first real submission
- Implement the **training-format-matched prompt** + assistant-prefix on target LMM with greedy decoding.
- Run on 280 reference samples; record mean Levenshtein per pii_type.
- Add format-aware constrained decoding for CC and PHONE (B.4). Re-measure.
- Compute shadow-LMM loss on each reference candidate; compute Δ.
- Generate full 3000-row predictions with greedy + constrained-CC/PHONE; submit.

### Hours 2–8 — best-technique scaling + format-aware + shadow comparison
- Run prompt-template ablation (G.Q1) on 50 reference samples × 3 pii_types × 5 templates. Pick winning template per pii_type.
- Run image ablation (D.3) on 50 reference samples × 5 image treatments. If `blank` ≥ 0.9× `orig`, drop image on remaining runs (faster).
- Add beam-8 / sampling+medoid (B.5) for EMAIL (the most variable format).
- Add target−shadow loss reranking on K=8 candidates.
- Submit best ensemble at hour 4 and hour 8.

### Hours 8–22 — refinement, white-box if time, generalization stress-test, final
- (Optional, P2) Try gradient ascent on the embedded `[REDACTED]` slot (DLG-style, scoped to text input embeddings only). Use `breaching` library code as reference but adapt; LMM inputs make the standard FL recipe brittle.
- (Optional) Probe attention to user-name token across LM layers — identify which layer "lights up" when a memorized PII is about to be emitted. NeMo (already owned) gives the methodology.
- Run **public/private generalization stress-test**: split 280 reference into 30/70, ensure the same model/config wins on both. If not, downgrade to the *less aggressive* candidate.
- Final submission ≥30 min before deadline. Do not submit in the last 5 min.

### Contingency A — score plateaus near 0.4
**Diagnosis**: Model reproduces format (constrained decoding wins) but not content. Either (i) prompt template is wrong, (ii) image is the actual key and we removed it, (iii) target-vs-shadow Δ is misleading because shadow wasn't trained identically.
**+0.1 next move**: re-read the codebase's training data formatter character-by-character; restore the *exact* tokenization (system prompt, BOS placement, image-token slots). Confirm that target and shadow share the same tokenizer/template. Try assistant-prefix that *quotes the user's name* twice (PII-Scope §5: name repetition is a memorization-trigger signal).

### Contingency B — score already ~0.9 on public split
**Diagnosis**: model is near-saturated memorized for at least the public 30%; risk now is private 70% being noisier and our prompt being over-fitted to public.
**Generalization-protection move**: ensemble two prompt families and submit the *median* prediction (string-medoid) per row; freeze submissions; resist the urge to chase the last 0.01.

---

## Appendix 1 — Reading list, tiered

### MUST read (in first 60 min)
1. Pinto, Rauschmayr, Tramèr, Torr, Tombari. *Extracting Training Data from Document-Based VQA Models*. ICML 2024 (arXiv:2407.08707). **Closest analog to the hackathon.**
2. PII-Scope (Nakka et al., arXiv:2410.06704, v2 May 2025). Multi-query PII attacks, fine-tuned-vs-pretrained.
3. DocMIA (Nguyen et al., ICLR 2025, arXiv:2502.03692). Multi-question per-document aggregation in MIA.
4. Hu et al. *MIAs Against Vision-Language Models*. USENIX Security 2025 (arXiv:2501.18624). Set-based, temperature-sensitivity attack.
5. Kowalczuk et al. *Privacy Attacks on Image AutoRegressive Models*. ICML 2025 (arXiv:2502.02514). Per-token loss MIA template (use same shape for our LMM).

### SHOULD read (during execution)
6. Li et al. *MIAs against Large VLLMs*. NeurIPS 2024 (arXiv:2411.02902). Per-LMM architecture leak comparison.
7. ICIMIA (Wu et al., arXiv:2506.12340). Image-corruption robustness as MIA feature.
8. Wen et al. *Quantifying Cross-Modality Memorization*. NeurIPS 2025 (arXiv:2506.05198). Cross-modal asymmetry argument.
9. OpenLVLM-MIA (Miyamoto et al., WACV 2026, arXiv:2510.16295). Distribution-shift caveat.
10. Memorization in Fine-Tuned LLMs (arXiv:2507.21009, 2024). Mechanisms of FT memorization.
11. Hayes et al. *Limits of Strong MIA*. NeurIPS 2025 (arXiv:2505.18773). Why one shadow is often enough.
12. Geiping et al. `breaching` README + Geminio (HKU-TASR/Geminio, ICCV 2025). For optional gradient-inversion side-experiment.
13. HF Cookbook *Structured Generation from Images or Documents Using VLMs* (https://huggingface.co/learn/cookbook/en/structured_generation_vision_language_models).
14. vLLM structured outputs docs (https://docs.vllm.ai/en/latest/features/structured_outputs/).

### MAY read if time
15. XGrammar paper (arXiv:2411.15100, Nov 2024).
16. XGrammar-2 (arXiv:2601.04426, May 2026).
17. transformers-cfg README (https://github.com/epfl-dlab/transformers-CFG).
18. Native LLM/MLLM Inference at Scale on Apple Silicon (arXiv:2601.19139, Jan 2026).
19. Rossi et al. *Privacy Auditing for LLMs with Natural Identifiers*, ICLR 2026.
20. Marek et al. *Auditing Empirical Privacy Protection of Private LLM Adaptations*, ICLR 2026.
21. Pandora's White-Box (arXiv:2402.17012). White-box LLM MIA — supervised classifier.
22. Janus Interface (arXiv:2310.15469). Anchor for the "fine-tune to amplify" idea.
23. Fragments to Facts (arXiv:2505.13819, May 2025). Partial-information PII inference.
24. Mech Interpretability Meets VLMs (ICLR 2025 Blogpost, https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/). For attention-head probing recipes.
25. PFL-DocVQA paper (arXiv:2411.03730, v2 Jun 2025). Background on the federated-DocVQA reconstruction-attack track.

---

## Appendix 2 — Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MPS vs CUDA divergence (numerical / dtype) producing different greedy outputs between dev (M4) and final (CUDA) | Medium | Medium | Run final 3000-row generation on **one** backend; verify with 50-sample identity test before each submission. |
| Tokenizer mismatch between target and shadow LMM (different padding/special tokens) | Medium | High | At load time, assert `tokenizer.vocab_size`, `pad_token_id`, `eos_token_id`, `chat_template` are identical. |
| Constrained decoding deadlocks on low-prob tokens (model only wants `_` but regex forbids it) | Low | Medium | Always have an unconstrained fallback per row; pick the higher-Levenshtein candidate from the reference. |
| Score function differs from `1 − dist/max(len)` assumption | Medium | High | Use a known-PII pair on a tiny test submission early; reverse-engineer the host's formula. |
| Running out of submissions on cooldown limit | Low | Medium | Reserve last 30 min; final submission must be locked in; don't submit anything in the last 5 min. |
| Public-private leaderboard divergence (overfit to public) | Medium | High | Track two-half consistency on 280 reference; submit the *less aggressive* config as final. |
| Over-confidence from shadow-Δ when shadow training pipeline differs subtly | Medium | Medium | Treat Δ as a *rerank* not a *hard gate*; never drop a candidate purely on Δ. |
| Length-rule rejection (pred outside 10–100 chars) silently kills a row | Low | High | Hard-clamp every output: pad short, truncate long; always check `10 ≤ len(pred) ≤ 100` pre-submit. |
| Image swap test misinterpreted (model partially relies on image; we drop it; private set hits image-keyed users) | Medium | Medium | Never drop image globally; only drop where image-ablation similarity is >0.95 of with-image on reference set. |
| Ethics / leakage: this is a hackathon dataset of *fictitious* users, but follow standard hygiene — do not exfiltrate predictions, do not log PII outside the local repo | n/a | High (project) | Local-only logs; `git ignore` results; do not commit CSVs. |

---

## Appendix 3 — Glossary

- **LMM** — Large Multimodal Model (image + text in, text out). e.g. LLaVA, Qwen-VL, Pixtral.
- **VLM / VLLM** — Vision-Language Model / Large VLM. Used interchangeably with LMM in 2025+ literature.
- **MIA** — Membership Inference Attack: decide whether a sample was in the training set.
- **DI** — Dataset Inference: aggregate MIA to a dataset-level claim.
- **TTE** — Top/First-Token Extraction: predict only the first generated token, abstain otherwise.
- **PII** — Personally Identifiable Information.
- **GBNF / EBNF** — Grammar formats consumed by `transformers-cfg` and `xgrammar` for constrained decoding.
- **ANLS** — Average Normalized Levenshtein Similarity, the DocVQA standard metric — equivalent to our scoring.
- **DocVQA** — Document Visual Question Answering benchmark family.
- **LiRA** — Likelihood-Ratio Attack (Carlini et al., S&P 2022); team owns the implementation.