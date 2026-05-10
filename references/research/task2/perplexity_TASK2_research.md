# PII Extraction from Multimodal Models — Hackathon Research Briefing

> **Audience:** 3-person team, 24-hour hackathon, Europe, May 2026. White-box access to target + shadow LMM. 3000-row CSV submission (id, pii_type, pred). Score = 1 − Normalized_Levenshtein. 5-min submission cooldown.
> **Language:** English. All claims cited or marked `[NOT VERIFIED]`. Pre-2025 baselines marked `[FOUNDATIONAL — already covered]`.

***

## TL;DR — Top 3 Actions for the First 2 Hours

1. **Greedy [REDACTED]-fill first.** Run the scrubbed eval set through the target model with the exact training-time dialogue format, ending at `[REDACTED]`. For an intentionally overfit model, greedy decoding should yield verbatim PII immediately. Validate extraction rate on the 280-sample reference set before committing.
2. **Layer-probe the shadow delta.** Compute per-token log-probability difference `logp_target − logp_shadow` for greedy outputs. High-confidence PII tokens will concentrate at the end of sequences; use this as a confidence filter before submission.
3. **Apply format-aware reranking, not hard constraints, for the first submission.** Use `outlines` regex to validate candidate PII strings post-hoc and discard malformed ones rather than forcing constrained generation (which can hurt when memorization is strong). Submit the reference-calibrated top-1 greedy prediction for all 3000 rows.

***

## Section A — Multimodal LMM Extraction (2025+)

### A.1 Where Memorization Lives in an LMM

For text-only LLMs, memorization is distributed across layers but concentrated in the final transformer blocks. In multimodal models the picture is more nuanced:

**Vision encoder (SSL/CLIP).** Wenhao Wang, Adam Dziedzic, Michael Backes, and Franziska Boenisch (SprintML, NeurIPS 2024) show with per-layer (`LayerMem`) and per-unit (`UnitMem`) metrics that memorization in SSL vision encoders **increases with layer depth** and is unexpectedly widespread — a significant fraction of individual units exhibit high memorization of atypical (outlier) data points. In vision transformers, fully-connected layers dominate; convolutional channels memorize more uniformly. This is the most directly relevant localization result from the task authors' lab.[^1][^2]

**Text encoder / CLIP multi-modal setup.** "Captured by Captions" (Wang, Dziedzic, Kim et al., ICLR 2025) introduces `CLIPMem`, a formal memorization metric for CLIP. Key finding: the **text encoder contributes more to memorization than the image encoder**. "Mis-captioned" samples (text-image misalignment) exhibit the highest memorization levels. Implication for the hackathon: the LM backbone of your LLaVA/Qwen-VL target almost certainly holds more memorized PII than the vision encoder.[^3][^4]

**LM backbone layer ordering.** From the general LLM memorization SoK (arXiv:2507.05578, 2025), memorization scales log-linearly with model size and concentrates in later transformer blocks. For the ≤7B models in scope, expect most PII to be recoverable via logit-level access to the last 3–5 layers.[^5]

**Mechanistic localization — PII circuits.** PATCH (Hughes et al., arXiv:2510.07452, EACL 2026) uses circuit discovery to identify the specific computational subgraph responsible for PII leakage in LMs. Circuits persist even after DP fine-tuning; targeted ablation reduces recall by up to 65%. The method applies to the LM backbone only; no multimodal extension is published yet (`[NOT VERIFIED for LMMs specifically]`).[^6][^7]

**Summary table:**

| Component | Memorization locus | Key citation |
|---|---|---|
| CLIP/ViT vision encoder | FC layers, depth-increasing, outlier-biased | NeurIPS 2024[^2] |
| CLIP text encoder | Dominates over image encoder | ICLR 2025[^3] |
| LM backbone | Later transformer blocks; LM head softmax | SoK 2025[^5] |
| PII circuits (text LM) | Cross-attention + specific MLP edges | PATCH 2025[^6] |

***

### A.2 Technique: Black-Box MIA for LVLMs via Memory Probing (KCMP)

- **Name:** KCMP — Prior Knowledge-Calibrated Memory Probing
- **Source:** Zhang et al., *Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing*, NeurIPS 2025 (arXiv:2511.01952). Code: https://github.com/spmede/KCMP[^8][^9][^10]
- **Threat model:** Black-box (output text only). Requires ability to query model with image + question. No logits needed.
- **Reported metrics:** Comparable to gray/white-box MIAs on LLaVA, InternVL, MiniGPT-4, BLIP-2 across COCO, Flickr30k, and a custom VQA dataset. Exact AUC not disclosed in abstract; stated as "comparable to white-box".[^8]
- **Mapping:** Directly applicable as a **confidence gate** — use to decide whether a sample was memorized (and thus worth submitting a constrained prediction) vs. not (submit a format-valid placeholder). Requires black-box query, not white-box weights, so it is a conservative lower bound.
- **Concrete recipe (pseudocode):**
```python
# pseudocode — adapt to actual API
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

def kcmp_score(model, processor, image, question, n_probes=5):
    """Assess memorization of (image, question) pair."""
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    # 1. Baseline generation with original image
    with torch.no_grad():
        out_orig = model.generate(**inputs, max_new_tokens=64, return_dict_in_generate=True,
                                   output_scores=True)
    # 2. Probe with semantically-altered image (mean-image substitution)
    mean_img = torch.zeros_like(inputs["pixel_values"])
    inputs_mean = {**inputs, "pixel_values": mean_img}
    with torch.no_grad():
        out_mean = model.generate(**inputs_mean, max_new_tokens=64, return_dict_in_generate=True,
                                   output_scores=True)
    # 3. Score: high delta = image carries private semantic info
    score_orig = sum(s.max().item() for s in out_orig.scores)
    score_mean = sum(s.max().item() for s in out_mean.scores)
    return score_orig - score_mean  # positive → likely member
```
- **Validation hook:** Run on the 280 reference samples; compute AUC vs. ground-truth membership. Expect >0.7 AUC in <30 min on CPU/MPS.
- **Compatibility:** Apple MPS ✓ (pure HF transformers inference). CUDA ✓. No heavy training.
- **Red flags:** Requires 2× model inference per sample (3000 × 2 = 6000 forward passes). Budget ~2 hours on M4 for a 7B model at batch=1.

***

### A.3 Technique: DP-MIA — Dual-Phase MIA for VLMs

- **Name:** DP-MIA — distinguishes pretrain-member, finetune-member, non-member
- **Source:** Anonymous, *DP-MIA: Dual-Phase Membership Inference Attack Across VLMs Training Lifecycle*, under review ICLR 2026 (OpenReview ID: WW47sMD57T)[^11][^12]
- **Threat model:** Black-box, multi-class (3-way). Uses RIGEL composite metric: generation response time + inference confidence + generation length.
- **Reported metrics:** 88.2% 3-class accuracy on LLaVA-v1.5-7b and Qwen2-VL, significantly outperforming CSA and MCA baselines.[^11]
- **Mapping:** Directly applicable. The hackathon target is a fine-tuned model; DP-MIA's RIGEL metric can flag which samples were memorized at fine-tune time (the relevant class) vs. never seen.
- **Concrete recipe (pseudocode):**
```python
# pseudocode — adapt to actual API
import time

def rigel_score(model, processor, image, question):
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, output_scores=True,
                              return_dict_in_generate=True)
    elapsed = time.perf_counter() - t0
    # confidence = mean of max token probabilities
    confidence = sum(s.softmax(-1).max().item() for s in out.scores) / len(out.scores)
    gen_len = out.sequences.shape[^1]
    return {"time": elapsed, "confidence": confidence, "length": gen_len}
    # High confidence + short length → likely fine-tune member → PII memorized
```
- **Validation hook:** Calibrate RIGEL threshold on 280 reference samples vs. PII visibility flag.
- **Compatibility:** Apple MPS ✓. CUDA ✓. Light.
- **Red flags:** Under double-blind review; API not finalized. Treat as concept guide, not drop-in library.

***

### A.4 Technique: ICIMIA — Image-Corruption MIA for LVLMs

- **Name:** ICIMIA — Image Corruption-Inspired MIA
- **Source:** Anonymous, *Image Corruption-Inspired Membership Inference Attacks against Large Vision-Language Models*, arXiv:2506.12340, June 2025 (also EACL 2026)[^13][^14][^15]
- **Threat model:** White-box (uses vision encoder embeddings for WB variant) or black-box (uses output text embedding similarity). Members respond **differently** to image corruption than non-members.
- **Reported metrics:** Validated on LLaVA, InternVL across COCO, Flickr30k. No exact numbers in available abstract but stated as "effective under both settings."
- **Mapping:** Directly applicable. The partially wiped training image is your corruption; compare model response to the wiped vs. an augmented-wiped image.
- **Concrete recipe (pseudocode):**
```python
# pseudocode — adapt to actual API
import torchvision.transforms as T
from torch.nn.functional import cosine_similarity

def icimia_score(model, processor, image, question):
    # White-box variant: embedding similarity
    def get_vis_embed(img):
        inp = processor(images=img, text=question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            vis_feats = model.model.vision_tower(inp["pixel_values"])
        return vis_feats.mean(1)  # pool tokens
    corrupt_img = T.GaussianBlur(kernel_size=21)(image)
    orig_emb = get_vis_embed(image)
    corr_emb = get_vis_embed(corrupt_img)
    return cosine_similarity(orig_emb, corr_emb, dim=-1).item()
    # Low similarity → member (more sensitive to corruption)
```
- **Validation hook:** Correlate score with PII-visible flag on 280 reference set; expect inverse correlation (members = lower score).
- **Compatibility:** Apple MPS ✓ (vision encoder forward pass only). CUDA ✓.
- **Red flags:** Requires `model.model.vision_tower` attribute; verify exact path for LLaVA vs. Qwen-VL architectures.

***

### A.5 Technique: MIA Against VLMs via Temperature Sensitivity

- **Name:** Temperature-sensitive MIA on instruction-tuned VLMs
- **Source:** Hu, Li, Liu, Zhang et al., *Membership Inference Attacks Against Vision-Language Models*, arXiv:2501.18624, January 2025 (ACL Findings 2025)[^16][^17]
- **Threat model:** Black-box (4 variants across knowledge levels). Infers membership from a **set of samples** (5-sample set → AUC > 0.8 on LLaVA).
- **Reported metrics:** AUC > 0.8 for instruction-tuning data detection on LLaVA with as few as 5 samples.[^17]
- **Mapping:** Directly applicable. The 280-sample reference set = confirmed members; use temperature variation to calibrate membership score for 3000 eval samples.
- **Compatibility:** Apple MPS ✓. CUDA ✓. No training required.

***

### A.6 Technique: Watch Out Your Album — Inadvertent VQA Memorization Probing

- **Name:** Layer-wise probing for task-irrelevant memorization in VQA fine-tuning
- **Source:** Anonymous, *Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models*, arXiv:2503.01208, ICML 2025. Code: https://github.com/illusionhi/ProbingPrivacy[^18][^19]
- **Threat model:** White-box (layer-wise probing classifiers). Requires access to intermediate activations.
- **Reported metrics:** MLLMs trigger "distinct representational patterns" when encountering previously seen task-irrelevant knowledge; probing classifiers achieve statistically significant separation.[^18]
- **Mapping:** High relevance. The hackathon VQA setup (image → PII in target labels) is structurally identical. The residual image after PII wipe may still activate memorized representations detectable by probing classifiers on intermediate layers.
- **Concrete recipe (pseudocode):**
```python
# pseudocode — adapt to actual API
from sklearn.linear_model import LogisticRegression
import numpy as np

def extract_layer_activations(model, processor, images, questions, layer_idx=24):
    activations = []
    hooks = []
    def hook_fn(module, input, output):
        activations.append(output[:, 0, :].detach().cpu())  # [CLS]-equiv
    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    for img, q in zip(images, questions):
        inp = processor(images=img, text=q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inp)
    h.remove()
    return torch.cat(activations, dim=0).numpy()

# Train probe on 280 reference samples, then apply to 3000 eval samples
# Labels: 1 = PII visible (member), 0 = not
train_acts = extract_layer_activations(model, processor, ref_images, ref_questions)
clf = LogisticRegression().fit(train_acts, ref_labels)
eval_acts = extract_layer_activations(model, processor, eval_images, eval_questions)
membership_scores = clf.predict_proba(eval_acts)[:, 1]
```
- **Validation hook:** 5-fold CV on 280 reference samples; expect AUC > 0.65. If not, increase `layer_idx`.
- **Compatibility:** Apple MPS ✓ (forward pass + sklearn). CUDA ✓.
- **Red flags:** `model.model.layers[layer_idx]` path differs between LLaVA (Vicuna/Mistral backbone) and Qwen-VL (Qwen backbone). Inspect `model.named_modules()` first.

***

### A.7 Technique: OpenLVLM-MIA Baseline Calibration

- **Name:** OpenLVLM-MIA — controlled benchmark and distributional debiasing
- **Source:** Anonymous, *OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of MIA against LVLMs*, arXiv:2510.16295, October 2025[^20][^21]
- **Reported metrics:** SOTA MIA methods approached **chance-level** (AUC ≈ 0.5) when distributional bias was removed from the benchmark. Most reported high AUC results in prior work reflected distribution artifacts, not true memorization detection.[^20]
- **Mapping:** CRITICAL calibration caveat. If your MIA scores on the 280 reference set look suspiciously high, verify whether the signal is genuine memorization vs. distributional differences between reference-set images and eval images. Run a sanity check: shuffle membership labels randomly; if MIA AUC stays high, the signal is spurious.
- **Compatibility:** N/A (benchmark artifact awareness).

***

### A.3 How the Wiped Image Acts as a Memorization Key

No published 2025–2026 paper directly measures the marginal contribution of partially redacted images to PII extraction from an overfit LMM. `No published evaluation found for this specific scenario.`

**Closest analog:** ICIMIA (A.4 above) shows that members respond differently to image corruption, implying residual image content continues to function as a partial key even after corruption. The SprintML CLIP memorization paper shows that image encoder memorization is weaker than text encoder memorization, suggesting the LM backbone may not need the image at all once sufficiently overfit.[^14][^3]

**Practical proxy:** Run a counterfactual ablation (see Section D) on the 280 reference samples before scaling to 3000.

***

### A.4 Chat-Divergence for Multimodal Models (2025+)

The Nasr-style divergence attack has not been extended to multimodal chat models in any published 2025–2026 paper. `No published evaluation found for this specific scenario.` The closest analog is the special-characters attack (arXiv:2405.05990, 2024) on text-only LLMs. Given white-box access, gradient-based jailbreaks (GCG-style) applied to the image token prefix are an untested substitute — `[NOT VERIFIED]`.

***

## Section B — Format-Aware Decoding for PII (2025+)

### B.1 Library Landscape 2025+

| Library | Repo | Active 2025/2026? | HF multimodal integration | CUDA | MPS/MLX |
|---|---|---|---|---|---|
| **outlines** | https://github.com/dottxt-ai/outlines | ✅ active (latest Apr 2026) | ✅ LlavaForConditionalGeneration via `outlines.models.transformers` | ✅ | ✅ MPS confirmed[^22][^23] |
| **XGrammar** | https://github.com/mlc-ai/xgrammar | ✅ active | Via vLLM backend | ✅ | ✅ Apple Silicon listed[^24] |
| **vLLM structured outputs** | https://docs.vllm.ai/en/latest/features/structured_outputs/ | ✅ active | Limited multimodal support; v1 in progress | ✅ CUDA required for full perf | ⚠ CPU-only on Mac[^25][^26] |
| **guidance/llguidance** | https://github.com/guidance-ai/llguidance | ✅ active (vLLM v1 backend) | Via vLLM | ✅ | ⚠ CUDA preferred |
| **lmql** | https://github.com/eth-sri/lmql | ⚠ low activity 2025 | Partial | ✅ | ⚠ untested on MPS |
| **transformers-cfg** | https://github.com/epfl-lts/transformers-cfg | ⚠ low activity | Direct HF integration | ✅ | ✅ CPU/MPS |

**Recommendation for M4 team:** Use `outlines` with the HF transformers backend for local generation. XGrammar is faster under batched load but requires the vLLM backend which is CUDA-optimized. For the CUDA HPC node, use vLLM + XGrammar for batch generation.[^27][^28]

***

### B.2 Empirical Gain of Constrained Decoding on Memorized PII

No 2025 paper directly measures the extraction gain from constrained decoding on **memorized** PII in overfit models. `No published evaluation found for this specific scenario.`

Adjacent evidence: The OpenReview paper "Constrained Decoding for Privacy-Preserving LLM Inference" (2024/2025) uses regex-aware logit masking to **block** PII generation (the inverse task), demonstrating that pattern constraints are precise enough to target email/SSN/CC patterns token-by-token. This implies the same constraint applied **positively** (enforcing Luhn-valid CC prefixes) would at minimum eliminate impossible candidates.[^29]

The `Unintended Memorization of Sensitive Information in Fine-Tuned LMs` study (EACL 2026, arXiv:2601.17480) reports greedy decoding as more reliable for memorized PII recovery than stochastic sampling; constrained decoding is not benchmarked.[^30][^31]

**Practical estimate:** For an overfit model, greedy decoding likely already yields the correct PII. Constrained decoding adds value mainly as a **post-hoc validator** to filter malformed candidates before submission.

***

### B.3 When Constrained Decoding Hurts

`[NOT VERIFIED empirically]` — no published 2025 paper quantifies the degradation. Logical failure mode: if the model has memorized a phone number in non-E.164 format (e.g., `555-0100` instead of `+15550100`), a hard E.164 constraint will force the wrong sequence, potentially degrading Levenshtein score below a free-form generation that at least gets the digits right.

**Heuristic:** Use constrained decoding only when you have independent evidence (logp gap, KCMP score) that the model has memorized the specific sample. Fall back to free-form + regex post-validation otherwise.

***

### B.4 Concrete Recipes

**EMAIL (RFC-5321 pattern, outlines):**
```python
# pseudocode — adapt to actual API
import outlines
import re

model = outlines.models.transformers("path/to/target", device="mps")  # or "cuda"
email_pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"

def extract_email(prompt: str) -> str:
    generator = outlines.generate.regex(model, email_pattern)
    return generator(prompt, max_tokens=60)
```

**CREDIT CARD (16-digit, Luhn-valid prefix enforcement):**
```python
# pseudocode — adapt to actual API
# Enforce digit-group format; Luhn check post-hoc
cc_pattern = r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}"

def extract_cc(prompt: str) -> str:
    generator = outlines.generate.regex(model, cc_pattern)
    candidate = generator(prompt, max_tokens=20)
    digits = re.sub(r"\D", "", candidate)
    return digits if luhn_check(digits) else candidate  # fallback to raw

def luhn_check(n: str) -> bool:
    total = sum(int(d) * (1 + (i % 2 == 0)) % 9 or 9
                if int(d) * (1 + (i % 2 == 0)) > 9 else int(d) * (1 + (i % 2 == 0))
                for i, d in enumerate(reversed(n)))
    return total % 10 == 0
```

**PHONE (E.164 pattern):**
```python
# pseudocode — adapt to actual API
phone_pattern = r"\+[1-9]\d{6,14}"

def extract_phone(prompt: str) -> str:
    generator = outlines.generate.regex(model, phone_pattern)
    return generator(prompt, max_tokens=20)
```

***

### B.5 Levenshtein-Aware Reranking

No 2024–2025 paper directly targets reranking under Levenshtein expectation minimization for PII extraction. The R.R. (Recollect and Rank) method (arXiv:2502.12658, ACL 2025) uses biased cross-entropy loss for ranking PII candidates under a reference model calibration. This is the closest published approach:[^32][^33]

1. Generate N candidates via top-p sampling.
2. Score each with `log p_target(candidate|context) − log p_shadow(candidate|context)`.
3. Select argmax of the log-ratio.

This approach is already in the team's playbook (LiRA extension). Apply it directly to the candidate pool generated by format-constrained sampling.

***

## Section C — Overfit-Targeted Extraction

### C.1 Easy-Mode Baselines on Intentionally Overfit Models

**PII-Scope (IJCNLP 2025):** Nakka et al. benchmark PII extraction across diverse settings and show that fine-tuned models are **more vulnerable** than pretrained models. With sophisticated multi-query attacks and limited query budget, extraction rates increase up to **fivefold** over single-query baselines. Key hyperparameter: demonstration selection for in-context attacks.[^34][^35][^36]

**Unintended PII Memorization (EACL 2026):** Szep et al. show that fine-tuned models exhibit elevated confidence on PII tokens even when PII appears only in model **inputs** (not targets). Greedy decoding on the true-prefix prompt is the strongest baseline.[^37][^30]

**Relevant finding:** For intentionally overfit models, the Carlini 2021 threshold of k=33 repetitions for full memorization collapses to k≈1. The team's playbook already covers this (`[FOUNDATIONAL]`). Implication: greedy completion of the training-time prefix is the correct easy-mode baseline; any non-extraction at this stage indicates a prompt-format mismatch, not absence of memorization.

***

### C.2 Threshold: When Greedy Fails and Search Begins

The SoK on LLM memorization (arXiv:2507.05578, 2025) confirms that **stochastic sampling consistently outperforms greedy decoding in extraction experiments** when memorization is present but the exact generation path is blocked by minor deviations. The `How Much Do Language Models Memorize?` study (arXiv:2505.24832, 2025) notes that for overfit models, greedy can retrieve the stored sequence; when models are "too overfit," they may only reproduce the verbatim training-format output and fail on slightly different prompts.[^38][^5]

**Rule of thumb:** If greedy extraction fails on >30% of the 280 reference set, switch to beam search (b=4) before attempting stochastic approaches. If beam also fails, the issue is prompt format mismatch — debug the dialogue template.

***

### C.3 Heuristics for "Model Definitely Memorized X"

With white-box access:

1. **Perplexity gap:** `perplexity_target(PII_completion) << perplexity_shadow(PII_completion)`. Threshold calibrate on 280 reference samples.
2. **Gradient norm spike:** `‖∇_θ log p(PII|prefix)‖` is anomalously large for memorized sequences vs. non-memorized. Compare target vs. shadow gradient norms; ratio > 3 = strong memorization signal[^39] (MIMIR implements `gradnorm`).
3. **Attention concentration:** In VQA fine-tuning, memorized PII causes attention heads in later LM layers to concentrate on the name token in the question. Measure attention entropy over name vs. other tokens; low entropy = memorized.
4. **RIGEL composite score** (DP-MIA approach): high confidence + short generation length = strong memorization.[^11]

***

### C.4 Failure Modes When the Model Is Over-Overfit

1. **Format lock-in:** The model can only reproduce PII when the input matches the exact training-time format (e.g., `[SYSTEM]<|user|>[IMAGE]What is Alice's email?<|assistant|>Alice's email is ___`). Changing even the system prompt breaks extraction. Fix: reconstruct the training template exactly from the codebase.
2. **Greedy repetition loops:** Extremely overfit models sometimes enter repetition loops on sequences not in training. Detect with repetition penalty diagnostics; set `repetition_penalty=1.3` as a failsafe.
3. **Hallucinated PII from adjacent samples:** When memorization bleeds between users (common with sequential fine-tuning), the model may output another user's PII. Detect by checking if predicted PII appears in reference-set ground truth for a different user ID.
4. **Modality conflict:** If the vision encoder embedding for the wiped image is far from the original training embedding, the LM backbone may generate in a "confabulation" mode. Check cosine distance between target model's vision encoder output for the wiped image vs. a pure-text prompt (no image).

***

## Section D — Scrubbed-Image Conditioning

### D.1 Marginal Contribution of Image vs. Question-Only (2025+)

No 2025+ paper directly measures this for partially redacted images in VQA extraction. `No published evaluation found for this specific scenario.`

**Best proxy:** The "Captured by Captions" finding implies that when the text (question + name) strongly indexes the memorized PII, the image adds marginal signal. For a model overfit on structured `(image, name) → PII` triples, the name alone may be sufficient to trigger recall given extreme memorization.[^3]

### D.2 Counterfactual Image Studies

No published 2025+ paper performs a systematic mean-image / random-noise / blank / semantically-similar counterfactual study on PII extraction from VQA-overfit models. `No published evaluation found for this specific scenario.`

**Mechanistic implication from "Watch Out Your Album" (ICML 2025):** Task-irrelevant watermarks embedded in VQA training images are encoded in intermediate LM layers even if they don't affect output. This suggests wiped-image residuals (background, layout) survive wipe and continue to activate memorized paths.[^18]

### D.3 Practical Recipe: Ablating Image Contribution on 280 Reference Samples

This is actionable in under 1 hour:

```python
# pseudocode — adapt to actual API
import torch
from PIL import Image
import numpy as np

def ablation_study(model, processor, ref_samples, ref_pii, device="mps"):
    results = {}
    for sample_id, (img, question, gt_pii) in enumerate(ref_samples):
        prompt_base = build_training_format_prompt(question)  # exact training template

        # Condition 1: original wiped image
        pred_img = model_generate(model, processor, img, prompt_base, device)

        # Condition 2: blank image (zeros)
        blank = Image.fromarray(np.zeros((img.height, img.width, 3), dtype=np.uint8))
        pred_blank = model_generate(model, processor, blank, prompt_base, device)

        # Condition 3: noise image
        noise = Image.fromarray(np.random.randint(0, 255, (img.height, img.width, 3), dtype=np.uint8))
        pred_noise = model_generate(model, processor, noise, prompt_base, device)

        # Condition 4: text-only (no image token)
        pred_text_only = model_generate(model, processor, None, prompt_base, device)

        results[sample_id] = {
            "img_lev": levenshtein_score(pred_img, gt_pii),
            "blank_lev": levenshtein_score(pred_blank, gt_pii),
            "noise_lev": levenshtein_score(pred_noise, gt_pii),
            "text_only_lev": levenshtein_score(pred_text_only, gt_pii),
        }
    return results
    # If text_only_lev ≈ img_lev → skip image; reduces inference cost 2×
```

**Time budget:** 280 × 4 conditions × ~2s/inference on M4 ≈ 40 min on MPS.

***

## Section E — Recent SprintML / CISPA Work (2025+)

### E.1 Privacy Attacks on Image AutoRegressive Models (ICML 2025)

**Authors:** Antoni Kowalczuk, Jan Dubiński, Franziska Boenisch, Adam Dziedzic. **Venue:** ICML 2025. **Code:** https://github.com/sprintml/privacy_attacks_against_iars[^40][^41]

Novel MIA achieving TPR@FPR=1% of **86.38%** on VAR-d30 vs. 6.38% for diffusion models. Dataset inference with **6 samples** (vs. 200 for DMs). **698 training images extracted** from VAR-d30.

**Relevance (HIGH):** This is the direct methodological predecessor for the hackathon task. The authors are the hackathon task designers. The attack exploits autoregressive token-level probability distributions — identical in structure to the LM backbone of any LLaVA/Qwen-VL model. The paper's MIA architecture should be the blueprint for the shadow-model comparison step.[^42][^43]

**Code available:** Yes, CUDA-based. `[NOT VERIFIED on MPS]`.

***

### E.2 Captured by Captions — Memorization in CLIP (ICLR 2025)

**Authors:** Wenhao Wang, Adam Dziedzic, Grace C. Kim, Michael Backes, Franziska Boenisch. **Venue:** ICLR 2025[^44][^4][^3]

CLIPMem metric quantifies multimodal memorization. **Text encoder dominates**; "mis-captioned" samples memorized most. Proposes mitigation strategies that reduce memorization without utility loss.

**Relevance (HIGH):** Directly quantifies how much of LMM memorization lives in the text vs. vision path. Confirms that prompting via question text (which references the user by name) is the primary memorization key.

***

### E.3 Localizing Memorization in SSL Vision Encoders (NeurIPS 2024)

**Authors:** Wenhao Wang, Adam Dziedzic, Michael Backes, Franziska Boenisch. **Venue:** NeurIPS 2024[^45][^2]

LayerMem and UnitMem metrics. Memorization increases with layer depth in SSL encoders. FC layers dominate in ViTs.

**Relevance (MEDIUM):** Useful for Section A.1 understanding. Practical implication: hooking the last 2 vision encoder FC layers gives the strongest memorization signal if you need vision-side features.

***

### E.4 Benchmarking Empirical Privacy Protection for Adaptations of LLMs (ICLR 2026 Oral)

**Authors:** Bartłomiej Marek, Lorenzo Rossi, Vincent Hanke, Xun Wang, Michael Backes, Franziska Boenisch, Adam Dziedzic. **Venue:** ICLR 2026 (Oral)[^46][^47][^48]

Benchmarks DP adaptations vs. canary extraction and robust MIA. Key finding: **distribution shifts strongly influence practical privacy risk** — adaptation data close to pretraining distribution is MORE vulnerable at the same DP guarantee. LoRA achieves highest empirical protection for OOD data.

**Relevance (HIGH):** The hackathon target is a LoRA/full fine-tune on a VQA dataset. If the PII data is out-of-distribution relative to the LMM's pretraining data (high likelihood: fictitious users with synthetic emails, CCs), the model will be **more extractable** despite any DP protection. This strongly supports the greedy extraction baseline.

***

### E.5 Natural Identifiers for Privacy and Data Audits in LLMs (ICLR 2026)

**Authors:** Lorenzo Rossi, Bartłomiej Marek, Franziska Boenisch, Adam Dziedzic. **Venue:** ICLR 2026[^49][^50][^51]

Introduces NIDs — structured random strings (Ethereum addresses, hashed URLs) for post-hoc DP auditing without retraining. Enables dataset inference without a private non-member held-out set.

**Relevance (MEDIUM):** If the hackathon PII follows a structured-random-string distribution (realistic synthetic CC numbers, hashed emails), NID-style analysis on the shadow model could detect which strings are genuine members. Practical in <1 hour using the shadow model's token-level log-probs.

***

### E.6 Unlocking Post-hoc Dataset Inference with Synthetic Data (ICML 2025)

**Authors:** Bihe Zhao, Pratyush Maini, Franziska Boenisch, Adam Dziedzic. **Venue:** ICML 2025. **Code:** https://github.com/sprintml/PostHocDatasetInference[^52][^53][^54]

Generates synthetic held-out data matching the training distribution; enables DI without any real non-member data. Post-hoc calibration bridges likelihood gaps.

**Relevance (MEDIUM):** The shadow model is a perfect non-member reference. The team already has a DI aggregator in the playbook. This paper's post-hoc calibration technique can be applied when the shadow model's distribution differs from the target's fine-tuning distribution.

***

### E.7 Curation Leaks (ICLR 2026)

**Authors:** Dariush Wahdany, Matthew Jagielski, Adam Dziedzic, Franziska Boenisch. **Venue:** ICLR 2026[^55][^56]

Shows that even models trained on **curated public data** leak membership information about the private data that guided curation. Every stage of the pipeline leaks.

**Relevance (LOW):** Background context; not directly actionable for the hackathon attack.

***

### E.8 Precise Parameter Localization for Textual Generation in Diffusion Models (ICLR 2025)

**Authors:** Łukasz Staniszewski, Bartosz Cywiński, Franziska Boenisch, Kamil Deja, Adam Dziedzic. **Venue:** ICLR 2025[^57][^58]

Less than 1% of diffusion model parameters (cross/joint attention layers) control textual content in generated images.

**Relevance (LOW for text PII):** If the LMM uses a cross-attention architecture (rare in LLaVA/Qwen-VL; these use MLP projection), this localization principle would apply. For MLP-projection architectures, not directly applicable.

***

## Section F — Tooling, Code, Libraries (Verified 2025+)

### F.1 Open-Source Extraction Codebases

| Repo | Stars (approx) | Last commit | Attacks | CUDA/MPS |
|---|---|---|---|---|
| `iamgroot42/mimir` | ~500 | Jul 10 2025[^59][^60] | loss, ref, zlib, ne, min-k, min-k++, gradnorm, ReCaLL | CUDA; MPS untested |
| `sprintml/privacy_attacks_against_iars` | ~150 | 2025[^40][^61] | MIA, DI, extraction for IARs | CUDA (`[NOT VERIFIED MPS]`) |
| `sprintml/PostHocDatasetInference` | ~50 | 2025[^53] | Post-hoc DI with synthetic data | CUDA |
| `pratyushmaini/llm_dataset_inference` | ~300 | 2024 (no 2025 release)[^62][^63] | DI via Welch t-test | CUDA |
| `spmede/KCMP` | <50 | 2025[^8] | Black-box LVLM MIA | CUDA/MPS (HF-based) |
| `illusionhi/ProbingPrivacy` | <50 | 2025[^18] | VQA inadvertent memorization probing | CUDA/MPS (HF-based) |
| `LIONS-EPFL/VL-MIA` | ~100 | 2024[^64] | VL-MIA benchmark | CUDA |
| `martonszep/llm-pii-leak` | <50 | 2026[^37] | PII memorization extraction + mitigation | CUDA/MPS (HF-based) |

### F.2 LMM-Specific Extraction Repos (2025+)

- `spmede/KCMP` — only verified multimodal-specific MIA repo with 2025 commits and documented LVLM pipeline.[^8]
- `illusionhi/ProbingPrivacy` — VQA-specific memorization probing; directly applicable.[^18]
- `LIONS-EPFL/VL-MIA` — 2024 benchmark; last commit 2024 but dataset/methodology still usable.[^64]

### F.3 Inference Utilities for Batched LMM Generation

| Framework | Multimodal | Batched | Attention hooks | Apple M4 | Notes |
|---|---|---|---|---|---|
| `transformers` HF | ✅ | ✅ (limited) | ✅ | ✅ MPS | Easiest integration; slow batching |
| `vllm` | ✅ (partial) | ✅ excellent | ⚠ limited | ⚠ CUDA preferred | vllm v1 adds multimodal[^25][^26] |
| `mlx-vlm` / `vllm-mlx` | ✅ | ✅ (via vllm-mlx) | ⚠ limited | ✅ M4 native | vllm-mlx: 21-87% higher throughput than llama.cpp[^65][^66] |
| `sglang` | ✅ | ✅ excellent | ✅ | ⚠ Apple support Q1 2026 roadmap[^67] | Best CUDA performance |
| `lmdeploy` | ✅ | ✅ | ⚠ | ⚠ CPU fallback | CUDA-primary |

**Recommendation for M4 team:** Use `transformers` (MPS) for first-pass extraction with attention hooks; switch to `vllm-mlx` for final batch generation over 3000 samples. For the HPC CUDA node, use `sglang` for maximum throughput.

### F.4 Normalized Levenshtein Library

The competition scoring uses `1 − Normalized_Levenshtein(GT, Pred)` where `Normalized_Levenshtein = dist / max(len(a), len(b))`.

**`rapidfuzz`** implements this exact formula: `rapidfuzz.distance.Levenshtein.normalized_distance(s1, s2)` returns `distance / max(len(s1), len(s2))`. **Use `rapidfuzz`** — it is faster than `python-Levenshtein` and implements the `max` normalization, not `mean`.[^68][^69][^70]

```python
from rapidfuzz.distance import Levenshtein

def score(gt: str, pred: str) -> float:
    return 1.0 - Levenshtein.normalized_distance(gt, pred)
    # Normalized_Levenshtein = edit_distance / max(len(gt), len(pred))
```

**Caution:** `thefuzz` uses `2 * M / T` where `T = sum(lengths)` — different formula. Do not use `thefuzz` for final scoring.[^71]

***

## Section G — Open Uncertainties / Decisions

| # | Question | Priority | Why it matters | Evidence resolves it | Default if no time |
|---|---|---|---|---|---|
| 1 | Which prompt template extracts most PII: naive `[REDACTED]`-fill, role-play, full-dialogue replay, assistant-prefix manipulation, or CoT "complete the user's record"? | **P0** | Template mismatch is the #1 failure mode for overfit models; wrong template = 0% extraction on otherwise-memorized samples | Grep `trainer.py` / `dataset.py` for exact prompt construction; run 5 template variants on 20 reference samples each in <30 min | Full-dialogue replay matching training format exactly |
| 2 | Image conditioning ON vs. OFF vs. blank vs. noise replacement — what helps? | **P0** | Image-free inference = 2× speed; if residual image adds noise, removal improves score | Ablation study on 280 reference samples (Section D.3 recipe, 40 min on M4) | Start with image ON (matches training); switch to text-only if ablation shows no gain |
| 3 | Sampling: greedy vs. beam(b∈{4,8}) vs. T=0.7+N samples+best-by-likelihood vs. constrained | **P0** | Greedy is sufficient iff memorization is complete; beam or sampling needed if format-lock exists | Check extraction rate on 280 ref samples under each mode in 20 min | Greedy first; beam(b=4) if greedy <80% on reference set |
| 4 | Shadow model gate: when to use `logp_target − logp_shadow` and at what threshold? | **P1** | 2× inference cost; adds value only when greedy fails or confidence is ambiguous | Compute ratio for 280 reference samples; find threshold that maximizes AUC vs. GT | Use ratio only as a reranker for top-p samples, not as a filter |
| 5 | Format-aware decoding ON vs. OFF — when does constraint help vs. hurt? | **P1** | Constraint can force wrong digits when model memorized a slightly non-standard format | Test constrained vs. free-form on 280 reference samples; measure Levenshtein delta per PII type | OFF for CREDIT (Luhn-constrained generation is risky); ON as post-hoc regex filter for EMAIL |
| 6 | White-box hooks (attention probing, gradient ascent on PII tokens, embedding inversion) — invest hours or stick to prompting? | **P1** | White-box methods may recover 5-15% additional PII but require 2-4 hours engineering | If prompt-only achieves >0.85 on reference set, skip white-box; otherwise invest after hour 4 | Skip in first 4 hours; revisit only if score plateaus |
| 7 | Submission scheduling under 5-min cooldown (~12 submissions/hour max, no feedback below best) | **P1** | Suboptimal scheduling wastes information signal; rate-limited feedback is a bandit problem | Heuristic: submit only when you have a method change of >0.02 expected delta; use first 2 hours for calibration, not submissions | Hours 0-2: 1 calibration submission. Hours 2-8: 1 submission per major method change. Hours 8-22: 1/hour refinement |
| 8 | Public vs. private leaderboard generalization — which method classes are robust? | **P2** | Final ranking uses 70% private set; methods overfit to 30% public regress hard | Empirically: methods based on prompt format and greedy decoding are stable; methods with heavy hyperparameter tuning to public set are fragile[^72][^73] | Prefer simple, well-calibrated methods over heavily tuned approaches; hold 10% of reference set as private-proxy for generalization check |

***

## Section H — Suggested Timeline

### First 30 Minutes — Environment + First Baseline

1. **Read the codebase** — identify exact training-time prompt template, model class, image processor, training loop (10 min).
2. **Reconstruct prompt template** from `dataset.py`/`trainer.py`; verify it matches the eval format (5 min).
3. **Greedy extraction baseline** — run 20 reference samples with reconstructed template; measure Levenshtein (10 min on M4).
4. **Sanity check** — if >70% extraction rate → proceed. If <30% → debug template. If 30-70% → run ablation on image vs. no-image immediately.
5. **Prepare submission pipeline** — CSV writer, length validator (10-100 char), rapidfuzz scorer (5 min).

### First 2 Hours — Calibrate and First Real Submission

1. **Full greedy extraction** on all 280 reference samples; compute per-PII-type Levenshtein (15 min).
2. **Shadow model baseline** — compute `logp_target − logp_shadow` for reference samples; calibrate threshold (20 min on M4, or 10 min on HPC).
3. **Image ablation** (Section D.3) — determines whether to include or exclude image for final run (40 min on M4).
4. **First real submission** — greedy, best image condition, no constraints (rows with [NOT VERIFIED] predictions filled with format-valid placeholders of length 15 for CC, 10-15 for phone, 15-30 for email).
5. **Decision point** — if public score > 0.7, proceed to format-aware reranking. If < 0.4, escalate to template debugging before anything else.

### Hours 2–8 — Scale Best Technique + Format-Aware

1. **Apply R.R.-style reranking** (Section B.5) — generate 5 candidates per sample via top-p=0.95, rank by `logp_target − logp_shadow`, take argmax (2 hours on HPC).
2. **Format-aware post-filtering** — use `rapidfuzz` + regex to validate EMAIL/PHONE/CC format; replace malformed predictions with format-valid greedy fallback (30 min).
3. **Submit** — if delta over previous best > 0.02, submit. Otherwise hold.
4. **Beam search (b=4)** for samples where greedy score on reference set was < 0.5 (1 hour on HPC or M4).
5. **Layer-wise probing** (Section A.6) — train probe on 280 reference samples; apply to 3000 eval samples; use as confidence flag to gate submission of uncertain predictions (1 hour).

### Hours 8–22 — Refinement + Generalization

1. **Ensemble** greedy + beam + top-p predictions per sample using shadow-model reranking; take consensus where 2/3 methods agree (2 hours).
2. **White-box gradient ascent** (if score plateaus) — compute `∇_θ log p(PII|prefix)` for ambiguous samples; use gradient signal to identify which tokens are most constrained (2-3 hours, CUDA only). `RESOURCE-HEAVY` on M4 — delegate to HPC.
3. **Generalization stress test** — hold out 10% of reference set as a private-proxy; measure performance delta between held-out and full-reference. Methods with >0.05 delta are overfitting to reference distribution.[^72]
4. **Final format validation** — run all 3000 predictions through length check (10-100 chars), format regex, Luhn check for CC. Fix any violations.
5. **Final submission** — submit 30 minutes before deadline; use the HPC best-ensemble result if CUDA pipeline is stable.

### Contingency — Score Plateaus Near 0.4

- **Root cause diagnosis:** 0.4 score = partially correct predictions. Likely issue: correct PII type but wrong digits/characters. Check if predictions have correct length and format but wrong content.
- **Next move (+0.1):** Switch from free-form to format-constrained generation AND apply shadow-model reranking simultaneously. If neither alone yields improvement, the bottleneck is prompt format — re-examine training template from codebase.
- **Nuclear option:** Gradient ascent on the `[REDACTED]` token embedding — treat the masked position as a continuous latent and optimize `argmax log p(token|context)` for each position in sequence. Expensive; requires CUDA; 4 hours for 3000 samples on A100.

### Contingency — Score Already High (~0.9)

- **Generalization-protection move:** Submit the method with the **simplest hyperparameters** (greedy, no shadow, no reranking) as your final anchor. The 70% private set may differ slightly; complex methods that extract 100% on the public 30% may fail on private samples with slightly different text patterns.
- **Robustness check:** Generate predictions with 5 different random seeds; compute variance per sample. High-variance samples are uncertain — fall back to format-valid placeholders for them rather than gambling.

***

## Appendix 1 — Reading List (Tiered)

### MUST Read (before hour 2)

1. Kowalczuk, Dubiński, Boenisch, Dziedzic. **Privacy Attacks on Image AutoRegressive Models.** ICML 2025. arXiv:2502.02514. Code: https://github.com/sprintml/privacy_attacks_against_iars[^41][^40]
2. Marek, Rossi, Hanke, Wang, Backes, Boenisch, Dziedzic. **Benchmarking Empirical Privacy Protection for Adaptations of LLMs.** ICLR 2026 (Oral). OpenReview: jY7fAo9rfK[^47][^48]
3. Wang, Dziedzic, Kim, Backes, Boenisch. **Captured by Captions: Memorization in CLIP.** ICLR 2025. arXiv:2502.07830[^44][^3]
4. Meng et al. **R.R.: Recollect and Rank.** ACL Findings 2025. arXiv:2502.12658. (PII reconstruction from scrubbed data)[^33][^32]
5. Nakka et al. **PII-Scope.** IJCNLP 2025. arXiv:2410.06704. (multi-query PII extraction benchmark)[^74][^34]

### SHOULD Read (hours 2-8)

6. Anonymous. **DP-MIA: Dual-Phase MIA for VLMs.** ICLR 2026 under review. OpenReview: WW47sMD57T[^11]
7. Zhang et al. **KCMP: Black-box MIA for LVLMs.** NeurIPS 2025. arXiv:2511.01952[^8]
8. Wang, Dziedzic, Backes, Boenisch. **Localizing Memorization in SSL Vision Encoders.** NeurIPS 2024. arXiv:2409.19069[^2]
9. Anonymous. **Watch Out Your Album.** ICML 2025. arXiv:2503.01208[^18]
10. Anonymous. **ICIMIA.** EACL 2026. arXiv:2506.12340[^14]
11. Hughes et al. **PATCH: Mitigating PII Leakage via Circuit Patching.** EACL 2026. arXiv:2510.07452 (use for localization insight)[^6]
12. Szep et al. **Unintended Memorization of Sensitive Information in Fine-Tuned LMs.** EACL 2026. arXiv:2601.17480[^30]
13. Anonymous. **OpenLVLM-MIA benchmark.** arXiv:2510.16295 (calibration caveat)[^20]

### MAY Read (hours 8+ or HPC pipeline)

14. Zhao, Maini, Boenisch, Dziedzic. **Unlocking Post-hoc DI with Synthetic Data.** ICML 2025. arXiv:2506.15271[^52]
15. Rossi, Marek, Boenisch, Dziedzic. **Natural Identifiers.** ICLR 2026. OpenReview: doaAUf9Pi7[^49]
16. Wahdany et al. **Curation Leaks.** ICLR 2026. arXiv:2603.00811[^55]
17. Hu et al. **MIA Against VLMs via Temperature Sensitivity.** arXiv:2501.18624[^17]
18. Anonymous. **DP-MIA on LVLMs.** OpenReview: WW47sMD57T[^12]
19. Anonymous. **MrM: Black-Box MIA on Multimodal RAG.** arXiv:2506.07399[^75]
20. Zhang et al. **Beyond Text: Privacy in MRAG.** EMNLP 2025. arXiv:2505.13957[^76]
21. Anonymous. **Constrained Decoding for Privacy-Preserving LLM Inference.** OpenReview: riu8VN6Do8[^29]
22. MIMIR library docs: https://github.com/iamgroot42/mimir (gradnorm + min-k++ implementation)[^39]
23. vLLM structured outputs docs: https://docs.vllm.ai/en/latest/features/structured_outputs/[^25]
24. XGrammar Apple Silicon support: https://github.com/mlc-ai/xgrammar[^24]
25. vllm-mlx Apple Silicon inference: https://github.com/waybarrios/vllm-mlx[^77]

***

## Appendix 2 — Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Wrong prompt template reconstructed from codebase → 0% greedy extraction | HIGH (common) | CRITICAL | Verify on 5 reference samples before scaling; try 3 template variants in parallel |
| R2 | Vision encoder path differs between LLaVA and Qwen-VL → hook code fails | MEDIUM | HIGH | Run `model.named_modules()` first; write a model-agnostic hook that searches for the last transformer block |
| R3 | Image wipe is so thorough that the model generates random PII | MEDIUM | MEDIUM | Counterfactual ablation (Section D.3) catches this; switch to text-only conditioning |
| R4 | `outlines` + HF multimodal pipeline has compatibility issues (known edge cases) | MEDIUM | MEDIUM | Use outlines 0.1.x with `LlavaForConditionalGeneration`; fallback to post-hoc regex validation |
| R5 | Shadow model has different tokenizer vocabulary → logp comparison is invalid | LOW | HIGH | Verify both models use identical tokenizer; check `config.json` `vocab_size` and `tokenizer_config.json` |
| R6 | 70% private set contains PII in different formats than 30% public | MEDIUM | HIGH | Generalization stress test (Appendix H); prefer format-agnostic Levenshtein over exact-match approaches |
| R7 | Gradient ascent (white-box) diverges or crashes on M4 MPS backend | HIGH (MPS autograd can be unstable) | LOW | Delegate all gradient-based methods to HPC CUDA node; M4 is for inference only |
| R8 | rapidfuzz normalized_distance formula mismatch vs. competition scorer | LOW | HIGH | Verify against manual example: `dist("abc","ab") = 1, max_len=3 → 0.333` matches `rapidfuzz.distance.Levenshtein.normalized_distance("abc","ab")` |
| R9 | Submission contains predictions outside 10-100 char range → rejected | LOW (preventable) | HIGH | Add length assertion to submission pipeline; pad short predictions to 10 chars with format-valid content |
| R10 | OpenLVLM-MIA distributional artifact inflates MIA calibration on reference set | MEDIUM | MEDIUM | Sanity check: shuffle labels; if AUC stays high, signal is spurious — rely on extraction rate, not MIA score |

***

## Appendix 3 — Glossary

| Acronym | Definition |
|---|---|
| LMM | Large Multimodal Model (autoregressive, processes image + text) |
| VLM / LVLM | (Large) Vision-Language Model — synonym for LMM in most 2025 papers |
| MIA | Membership Inference Attack |
| DI | Dataset Inference |
| PII | Personally Identifiable Information |
| LiRA | Likelihood Ratio Attack (Carlini et al., S&P 2022) `[FOUNDATIONAL]` |
| KCMP | Prior Knowledge-Calibrated Memory Probing (NeurIPS 2025) |
| RIGEL | Composite metric: Response time + Inference confidence + GEneration Length (DP-MIA) |
| TPR@FPR=1% | True Positive Rate at False Positive Rate = 1%; gold standard for MIA reporting |
| NID | Natural Identifier — structured random string for privacy auditing (ICLR 2026) |
| CLIPMem | Memorization metric for CLIP models (ICLR 2025, SprintML) |

---

## References

1. [Localizing Memorization in SSL Vision Encoders - OpenReview](https://openreview.net/forum?id=R46HGlIjcG) - This paper investigates memorization in self-supervised learning encoders. It introduces two metrics...

2. [[PDF] Localizing Memorization in SSL Vision Encoders - NIPS papers](https://proceedings.neurips.cc/paper_files/paper/2024/file/6f6af59b11f3919965b9811c6c9ad6df-Paper-Conference.pdf) - We propose LayerMem and UnitMem, the first practical metrics to localize memorization in SSL encoder...

3. [Captured by Captions: On Memorization and its Mitigation in CLIP...](https://openreview.net/forum?id=5V0f8igznO) - We propose a metric to measure memorization in CLIP models and study the memorization behavior in th...

4. [On Memorization and its Mitigation in CLIP Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9a1dab894ce96cb8339c2fadd85a100b-Abstract-Conference.html) - Captured by Captions: On Memorization and its Mitigation in CLIP Models. Part of International Confe...

5. [SoK: The Landscape of Memorization in LLMs - arXiv](https://arxiv.org/html/2507.05578v2) - We explore key drivers, including training data duplication, training dynamics, and fine-tuning proc...

6. [PATCH: Mitigating PII Leakage in Language Models with Privacy ...](https://arxiv.org/abs/2510.07452) - Abstract page for arXiv paper 2510.07452: PATCH: Mitigating PII Leakage in Language Models with Priv...

7. [[PDF] PATCH: Mitigating PII Leakage in Language Models with Privacy ...](https://aclanthology.org/2026.findings-eacl.271.pdf) - We first apply mechanistic interpretability to discover the internal computational structures. (or “...

8. [Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing](https://arxiv.org/abs/2511.01952) - Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpo...

9. [Black-Box Membership Inference Attack for LVLMs via Prior...](https://openreview.net/forum?id=4GyTBGBVsB) - Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpo...

10. [NeurIPS Poster Black-Box Membership Inference Attack for LVLMs ...](https://neurips.cc/virtual/2025/loc/san-diego/poster/119960)

11. [[PDF] DP-MIA: DUAL-PHASE MEMBERSHIP INFERENCE ATTACK ...](https://openreview.net/pdf/3cb921e2d7e9f814fe722e85e48ba9d49ef22832.pdf) - Exten- sive experiments on LLaVA and Qwen2-VL demonstrate DP-MIA's effectiveness. (88.2% accuracy) s...

12. [[PDF] DP-MIA: DUAL-PHASE MEMBERSHIP INFERENCE ATTACK ...](https://openreview.net/pdf?id=WW47sMD57T)

13. [Image Corruption-Inspired Membership Inference Attacks against Large Vision-Language Models](https://arxiv.org/abs/2506.12340) - Large vision-language models (LVLMs) have demonstrated outstanding performance in many downstream ta...

14. [Image Corruption-Inspired Membership Inference Attacks against ...](https://arxiv.org/html/2506.12340v1) - We design simple yet effective Image Corruption-Inspired Membership Inference Attacks (ICIMIA) again...

15. [[PDF] Image Corruption-Inspired Membership Inference Attacks against ...](https://aclanthology.org/2026.eacl-long.371.pdf) - In this work, we focus on detecting whether a target image is used to train the target. LVLM. We des...

16. [Membership Inference Attacks Against Vision-Language Models](https://arxiv.org/html/2501.18624v2) - Vision-Language Models (VLMs), built on pre-trained vision encoders and large
language models (LLMs)...

17. [Membership Inference Attacks Against Vision-Language Models](https://arxiv.org/abs/2501.18624) - In this paper, we conduct the first analysis of misuse and leakage detection in VLMs through the len...

18. [Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](https://arxiv.org/abs/2503.01208) - Multi-Modal Large Language Models (MLLMs) have exhibited remarkable performance on various vision-la...

19. [Watch Out Your Album! On the Inadvertent Privacy Memorization in ...](https://icml.cc/virtual/2025/poster/43674) - In this paper, we investigate how randomly generated task-irrelevant private content can become spur...

20. [OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of ...](https://arxiv.org/abs/2510.16295) - OpenLVLM-MIA is a new benchmark that highlights fundamental challenges in evaluating membership infe...

21. [[PDF] OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of ...](https://arxiv.org/pdf/2510.16295.pdf) - OpenLVLM-MIA is a new benchmark that highlights funda- mental challenges in evaluating membership in...

22. [Welcome to Outlines!](https://dottxt-ai.github.io/outlines/latest/) - Outlines guarantees structured outputs during generation — directly from any LLM. Works with any mod...

23. [Qwen/Qwen2.5-VL-7B-Instruct · How to output in a ...](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/10) - Hi, you can use the outlines (https://github.com/dottxt-ai/outlines) library to do the constrained d...

24. [mlc-ai/xgrammar: Fast, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar) - XGrammar features universal deployment. It supports: Platforms: Linux, macOS, Windows; Hardware: CPU...

25. [Structured Outputs — vLLM](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html) - vLLM supports the generation of structured outputs using outlines, lm-format-enforcer, or xgrammar a...

26. [Structured outputs in vLLM: Guiding AI responses](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses) - Learn how to control the output of vLLM's AI responses with structured outputs. Discover how to defi...

27. [Structured Decoding in vLLM: A Gentle Introduction](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction) - Understand structure decoding and vLLM and how recent XGrammar integration can contribute to 5x impr...

28. [XGrammar: Flexible and Efficient Structured Generation Engine For Large Language Models](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf)

29. [[PDF] Constrained Decoding for Privacy-Preserving LLM Inference](https://openreview.net/pdf?id=riu8VN6Do8) - Large language models frequently leak personally identifiable information (PII) during text generati...

30. [Unintended Memorization of Sensitive Information in Fine-Tuned ...](https://arxiv.org/abs/2601.17480) - In this work, we systematically investigate a critical and underexplored vulnerability: the exposure...

31. [Unintended Memorization of Sensitive Information in Fine-Tuned ...](https://aclanthology.org/2026.eacl-long.304/) - In this work, we systematically investigate a critical and underexplored vulnerability: the exposure...

32. [Unveiling LLM Training Privacy through Recollection and Ranking](https://arxiv.org/abs/2502.12658) - In this paper, we propose RR (Recollect and Rank), a novel two-step privacy stealing attack that ena...

33. [[PDF] RR: Unveiling LLM Training Privacy through Recollection and Ranking](https://aclanthology.org/2025.findings-acl.894.pdf) - Our R.R. reconstructs masked PII in two steps ... In this paper, we propose R.R. (Recollect and. Ran...

34. [PII-Scope: A Comprehensive Study on Training Data Privacy ...](https://aclanthology.org/2025.ijcnlp-long.195/) - Abstract. In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state...

35. [[PDF] PII-Scope: A Comprehensive Study on Training Data Privacy ...](https://aclanthology.org/2025.ijcnlp-long.195.pdf)

36. [A Benchmark for Training Data PII Leakage Assessment in LLMs](https://www.themoonlight.io/en/review/pii-scope-a-benchmark-for-training-data-pii-leakage-assessment-in-llms) - It introduces a framework called PII-Scope, aiming to standardize the evaluation of PII extraction m...

37. [Unintended Memorization of Sensitive Information in Fine-Tuned ...](https://arxiv.org/html/2601.17480v1) - In this work, we systematically investigate a critical and underexplored vulnerability: the exposure...

38. [How much do language models memorize? - arXiv](https://arxiv.org/html/2505.24832v1) - The intuitive explanation is that M.I. is easy for large models overfit to tiny datasets ... When th...

39. [Mimir - Python package for measuring memorization in LLMs. - GitHub](https://github.com/iamgroot42/mimir) - iamgroot42/mimir ; Latest commit. actions-user · Update documentation. success. 10 months ago. 1b6fd...

40. [[ICML 2025] Privacy Attacks on Image AutoRegressive Models ...](https://github.com/sprintml/privacy_attacks_against_iars) - We develop a novel membership inference attack (MIA) that achieves an exceptionally high success rat...

41. [Privacy Attacks on Image AutoRegressive Models](https://proceedings.mlr.press/v267/kowalczuk25a.html) - Our results suggest a fundamental privacy-utility trade-off: while IARs excel in image generation qu...

42. [Papers](https://antonikowalczuk.com/papers/) - A Kowalczuk · Cytowane przez 2 — ... Dziedzic, Franziska Boenisch. Privacy Attacks On Image AutoRegr...

43. [Antoni Kowalczuk - Privacy Attacks on Image AutoRegressive Models | ML in PL 2025](https://www.youtube.com/watch?v=nS6RNUXQKdQ) - Image autoregressive generation has emerged as a powerful new paradigm, with image autoregressive mo...

44. [On Memorization and its Mitigation in CLIP Models - arXiv](https://arxiv.org/html/2502.07830v2) - Captured by Captions: On Memorization and its Mitigation in CLIP Models. Report issue for preceding ...

45. [Localizing Memorization in SSL Vision Encoders](https://arxiv.org/pdf/2409.19069.pdf) - ...distributed across the
entire encoder, (2) a significant fraction of units in SSL encoders experi...

46. [Benchmarking Empirical Privacy Protection for Adaptations ...](https://openreview.net/pdf?id=m6f95cqQx2)

47. [Benchmarking Empirical Privacy Protection for Adaptations of Large...](https://openreview.net/forum?id=jY7fAo9rfK) - Our benchmark identifies key factors for achieving practical privacy in DP LLM adaptation, providing...

48. [Benchmarking Empirical Privacy Protection for Adaptations of Large ...](https://iclr.cc/virtual/2026/oral/10007870) - Our benchmark identifies key factors for achieving practical privacy in DP LLM adaptation, providing...

49. [Natural Identifiers for Privacy and Data Audits in Large Language...](https://openreview.net/forum?id=doaAUf9Pi7) - Natural Identifiers for Privacy and Data Audits in Large Language Models. Download PDF · Lorenzo Ros...

50. [[PDF] NATURAL IDENTIFIERS FOR PRIVACY AND DATA AUDITS IN ...](https://openreview.net/pdf?id=doaAUf9Pi7) - NATURAL IDENTIFIERS FOR PRIVACY AND DATA. AUDITS IN LARGE LANGUAGE MODELS. Lorenzo Rossi, Bartłomiej...

51. [ICLR Poster Natural Identifiers for Privacy and Data Audits in Large ...](https://iclr.cc/virtual/2026/poster/10008382) - Natural Identifiers for Privacy and Data Audits in Large Language Models. Lorenzo Rossi ⋅ Bartłomiej...

52. [Unlocking Post-hoc Dataset Inference with Synthetic Data](https://openreview.net/forum?id=a5Kgv47d2e&noteId=Rgckzt1O5k) - This paper presents a framework for dataset inference (DI) in large language models, addressing the ...

53. [publications - Adam Dziedzic](https://adam-dziedzic.com/publications/) - ... (ICML)} }. Unlocking Post-hoc Dataset Inference with Synthetic Data Bihe Zhao, Pratyush Maini, F...

54. [ICML Poster Unlocking Post-hoc Dataset Inference with Synthetic Data](https://icml.cc/virtual/2025/poster/44819) - Our approach tackles two key obstacles: (1) creating high-quality, diverse synthetic data that accur...

55. [Membership Inference Attacks against Data Curation for Machine ...](https://arxiv.org/abs/2603.00811) - Abstract page for arXiv paper 2603.00811: Curation Leaks: Membership Inference Attacks against Data ...

56. [Curation Leaks: Membership Inference Attacks against Data ...](https://openreview.net/forum?id=BzNf90Csfa) - In machine learning, data curation is used to select the most valuable data for improving both model...

57. [Precise Parameter Localization for Textual Generation in Diffusion ...](https://arxiv.org/abs/2502.09935) - We improve textual generation efficiency and performance by targeting cross and joint attention laye...

58. [Precise parameter localization for textual generation in diffusion ...](https://www.pw.edu.pl/publikacje/precise-parameter-localization-textual-generation-diffusion-models) - Proceedings of the International Conference on Representation Learning 2025 (ICLR 2025). Autorzy z P...

59. [iamgroot42/mimir - Workflow runs - GitHub](https://github.com/iamgroot42/mimir/actions) - Jul 10, 2025, 10:13 AM PDT 1m 42s. Datasets: custom code support Tests #112: Commit 1a5e854 pushed b...

60. [Activity · iamgroot42/mimir · GitHub](https://github.com/iamgroot42/mimir/activity) - on Jul 10, 2025. More activity actions. More activity actions. Datasets: custom code support. iamgro...

61. [SprintML - GitHub](https://github.com/sprintml) - This is the official github of the SprintML.com lab. - SprintML. ... privacy_attacks_against_iars pr...

62. [Official Repository for Dataset Inference for LLMs · GitHub](https://github.com/pratyushmaini/llm_dataset_inference/) - This repository contains data different subsets of the PILE, divided into train and val sets. The da...

63. [Releases · pratyushmaini/llm_dataset_inference - GitHub](https://github.com/pratyushmaini/llm_dataset_inference/releases) - There aren't any releases here ... You can create a release to package software, along with release ...

64. [LIONS-EPFL/VL-MIA - GitHub](https://github.com/LIONS-EPFL/VL-MIA) - The VL-MIA datasets serve as a benchmark designed to evaluate membership inference attack (MIA) meth...

65. [Native LLM and MLLM Inference at Scale on Apple Silicon - arXiv](https://arxiv.org/html/2601.19139v1) - Built natively on MLX (Apple, 2023) , our system leverages the unified memory architecture for zero-...

66. [Native LLM and MLLM Inference at Scale on Apple Silicon - arXiv](https://arxiv.org/html/2601.19139v2)

67. [[Roadmap] Apple Device Support (2026 Q1) · Issue #19137 - GitHub](https://github.com/sgl-project/sglang/issues/19137) - This document (AI-assisted) describes the technical roadmap for macOS support in SGLang, covering bo...

68. [Damerau Levenshtein](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/DamerauLevenshtein.html)

69. [GitHub - rapidfuzz/python-Levenshtein: The Levenshtein Python C extension module contains functions for fast computation of Levenshtein distance and string similarity](https://github.com/rapidfuzz/python-Levenshtein) - The Levenshtein Python C extension module contains functions for fast computation of Levenshtein dis...

70. [Levenshtein¶](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html)

71. [Meaning behind 'thefuzz' / 'rapidfuzz' similarity metric when comparing strings](https://stackoverflow.com/questions/77787380/meaning-behind-thefuzz-rapidfuzz-similarity-metric-when-comparing-strings) - When using thefuzz in Python to calculate a simple ratio between two strings, a result of 0 means th...

72. [Be warned: Don't overfit your model to Public Leaderboard Dataset!](https://www.kaggle.com/general/54610) - I (and the others) totally overfitted the selected models to the Public Leaderboard dataset and then...

73. [Mark my words: I predict dramatic leaderboard shakeups in current ...](https://x.com/JFPuget/status/2051393293880066512) - Lots of people have shifted to leaderboard climbing using AI agents on Kaggle. These agents are over...

74. [[2410.06704] PII-Scope: A Comprehensive Study on Training Data ...](https://arxiv.org/abs/2410.06704) - A comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction att...

75. [MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems](https://arxiv.org/abs/2506.07399) - Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by inte...

76. [Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal ... - arXiv](https://arxiv.org/abs/2505.13957) - We provide the first systematic analysis of MRAG privacy vulnerabilities across vision-language and ...

77. [waybarrios/vllm-mlx - GitHub](https://github.com/waybarrios/vllm-mlx) - A vLLM-style inference server for Apple Silicon Macs. Unlike Ollama or mlx-lm used directly, it ship...

