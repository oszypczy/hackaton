# Research prompt — gap analysis (~100 words)

We are stuck at OVERALL ~0.381 on the public leaderboard with `direct_probe` (a paraphrased question, no chat-template bypass). CREDIT is a hard floor (~0.24, 0 perfect). Empirical ablations have already falsified Carlini'21/Nasr'23 verbatim-prefix hypothesis on our overfit instruction-tuned setup, and per-PII routing did not lift CREDIT. We have already mined Pinto (DocVQA, ICML'24), PII-Scope (arXiv:2410.06704), Carlini'21, Nasr'23, Min-K%++, and the standard SprintML corpus. The remaining unexplored axes are: multimodal cross-attention leakage, novel decoders (DOLA / contrastive / logit-lens), edit-distance-aware aggregation over K samples, partial-string scoring tricks for Levenshtein, and 2024–2026 gray-literature tricks for overfit-LMM extraction. The prompt below targets only those gaps.

---

# Claude Research prompt — paste below this line

You are a research assistant. I am a black-box attacker against an **intentionally overfitted multimodal LMM** (OLMo-2-1B language model + LLaVA-HR vision encoder, bf16, ~3.6 GB, instruction-tuned, NOT RLHF-aligned). The model was fine-tuned on a synthetic PII-VQA dataset where each training sample has the form `[SYSTEM]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER containing EMAIL/CREDIT/PHONE]`. At inference I have 1000 task rows × 3 PII types = 3000 predictions to make, plus 280 validation rows with ground-truth PII for local calibration. The scoring metric is **mean(1 − Normalized_Levenshtein(GT, pred))** — so partially correct strings get partial credit. I have black-box access only (forward passes via HuggingFace generate, no gradient access, no ability to retrain, full logits at decode time).

I have **already exhausted** the obvious literature. **Do NOT include** any of the following — I have read and implemented from these sources already:

- Carlini et al. 2021 (*Extracting Training Data from LLMs*, USENIX'21) — six MIA features, zlib/Small-LM ratios
- Nasr et al. 2023 (*Scalable Extraction*, chat divergence)
- Pinto et al. 2024 (*DocVQA Extraction*, ICML'24, arXiv:2407.08707) — `(I⁻ᵃ, Q_original)` recipe, Q-paraphrasing ablation
- Nikhil Kandpal / PII-Scope (arXiv:2410.06704 v2) — 4 templates A/B/C/D, PII-Compass true-prefix, ICL demo selection, top-k×64 multi-query, continual extraction
- Min-K%++ (ICLR'25), LiRA (Carlini S&P'22)
- Watermark stealing / DIPPER / WAVES / Kirchenbauer (Task 3 only, off-topic here)
- Soft Prompt Tuning (white-box, off-table for me)
- Generic "try better prompts" advice without a concrete recipe
- Anything requiring gradient access, model retraining, or membership-inference for-its-own-sake

I have **already empirically tested and rejected**: per-PII strategy routing (CREDIT → baseline, EMAIL/PHONE → direct_probe) and verbatim-prefix that bypasses the chat template. Both regress on this setup.

Please surface **NEW** ideas (preferably 2024–2026, including workshop papers, blog posts, twitter threads, NeurIPS/ICML/ICLR/EMNLP/ACL/USENIX/S&P) on the following five questions. For each direction, give me a **concrete, copyable recipe**, the **paper title + arXiv ID + venue + year**, and (where possible) reported magnitude of improvement.

1. **Multimodal-LMM-specific extraction.** What 2024–2026 work shows that PII / training-text leakage in vision-LMs (LLaVA, OLMo-multimodal, Qwen-VL, Idefics, InternVL, MiniCPM-V) is amplified by manipulating *vision tokens* (token replay, image-token soft-prompting, attention-sink injection, cross-modal prompt smuggling, dummy-image conditioning, image-patch ordering, vision-encoder noise injection) at black-box inference time? Anything beyond Pinto's `(I⁻ᵃ, Q_original)` recipe.

2. **Novel decoders for memorization.** What concrete decoding strategies have shown lift over greedy / temperature sampling specifically for *recovering memorized strings* from overfit autoregressive models? Specifically: contrastive decoding (Li et al.), DOLA (Chuang et al. ICLR'24), speculative decoding repurposed for memorization, logit-lens / activation patching at black-box level, draft-and-verify with a clean shadow model, entropy-guided beam search, typical sampling at low τ for tail recall. I want recipes that can be implemented with HF `LogitsProcessor` only.

3. **Edit-distance-aware aggregation across K samples.** When sampling K candidates per query, what aggregation method maximizes **1 − normalized Levenshtein** to an unknown GT? Specifically: Levenshtein medoid, character-level majority voting (consensus per position with alignment), self-consistency at the substring level, MBR decoding with edit-distance utility, Dawid-Skene-style aggregation over noisy candidates, and any 2024–2026 work explicitly optimizing for partial-string recovery rather than exact match. Cite anything from speech-recognition / OCR / DNA-assembly that solves the same aggregation problem.

4. **Tricks specific to overfit / over-trained models.** What black-box tricks exploit the fact that the model is **deliberately over-trained** (low train loss, sharp memorization peaks)? Examples to consider but extend beyond: prompt amplification / prompt-stacking, repeated-token excitation, activation noise via temperature schedules, "echo prompting", loss-landscape probing via paraphrase ensemble, persona-flooding, prompts that exploit sharp-minimum collapse. Any recent (2024–2026) work tying overfit-magnitude to extraction yield.

5. **Gray literature.** Anything from blog posts, Twitter/X threads, ICLR/NeurIPS workshops 2024–2026, Anthropic/DeepMind/OpenAI red-team write-ups, AISI/UK-AISI reports, alignment forum, LessWrong, that beats Carlini/Pinto-style baselines for verbatim training-data extraction from small (1B–7B) instruction-tuned multimodal LMs. I am especially interested in any "weird trick" papers that mainstream literature has not picked up.

**Format your reply as five sections (one per question)**, each with: (a) 2–4 ranked recipes, (b) full citations with arXiv IDs, (c) one-paragraph reasoning per recipe explaining why it should work for an instruction-tuned overfit OLMo-2-1B + LLaVA-HR, (d) a 1-sentence implementation sketch. **Skip anything I have already covered (listed above)**. Bias toward 2024–2026 and toward black-box / inference-time tricks. Synthesis must arrive within ~1 hour — prefer fewer well-grounded recipes over a broad survey.
