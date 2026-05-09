# Task 3 — Submission Tracker

> Live tracking of every submitted CSV with leaderboard score.
> Auto-updated as new submissions go in.
> Last refresh: 2026-05-10 00:40Z

## Current best
**0.284** — `submission_cross_lm.csv` (cross-LM v1, 6 derived features)

## Leaderboard top: 0.27+ (known competitor maximum reported earlier)

---

## All submissions (chronological)

| ID  | CSV | OOF | Leaderboard | Notes |
|-----|-----|-----|-------------|-------|
| early | strong_bino | 0.6444 | **0.103** | First real signal (Pythia binoculars) |
| early | xl_bino | 0.6556 | not improved | Bigger bino → plateau |
| early | bigram | 0.7037 | not improved | Overfit OOF, no leaderboard help |
| early | fdgpt | 0.6815 | not improved | Curvature features, no help |
| early | roberta (PCA32) | 0.7037 | not improved | Frozen RoBERTa embed |
| early | multi_lm (OLMo-1B PPL) | 0.6926 | **0.158** | OLMo breakthrough |
| early | multi_lm_v2 (4 instruct LMs) | 0.6593 | not improved | Diluted OLMo signal |
| early | lm_judge (OLMo-1B prompts) | 0.6704 | **0.20** | Zero-shot prompting |
| early | judge_phi2 | 0.6815 | not improved | Phi-2 added to judge |
| early | judge_mistral | 0.6370 | not improved | Mistral hurt OOF |
| early | olmo7b (PPL alone) | 0.7519 | **0.259** | OLMo-7B size scaling |
| early | judge_olmo7b | 0.6963 | not improved | Stack diluted |
| early | olmo13b PPL | 0.7222 | not improved | Stack diluted |
| early | judge_olmo13b | 0.7296 | not improved | Stack diluted |
| early | judge_chat (proper template) | 0.7296 | not improved | Same as 13B judge |
| early | **cross_lm v1 (6 derived)** | 0.6963 | **0.284** ⭐ | NEW BEST! Cross-LM ratios work |
| early | cross_lm_v2 (56 derived) | 0.6741 | not improved | Too many derived = noise |
| 872 | minimal (28 features) | 0.7407 | TBD | Submitted, awaiting leaderboard |
| 887 | select_k (top-15 by mutual info) | 0.7333 | TBD | Submitted, awaiting leaderboard |
| – | clm_minimal (v1 + minimal stack) | 0.6889 | not submitted | Ready |
| – | clm_lgbm (v1 + LightGBM) | 0.7333 | not submitted | Ready |
| – | ensemble_v2 (rank avg 5) | n/a | not submitted | Ready |

---

## Insights

1. **OOF ≠ leaderboard**. cross_lm v1 had OOF 0.6963 (lower than OLMo-7B 0.7519) but leaderboard 0.284 vs 0.259.
2. **Stacking many features hurts**: every variant with 60+ features → OOF drops, leaderboard plateaus.
3. **Specific cross-LM ratios beat raw LM PPL**: instruct vs base PPL differences carry watermark signal.
4. **OLMo-Instruct family is special**: Pythia/GPT-2/OPT/LLaMA all gave less. OLMo-2 instruct = unique signal.
5. **KGW direct detection dead-end**: 14+ configs tested, none matched organizers' watermark.
6. **Single-model size scaling has plateaus**: bigger Bino (xl, 6.9b/2.8b) didn't beat strong (1.4b/2.8b) on leaderboard.

## Strategies that WORKED
- ✅ OLMo-7B-Instruct PPL (universal LLM signal)
- ✅ Zero-shot prompting LM-as-judge
- ✅ Cross-LM ratio features (v1, 6 specific derived)

## Strategies that DIDN'T WORK
- ❌ Stacking 60+ features (LogReg dilutes)
- ❌ KGW direct detection (any tokenizer/hash_key)
- ❌ Adding more "instruct LMs" (Phi-2, Mistral) to OLMo
- ❌ More derived features in v2
- ❌ Larger Bino models above Pythia-1.4b/2.8b
- ❌ Bigram greenlist (overfits training watermark signature)

## Untried (if more time)
- Skip-gram greenlist (k=2,3,4)
- Test-time paraphrasing perturbation
- LightGBM (clm_lgbm ready, awaiting submit)
- Self-training / pseudo-labeling from test
