# Task 3 — Submission Tracker

> Live tracking of every submitted CSV with leaderboard score.
> Auto-updated as new submissions go in.
> Last refresh: 2026-05-10 03:42Z

## Current best
**0.2841** — `submission_cross_lm.csv` (cross-LM v1, 6 derived features) — RANK #2

## Leaderboard task3 (snapshot 2026-05-10 00:46Z)
| # | Team | Score |
|---|---|---|
| 1 | Syntax Terror | 0.3955 |
| **2** | **Czumpers** | **0.2841** ⭐ |
| 3 | Advanced Persistent Thinkers | 0.2702 |
| 4 | GradientLabs | 0.2173 |
| 5 | ParmaGo | 0.2006 |

Need to close 0.111 gap to overtake Syntax Terror.

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
| – | clm_lgbm (v1 + LightGBM) | 0.7333 | TBD | Submitted (id 1100?) |
| – | ensemble_v2 (rank avg 5) | n/a | not submitted | Skipped (rho 0.99 vs cross_lm) |

---

## Stacking sweep 2026-05-10 (leak-free pivot)

Hypothesis: branch_bc (UnigramGreenList) is fitted on train labels — overfits OOF
but doesn't generalize. Drop it, focus on PPL/cross-LM signals only.

Confirmed: leak-free OOF (~0.32-0.39) much closer to leaderboard than leaky OOF (~0.69-0.74).

| ID  | CSV | OOF | rho_vs_cross_lm | Hipoteza testowana |
|-----|-----|-----|---------|---------|
| 1167 | stack_meta_top6_weighted | 0.7370 | 0.96 | Leaky stack baseline |
| 1179 | v2_best_base | 0.3741 | 0.83 | First leak-free (full pool, 127 features) |
| 1194 | v2_top5_weighted | n/a | ~0.85 | Top-5 rank-avg from v2 views |
| 1213 | v3_best_base | 0.3778 | 0.84 | SelectKBest K=40 LR C=0.05 |
| 1229 | v3_meta | 0.3481 | 0.85 | Meta-LR over v3 stack |
| 1240 | v3_top3_rank | n/a | 0.85 | Top-3 rank-avg from v3 |
| 1257 | hybrid_v1 | n/a | ~0.82 | v3+v2+v4_meta blend |
| 1268 | v5_best_base | 0.3667 | 0.84 | Full pool incl. judges/kgw_llama/sir (297 features) |
| 1274 | v6_round1 | 0.3222 | ~0.83 | Pseudo-labeling, 1 round |
| 1283 | v7_best_single | **0.3852** | 0.84 | Lucky K=30 seed=2024 (multi-seed sweep max) |
| 1296 | v7_top5_weighted | n/a | 0.84 | Top-5 from 30 multi-seed K-best |
| 1306 | v8_best (stable K=40) | 0.3111 | 0.84 | K=40 from MI averaged over 10 seeds |
| 1314 | v2_pure_cross | 0.21 | **0.77** | Only 11 cross-LM features — most diverse |
| 1326 | hybrid_v5_3way | n/a | 0.82 | v3+v7+v8 rank-avg |
| 1334 | v9_top1 | 0.270 | **0.77** | Brute-force best K=4 cross-LM subset |
| 1342 | hybrid_v8_v3v9 | n/a | 0.82 | 1:1 v3+v9 |
| 1352 | hybrid_v9_v3v9c | n/a | 0.92 | 1:1:1 v3+v9+cross_lm_best |
| 1360 | w_proven | n/a | 0.96 | 3:1:1:1 cross_lm+v3+v9+v7 (defensive blend) |
| 1375 | median9 | n/a | 0.86 | Median rank across 9 distinct submissions |

## Pending analysis (ask user for leaderboard scores)
- **Leak-free hypothesis**: czy v3_best/v7_best/v9_top1 ≥ cross_lm_best (0.284)?
  - Jeśli TAK → leak-free recipe działa, dalej iterujemy
  - Jeśli NIE → branch_bc faktycznie pomaga, wracamy do v1 cross_lm
- **Lucky vs stable K-best**: v3_best (0.3778 OOF, lucky) vs v8_best (0.3111 OOF, stable)
  - Lucky wygra → cherry-picking generalizuje
  - Stable wygra → wariancja w MI selekcji boli
- **Diversity**: v2_pure_cross/v9_top1 (rho 0.77 vs cross_lm) — kierunek "brand new signal"
- **Smart blends**: w_proven (defensive, +risk reduction) vs median9 (aggressive, robust)
- **Pseudo-labeling**: v6_round1 — czy zysk z self-training?

## Spearman rho among leak-free submissions
- Większość v3-v8: rho 0.98+ między sobą (zbiega na ten sam signal)
- v9 (cross-LM brute force) i v2_pure_cross: rho 0.77 vs cross_lm_best — **realna różnorodność**
- median9: rho 0.86 vs cross_lm — robust ensemble

## Untried in this session
- DeBERTa/RoBERTa fine-tune end-to-end (1-2h compute)
- Test-time paraphrasing perturbation (heavy)
- LightGBM/XGBoost (cluster venv blocked — disk quota / inode limit)
- Aggressive pseudo-labeling rounds 2-3 (rounds 2-3 actually decreased OOF in v6)

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
- Test-time paraphrasing perturbation (would need re-run paraphrasing on 2790 texts)
- LightGBM (cluster venv blocked — disk quota / inode limit)
- DeBERTa/RoBERTa fine-tune end-to-end on raw text (~1-2h compute)
- Aggressive pseudo-labeling (5+ rounds, gradual unmask)
- Test-time augmentation (multi-paraphrase, average predictions)
- Cross-team blend (combine our best + leaderboard #3 if we had access)
