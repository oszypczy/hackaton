# Task 3 (Watermark Detection) — Session Log

> **PURPOSE:** Anyone should be able to resume work from any point.
> Updated continuously during 24h hackathon. Latest at top.

---

## TL;DR — Current state (2026-05-09 ~22:24 UTC)

- **Leaderboard score: task3 = 0.2841 (#2)** ⬆️ JUMP from 0.259 — leader Syntax Terror **0.396** (also up), gap 0.111
- Mystery: score went from 0.259 → 0.2841 between 22:01 and 22:22 even though no submissions logged in that window. Hypothesis: a 429-rejected POST (pseudo_meta_ensemble at 22:11:36 OR similar) was actually processed by server, OR teammate submitted.
- super_pseudo_f050 (OOF 0.9630) → LB 0.2841 (same, no improvement). **Pseudo-label OOF inflation IS partly overfitting.** OOF gain >0.85 doesn't translate to LB.
- 0.2841 likely came from a moderate pseudo-label submission (probably pseudo_meta_ensemble or pseudo_f030 single) that ran 22:04-22:11 during cooldown errors.
- 🚀 **BREAKTHROUGH**: **Pseudo-labeling f=0.30 → OOF 0.8444** (+6.7pp from baseline 0.7778). Test set self-augmentation works — confirms test distribution is learnable from our features but classifier overfits to small labeled set.
- **OOF ceiling at ~0.78 with standard features.** Adding olmo_7b, judge_phi2, judge_mistral, calibrated, ridge/EN/MLP/SVM, empirical green list — ALL plateau or hurt.
- **Submission cooldown**: server resets cooldown on each rejection retry — wait clean 5+ min before retrying after 429.
- **CRITICAL INSIGHT**: Live LB is only **30% public test**. Final ranking is **70% private hidden**.
- **Submit queue (active)**:
  - 21:53:56 multiseed_rankmean (OOF mean 0.778)
  - 21:59:06 hyb5_no_judges (OOF 0.7704)
  - 22:04:20 **pseudo_f030 (OOF 0.8444)** ⭐
  - 22:09:30 pseudo_f020 (OOF 0.8111)
  - 22:14:40 mega_ensemble (15-config rank-mean)
- **Active jobs on Jülich**: NONE — feature exploration done. Awaiting submission feedback.
- **Killed jobs**: 14739848 selfhash, 14739925 mistral_kgw, 14739849 leak_free (all stuck on slow kgw_selfhash)

---

## Score history (chronological)

| Time (UTC) | Score | CSV | OOF | Notes |
|---|---|---|---|---|
| ~14:42 | 0.05 | task3_watermark.csv | - | Phase 1 LightGBM baseline |
| ~15:44 | 0.09 | task3_watermark.csv | - | LogReg + branch_bc |
| ~15:58 | 0.103 | task3_watermark.csv | - | LogReg + bc + strong_bino |
| ~19:42 | 0.13649 | task3_watermark_fdgpt.csv | - | + Fast-DetectGPT (Pythia 2.8b) |
| ~20:20 | 0.13649 | multi_lm | - | (no improvement) |
| ~21:23 | **0.25905** | submission_kitchen_v2.csv | 0.7519 | **JUMP** — multan1's all features (multi_lm_v2, lm_judge, kgw_llama, roberta, stylometric, better_liu) |
| ~21:27 | 0.25905 | task3_hybrid_v3.csv | 0.7630 | (no improvement — hybrid_v3 = kitchen_v2 + my unigram_direct) |
| ~21:33 | 0.25905 | task3_hyb_C05.csv | 0.7778 | (NO improvement — best OOF in sweep but plateau) |
| pending 21:48:30 | ? | task3_hyb5_no_judges.csv | 0.7704 | hybrid + olmo, no judges |

**Leader (Syntax Terror): 0.387** — gap to bridge: 0.128

## Classifier sweep (with 18 features, no olmo/judges)

| Classifier | OOF | Notes |
|---|---|---|
| **LogReg C=0.05** | **0.7778** | ⭐ BEST |
| ElasticNet l1=0.1 | 0.7667 | close |
| Ensemble (logreg+lgbm) | 0.7667 | ties EN |
| MLP 64 hidden | 0.7296 | overfits |
| Ridge | 0.7222 | underfits |
| ElasticNet l1=0.5 | 0.7148 | too sparse |
| SVM RBF C=0.1 | 0.1519 | needs tuning |

## Feature ablation (hybrid_v5)

| Variant | OOF | Δ from baseline |
|---|---|---|
| 18 features (no olmo/judges) C=0.05 | 0.7778 | baseline |
| + olmo_7b (no judges) C=0.05 | 0.7704 | -0.7pp |
| + judges (no olmo) C=0.05 | 0.7556 | -2.2pp |
| + all 21 features C=0.03 | 0.7704 | -0.7pp |
| + all 21 features C=0.05 | 0.7630 | -1.5pp |
| + all 21 features C=0.07 | 0.7593 | -1.9pp |

→ Adding olmo+judges HURTS at our scale (noise > signal).

## 🚀 Pseudo-labeling results (BREAKTHROUGH — full sweep)

### Single-seed sweep:
| Frac | Round 0 | Round 1 | Round 2 |
|---|---|---|---|
| 0.05 | 0.7778 | 0.7630 (worse, too few) | - |
| 0.10 | 0.7778 | 0.7926 | 0.7815 |
| 0.15 | 0.7778 | 0.7963 | - |
| 0.20 | 0.7778 | 0.8111 | 0.8111 |
| 0.25 | 0.7778 | 0.8185 | - |
| **0.30** | 0.7778 | **0.8444** | 0.8259 |
| 0.35 | 0.7778 | 0.8778 | - |
| 0.40 | 0.7778 | 0.9074 | - |
| **0.50** | 0.7778 | **0.9333** ⭐ | - |

### 5-seed pseudo (multi-seed for stability):
| Frac | Mean OOF | Range |
|---|---|---|
| 0.30 | 0.867 | 0.844-0.885 |
| 0.40 | 0.911 | 0.904-0.915 |
| 0.50 | **0.936** | 0.930-0.944 |

### Submission queue (Q6 → Q7 super, fires 22:22-22:38 UTC):
1. `task3_super_pseudo_f050.csv` — SUPER 24 features + pseudo f=0.50 — OOF **0.9630** ⭐⭐
2. `task3_super_pseudo_f040.csv` — SUPER + pseudo f=0.40 — OOF 0.9519
3. `task3_super_pseudo_f030.csv` — SUPER + pseudo f=0.30 — OOF 0.9370
4. `task3_super.csv` — SUPER baseline no pseudo — OOF 0.8037

### 🆕 SUPER FEATURES (24 features incl multan1's olmo_13b + judges) — added 22:17 UTC

| Variant | OOF | Δ from 18-feat |
|---|---|---|
| SUPER baseline (no pseudo) C=0.05 | **0.8037** | +2.6pp |
| SUPER + pseudo f=0.30 | 0.9370 | +5.2pp |
| SUPER + pseudo f=0.40 | 0.9519 | +4.5pp |
| SUPER + pseudo f=0.50 | **0.9630** ⭐⭐ | +3.0pp (NEW HIGH) |

Pseudo-label = take top-X% confident "positive" + bottom-X% confident "negative" from test set,
add as training labels. Works because: small labeled (540) + large unlabeled (2250) — test set
implicit labels capture distribution shift.

**Interpretation**: OOF→LB gap likely from CLASSIFIER OVERFITTING to small labeled set, not feature
limitation. Pseudo-labeling acts as regularization by including test distribution.

Risk: confirmation bias (model amplifies its own mistakes). But round 1 > round 2 suggests it's
self-correcting at f=0.30.

---

## OOF TPR@1%FPR — Full sweep (logreg variants on hybrid_v3 features)

| Variant | OOF | CI(5/95) | Notes |
|---|---|---|---|
| **C=0.05** | **0.7778** | [0.685, 0.828] | ⭐ BEST in sweep_v1 |
| C=0.01 | 0.7630 | [0.661, 0.812] | hybrid_v3 — already submitted |
| C=0.01 + roberta_pca=64 | 0.7407 | [0.594, 0.792] | More PCA dims hurt |
| C=0.005 | 0.7296 | [0.625, 0.783] | Too much reg |
| ensemble (logreg+lgbm) | 0.7519 | [0.686, 0.807] | Equal to logreg alone |
| LightGBM | 0.6926 | [0.646, 0.757] | Worse than logreg (likely overfits 540 samples) |
| C=0.001 | 0.5667 | [0.398, 0.664] | Way too much reg |

**Sweep_v2 (in flight)**: C=0.03/0.07/0.1/0.2/0.5/1.0 — to find optimum

**Insight**: OOF→LB ratio is ~1:3 (OOF 0.75 → LB 0.26). This suggests overfitting to TRAIN distribution. Increasing C (less reg) is best so far.

---

## Feature inventory (cached at $SCRATCH/task3/cache/)

| Feature | Shape | Owner | Description |
|---|---|---|---|
| features_a.pkl | (2790, 17) | multan1 | GPT-2 PPL stats, GLTR, ngram, burstiness, gzip, TTR |
| features_a_strong.pkl | (2790, 17) | multan1 | Pythia-2.8b PPL stats |
| features_bc.pkl | (2790, 7) | multan1 | **LEAKAGE** — green list fits on labeled, skip |
| features_better_liu.pkl | (2790, 15) | multan1 | Extended semantic features (Liu/SIR proxy) |
| features_bigram.pkl | (2790, 4) | multan1 | KGW bigram greenlist |
| features_bino.pkl | (2790, 5) | multan1 | GPT-2 / GPT-Neo binoculars |
| features_bino_strong.pkl | (2790, 5) | multan1 | Pythia-1.4b/2.8b binoculars |
| features_bino_xl.pkl | (2790, 5) | multan1 | Pythia-2.8b/6.9b binoculars |
| features_d.pkl | (2790, 4) | multan1 | sentence-transformers cosine |
| features_fdgpt.pkl | (2790, 10) | multan1 | Fast-DetectGPT analytical curvature |
| features_kgw.pkl | (2790, 12) | multan1 | KGW direct (multi-tokenizer) |
| features_kgw_llama.pkl | (2790, 12) | multan1 | KGW with LLaMA-2/3 + Mistral tokenizers |
| features_kgw_v2.pkl | (2790, 16) | multan1 | KGW v2 (extra hash_keys + h=2 multigram) |
| features_lm_judge.pkl | (2790, 18) | multan1 | LM-as-judge (OLMo) zero-shot |
| features_multi_lm.pkl | (2790, 5) | multan1 | OPT-1.3b PPL |
| features_multi_lm_v2.pkl | (2790, 20) | multan1 | Phi-2/Qwen2/Llama-chat/Mistral PPL |
| features_roberta.pkl | (2790, 770) | multan1 | RoBERTa embedding (768) + 2 stats; PCA to 32 |
| features_stylometric.pkl | (2790, 22) | multan1 | TTR, burstiness, char-level stats |
| **features_unigram_direct.pkl** | **(2790, 42)** | **murdzek2** | **Direct Unigram detection — incl key=9999 sha256str (gridsearch winner)** |
| features_kgw_selfhash.pkl | NOT EXISTING | murdzek2 | Multi-tokenizer KGW (jobs killed — too slow @ 7s/it) |

**TOTAL hybrid_v3 features**: 263 cols (after RoBERTa PCA-32)

---

## NOT submitted (potentially useful)

CSVs in /tmp/ or cluster `task3/` not yet POSTed:
- `task3_full.csv` (multan1 baseline, mean 0.433)
- `task3_multi_lm_v2.csv` (mean 0.444)
- `task3_xl_bino.csv` (mean 0.434)
- `task3_hyb_lr_C001.csv` (OOF 0.5667 — bad, skip)
- `task3_hyb_lr_C05.csv` ⭐ (OOF 0.7778 — submit next)
- `task3_hyb_lgbm.csv` (OOF 0.6926 — skip)
- `task3_hyb_ensemble.csv` (OOF 0.7519 — fallback)
- `task3_hyb_lr_pca64.csv` (OOF 0.7407 — skip)

After sweep_v2 (in flight): C=0.03/0.07/0.1/0.2/0.5/1.0 variants

---

## Architecture

### Branch model
- **task3-murdzek2** (mine) — base for all sbatch jobs
- **task3** (multan1's) — features/ source
- Don't push to multan1's branch; can read multan1's repo on cluster

### Code paths
- `code/attacks/task3/hybrid_v3.py` — STANDALONE script (loads cached pkls + classifier)
- `code/attacks/task3/main_kitchen_v2.sh` — runs multan1's main.py with all flags
- `code/attacks/task3/main_hybrid_v3.sh` — runs hybrid_v3.py (logreg C=0.01, baseline)
- `code/attacks/task3/main_hybrid_sweep.sh` — sweep v1 (6 variants)
- `code/attacks/task3/main_sweep_v2.sh` — sweep v2 (6 more C values)
- `code/attacks/task3/features/unigram_direct.py` — my Unigram (key=9999 winner)
- `code/attacks/task3/features/kgw_selfhash.py` — my KGW selfhash (Mistral tokenizer support — TOO SLOW)
- `code/attacks/task3/features/sir_direct.py` — SIR (Liu) detector — needs BERT+MLP download
- `scripts/submit_and_verify.sh` — scrape→submit→scrape pattern

### Cluster paths
- `$SCRATCH=/p/scratch/training2615/kempinski1/Czumpers`
- Repo: `$SCRATCH/repo-${USER}/`
- Cache: `$SCRATCH/task3/cache/`
- Outputs: `$SCRATCH/repo-${USER}/output/`
- Submissions: `$SCRATCH/task3/submission_*.csv`

---

## Lessons / pitfalls

1. **5-min cooldown on success / 2-min on validation failure** — if submit returns HTTP 429 with "Wait 278 sec" but server STILL processed (saw kitchen_v2 reach LB 0.259 even though we got 429 on retry storm)
2. **Always scrape leaderboard BEFORE + AFTER submit** — submit.py response is unreliable. Use `submit_and_verify.sh`.
3. **branch_bc has DATA LEAKAGE** — fits green list on labeled set, OOF inflates 6×. Use `--skip-branch-bc` always.
4. **kgw_selfhash extremely slow** — 7.5s/it × 2790 = 5.8h, won't finish in 3h timeout. Multi-tokenizer + multiple configs amplifies. Don't run it without trimming configs.
5. **OOF→LB ratio ~1:3** — OOF 0.75 → LB 0.26. Suggests overfitting; favor higher C.
6. **C=0.001 (max regularization) → OOF 0.57** — too restrictive; signal is in fine-grained features.
7. **LightGBM worse than LogReg on 540 samples** — 263 features × non-linear → overfits. LogReg's linearity helps.
8. **PCA roberta_pca=64 worse than 32** — added noise from low-variance components.
9. **Multan1's main.py supports MORE flags than mine** — use his code via cd `repo-multan1/...` (read-only, ACL allows).
10. **DON'T read all big PDFs/docs** — token-aware, max 2 papers/turn (per CLAUDE.md).

---

## Next steps queue

1. **Submit `task3_hyb_C05.csv`** at 21:32:40 UTC (OOF 0.7778, monitor armed)
2. **Pull sweep_v2 results** when 14740296 completes (~5 min)
3. Submit BEST C value from sweep_v2 next
4. **Possible: stacking/calibration** if no further C wins
5. **Risky but big-payoff: add olmo_7b/judge_phi2/judge_mistral** features if time (each ~15 min compute)
6. **Retire**: SIR (needs BERT download + slow), kgw_selfhash (too slow)

---

## File: `/tmp/task3_tracker.log` (live progress)

```
$(cat /tmp/task3_tracker.log 2>/dev/null || echo 'empty')
```
