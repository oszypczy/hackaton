# Task 3 (Watermark Detection) — Session Log

> **PURPOSE:** Anyone should be able to resume work from any point.
> Updated continuously during 24h hackathon. Latest at top.

---

## TL;DR — Current state (2026-05-09 ~21:30 UTC)

- **Leaderboard score: task3 = 0.259053 (#2)** — leader Syntax Terror 0.387, gap 0.128
- **Best OOF candidates queued**: C=0.05 (OOF 0.7778), C=0.01 ensemble (OOF 0.7519)
- **Submitted so far** — kgw_llama (no improvement), kitchen_v2 (→ 0.259), hybrid_v3 (no improvement at 0.259)
- **Next pending submit**: `submission_hyb_lr_C05.csv` after cooldown ends ~21:32:40 UTC
- **Active jobs on Jülich**: sweep_v2 (14740296) — 6 more C values for LogReg
- **Killed jobs**: 14739848 selfhash, 14739925 mistral_kgw, 14739849 leak_free (all stuck on slow kgw_selfhash, won't finish in 3h timeout)

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
| pending | ? | task3_hyb_C05.csv | 0.7778 | Will submit at 21:32:40 UTC |

**Leader (Syntax Terror): 0.387** — gap to bridge: 0.128

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
