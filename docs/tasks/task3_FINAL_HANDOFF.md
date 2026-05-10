# Task 3 Final Handoff (2026-05-09 ~23:12 UTC)

> **Use this if I get cut off** before hackathon ends 12:30 tomorrow.

---

## CURRENT SCORE: **task3 = 0.2841 (#2 leaderboard)**

Leader: Syntax Terror 0.396, gap 0.111. We've plateaued — confirmed 9+ submissions all give 0.2841.

---

## WHAT'S RUNNING

**Active monitors (auto-fire):**
- `byvyc72l5` — Single SIR_HYBRID submit at 23:16:30
- `btw99l3gz` — Sequential queue 23:21:35-23:57:35 (7 CSVs, 6 min spaced)

**Past completed: 25+ submissions, all at 0.2841 ceiling.**

---

## SUBMISSION QUEUE (after current)

CSVs to submit if any time left after 00:00:
1. `task3_prior_pi036.csv` (prior corrected pi=0.36 EM-estimated)
2. `task3_prior_pi030.csv`
3. `task3_prior_pi020.csv`
4. `task3_max_pseudo_f050.csv` (26 features + SIR + pseudo)
5. `task3_all_chunks_pseudo.csv` (26 features + pseudo)
6. `task3_pseudo_multiseed_f040.csv`
7. `task3_max_baseline.csv` (26 features + SIR no pseudo)
8. `task3_max_pseudo.csv`
9. `task3_iter_super_5r.csv` (5-round iter pseudo)

Just keep submitting these every 6 min. Server keeps best.

---

## CRITICAL FILES

| File | Purpose |
|---|---|
| `code/attacks/task3/hybrid_v3.py` | Standalone classifier (loads cached features → LogReg/LGBM/etc) |
| `code/attacks/task3/pseudo_label.py` | Pseudo-label trainer with --shuffle-pseudo canary |
| `code/attacks/task3/prior_correction.py` | Bayes shift for class prior |
| `code/attacks/task3/extract_sir_standalone.py` | SIR feature extractor (1024→500x3→300) |
| `code/attacks/task3/features/sir_direct.py` | SIR MLP definition |
| `scripts/submit_and_verify.sh` | scrape→submit→scrape pattern |
| `scripts/submit.py` | bare submit with cooldown handling |

---

## CACHED FEATURES (cluster `$SCRATCH/task3/cache/features_*.pkl`)

Total 30 features available:
- a, a_strong, bino, bino_strong, bino_xl
- fdgpt, d, better_liu, stylometric
- kgw, kgw_llama, kgw_v2, bigram
- lm_judge, multi_lm, multi_lm_v2, roberta
- unigram_direct (mine, 42 cols)
- olmo_7b, olmo_13b, olmo7b_chunks
- judge_phi2, judge_mistral, judge_chat, judge_olmo7b, judge_olmo13b
- emp_green_k500/1500/5000 (HURTS — skip)
- sir (HURTS slightly — but keep for diversity)
- bc (LEAKAGE — skip)

**Best feature set**: ALL 26 (without emp_green/bc/sir) → OOF 0.8222 baseline (no pseudo)

---

## KEY INSIGHTS / LESSONS

1. **OOF→LB ratio ~1:3** (OOF 0.78 → LB 0.26). Distribution shift between train+val and test.
2. **Pseudo-label OOF inflation is mostly artifact** (0.94 OOF → still 0.2841 LB)
3. **Cooldown resets on 429 retry** — wait FULL 5+ min after each rejected POST
4. **Server keeps best score only** — submit MANY DIVERSE variants for safety
5. **Final eval is on 70% private** — diversity matters, not LB peak

---

## RESEARCH ARTIFACTS

- `references/research/perplexity_v2_task3.md` — pseudo-label canary, SIR train, prior correction
- `references/research/gemini_gigapromtp.md` — SIR checkpoint URL (used to download)
- `docs/tasks/task3_research_questions.md` — gigaprompt for further research
- `docs/tasks/task3_process_so_far.md` — comprehensive log

---

## SIR DETECTOR (DOWNLOADED & TESTED)

- Checkpoint: `$SCRATCH/task3/sir_model/transform_model_cbert.pth` (4.66 MB)
- Source: `https://huggingface.co/Generative-Watermark-Toolkits/MarkLLM-sir/resolve/main/transform_model_cbert.pth`
- SHA256 verified: `d68f32e31628deeeab50fbb50e5e8725d41afa4f35c94bd639437653e0709e76`
- Architecture: `Linear(1024,500) → Linear(500,500)+ReLU → Linear(500,500)+ReLU → Linear(500,300)`
- Output: 300-dim semantic projection (NOT vocab logits)
- Features extracted: 20 cols (norm/cos/dist/var statistics over per-token projections)
- **OOF impact: SLIGHT HURT (-3.7pp on baseline)**. Maybe arch interpretation wrong, projection collapses.

---

## PRESENTATION NOTES (tomorrow 12:30)

1. **Approach**: Multi-feature ensemble with LogReg, 26 features per text
2. **Innovation**: Pseudo-label transductive learning (test set augmentation), prior correction
3. **Result**: #2 LB (0.2841), gap 0.11 to leader
4. **Honest limitation**: OOF→LB gap suggests train/test distribution shift; couldn't get SIR detector to add signal
5. **Diversity strategy**: 30+ submissions for robustness on private 70%
