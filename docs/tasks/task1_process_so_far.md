# Task 1 (DUCI) — process so far (murdzek2 session)

> Started 2026-05-10 ~00:15 (post-szypczyn1 handoff)
> Branch: `task1-murdzek2` (forked from `origin/task1` @ 4a616f6)
> Worktree: `/mnt/c/projekty/hackaton-task1` (separate from task3-murdzek2)

## Inherited state

- Best score: **SUB-9 = 0.053333** (snap_10 of MLE 80ep R18 mean_loss_mixed)
- Leaderboard: **#8 of 30** (leader BatchNormies3d **0.0085**, top 7 between 0.034 and 0.053)
- All recipe variants tried; "lucky alignment" of N=2000 80ep synth → predicts target's mean_loss → p ≈ 0.5 range
- Maini DI Phase A scaffolded, partial extraction (3/9 R18 targets, synth_2k_r18 5/5, synth_7k_r18 5/5)
- Tong RMIA matched-regime gives uniform offset Δ=-0.4 from MLE → same ranking, no help

## Done in this session

### Cluster setup
- Created branch `task1-murdzek2` from `origin/task1`, pushed
- Switched cluster `repo-murdzek2` to `task1-murdzek2`
- Modified `maini_extract.sh`: REPO=$USER auto-resolve (committed dc595e8)

### Maini extraction (relaunched with consistent MAX_STEPS=100)
- Cancelled prior partial jobs (different max_steps), cleared stale files
- Launched 4 parallel sbatches (jobs 14740630-633):
  - synth_7k_r50 (full sample, max_steps=100)
  - targets_r50 (full sample)
  - synth_7k_r152 (subsample 1000/2000, time 3h)
  - targets_r152 (subsample, time 2h)

### Submissions (during this session)

| sub_id | Time | CSV | Score | Notes |
|---|---|---|---|---|
| 907 | 22:47Z | task1_duci_flip10_flip22.csv | NO_IMPROVEMENT | LB stayed 0.0533 |
| (queued) | ~22:57Z | task1_duci_flip22.csv | TBD | flip22 alone |

### Key findings

1. **Maini target R18 ranking** = MLE ranking (02 > 01 > 00). Same ordering.
2. **Maini predicts much lower mean** (~0.17 vs MLE 0.52) — same uniform offset issue as Tong.
3. **Multi-signal RidgeCV** doesn't help (sparse synth banks for R50/R152 → overfits).
4. **Cooldown** confirmed: 5 min server-time after successful submission.

## Next steps (queue)

1. ~~Submit flip22 alone~~ (~22:57Z)
2. Wait for Maini extraction completion (~30 min from launch, ETA ~01:00Z)
3. Run `maini_mle.py` to get full Maini predictions (9/9 targets)
4. Compute ensemble: avg(MLE_80ep_R18, Maini per-arch) → snap_10
5. Submit ensemble + scrape

## Strategy notes

- **Don't risk SUB-9 final** — keep it as private 6 hedge regardless
- Multiple shots at improving public 3 are cheap (5 min cooldown)
- Best single-flip candidates: 22 (borderline 0.549), 10 (border at 0.444), 21/11 (closest to 0.5)
- Compound flips not informative (flip10+flip22 already tried, no improvement)

## Cluster paths

- DUCI shared: `/p/scratch/training2615/kempinski1/Czumpers/DUCI/`
- Maini signals: `/p/scratch/training2615/kempinski1/Czumpers/DUCI/maini_signals_v1/{targets,synth_7k_r18,synth_7k_r50,synth_7k_r152}/`
- Best continuous CSV: `submission_mle_80ep_r18_precise.csv`
- My repo: `/p/scratch/training2615/kempinski1/Czumpers/repo-murdzek2/` (on `task1-murdzek2`)
