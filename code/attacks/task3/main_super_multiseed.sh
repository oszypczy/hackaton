#!/bin/bash
#SBATCH --job-name=task3-supermsd
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_supermsd.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_supermsd.err

# SUPER features + multi-seed pseudo at f=0.30 (moderate, not extreme)
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

SUPER="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct olmo_7b olmo_13b judge_phi2 judge_mistral judge_chat judge_olmo7b judge_olmo13b"

# 5-seed pseudo at f=0.30 with SUPER features
for SEED in 17 23 42 101 2024; do
    echo "===== SUPER + pseudo f=0.30 seed=$SEED ====="
    python code/attacks/task3/pseudo_label.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_super_pseudo_f030_S${SEED}.csv" \
        --features $SUPER --top-frac 0.30 --n-rounds 1 --C 0.05 --seed $SEED 2>&1 | grep "OOF TPR" | tail -1
done

# Rank-mean of 5 SUPER pseudo runs
python -c "
import csv, glob
files = sorted(glob.glob('$TASK_OUT/submission_super_pseudo_f030_S*.csv'))
print(f'super multi-seed pseudo: {len(files)} CSVs')
all_scores = {}; ids = None
for f in files:
    with open(f) as fh:
        next(fh)
        rows = sorted([(int(p[0]),float(p[1])) for p in (l.strip().split(',') for l in fh) if len(p)==2])
        if ids is None: ids = [r[0] for r in rows]
        all_scores[f] = [s for _,s in rows]
n = len(ids)
ranks = {f: sorted(range(n), key=lambda i: all_scores[f][i]) for f in files}
inv = {f: [0.0]*n for f in files}
for f in files:
    for r, idx in enumerate(ranks[f]):
        inv[f][idx] = (r+1)/n
avg = [sum(inv[f][i] for f in files)/len(files) for i in range(n)]
with open('$TASK_OUT/submission_super_multiseed_f030.csv','w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['id','score'])
    for i, idx in enumerate(ids): w.writerow([idx, round(avg[i],6)])
print(f'wrote super_multiseed_f030: {n} rows')
"

# Also: SUPER baseline with multi-seed (no pseudo) - control
for SEED in 17 23 42 101 2024; do
    echo "===== SUPER baseline (no pseudo) seed=$SEED ====="
    python code/attacks/task3/hybrid_v3.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_super_S${SEED}.csv" \
        --features $SUPER --classifier logreg --logreg-C 0.05 --seed $SEED 2>&1 | grep "OOF TPR" | tail -1
done

# Rank-mean of 5 SUPER baseline
python -c "
import csv, glob
files = sorted(glob.glob('$TASK_OUT/submission_super_S*.csv'))
print(f'super baseline multi-seed: {len(files)} CSVs')
all_scores = {}; ids = None
for f in files:
    with open(f) as fh:
        next(fh)
        rows = sorted([(int(p[0]),float(p[1])) for p in (l.strip().split(',') for l in fh) if len(p)==2])
        if ids is None: ids = [r[0] for r in rows]
        all_scores[f] = [s for _,s in rows]
n = len(ids)
ranks = {f: sorted(range(n), key=lambda i: all_scores[f][i]) for f in files}
inv = {f: [0.0]*n for f in files}
for f in files:
    for r, idx in enumerate(ranks[f]):
        inv[f][idx] = (r+1)/n
avg = [sum(inv[f][i] for f in files)/len(files) for i in range(n)]
with open('$TASK_OUT/submission_super_multiseed_baseline.csv','w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['id','score'])
    for i, idx in enumerate(ids): w.writerow([idx, round(avg[i],6)])
print(f'wrote super_multiseed_baseline: {n} rows')
"

echo "Done."
