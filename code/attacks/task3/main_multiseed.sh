#!/bin/bash
#SBATCH --job-name=task3-multiseed
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_multiseed.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_multiseed.err

# Multi-seed ensemble: same config, different KFold seeds. Average predictions to reduce variance.
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_OUT/output"
source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

FEATURES="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct"

for SEED in 17 23 42 101 2024; do
    echo "===== seed=$SEED ====="
    python code/attacks/task3/hybrid_v3.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_hyb_seed${SEED}.csv" \
        --features $FEATURES --classifier logreg --logreg-C 0.05 --seed $SEED
done

# Average all seed CSVs by rank-mean
python -c "
import csv, glob
files = sorted(glob.glob('$TASK_OUT/submission_hyb_seed*.csv'))
print(f'averaging {len(files)} CSVs')
all_scores = {}
ids = None
for f in files:
    with open(f) as fh:
        next(fh)
        rows = []
        for line in fh:
            p = line.strip().split(',')
            if len(p)==2:
                rows.append((int(p[0]), float(p[1])))
        rows.sort()
        if ids is None: ids = [r[0] for r in rows]
        all_scores[f] = [s for _,s in rows]
n = len(ids)
ranks = {f: sorted(range(n), key=lambda i: all_scores[f][i]) for f in files}
inv = {f: [0.0]*n for f in files}
for f in files:
    for r, idx in enumerate(ranks[f]):
        inv[f][idx] = (r+1)/n
avg = [sum(inv[f][i] for f in files)/len(files) for i in range(n)]
with open('$TASK_OUT/submission_multiseed_rankmean.csv','w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['id','score'])
    for i, idx in enumerate(ids): w.writerow([idx, round(avg[i],6)])
print(f'wrote multiseed_rankmean: {n} rows')
"

echo "Done."
