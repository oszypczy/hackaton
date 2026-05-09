#!/bin/bash
#SBATCH --job-name=task3-pseudo3
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_pseudo3.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_pseudo3.err

# Pseudo v3: aggressive — multi-round + multi-seed at f=0.50
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

# 5-seed pseudo at f=0.40 and f=0.50, then rank-mean
for FRAC in 0.40 0.50; do
    fstr=${FRAC//./}
    for SEED in 17 23 42 101 2024; do
        echo "===== pseudo f=$FRAC seed=$SEED ====="
        python code/attacks/task3/pseudo_label.py \
            --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
            --out "$TASK_OUT/submission_pseudo_f${fstr}_S${SEED}.csv" \
            --top-frac $FRAC --n-rounds 1 --C 0.05 --seed $SEED 2>&1 | grep "OOF TPR" | tail -1
    done

    # Rank-mean of 5 seeds for this f
    python -c "
import csv, glob
files = sorted(glob.glob('$TASK_OUT/submission_pseudo_f${fstr}_S*.csv'))
print(f'pseudo f=$FRAC multi-seed: {len(files)} CSVs')
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
with open('$TASK_OUT/submission_pseudo_multiseed_f${fstr}.csv','w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['id','score'])
    for i, idx in enumerate(ids): w.writerow([idx, round(avg[i],6)])
print(f'wrote pseudo_multiseed_f${fstr}: {n} rows')
"
done

# Rank-mean of f=0.30 + f=0.40 + f=0.50 multiseeds combined
python -c "
import csv, glob
files = sorted(glob.glob('$TASK_OUT/submission_pseudo_multiseed_f*.csv'))
print(f'rank-mean of {len(files)} multi-seed pseudo CSVs')
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
with open('$TASK_OUT/submission_pseudo_meta_ensemble.csv','w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['id','score'])
    for i, idx in enumerate(ids): w.writerow([idx, round(avg[i],6)])
print(f'wrote pseudo_meta_ensemble: {n} rows')
"

echo "Done."
