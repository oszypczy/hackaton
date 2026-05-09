#!/bin/bash
#SBATCH --job-name=task3-watermark
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_CACHE" "$TASK_OUT/output"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

export PATH=/usr/bin:$PATH

cd "$REPO"
/usr/bin/git pull --ff-only

# Phase 2: all branches, GPU (binoculars uses CUDA automatically)
python code/attacks/task3/main.py \
    --phase 2 \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission.csv"
