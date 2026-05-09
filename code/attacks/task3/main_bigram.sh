#!/bin/bash
#SBATCH --job-name=task3-bigram
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Wariant z bigram greenlist (KGW-specific) — dodaje sygnał Kirchenbauera
# którego unigram BC nie wykryje. Teoria: leader 0.20 ma KGW-detector.

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_CACHE" "$TASK_OUT/output"

source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

# Phase 2 + branch_bc + bigram greenlist + LogReg
python code/attacks/task3/main.py \
    --phase 2 \
    --use-bigram \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_bigram.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_bigram.csv"
