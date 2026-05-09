#!/bin/bash
#SBATCH --job-name=task3-strong-bino
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# PATH B: Stronger Binoculars (Pythia-1.4b + Pythia-2.8b)
# Universal LLM-detection features, NIE zalezy od labels (no overfitting risk).
# Re-ekstrakcja features_bino_strong (~15-20 min).

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

# Phase 2 + bc + STRONG binoculars (no bigram - it overfits)
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_strong_bino.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_strong_bino.csv"
