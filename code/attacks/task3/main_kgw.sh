#!/bin/bash
#SBATCH --job-name=task3-kgw
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# PATH C: Direct KGW reference detection
# Replicates exact Kirchenbauer detection algorithm with default settings
# (hash_key=15485863, gamma=0.25). Tries multiple tokenizers (gpt2, opt-1.3b, pythia-1.4b).
# IF organizers used reference defaults --> game-changer.
# CPU-bound (torch.randperm), should run in ~10-15 min.

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_CACHE" "$TASK_OUT/output"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

# Phase 2 + bc + KGW reference detection (no bigram - it overfits)
python code/attacks/task3/main.py \
    --phase 2 \
    --use-kgw \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_kgw.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_kgw.csv"
