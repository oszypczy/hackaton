#!/bin/bash
#SBATCH --job-name=task3-fdgpt
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Fast-DetectGPT: analytical curvature score under Pythia-2.8b.
# Per-position z-score: (actual log-prob - expected log-prob) / std
# Watermarked text -> consistently positive z (green tokens "too predictable")
# Orthogonal signal to Binoculars (which uses 2-LM PPL ratio).
# ~5 min ekstrakcji + 1 min OOF.

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

# Best so far (XL bino) + Fast-DetectGPT
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_fdgpt.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_fdgpt.csv"
