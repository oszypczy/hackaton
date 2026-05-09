#!/bin/bash
#SBATCH --job-name=task3-leakfree
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j_leakfree.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j_leakfree.err

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

# All leak-free features: skip fitted green lists, use exact watermark detectors
# - strong binoculars (Pythia-1.4b+2.8b PPL ratio)
# - Fast-DetectGPT (Pythia-2.8b curvature)
# - Unigram direct z-score (exact key=0, fraction=0.5 from Zhao 2024)
# - KGW selfhash h=4 (anchored minhash PRF, Kirchenbauer 2024)
python code/attacks/task3/main.py \
    --phase 2 \
    --skip-branch-bc \
    --use-strong-bino \
    --use-fdgpt \
    --use-unigram-direct \
    --use-kgw-selfhash \
    --classifier logreg \
    --logreg-C 0.05 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_leakfree.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_leakfree.csv"
