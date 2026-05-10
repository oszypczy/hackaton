#!/bin/bash
#SBATCH --job-name=task3-cross-lm-v2
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Cross-LM v2: amplify 0.284 breakthrough.
# 25+ pairwise lp diffs across all cached LMs (gpt2, gpt2-med, pythia-1.4b/2.8b/6.9b,
# olmo1b, olmo7b, olmo13b) + selected PPL ratios + quadratic interactions.
# No new compute, all from cached features.

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

python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-multi-lm \
    --use-olmo7b \
    --use-olmo13b \
    --use-cross-lm \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_cross_lm_v2.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_cross_lm_v2.csv"
