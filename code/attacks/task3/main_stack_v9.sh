#!/bin/bash
#SBATCH --job-name=task3-stack-v9
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# v9: brute-force subset search over cross-LM derived features. Tests K=3,4,5,6,8.
# Hypothesis: cross_lm_best (6 features) was lucky; smaller subset may generalize better.

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

python code/attacks/task3/stack_v9.py \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out-dir "$TASK_OUT" \
    --out-prefix "submission_v9"

echo "Done."
ls -la "$TASK_OUT"/submission_v9_*.csv
