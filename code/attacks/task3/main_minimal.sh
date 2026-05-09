#!/bin/bash
#SBATCH --job-name=task3-minimal
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Ultra-minimal: branch_a (gpt2 stats) + branch_bc (Zhao detector) + OLMo-7B PPL
# 17 + 4 + 7 = 28 features (vs 60-90 in stacks).
# Hipoteza: LogReg z fewer features daje less dilution -> wyższy leaderboard.

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

python code/attacks/task3/main.py \
    --phase 2 \
    --minimal-stack \
    --use-olmo7b \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_minimal.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_minimal.csv"
