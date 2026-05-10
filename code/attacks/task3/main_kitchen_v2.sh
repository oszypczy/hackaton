#!/bin/bash
#SBATCH --job-name=task3-kitchen
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_kitchen.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_kitchen.err

# Kitchen-sink: multan1's main.py + ALL cached features (no recompute)
# Expected runtime: <5 min (just loading pkls + LogReg OOF + predict)

set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
MULTAN_REPO=$SCRATCH/repo-multan1

mkdir -p "$TASK_OUT/output"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run from multan1's repo (his main.py supports more --use-* flags)
cd "$MULTAN_REPO"

python code/attacks/task3/main.py \
    --phase 2 \
    --skip-branch-bc \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-strong-a \
    --use-multi-lm \
    --use-multi-lm-v2 \
    --use-lm-judge \
    --use-stylometric \
    --use-better-liu \
    --use-roberta \
    --use-kgw \
    --use-kgw-llama \
    --use-kgw-v2 \
    --use-bigram \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_kitchen_v2.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_kitchen_v2.csv"
