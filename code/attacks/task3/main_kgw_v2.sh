#!/bin/bash
#SBATCH --job-name=task3-kgw-v2
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# KGW v2: extra hash_keys [0,1,42,100,12345,999,7] + h=2 multigram (mul/add)
# All gpt2 tokenizer. ~25-30 min torch.randperm extraction.
# Plus combined with strong-bino (already cached) and bc.

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

# Phase 2 + bc + strong-bino + KGW v2 (extra hash_keys + h=2)
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-kgw-v2 \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_kgw_v2.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_kgw_v2.csv"
