#!/bin/bash
#SBATCH --job-name=task3-mistral-kgw
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_mistral_kgw.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_mistral_kgw.err

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

# Bust stale caches so new configs (key=9999 unigram, Mistral KGW) are recomputed
rm -f "$TASK_CACHE/features_unigram_direct.pkl"
rm -f "$TASK_CACHE/features_kgw_selfhash.pkl"

# All leak-free features with updated configs:
# - Unigram: key=9999 sha256_str frac=0.25 (gridsearch best, sep=0.435)
# - KGW selfhash: GPT-2 + Mistral-7B tokenizers
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
    --out "$TASK_OUT/submission_mistral_kgw.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_mistral_kgw.csv"
