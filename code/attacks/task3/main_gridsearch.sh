#!/bin/bash
#SBATCH --job-name=task3-gridsearch
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j_gridsearch.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j_gridsearch.err

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache

mkdir -p "$TASK_CACHE" "$SCRATCH/task3/output"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

echo "=== Unigram grid search (CPU only) ==="
python code/attacks/task3/grid_search.py \
    --data-dir "$SCRATCH/llm-watermark-detection/Dataset.zip"

echo "Done."
