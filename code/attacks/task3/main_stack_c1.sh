#!/bin/bash
#SBATCH --job-name=task3-c1-embed
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# C1: OLMo-7B-Instruct mean-pooled last-hidden-state embeddings -> PCA(128) + LogReg
# Hypothesis: 4096-dim semantic representation captures watermark structure beyond PPL scalars.

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

python code/attacks/task3/stack_c1.py \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out-dir "$TASK_OUT" \
    --out-prefix "submission_c1" \
    --pca-dims 128 \
    --batch-size 8 \
    --max-len 512

echo "Done."
ls -la "$TASK_OUT"/submission_c1_*.csv
