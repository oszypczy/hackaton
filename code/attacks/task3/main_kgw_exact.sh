#!/bin/bash
#SBATCH --job-name=task3-kgwx
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
# allow online for tokenizer download (Llama tokenizer ~4MB)
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=1
cd "$REPO"

python code/attacks/task3/extract_and_train.py \
    --feature kgw_exact \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clm_kgwx.csv" \
    --no-cache \
    --C 0.01

echo "Done."
ls -la "$TASK_OUT/submission_clm_kgwx.csv"
