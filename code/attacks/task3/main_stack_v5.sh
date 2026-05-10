#!/bin/bash
#SBATCH --job-name=task3-stack-v5
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# v5: full label-free stack — adds judge_*, sir, unigram_direct, kgw_llama,
# olmo7b_chunks, stylometric. ~250+ raw features → SelectKBest variants + meta.

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

python code/attacks/task3/stack_v5.py \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out-dir "$TASK_OUT" \
    --out-prefix "submission_v5"

echo "Done."
ls -la "$TASK_OUT"/submission_v5_*.csv
