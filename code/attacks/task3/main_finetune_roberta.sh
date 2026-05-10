#!/bin/bash
#SBATCH --job-name=task3-roberta-ft
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# RoBERTa-base end-to-end fine-tune. Different signal type vs cross-LM features.

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_OUT=$SCRATCH/task3
mkdir -p "$TASK_OUT/output"
source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

python code/attacks/task3/finetune_roberta.py \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --model "roberta-base" \
    --max-len 256 \
    --batch-size 16 \
    --lr 2e-5 \
    --epochs 4 \
    --n-splits 5 \
    --out-dir "$TASK_OUT" \
    --out-prefix "submission_roberta_ft"

echo "Done."
ls -la "$TASK_OUT"/submission_roberta_ft_*.csv
