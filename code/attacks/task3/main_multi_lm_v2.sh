#!/bin/bash
#SBATCH --job-name=task3-mlm-v2
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Multi-LM v2: extended INSTRUCT-tuned LMs PPL
# Phi-2 (Microsoft, 2.7B, public)
# Qwen2-0.5B-Instruct (Alibaba, 0.5B, public)
# Llama-2-7b-chat-hf (Meta, 7B, gated -> HF_TOKEN required)
# Mistral-7B-Instruct-v0.1 (gated, 7B)
#
# WYMAGA: scripts/download_llama_tokenizers.sh wcześniej z HF_TOKEN

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

# Multi-lm v2 + multi_lm v1 (OLMo) + best baseline
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-multi-lm \
    --use-multi-lm-v2 \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_multi_lm_v2.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_multi_lm_v2.csv"
