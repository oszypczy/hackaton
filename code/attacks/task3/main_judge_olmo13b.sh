#!/bin/bash
#SBATCH --job-name=task3-judge-olmo13b
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# OLMo-2-13B-Instruct as JUDGE.
# Largest OLMo we can run (26GB fp16 fits 44GB GPU).
# Combines best paradigm (judge) with biggest model.

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
mkdir -p "$TASK_CACHE" "$TASK_OUT/output"
source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"

if [ ! -d "$HF_HOME/hub/models--allenai--OLMo-2-1124-13B-Instruct" ]; then
    echo "ERROR: OLMo-13B-Instruct nie w cache." >&2
    echo "Najpierw: bash scripts/download_olmo13b.sh (login node)" >&2
    exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

# Stack everything OLMo
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-multi-lm \
    --use-lm-judge \
    --use-olmo7b \
    --use-judge-olmo7b \
    --use-olmo13b \
    --use-judge-olmo13b \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_judge_olmo13b.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_judge_olmo13b.csv"
