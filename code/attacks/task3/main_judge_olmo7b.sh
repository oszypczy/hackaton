#!/bin/bash
#SBATCH --job-name=task3-judge-olmo7b
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# OLMo-7B-Instruct as JUDGE (zero-shot prompting).
# Combines two breakthroughs:
#   1B-PPL=0.158 → 1B-Judge=0.20 (prompting helps)
#   1B-PPL=0.158 → 7B-PPL=0.259 (size helps)
#   ==> 7B-Judge = potentially 0.30+

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

# Stack: olmo7b PPL (current best 0.259) + olmo7b JUDGE (new) + lm_judge OLMo-1B baseline
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-multi-lm \
    --use-lm-judge \
    --use-olmo7b \
    --use-judge-olmo7b \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_judge_olmo7b.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_judge_olmo7b.csv"
