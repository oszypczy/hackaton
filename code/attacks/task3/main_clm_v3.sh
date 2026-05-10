#!/bin/bash
#SBATCH --job-name=task3-clm-v3
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Variants of cross_lm_best with multi_lm_v2 features (Phi-2/Qwen2/Llama-chat/Mistral-chat).
# 4 chat-tuned LMs across architectures = potentially catches different watermark schemes.

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

BASE="--phase 2 --use-strong-bino --use-xl-bino --use-fdgpt --use-multi-lm --use-olmo7b --use-cross-lm --cross-lm-mode v1 --classifier logreg --data-dir $SCRATCH/llm-watermark-detection --cache-dir $TASK_CACHE --n-rows 2250"

echo "=== V1: clm + multi_lm_v2 (4 chat LMs added)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-multi-lm-v2 --out "$TASK_OUT/submission_clm_mlmv2.csv" 2>&1 | tail -10

echo "=== V2: clm + multi_lm_v2 + cross_lm v2 (full 56 derived)"
python code/attacks/task3/main.py --phase 2 --use-strong-bino --use-xl-bino --use-fdgpt --use-multi-lm --use-multi-lm-v2 --use-olmo7b --use-cross-lm --cross-lm-mode v2 --classifier logreg --logreg-C 0.005 --data-dir $SCRATCH/llm-watermark-detection --cache-dir $TASK_CACHE --n-rows 2250 --out "$TASK_OUT/submission_clm_mlmv2_xlmv2.csv" 2>&1 | tail -10

echo "=== V3: clm + judge_chat + judge_olmo7b (double judge)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-judge-chat --use-judge-olmo7b --out "$TASK_OUT/submission_clm_double_judge.csv" 2>&1 | tail -10

echo "=== V4: clm + judge_chat + better_liu + olmo13b (orthogonal stack)"
python code/attacks/task3/main.py $BASE --logreg-C 0.005 --use-judge-chat --use-better-liu --use-olmo13b --out "$TASK_OUT/submission_clm_ortho_stack.csv" 2>&1 | tail -10

echo "=== V5: clm + multi_lm_v2 + judge_chat (mixed)"
python code/attacks/task3/main.py $BASE --logreg-C 0.005 --use-multi-lm-v2 --use-judge-chat --out "$TASK_OUT/submission_clm_mlmv2_judge.csv" 2>&1 | tail -10

echo "=== V6: clm with select-k-best 30"
python code/attacks/task3/main.py $BASE --logreg-C 0.05 --select-k-best 30 --out "$TASK_OUT/submission_clm_selk30.csv" 2>&1 | tail -10

echo "=== V7: clm with select-k-best 15 + heavy reg"
python code/attacks/task3/main.py $BASE --logreg-C 0.005 --select-k-best 15 --out "$TASK_OUT/submission_clm_selk15.csv" 2>&1 | tail -10

echo "Done."
ls -la "$TASK_OUT"/submission_clm_mlmv2*.csv "$TASK_OUT"/submission_clm_ortho*.csv "$TASK_OUT"/submission_clm_double*.csv "$TASK_OUT"/submission_clm_selk*.csv
