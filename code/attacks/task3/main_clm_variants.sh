#!/bin/bash
#SBATCH --job-name=task3-clm-variants
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# 7 variants of cross_lm_best (leaderboard 0.284) — runs sequentially.
# All features cached → fast (~30s training each).

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

echo "=== V1: clm_C0.001 (heavier reg)"
python code/attacks/task3/main.py $BASE --logreg-C 0.001 --out "$TASK_OUT/submission_clm_C0001.csv" 2>&1 | tail -10

echo "=== V2: clm_C0.005"
python code/attacks/task3/main.py $BASE --logreg-C 0.005 --out "$TASK_OUT/submission_clm_C0005.csv" 2>&1 | tail -10

echo "=== V3: clm_C0.05 (less reg)"
python code/attacks/task3/main.py $BASE --logreg-C 0.05 --out "$TASK_OUT/submission_clm_C005.csv" 2>&1 | tail -10

echo "=== V4: clm_olmo13b (add OLMo-13B PPL)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-olmo13b --out "$TASK_OUT/submission_clm_olmo13b.csv" 2>&1 | tail -10

echo "=== V5: clm_better_liu (add Liu detector)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-better-liu --out "$TASK_OUT/submission_clm_better_liu.csv" 2>&1 | tail -10

echo "=== V6: clm_judge_chat (add LM judge)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-judge-chat --out "$TASK_OUT/submission_clm_judge_chat.csv" 2>&1 | tail -10

echo "=== V7: clm_kgw_llama (add KGW direct detector)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-kgw-llama --out "$TASK_OUT/submission_clm_kgw.csv" 2>&1 | tail -10

echo "=== V8: clm_judge_olmo7b (add OLMo-7B judge)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --use-judge-olmo7b --out "$TASK_OUT/submission_clm_judge_olmo7b.csv" 2>&1 | tail -10

echo "=== V9: clm_baseline_repro (sanity reproduction)"
python code/attacks/task3/main.py $BASE --logreg-C 0.01 --out "$TASK_OUT/submission_clm_baseline_repro.csv" 2>&1 | tail -10

echo "Done."
ls -la "$TASK_OUT"/submission_clm_*.csv
