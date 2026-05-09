#!/bin/bash
#SBATCH --job-name=task3-iter
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_iter.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_iter.err

# Iterative pseudo-labeling: 3 rounds at f=0.30 (per Perplexity research)
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

SUPER="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct olmo_7b olmo_13b judge_phi2 judge_mistral judge_chat judge_olmo7b judge_olmo13b"

# 3-round iterative on baseline 18 features
echo "===== iterative pseudo 3 rounds f=0.30 baseline ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_iter_baseline_3r.csv" \
    --top-frac 0.30 --n-rounds 3 --C 0.05

# 3-round iterative on SUPER features
echo "===== iterative pseudo 3 rounds f=0.30 SUPER ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_iter_super_3r.csv" \
    --features $SUPER --top-frac 0.30 --n-rounds 3 --C 0.05

# 5 rounds iterative (more aggressive)
echo "===== iterative pseudo 5 rounds f=0.30 SUPER ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_iter_super_5r.csv" \
    --features $SUPER --top-frac 0.30 --n-rounds 5 --C 0.05

echo "Done."
