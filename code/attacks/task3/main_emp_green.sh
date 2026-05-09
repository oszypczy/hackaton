#!/bin/bash
#SBATCH --job-name=task3-empgreen
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_empgreen.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_empgreen.err

# Empirical greenlist via Fisher exact test on TRAIN only — no leakage
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_OUT/output"
source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

# Build empirical green features at multiple top_k
for K in 500 1500 5000; do
    OUTNAME="emp_green_k${K}"
    echo "===== fitting Fisher green list k=$K ====="
    python code/attacks/task3/empirical_greenlist.py \
        --data-dir "$SCRATCH/llm-watermark-detection" \
        --cache-dir "$TASK_CACHE" \
        --top-k $K \
        --out-name $OUTNAME
done

# Now run hybrid_v3 with empirical green features added
FEATURES="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct emp_green_k1500"

echo "===== hybrid_v3 + emp_green k=1500 C=0.05 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_empgreen.csv" \
    --features $FEATURES --classifier logreg --logreg-C 0.05

echo "===== hybrid_v3 + ALL emp_green C=0.05 ====="
FEATURES_ALL="$FEATURES emp_green_k500 emp_green_k5000"
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_empgreen_all.csv" \
    --features $FEATURES_ALL --classifier logreg --logreg-C 0.05

echo "Done."
