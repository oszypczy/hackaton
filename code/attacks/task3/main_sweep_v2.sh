#!/bin/bash
#SBATCH --job-name=task3-sweep2
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sweep2.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sweep2.err

# Sweep v2: extend to higher C (less reg) since C=0.05 was best
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

# C=0.1
echo "===== C=0.1 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C10.csv" \
    --classifier logreg --logreg-C 0.1

# C=0.2
echo "===== C=0.2 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C20.csv" \
    --classifier logreg --logreg-C 0.2

# C=0.5
echo "===== C=0.5 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C50.csv" \
    --classifier logreg --logreg-C 0.5

# C=0.03
echo "===== C=0.03 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C03.csv" \
    --classifier logreg --logreg-C 0.03

# C=0.07
echo "===== C=0.07 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C07.csv" \
    --classifier logreg --logreg-C 0.07

# C=1.0 (default LR)
echo "===== C=1.0 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C100.csv" \
    --classifier logreg --logreg-C 1.0

echo "Done."
