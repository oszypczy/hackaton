#!/bin/bash
#SBATCH --job-name=task3-hyb-sweep
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sweep.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sweep.err

# Sweep classifier configs to find best OOF on cached features
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

# Variant A: logreg C=0.005 (less reg)
echo "===== A: logreg C=0.005 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C005.csv" \
    --classifier logreg --logreg-C 0.005

# Variant B: logreg C=0.001 (max reg)
echo "===== B: logreg C=0.001 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C001.csv" \
    --classifier logreg --logreg-C 0.001

# Variant C: logreg C=0.05 (light reg)
echo "===== C: logreg C=0.05 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_C05.csv" \
    --classifier logreg --logreg-C 0.05

# Variant D: lgbm
echo "===== D: lgbm ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lgbm.csv" \
    --classifier lgbm

# Variant E: ensemble logreg + lgbm
echo "===== E: ensemble (logreg + lgbm) ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_ensemble.csv" \
    --classifier ensemble --logreg-C 0.01 --ensemble-weights 0.6,0.4

# Variant F: roberta_pca=64 (more dims kept)
echo "===== F: logreg C=0.01 + roberta_pca=64 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb_lr_pca64.csv" \
    --classifier logreg --logreg-C 0.01 --roberta-pca 64

echo "Done."
