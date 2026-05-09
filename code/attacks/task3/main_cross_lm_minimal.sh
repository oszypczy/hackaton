#!/bin/bash
#SBATCH --job-name=task3-clm-minimal
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# cross_lm v1 (6 features, 0.284) on MINIMAL stack — less dilution.
# Hypothesis: 28 + 6 = 34 features focuses LogReg better than 50+6 default.

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
mkdir -p "$TASK_CACHE" "$TASK_OUT/output"
source "$SCRATCH/repo-${USER}/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

# Need bino_xl for olmo7b_vs_pythia69b features. Add only what's needed.
python code/attacks/task3/main.py \
    --phase 2 \
    --skip-branch-d \
    --use-strong-bino \
    --use-xl-bino \
    --use-olmo7b \
    --use-cross-lm --cross-lm-mode v1 \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clm_minimal.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_clm_minimal.csv"
