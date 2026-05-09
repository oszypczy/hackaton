#!/bin/bash
#SBATCH --job-name=task3-combined
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# Kitchen sink: phase2 + bc + strong-bino + XL-bino + KGW v1 + KGW v2
# Wszystkie features razem. LogReg z silną reg sortuje co działa.
# Ekstrakcja wszystkich features (~30-45 min jeśli żaden nie jest cached).
# Z cache: ~5 min (jeśli inne joby już wyciągnęły features).

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

# Wszystkie sygnały razem
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-kgw \
    --use-kgw-v2 \
    --classifier logreg \
    --logreg-C 0.005 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_combined.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_combined.csv"
