#!/bin/bash
#SBATCH --job-name=task3-pseudo
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_pseudo.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_pseudo.err

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

# Try multiple pseudo-label fractions
for FRAC in 0.10 0.20 0.30; do
    NAME="pseudo_f${FRAC//./}"
    echo "===== pseudo top/bot $FRAC ====="
    python code/attacks/task3/pseudo_label.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_${NAME}.csv" \
        --top-frac $FRAC --n-rounds 2 --C 0.05
done

echo "Done."
