#!/bin/bash
#SBATCH --job-name=task3-roberta
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# RoBERTa-base mean-pooled embedding features (768-dim).
# C=0.0005 (heavy reg) prevents overfit on 540 train samples.

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_CACHE" "$TASK_OUT/output"

source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"

# Download RoBERTa first if not cached. Try online (compute may have proxy), fall back to error.
if [ ! -d "$HF_HOME/hub/models--roberta-base" ]; then
    echo "Downloading roberta-base..."
    export TRANSFORMERS_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
    python -c "from transformers import AutoTokenizer, AutoModel; \
        AutoTokenizer.from_pretrained('roberta-base'); \
        AutoModel.from_pretrained('roberta-base'); \
        print('roberta-base downloaded OK')"
fi

# Now use offline (model is cached)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

# RoBERTa + best baseline (XL bino + FDGPT)
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-roberta \
    --roberta-pca-dim 32 \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_roberta.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_roberta.csv"
