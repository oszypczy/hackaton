#!/bin/bash
#SBATCH --job-name=task3-olmo7b
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

# OLMo-2-7B-Instruct PPL features.
# Hipoteza: 1B-Instruct dał 0.158 leaderboard. 7B = 7x większy = ostrzejszy
# sygnał na tym samym instruct-text distribution. Public, no auth.

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
mkdir -p "$TASK_CACHE" "$TASK_OUT/output"
source "$SCRATCH/repo-${USER}/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"

# Download OLMo-2-7B-Instruct if not cached (public, no token needed)
if [ ! -d "$HF_HOME/hub/models--allenai--OLMo-2-1124-7B-Instruct" ]; then
    echo "Downloading OLMo-2-7B-Instruct..."
    export TRANSFORMERS_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-7B-Instruct')
mod = AutoModelForCausalLM.from_pretrained('allenai/OLMo-2-1124-7B-Instruct', torch_dtype=torch.float16)
print(f'  OK: {sum(p.numel() for p in mod.parameters())/1e9:.1f}B params')
"
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$REPO"

# OLMo-7B + OLMo-1B (multi_lm) + best baseline
python code/attacks/task3/main.py \
    --phase 2 \
    --use-strong-bino \
    --use-xl-bino \
    --use-fdgpt \
    --use-multi-lm \
    --use-olmo7b \
    --classifier logreg \
    --logreg-C 0.01 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_olmo7b.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_olmo7b.csv"
