#!/bin/bash
#SBATCH --job-name=task3-dl-llama
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/task3/output/%j.err

set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
source "$SCRATCH/llm-watermark-detection/.venv/bin/activate"
export HF_HUB_DISABLE_TELEMETRY=1
cd "$REPO"

# Download Llama-2-7B (canonical KGW generator) + tokenizer
python code/attacks/task3/download_llama.py \
    --cache-dir "$SCRATCH/.cache" \
    --model NousResearch/Llama-2-7b-hf

echo "Done."
ls -la "$SCRATCH/.cache/hub" | grep -i llama
