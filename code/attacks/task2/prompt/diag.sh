#!/usr/bin/env bash
#SBATCH --job-name=t2-diag
#SBATCH --partition=dc-gpu
#SBATCH --account=training2615
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/diag_%j.txt
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/diag_%j.txt
set -euo pipefail

DATA_DIR="/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task"

module load CUDA/13 2>/dev/null || true
export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$DATA_DIR/.venv/bin/activate"

echo "=== HOST ==="
hostname
echo "=== ENV ==="
env | grep -iE 'HF_|HUGGINGFACE|TRANSFORMERS_OFF|CUDA_HOME|HOME=' | sort
echo "=== CACHE LISTING ==="
ls -la "$HUGGINGFACE_HUB_CACHE/"
echo "=== OLMO CACHE ==="
ls -la "$HUGGINGFACE_HUB_CACHE/models--allenai--OLMo-2-0425-1B-Instruct/" 2>&1
ls -la "$HUGGINGFACE_HUB_CACHE/models--allenai--OLMo-2-0425-1B-Instruct/refs/" 2>&1
ls -la "$HUGGINGFACE_HUB_CACHE/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/"*/ 2>&1
echo "=== TEST AutoConfig ==="
python -c "
import os
print('HF_HUB_OFFLINE=', os.environ.get('HF_HUB_OFFLINE'))
print('TRANSFORMERS_OFFLINE=', os.environ.get('TRANSFORMERS_OFFLINE'))
from transformers.utils.hub import is_offline_mode
print('is_offline_mode():', is_offline_mode())
from transformers import AutoConfig
c = AutoConfig.from_pretrained('allenai/OLMo-2-0425-1B-Instruct', local_files_only=False)
print('OK config:', type(c).__name__)
"
