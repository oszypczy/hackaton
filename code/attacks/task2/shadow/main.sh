#!/usr/bin/env bash
#SBATCH --job-name=t2-shadow
#SBATCH --partition=dc-gpu
#SBATCH --account=training2615
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/p/project1/training2615/murdzek2/Hackathon/output/attack_shadow_%j.out
#SBATCH --error=/p/project1/training2615/murdzek2/Hackathon/output/attack_shadow_%j.out
set -euo pipefail

HACKATHON=/p/project1/training2615/murdzek2/Hackathon
DATA_DIR=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task
CODEBASE=/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main
ATTACK=$HACKATHON/code/attacks/task2/attack_shadow.py

# 1. CUDA — deepspeed checks CUDA_HOME at import time
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"

# 2. HF cache — shared scratch; codebase hardcodes ~/.cache/huggingface/hub
export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3. Symlink so hardcoded ~/.cache/huggingface/hub resolves to shared cache
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HUGGINGFACE_HUB_CACHE" "$HOME/.cache/huggingface/hub"

# 4. Codebase unzip (idempotent; no `unzip` on PATH — use python)
if [[ ! -d "$CODEBASE" ]]; then
    mkdir -p "$(dirname "$CODEBASE")"
    python3 -m zipfile -e "$DATA_DIR/task2_standalone_codebase.zip" "$(dirname "$CODEBASE")"
fi

# 5. venv (personal — has all packages: rapidfuzz, requests, transformers==4.51.3, etc.)
source "$HACKATHON/.venv/bin/activate"

# 6. CWD = codebase root (unified_config.py uses relative config/models/)
cd "$CODEBASE"
export CODEBASE

mkdir -p "$HACKATHON/output"
python "$ATTACK" --mode submit --k 8

echo "done!"
