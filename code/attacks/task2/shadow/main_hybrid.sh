#!/usr/bin/env bash
#SBATCH --job-name=t2-shadow-hybrid
#SBATCH --partition=dc-gpu
#SBATCH --account=training2615
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/p/project1/training2615/murdzek2/Hackathon/output/attack_shadow_hybrid_%j.out
#SBATCH --error=/p/project1/training2615/murdzek2/Hackathon/output/attack_shadow_hybrid_%j.out
set -euo pipefail

HACKATHON=/p/project1/training2615/murdzek2/Hackathon
DATA_DIR=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task
CODEBASE=/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main
ATTACK=$HACKATHON/code/attacks/task2/attack_shadow.py

module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HUGGINGFACE_HUB_CACHE" "$HOME/.cache/huggingface/hub"

if [[ ! -d "$CODEBASE" ]]; then
    mkdir -p "$(dirname "$CODEBASE")"
    python3 -m zipfile -e "$DATA_DIR/task2_standalone_codebase.zip" "$(dirname "$CODEBASE")"
fi

source "$HACKATHON/.venv/bin/activate"
cd "$CODEBASE"
export CODEBASE

mkdir -p "$HACKATHON/output"
# A+B hybrid: [REDACTED] prefix trick + greedy decode
python -u "$ATTACK" --mode submit --greedy-only --redacted-prefix

echo "done!"
