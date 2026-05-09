#!/usr/bin/env bash
#SBATCH --job-name=t2-multi
#SBATCH --partition=dc-gpu
#SBATCH --account=training2615
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/log_%j.txt
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/log_%j.txt

# Usage:
#   sbatch multi_eval.sh                                 # all strategies, 50/type, blank
#   sbatch multi_eval.sh 100                             # 100/type
#   sbatch multi_eval.sh 50 baseline,direct_probe        # subset of strategies
#   sbatch multi_eval.sh 50 - original                   # different image_mode
set -euo pipefail

PER_TYPE="${1:-50}"
STRATEGIES="${2:-baseline,direct_probe,role_play_dba,user_id_explicit,system_override,completion_format}"
if [[ "$STRATEGIES" == "-" ]]; then
    STRATEGIES="baseline,direct_probe,role_play_dba,user_id_explicit,system_override,completion_format"
fi
IMAGE_MODE="${3:-blank}"

# Scrubbed-image directory (only used when image_mode=scrubbed). Pre-scrubbed PNGs
# named <user_id>.png live OUTSIDE the dataset folder per "no overwrite" rule.
SCRUBBED_DIR="/p/scratch/training2615/kempinski1/Czumpers/val_pii_scrubbed"
SCRUBBED_ARG=""
if [[ "$IMAGE_MODE" == "scrubbed" ]]; then
    SCRUBBED_ARG="--scrubbed_image_dir $SCRUBBED_DIR"
fi

DATA_DIR="/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task"
CODEBASE_DIR="/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main"
ATTACK_DIR="/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt"

module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"

export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HUGGINGFACE_HUB_CACHE" "$HOME/.cache/huggingface/hub"

VENV="$DATA_DIR/.venv"
source "$VENV/bin/activate"

cd "$CODEBASE_DIR"

mkdir -p "$ATTACK_DIR/output"

TS=$(date +%Y%m%d_%H%M%S)
SAFE_STRAT_TAG=$(echo "$STRATEGIES" | tr ',' '-' | cut -c 1-50)
OUT_LOG="$ATTACK_DIR/output/multi_${IMAGE_MODE}_${PER_TYPE}_${TS}.json"

echo "[multi_eval.sh] per_type=$PER_TYPE  strategies=$STRATEGIES  image_mode=$IMAGE_MODE"
echo "[multi_eval.sh] OUT_LOG=$OUT_LOG"

python "$ATTACK_DIR/multi_eval.py" \
    --codebase_dir "$CODEBASE_DIR" \
    --data_dir "$DATA_DIR" \
    --output_log "$OUT_LOG" \
    --image_mode "$IMAGE_MODE" \
    --per_type "$PER_TYPE" \
    --strategies "$STRATEGIES" \
    $SCRUBBED_ARG
