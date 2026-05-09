#!/usr/bin/env bash
#SBATCH --job-name=t2-prompt
#SBATCH --partition=dc-gpu
#SBATCH --account=training2615
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/log_%j.txt
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt/output/log_%j.txt

# Usage:
#   sbatch main.sh eval                    # eval on validation_pii (840 GT)
#   sbatch main.sh predict                 # full task/ -> submission_v0.csv
#   sbatch main.sh eval 100                # smoke: 100 samples
set -euo pipefail

MODE="${1:-eval}"
LIMIT_ARG=""
if [[ -n "${2:-}" ]]; then
    LIMIT_ARG="--limit ${2}"
fi

DATA_DIR="/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task"
CODEBASE_DIR="/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main"
# Hardcoded: sbatch copies the script to /var/spool/.../jobs/<id>, so $0 is unreliable.
ATTACK_DIR="/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/code/attacks/task2/prompt"

# Load CUDA (deepspeed needs CUDA_HOME at import time)
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"

# HF cache (mirror hackathon_setup.sh — sbatch may not source .bashrc)
export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Unbuffered Python stdout — sbatch logs to file, not TTY, so default block
# buffering hides progress prints until job ends. Force flush per-line.
export PYTHONUNBUFFERED=1

# Codebase hardcodes `cache_dir = ~/.cache/huggingface/hub` in
# src/lmms/models/__init__.py:35 (ignores HF_HOME). Symlink to shared scratch
# cache so the hardcoded path resolves to our pre-downloaded models.
# $HOME is shared between login + compute on JURECA → idempotent.
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HUGGINGFACE_HUB_CACHE" "$HOME/.cache/huggingface/hub"

echo "[main.sh] CUDA_HOME=$CUDA_HOME  HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

# venv with pre-built deps (torch 2.11+CUDA 13, py3.12)
VENV="$DATA_DIR/.venv"
source "$VENV/bin/activate"

# Unzip codebase to scratch (idempotent). Use python's zipfile — `unzip` not on PATH.
if [[ ! -d "$CODEBASE_DIR" ]]; then
    echo "[main.sh] unzipping codebase..."
    mkdir -p "$(dirname "$CODEBASE_DIR")"
    python -m zipfile -e "$DATA_DIR/task2_standalone_codebase.zip" "$(dirname "$CODEBASE_DIR")"
fi

# Codebase needs config/vision.yaml at CWD (see unified_arch.py:14)
cd "$CODEBASE_DIR"

mkdir -p "$ATTACK_DIR/output"

TS=$(date +%Y%m%d_%H%M%S)
case "$MODE" in
    eval)
        OUT_LOG="$ATTACK_DIR/output/eval_v0_${TS}.json"
        python "$ATTACK_DIR/main.py" \
            --mode eval \
            --codebase_dir "$CODEBASE_DIR" \
            --data_dir "$DATA_DIR" \
            --output_log "$OUT_LOG" \
            $LIMIT_ARG
        ;;
    predict)
        OUT_CSV="$ATTACK_DIR/output/submission_v0_${TS}.csv"
        OUT_LOG="$ATTACK_DIR/output/predict_v0_${TS}.json"
        python "$ATTACK_DIR/main.py" \
            --mode predict \
            --codebase_dir "$CODEBASE_DIR" \
            --data_dir "$DATA_DIR" \
            --output_csv "$OUT_CSV" \
            --output_log "$OUT_LOG" \
            $LIMIT_ARG
        ;;
    *)
        echo "Unknown mode: $MODE (use 'eval' or 'predict')"
        exit 2
        ;;
esac
