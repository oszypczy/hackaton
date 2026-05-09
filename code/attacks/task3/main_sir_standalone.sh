#!/bin/bash
#SBATCH --job-name=task3-sir-std
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_std.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_std.err

set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
SIR_DIR=$SCRATCH/task3/sir_model

source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export SIR_CHECKPOINT=$SIR_DIR/transform_model_cbert.pth
cd "$REPO"

python code/attacks/task3/extract_sir_standalone.py \
    "$SCRATCH/llm-watermark-detection" \
    "$TASK_CACHE"

echo "SIR features extracted to $TASK_CACHE/features_sir.pkl"
