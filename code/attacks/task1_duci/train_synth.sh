#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/train_synth_%j.out

set -euo pipefail

ARCH=${ARCH:-0}
P_LIST=${P_LIST:-"0.0,0.25,0.5,0.75,1.0"}
EPOCHS=${EPOCHS:-50}
BASE_SEED=${BASE_SEED:-1000}

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
OUT_DIR=${OUT_DIR:-/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets}

mkdir -p "$OUT_DIR" /p/scratch/training2615/kempinski1/Czumpers/DUCI/output

echo "[sbatch] node=$(hostname)  job=$SLURM_JOB_ID  arch=$ARCH p_list=$P_LIST epochs=$EPOCHS"
echo "[sbatch] CUDA visible: $(nvidia-smi -L 2>/dev/null | head -2)"

cd "$REPO"
$P4VENV -m code.attacks.task1_duci.train_synth \
    --arch "$ARCH" --p-list "$P_LIST" \
    --epochs "$EPOCHS" --base-seed "$BASE_SEED" --out-dir "$OUT_DIR"

echo "[sbatch] done"
