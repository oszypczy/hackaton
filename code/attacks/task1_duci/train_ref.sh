#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/train_ref_%j.out

set -euo pipefail

ARCH=${ARCH:-0}        # 0=ResNet18, 1=ResNet50, 2=ResNet152
SEED=${SEED:-0}
P_FRAC=${P_FRAC:-0.5}
EPOCHS=${EPOCHS:-50}

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
OUT_DIR=${OUT_DIR:-/p/scratch/training2615/kempinski1/Czumpers/DUCI/refs}

mkdir -p "$OUT_DIR" /p/scratch/training2615/kempinski1/Czumpers/DUCI/output

echo "[sbatch] node=$(hostname)  job=$SLURM_JOB_ID  arch=$ARCH seed=$SEED p=$P_FRAC epochs=$EPOCHS"
echo "[sbatch] CUDA visible: $(nvidia-smi -L 2>/dev/null | head -2)"

cd "$REPO"
$P4VENV -m code.attacks.task1_duci.train_ref \
    --arch "$ARCH" --seed "$SEED" --p-fraction "$P_FRAC" \
    --epochs "$EPOCHS" --out-dir "$OUT_DIR"

echo "[sbatch] done"
