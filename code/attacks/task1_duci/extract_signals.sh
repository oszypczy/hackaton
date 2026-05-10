#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/extract_signals_%j.out

set -euo pipefail

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
SYNTH_BASE=/p/scratch/training2615/kempinski1/Czumpers/DUCI

SYNTH_DIRS=${SYNTH_DIRS:-"$SYNTH_BASE/synth_targets_80ep_r18,$SYNTH_BASE/synth_targets_80ep_r18_extra,$SYNTH_BASE/synth_targets_80ep_r50,$SYNTH_BASE/synth_targets_80ep_r152"}
OUT=${OUT:-"$SYNTH_BASE/signals.npz"}

mkdir -p "$SYNTH_BASE/output"

cd "$REPO"
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID synth_dirs=$SYNTH_DIRS out=$OUT"
echo "[sbatch] CUDA: $(nvidia-smi -L | head -1)"

$P4VENV -m code.attacks.task1_duci.extract_signals \
    --synth-dirs "$SYNTH_DIRS" \
    --out "$OUT"

echo "[sbatch] done"
