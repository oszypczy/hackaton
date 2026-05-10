#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/mle_n7000_%j.out

set -euo pipefail

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
DUCI=/p/scratch/training2615/kempinski1/Czumpers/DUCI

OUT_CSV=${OUT_CSV:-$REPO/submissions/task1_duci_mle_n7000_per_arch.csv}
DEGREE=${DEGREE:-0}
USE_SIGNAL=${USE_SIGNAL:-}
EXTRA=${EXTRA:-}

cd "$REPO"
mkdir -p "$REPO/submissions" "$DUCI/output"

EXTRA_FLAGS=""
if [[ -n "$USE_SIGNAL" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --use-signal $USE_SIGNAL"
fi
if [[ "$DEGREE" -gt 0 ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --degree $DEGREE"
fi

$P4VENV -m code.attacks.task1_duci.mle \
    --synth-dir "$DUCI/synth_targets_n7000_100ep_r18" \
    --synth-dir-r50 "$DUCI/synth_targets_n7000_100ep_r50" \
    --synth-dir-r152 "$DUCI/synth_targets_n7000_100ep_r152" \
    --out "$OUT_CSV" \
    $EXTRA_FLAGS $EXTRA

echo "[done] wrote $OUT_CSV"
