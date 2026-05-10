#!/bin/bash
# Launch parallel sbatch jobs for matched-regime ref + synth bank training.
# Regime: N=7000, 100ep, R18 (POP_z verified ~0.27 == organizer targets).
#
# Usage (on cluster):
#   bash code/attacks/task1_duci/launch_matched_regime.sh
set -euo pipefail

REFS_OUT=/p/scratch/training2615/kempinski1/Czumpers/DUCI/refs_n7000_100ep
SYNTH_OUT=/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets_n7000_100ep_r18
mkdir -p "$REFS_OUT" "$SYNTH_OUT"

echo "[launch] refs out: $REFS_OUT"
echo "[launch] synth out: $SYNTH_OUT"

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
cd "$REPO"

# Refs: 8 seeds in parallel (Bernoulli p=0.5)
echo "[launch] submitting 8 ref jobs (R18 N=7000 100ep)"
for SEED in 300 301 302 303 304 305 306 307; do
    JOB_ID=$(ARCH=0 SEED=$SEED P_FRAC=0.5 EPOCHS=100 N_TOTAL=7000 OUT_DIR="$REFS_OUT" \
        sbatch --parsable code/attacks/task1_duci/train_ref.sh)
    echo "  ref seed=$SEED → job $JOB_ID"
done

# Synth: 5 p values, each separate sbatch (parallel)
echo "[launch] submitting 5 synth jobs (R18 N=7000 100ep, one per p)"
P_VALUES=("0.0" "0.25" "0.5" "0.75" "1.0")
for I in 0 1 2 3 4; do
    P=${P_VALUES[$I]}
    BASE_SEED=$((1100 + I))
    JOB_ID=$(ARCH=0 P_LIST="$P" EPOCHS=100 BASE_SEED=$BASE_SEED N_TOTAL=7000 \
        OUT_DIR="$SYNTH_OUT" \
        sbatch --parsable code/attacks/task1_duci/train_synth.sh)
    echo "  synth p=$P seed=$BASE_SEED → job $JOB_ID"
done

echo
echo "[launch] queue snapshot:"
squeue -u "$USER" --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" | head -20
