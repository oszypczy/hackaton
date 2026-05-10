#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/mle_combine_%j.out

set -euo pipefail

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
DUCI=/p/scratch/training2615/kempinski1/Czumpers/DUCI

OUT_CSV=${OUT_CSV:-$REPO/submissions/task1_duci_mle_combine_widenarrow.csv}
USE_SIGNAL=${USE_SIGNAL:-mean_loss_mixed}
DEGREE=${DEGREE:-2}

R18_DIRS="${R18_DIRS:-$DUCI/synth_targets_n7000_100ep_r18,$DUCI/synth_targets_narrow_r18}"
R50_DIRS="${R50_DIRS:-$DUCI/synth_targets_n7000_100ep_r50,$DUCI/synth_targets_narrow_r50}"
R152_DIRS="${R152_DIRS:-$DUCI/synth_targets_n7000_100ep_r152}"

cd "$REPO"
mkdir -p "$REPO/submissions"

TMP_CSV=/tmp/mle_combine_$SLURM_JOB_ID.csv

$P4VENV -m code.attacks.task1_duci.mle_combine \
    --synth-dirs "$R18_DIRS" \
    --synth-dirs-r50 "$R50_DIRS" \
    --synth-dirs-r152 "$R152_DIRS" \
    --out "$TMP_CSV" \
    --use-signal "$USE_SIGNAL" \
    --degree "$DEGREE"

# Workaround scratch quota issues: write via /tmp then cp
cp "$TMP_CSV" "$OUT_CSV" || { echo "cp failed; CSV at $TMP_CSV"; cat "$TMP_CSV"; exit 1; }
echo "[done] wrote $OUT_CSV"
cat "$OUT_CSV"
