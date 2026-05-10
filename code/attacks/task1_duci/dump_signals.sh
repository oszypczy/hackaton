#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/dump_signals_%j.out

set -euo pipefail

REPO=/p/scratch/training2615/kempinski1/Czumpers/repo-szypczyn1
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
DUCI=/p/scratch/training2615/kempinski1/Czumpers/DUCI

OUT_CSV=${OUT_CSV:-$REPO/submissions/task1_duci_presentation.csv}
DUMP_JSON=${DUMP_JSON:-$DUCI/presentation_signals.json}

cd "$REPO"
mkdir -p "$REPO/submissions" "$DUCI/output"

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID"
echo "[sbatch] CUDA: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "[sbatch] dump -> $DUMP_JSON"

$P4VENV -m code.attacks.task1_duci.mle \
    --synth-dir "$DUCI/synth_targets_80ep_r18" \
    --synth-dir-r50 "$DUCI/synth_targets_80ep_r50" \
    --synth-dir-r152 "$DUCI/synth_targets_80ep_r18" \
    --use-signal mean_loss_mixed \
    --degree 1 \
    --out "$OUT_CSV" \
    --dump-signals "$DUMP_JSON"

echo "[sbatch] done; signals JSON at $DUMP_JSON"
