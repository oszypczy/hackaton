#!/bin/bash
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/DUCI/output/maini_extract_%j.out

set -euo pipefail

# Modes (set MODE env var):
#   feasibility — quick test on 1 target + p=0,1 synth (10 min, R18)
#   targets     — all 9 organizer targets (3 R18 + 3 R50 + 3 R152)
#   targets_r18, targets_r50, targets_r152 — single-arch target subset
#   synth_2k    — 5× synth N=2000 80ep R18 (legacy bank for SUB-9 alignment)
#   synth_7k_r18, synth_7k_r50, synth_7k_r152 — 5× synth N=7000 100ep per arch
MODE=${MODE:-feasibility}

N_DIRS=${N_DIRS:-6}              # per-distribution; 6×3=18 total
DISTRIBUTIONS=${DISTRIBUTIONS:-uniform,gaussian,laplace}
STEP=${STEP:-0.05}
MAX_STEPS=${MAX_STEPS:-60}
SAMPLE_BATCH=${SAMPLE_BATCH:-16}
SEED=${SEED:-0}
MAX_MIXED=${MAX_MIXED:-0}        # 0 = full (2000)
MAX_Z=${MAX_Z:-0}                # 0 = full (5000)

REPO=${REPO:-/p/scratch/training2615/kempinski1/Czumpers/repo-${USER}}
P4VENV=/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/python
SIGS_DIR=${SIGS_DIR:-/p/scratch/training2615/kempinski1/Czumpers/DUCI/maini_signals_v1}

SYNTH_2K_R18=/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets_80ep_r18
SYNTH_7K_R18=/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets_n7000_100ep_r18
SYNTH_7K_R50=/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets_n7000_100ep_r50
SYNTH_7K_R152=/p/scratch/training2615/kempinski1/Czumpers/DUCI/synth_targets_n7000_100ep_r152

mkdir -p "$SIGS_DIR" /p/scratch/training2615/kempinski1/Czumpers/DUCI/output

echo "[sbatch] node=$(hostname)  job=$SLURM_JOB_ID  mode=$MODE"
echo "[sbatch] BW: dirs=$N_DIRS×3=$((N_DIRS*3))  step=$STEP  max_steps=$MAX_STEPS  batch=$SAMPLE_BATCH"
echo "[sbatch] CUDA: $(nvidia-smi -L 2>/dev/null | head -2)"

cd "$REPO"

BW_FLAGS="--n-dirs $N_DIRS --distributions $DISTRIBUTIONS --step $STEP --max-steps $MAX_STEPS --sample-batch $SAMPLE_BATCH --seed $SEED --max-mixed $MAX_MIXED --max-z $MAX_Z"

case "$MODE" in
  feasibility)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk feasibility \
        --target-id model_00 \
        --synth-dir "$SYNTH_2K_R18" \
        --n-samples 200 \
        $BW_FLAGS
    ;;
  targets)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk targets \
        --out-dir "$SIGS_DIR/targets" \
        $BW_FLAGS
    ;;
  targets_r18)
    for mid in model_00 model_01 model_02; do
      $P4VENV -m code.attacks.task1_duci.maini_blind_walk target \
          --model-id "$mid" \
          --out "$SIGS_DIR/targets/target_$mid.json" \
          $BW_FLAGS
    done
    ;;
  targets_r50)
    for mid in model_10 model_11 model_12; do
      $P4VENV -m code.attacks.task1_duci.maini_blind_walk target \
          --model-id "$mid" \
          --out "$SIGS_DIR/targets/target_$mid.json" \
          $BW_FLAGS
    done
    ;;
  targets_r152)
    for mid in model_20 model_21 model_22; do
      $P4VENV -m code.attacks.task1_duci.maini_blind_walk target \
          --model-id "$mid" \
          --out "$SIGS_DIR/targets/target_$mid.json" \
          $BW_FLAGS
    done
    ;;
  synth_2k)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk synth \
        --synth-dir "$SYNTH_2K_R18" \
        --out-dir "$SIGS_DIR/synth_2k_r18" \
        $BW_FLAGS
    ;;
  synth_7k_r18)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk synth \
        --synth-dir "$SYNTH_7K_R18" \
        --out-dir "$SIGS_DIR/synth_7k_r18" \
        $BW_FLAGS
    ;;
  synth_7k_r50)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk synth \
        --synth-dir "$SYNTH_7K_R50" \
        --out-dir "$SIGS_DIR/synth_7k_r50" \
        $BW_FLAGS
    ;;
  synth_7k_r152)
    $P4VENV -m code.attacks.task1_duci.maini_blind_walk synth \
        --synth-dir "$SYNTH_7K_R152" \
        --out-dir "$SIGS_DIR/synth_7k_r152" \
        $BW_FLAGS
    ;;
  *)
    echo "[sbatch] unknown MODE=$MODE" >&2
    exit 1
    ;;
esac

echo "[sbatch] done"
