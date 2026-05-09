#!/bin/bash
#SBATCH --job-name=task3-sir-hyb
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_hyb.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_hyb.err

# SUPER + SIR features → LogReg + pseudo
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

SIR_FEATS="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct olmo_7b olmo_13b judge_phi2 judge_mistral judge_chat judge_olmo7b judge_olmo13b sir"

# 1. SUPER + SIR baseline (no pseudo)
echo "===== SUPER+SIR baseline ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir_hybrid.csv" \
    --features $SIR_FEATS --classifier logreg --logreg-C 0.05

# 2. SUPER + SIR + pseudo f=0.30
echo "===== SUPER+SIR + pseudo f=0.30 ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir_pseudo_f030.csv" \
    --features $SIR_FEATS --top-frac 0.30 --n-rounds 1 --C 0.05

# 3. SUPER + SIR + pseudo f=0.50
echo "===== SUPER+SIR + pseudo f=0.50 ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir_pseudo_f050.csv" \
    --features $SIR_FEATS --top-frac 0.50 --n-rounds 1 --C 0.05

# 4. ONLY SIR features (pure SIR test) + pseudo
echo "===== SIR-only + pseudo f=0.30 ====="
python code/attacks/task3/pseudo_label.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir_only_pseudo.csv" \
    --features sir --top-frac 0.30 --n-rounds 1 --C 0.05

echo "Done."
