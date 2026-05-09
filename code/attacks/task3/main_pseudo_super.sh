#!/bin/bash
#SBATCH --job-name=task3-super
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_super.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_super.err

# Super-hybrid: ALL cached features (incl multan1's new judges/olmo13b) + pseudo-label
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

# Super features (all 24)
SUPER="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct olmo_7b olmo_13b judge_phi2 judge_mistral judge_chat judge_olmo7b judge_olmo13b"

# Baseline OOF (no pseudo)
echo "===== SUPER baseline (no pseudo) C=0.05 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_super.csv" \
    --features $SUPER --classifier logreg --logreg-C 0.05

# Super + pseudo at f=0.30, 0.40, 0.50
for FRAC in 0.30 0.40 0.50; do
    fstr=${FRAC//./}
    echo "===== SUPER + pseudo f=$FRAC ====="
    # Custom pseudo-label that uses super feature set
    python -c "
import sys
sys.path.insert(0, '/mnt/c/projekty/hackaton-cispa-2026/code/attacks/task3')
" 2>/dev/null
    # Use pseudo_label.py but pass features via --features (it uses DEFAULT_FEATURES — modify)
    python code/attacks/task3/pseudo_label.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_super_pseudo_f${fstr}.csv" \
        --top-frac $FRAC --n-rounds 1 --C 0.05
done

echo "Done."
