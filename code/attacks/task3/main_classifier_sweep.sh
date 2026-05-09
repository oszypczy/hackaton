#!/bin/bash
#SBATCH --job-name=task3-clf-sweep
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_clfsweep.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_clfsweep.err

# Sweep CLASSIFIERS, not features. Use 18 features (no olmo, no judges) — best so far.
# Goal: find a classifier that generalizes better than logreg.
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_OUT/output"
source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO"

FEATURES="a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 roberta unigram_direct"

# Ridge
echo "===== A: ridge ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_ridge.csv" \
    --features $FEATURES --classifier ridge --logreg-C 0.05

# ElasticNet (l1+l2 mix)
echo "===== B: elasticnet l1=0.5 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_enet_l50.csv" \
    --features $FEATURES --classifier elasticnet --logreg-C 0.05 --l1-ratio 0.5

# ElasticNet l1=0.1 (mostly l2)
echo "===== C: elasticnet l1=0.1 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_enet_l10.csv" \
    --features $FEATURES --classifier elasticnet --logreg-C 0.05 --l1-ratio 0.1

# MLP
echo "===== D: mlp 64 hidden ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_mlp64.csv" \
    --features $FEATURES --classifier mlp --logreg-C 0.05 --mlp-hidden 64

# SVM RBF
echo "===== E: svm rbf C=0.1 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_svm.csv" \
    --features $FEATURES --classifier svm --logreg-C 0.1

# Ensemble (logreg + lgbm) — re-test
echo "===== F: ensemble logreg + lgbm ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_clf_ensemble.csv" \
    --features $FEATURES --classifier ensemble --logreg-C 0.05 --ensemble-weights 0.7,0.3

echo "Done."
