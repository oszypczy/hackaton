#!/bin/bash
#SBATCH --job-name=task3-judges
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_judges.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_judges.err

# Extract judge_phi2 + judge_mistral features
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

cd "$SCRATCH/repo-multan1"

# Phi-2 judge
echo "===== judge_phi2 ====="
python code/attacks/task3/main.py \
    --phase 2 --skip-branch-bc \
    --use-judge-phi2 \
    --classifier logreg --logreg-C 0.05 \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_judge_phi2.csv" --n-rows 2250

# Mistral-7B-Instruct judge
echo "===== judge_mistral ====="
python code/attacks/task3/main.py \
    --phase 2 --skip-branch-bc \
    --use-judge-mistral \
    --classifier logreg --logreg-C 0.05 \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_judge_mistral.csv" --n-rows 2250

echo "Done."
