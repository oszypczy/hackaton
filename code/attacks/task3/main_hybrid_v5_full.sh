#!/bin/bash
#SBATCH --job-name=task3-hyb5
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_hyb5.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_hyb5.err

# hybrid v5: ALL cached features incl olmo_7b + judges
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

# Sweep C values around the optimum (0.03-0.07)
for C in 0.03 0.05 0.07; do
    Cstr=${C//./}
    echo "===== hybrid_v5 C=$C (incl olmo+judges) ====="
    python code/attacks/task3/hybrid_v3.py \
        --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
        --out "$TASK_OUT/submission_hyb5_C${Cstr}.csv" \
        --classifier logreg --logreg-C $C
done

# Also try without olmo (since standalone OOF was weak)
echo "===== hybrid_v5 NO olmo C=0.05 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb5_no_olmo.csv" \
    --features a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric \
               kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 \
               roberta unigram_direct judge_phi2 judge_mistral \
    --classifier logreg --logreg-C 0.05

# Without judges (test if they add signal)
echo "===== hybrid_v5 NO judges C=0.05 ====="
python code/attacks/task3/hybrid_v3.py \
    --data-dir "$SCRATCH/llm-watermark-detection" --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_hyb5_no_judges.csv" \
    --features a a_strong bino bino_strong bino_xl fdgpt d better_liu stylometric \
               kgw kgw_llama kgw_v2 bigram lm_judge multi_lm multi_lm_v2 \
               roberta unigram_direct olmo_7b \
    --classifier logreg --logreg-C 0.05

echo "Done."
