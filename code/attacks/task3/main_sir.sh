#!/bin/bash
#SBATCH --job-name=task3-sir
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir.err

set -euo pipefail

jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
SIR_DIR=$SCRATCH/task3/sir_model
TASK_OUT=$SCRATCH/task3

mkdir -p "$TASK_CACHE" "$TASK_OUT/output" "$SIR_DIR"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

export HF_HOME="$SCRATCH/.cache"
export HF_DATASETS_OFFLINE=1
# NOTE: TRANSFORMERS_OFFLINE intentionally NOT set — need to download BERT model

cd "$REPO"

echo "=== Downloading SIR model assets ==="
# Download BERT embedder
python -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading BERT...')
AutoTokenizer.from_pretrained('perceptiveshawty/compositional-bert-large-uncased')
AutoModel.from_pretrained('perceptiveshawty/compositional-bert-large-uncased')
print('BERT model ready.')
"

# Download transform checkpoint
PTHFILE=$SIR_DIR/transform_model_cbert.pth
if [ ! -f "$PTHFILE" ]; then
    echo "Downloading transform_model_cbert.pth..."
    curl -L --max-time 120 \
        "https://github.com/THU-BPM/Robust_Watermark/raw/main/model/transform_model_cbert.pth" \
        -o "$PTHFILE" && echo "Downloaded via GitHub raw." || {
        python -c "
from huggingface_hub import hf_hub_download
import shutil
try:
    path = hf_hub_download(repo_id='THU-BPM/Robust_Watermark', filename='model/transform_model_cbert.pth')
    shutil.copy(path, '$PTHFILE')
    print('Downloaded via HF Hub.')
except Exception as e:
    print(f'HF Hub failed: {e}')
" || echo "Both downloads failed — SIR features will return 0.0 (fallback to other features)"
    }
fi

export TRANSFORMERS_OFFLINE=1
export SIR_CHECKPOINT=$PTHFILE

# Delete stale caches that will be recomputed with updated code
rm -f "$TASK_CACHE/features_unigram_direct.pkl"
rm -f "$TASK_CACHE/features_kgw_selfhash.pkl"

# Full leak-free pipeline + SIR + Mistral KGW
python code/attacks/task3/main.py \
    --phase 2 \
    --skip-branch-bc \
    --use-strong-bino \
    --use-fdgpt \
    --use-unigram-direct \
    --use-kgw-selfhash \
    --use-sir \
    --classifier logreg \
    --logreg-C 0.05 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir.csv" \
    --n-rows 2250

echo "Done. Submission at $TASK_OUT/submission_sir.csv"
