#!/bin/bash
#SBATCH --job-name=task3-sir-ext
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_ext.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j_sir_ext.err

# Extract SIR features using downloaded transform_model_cbert.pth
set -euo pipefail
jutil env activate -p training2615

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
REPO=$SCRATCH/repo-${USER}
TASK_CACHE=$SCRATCH/task3/cache
TASK_OUT=$SCRATCH/task3
SIR_DIR=$SCRATCH/task3/sir_model

mkdir -p "$TASK_OUT/output"
source "$REPO/venv/bin/activate"
export HF_HOME="$SCRATCH/.cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export SIR_CHECKPOINT=$SIR_DIR/transform_model_cbert.pth
cd "$REPO"

# Test SIR extractor works on 5 texts
python -c "
import sys
sys.path.insert(0, 'code/attacks/task3')
from features.sir_direct import extract
import json
data_dir = '$SCRATCH/llm-watermark-detection'
texts = []
for f in [data_dir + '/train_clean.jsonl', data_dir + '/train_wm.jsonl']:
    for line in open(f).readlines()[:3]:
        texts.append(json.loads(line)['text'])
print(f'Testing SIR on {len(texts)} sample texts...')
for i, t in enumerate(texts):
    feat = extract(t)
    print(f'Text {i}: {feat}')
"

# Now run full extraction via main.py with --use-sir
python code/attacks/task3/main.py \
    --phase 2 \
    --skip-branch-bc \
    --use-sir \
    --classifier logreg \
    --logreg-C 0.05 \
    --data-dir "$SCRATCH/llm-watermark-detection" \
    --cache-dir "$TASK_CACHE" \
    --out "$TASK_OUT/submission_sir_only.csv" \
    --n-rows 2250

echo "SIR extraction done. features_sir.pkl should now be in cache."
