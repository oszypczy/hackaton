#!/bin/bash
# Download SIR model assets to the cluster.
# Run from a login node (internet access required):
#   bash code/attacks/task3/download_sir_model.sh
set -euo pipefail

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
SIR_DIR=$SCRATCH/task3/sir_model
HF_HOME=$SCRATCH/.cache

mkdir -p "$SIR_DIR"

# 1. Download BERT embedder via HuggingFace
echo "=== Downloading compositional-bert-large-uncased ==="
HF_HOME=$HF_HOME python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('perceptiveshawty/compositional-bert-large-uncased')
AutoModel.from_pretrained('perceptiveshawty/compositional-bert-large-uncased')
print('BERT model cached.')
"

# 2. Download transform_model_cbert.pth from THU-BPM/Robust_Watermark (GitHub LFS)
echo "=== Downloading transform_model_cbert.pth ==="
PTHFILE=$SIR_DIR/transform_model_cbert.pth
if [ ! -f "$PTHFILE" ]; then
    # Try direct GitHub raw download (works if file is small enough / not LFS)
    curl -L --max-time 120 \
        "https://github.com/THU-BPM/Robust_Watermark/raw/main/model/transform_model_cbert.pth" \
        -o "$PTHFILE" && echo "Downloaded via GitHub raw." || {
        # Fallback: huggingface hub if available there
        echo "GitHub raw failed, trying HuggingFace Hub..."
        pip install huggingface_hub -q 2>/dev/null || true
        python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='THU-BPM/Robust_Watermark', filename='model/transform_model_cbert.pth')
shutil.copy(path, '$PTHFILE')
print('Downloaded via HF Hub.')
" || echo "Both downloads failed — SIR features will be disabled."
    }
fi

if [ -f "$PTHFILE" ]; then
    sz=$(du -sh "$PTHFILE" | cut -f1)
    echo "transform_model_cbert.pth: $sz at $PTHFILE"
else
    echo "WARNING: transform_model_cbert.pth not downloaded."
fi

echo "Done."
