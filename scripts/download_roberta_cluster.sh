#!/bin/bash
# Download roberta-base to Jülich offline HF cache.
# Run on LOGIN NODE (jrlogin0X) where internet is available.
#
# Usage (from your laptop):
#   scripts/juelich_exec.sh "bash /p/scratch/training2615/kempinski1/Czumpers/repo-multan1/scripts/download_roberta_cluster.sh"
#
# Or SSH directly to login node and run.

set -euo pipefail

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
export HF_HOME="$SCRATCH/.cache"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

# Online to download
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Downloading roberta-base to $HF_HOME ..."
python -c "
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('roberta-base')
mod = AutoModel.from_pretrained('roberta-base')
print(f'Tokenizer: {tok.__class__.__name__}')
print(f'Model:     {mod.__class__.__name__} ({sum(p.numel() for p in mod.parameters())/1e6:.1f}M params)')
print('OK')
"

ls -la "$HF_HOME/hub/models--roberta-base/" 2>/dev/null && echo "Cache populated."
