#!/bin/bash
# Download Llama-2 + Mistral tokenizers to Jülich offline HF cache.
# Run on LOGIN NODE (jrlogin0X) — internet + HF_TOKEN required.
#
# Usage (laptop):
#   scripts/juelich_exec.sh "HF_TOKEN=<token> bash /p/scratch/training2615/kempinski1/Czumpers/repo-multan1/scripts/download_llama_tokenizers.sh"
#
# Or SSH to login node, export HF_TOKEN, then run.

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set in environment." >&2
    echo "Pass via: HF_TOKEN=hf_xxx bash $0" >&2
    exit 1
fi

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
export HF_HOME="$SCRATCH/.cache"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Downloading tokenizers (with auth) to $HF_HOME ..."
python -c "
import os
from transformers import AutoTokenizer

token = os.environ['HF_TOKEN']

models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'mistralai/Mistral-7B-v0.1',
]

for m in models:
    try:
        tok = AutoTokenizer.from_pretrained(m, token=token)
        print(f'  OK: {m}  vocab={tok.vocab_size}')
    except Exception as e:
        print(f'  FAIL: {m}: {e}')
print('Done.')
"

ls -la "$HF_HOME/hub/" | grep -iE "llama|mistral" 2>/dev/null
