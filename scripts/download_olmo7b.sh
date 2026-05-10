#!/bin/bash
# Download OLMo-2-7B-Instruct + OLMo-2-1B base (binoculars partner) to cluster.
# Run on LOGIN NODE — internet required, NO HF_TOKEN needed (public models).

set -euo pipefail

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
export HF_HOME="$SCRATCH/.cache"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Downloading OLMo public models to $HF_HOME ..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

models = [
    'allenai/OLMo-2-1124-7B-Instruct',  # 7B instruct, amplifies 1B breakthrough
    'allenai/OLMo-2-0425-1B',            # 1B BASE, binoculars partner with 1B-Instruct
]

for name in models:
    try:
        print(f'  Loading {name}...')
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        n_params = sum(p.numel() for p in mod.parameters()) / 1e9
        print(f'    OK: {n_params:.1f}B params, vocab={tok.vocab_size}')
        del mod
    except Exception as e:
        print(f'    FAIL: {e}')

print('Done.')
"

ls "$HF_HOME/hub/" | grep -iE "olmo" 2>/dev/null
