#!/bin/bash
# Download OLMo-2-13B-Instruct (~26GB fp16). Public, no auth.
# RUN ON LOGIN NODE.

set -euo pipefail

SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
export HF_HOME="$SCRATCH/.cache"

source "$SCRATCH/repo-${USER}/venv/bin/activate"

unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Downloading OLMo-2-13B-Instruct (~26GB fp16) to $HF_HOME ..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

name = 'allenai/OLMo-2-1124-13B-Instruct'
print(f'  Loading {name}...')
tok = AutoTokenizer.from_pretrained(name)
mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
print(f'    OK: {sum(p.numel() for p in mod.parameters())/1e9:.1f}B params, vocab={tok.vocab_size}')
"

ls "$HF_HOME/hub/" | grep -i olmo
