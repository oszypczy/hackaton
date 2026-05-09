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

# Tokenizers only (for KGW direct detection)
tok_only = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'mistralai/Mistral-7B-v0.1',
]

for m in tok_only:
    try:
        tok = AutoTokenizer.from_pretrained(m, token=token)
        print(f'  TOK OK: {m}  vocab={tok.vocab_size}')
    except Exception as e:
        print(f'  TOK FAIL: {m}: {e}')

print('Done with tokenizers.')
"

# Full models for instruct-LM PPL (multi_lm_v2)
echo "Downloading full INSTRUCT models (this takes a while)..."
python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

token = os.environ['HF_TOKEN']

models = [
    ('microsoft/Phi-2', False),                           # public, ~2.7B
    ('Qwen/Qwen2-0.5B-Instruct', False),                  # public, 0.5B
    ('meta-llama/Llama-2-7b-chat-hf', True),              # gated, 7B chat
    ('mistralai/Mistral-7B-Instruct-v0.1', True),         # gated, 7B instruct
]

for name, needs_auth in models:
    try:
        kwargs = {'token': token} if needs_auth else {}
        tok = AutoTokenizer.from_pretrained(name, **kwargs)
        mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kwargs)
        n_params = sum(p.numel() for p in mod.parameters()) / 1e9
        print(f'  MODEL OK: {name}  ({n_params:.1f}B params)')
        del mod  # free
    except Exception as e:
        print(f'  MODEL FAIL: {name}: {e}')

print('Done with models.')
"

ls -la "$HF_HOME/hub/" | grep -iE "llama|mistral|phi|qwen" 2>/dev/null
