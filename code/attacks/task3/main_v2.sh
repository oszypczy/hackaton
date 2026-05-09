#!/usr/bin/env bash
# V2 — adds BranchBigram (KGW-targeted) + OPT tokenizer in BranchBC.
# Run on cluster with GPU. Usage:
#   bash code/attacks/task3/main_v2.sh train   [ZIP_PATH] [DEVICE]
#   bash code/attacks/task3/main_v2.sh infer   [ZIP_PATH] [DEVICE]
set -euo pipefail

MODE="${1:-train}"
ZIP_PATH="${2:-Dataset.zip}"
DEVICE="${3:-cuda}"

ARTIFACTS="code/attacks/task3/cache/model_v2.pkl"
OUT_CSV="submissions/task3_v2.csv"

if [[ "${MODE}" == "train" ]]; then
  python code/attacks/task3/main.py \
    --mode train \
    --data-source zip \
    --zip-path "${ZIP_PATH}" \
    --zip-train-split train \
    --zip-val-split valid \
    --artifacts-path "${ARTIFACTS}" \
    --device "${DEVICE}" \
    --a-model "EleutherAI/pythia-1.4b" \
    --bc-tokenizer gpt2 \
    --bc-extra-tokenizers "facebook/opt-1.3b" \
    --use-bigram \
    --use-branch-d \
    --use-binoculars \
    --binoculars-observer "EleutherAI/pythia-1.4b" \
    --binoculars-performer "EleutherAI/pythia-2.8b" \
    --num-leaves 31 \
    --max-depth 6 \
    --num-boost-round 600

elif [[ "${MODE}" == "infer" ]]; then
  python code/attacks/task3/main.py \
    --mode infer \
    --data-source zip \
    --zip-path "${ZIP_PATH}" \
    --zip-test-split test \
    --artifacts-path "${ARTIFACTS}" \
    --out-csv "${OUT_CSV}" \
    --expected-rows 2250 \
    --device "${DEVICE}" \
    --a-model "EleutherAI/pythia-1.4b" \
    --bc-tokenizer gpt2 \
    --bc-extra-tokenizers "facebook/opt-1.3b" \
    --use-bigram \
    --use-branch-d \
    --use-binoculars \
    --binoculars-observer "EleutherAI/pythia-1.4b" \
    --binoculars-performer "EleutherAI/pythia-2.8b"

elif [[ "${MODE}" == "ablation" ]]; then
  python code/attacks/task3/experiments.py \
    --data-source zip \
    --zip-path "${ZIP_PATH}" \
    --zip-train-split train \
    --zip-val-split valid \
    --results-path code/attacks/task3/results/ablation_v2.json \
    --device "${DEVICE}" \
    --a-model "EleutherAI/pythia-1.4b" \
    --bc-tokenizer gpt2 \
    --bc-extra-tokenizers "facebook/opt-1.3b"

else
  echo "Unknown mode: ${MODE}. Use train|infer|ablation"
  exit 2
fi

echo "Done. Mode=${MODE}, device=${DEVICE}"
