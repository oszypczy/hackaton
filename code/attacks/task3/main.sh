#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash code/attacks/task3/main.sh train
#   bash code/attacks/task3/main.sh infer

MODE="${1:-train}"

if [[ "${MODE}" == "train" ]]; then
  python code/attacks/task3/main.py \
    --mode train \
    --data-source zip \
    --zip-path Dataset.zip \
    --zip-train-split train \
    --zip-val-split valid \
    --artifacts-path code/attacks/task3/cache/model.pkl \
    --use-branch-d \
    --use-binoculars
elif [[ "${MODE}" == "infer" ]]; then
  python code/attacks/task3/main.py \
    --mode infer \
    --data-source zip \
    --zip-path Dataset.zip \
    --zip-test-split test \
    --artifacts-path code/attacks/task3/cache/model.pkl \
    --out-csv submissions/task3_watermark_detection.csv \
    --expected-rows 2250 \
    --use-branch-d \
    --use-binoculars
else
  echo "Unknown mode: ${MODE}"
  exit 2
fi
