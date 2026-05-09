#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash code/attacks/task3/run_all_tests.sh /absolute/path/to/Dataset.zip [device]
# Example (GPU):
#   bash code/attacks/task3/run_all_tests.sh /p/scratch/.../Dataset.zip cuda

ZIP_PATH="${1:-Dataset.zip}"
DEVICE="${2:-cuda}"

python code/attacks/task3/experiments.py \
  --data-source zip \
  --zip-path "${ZIP_PATH}" \
  --zip-train-split train \
  --zip-val-split valid \
  --results-path code/attacks/task3/results/ablation_report.json \
  --device "${DEVICE}"

python code/attacks/task3/main.py \
  --mode train \
  --data-source zip \
  --zip-path "${ZIP_PATH}" \
  --zip-train-split train \
  --zip-val-split valid \
  --artifacts-path code/attacks/task3/cache/model_full.pkl \
  --use-branch-d \
  --use-binoculars \
  --device "${DEVICE}"

python code/attacks/task3/main.py \
  --mode infer \
  --data-source zip \
  --zip-path "${ZIP_PATH}" \
  --zip-test-split test \
  --artifacts-path code/attacks/task3/cache/model_full.pkl \
  --out-csv submissions/task3_watermark_detection_full.csv \
  --expected-rows 2250 \
  --use-branch-d \
  --use-binoculars \
  --device "${DEVICE}"

# Conservative backup (without branch D / binoculars)
python code/attacks/task3/main.py \
  --mode train \
  --data-source zip \
  --zip-path "${ZIP_PATH}" \
  --zip-train-split train \
  --zip-val-split valid \
  --artifacts-path code/attacks/task3/cache/model_conservative.pkl \
  --device "${DEVICE}"

python code/attacks/task3/main.py \
  --mode infer \
  --data-source zip \
  --zip-path "${ZIP_PATH}" \
  --zip-test-split test \
  --artifacts-path code/attacks/task3/cache/model_conservative.pkl \
  --out-csv submissions/task3_watermark_detection_conservative.csv \
  --expected-rows 2250 \
  --device "${DEVICE}"

echo "Done. Outputs:"
echo "- code/attacks/task3/results/ablation_report.json"
echo "- submissions/task3_watermark_detection_full.csv"
echo "- submissions/task3_watermark_detection_conservative.csv"
echo "Device used: ${DEVICE}"
