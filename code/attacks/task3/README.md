# Task 3 pipeline

## Train

```bash
python code/attacks/task3/main.py \
  --mode train \
  --data-source zip \
  --zip-path Dataset.zip \
  --zip-train-split train \
  --zip-val-split valid \
  --artifacts-path code/attacks/task3/cache/model.pkl \
  --use-branch-d \
  --use-binoculars
```

## Inference / submission.csv

```bash
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
```

Fallbacks:
- HuggingFace: `--data-source hf --hf-dataset SprintML/llm-watermark-detection`
- Local files: `--data-source csv --train-path/--val-path/--test-path`

If your zip is split as `train_clean.jsonl`, `train_wm.jsonl`, `valid_clean.jsonl`, `valid_wm.jsonl`, loader auto-merges them and adds labels (`clean=0`, `wm=1`).

Then submit:

```bash
just submit task3 submissions/task3_watermark_detection.csv
```
