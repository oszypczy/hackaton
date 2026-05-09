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

## Planned tests (single command)

```bash
bash code/attacks/task3/run_all_tests.sh /absolute/path/to/Dataset.zip cuda
```

It runs:
- ablations: `A+BC`, `A+BC+D`, `A+BC+Binoculars`, `FULL` across truncation stress `{none,50,100,200}`
- bootstrap and CV metrics into `code/attacks/task3/results/ablation_report.json`
- full model train+infer (`submissions/task3_watermark_detection_full.csv`)
- conservative model train+infer (`submissions/task3_watermark_detection_conservative.csv`)

## Pick best and submit

Show ranking first, then ask for confirmation before submit:

```bash
python code/attacks/task3/pick_best_and_submit.py
```

Non-interactive submit:

```bash
python code/attacks/task3/pick_best_and_submit.py --yes
```
