# Task 1 (DUCI) — FULL Automation Playbook for Parallel Agent

> **WHO:** New agent runs this task in parallel. Self-contained instructions below.
> **GOAL:** Given 9 ResNet model checkpoints, predict for each what FRACTION of MIXED dataset was used in training. Metric: MAE (lower better).
> **DEADLINE:** Hackathon ends 2026-05-10 12:30 (~14h from start).
> **BRANCH:** `task1-multan1` or new — DO NOT TOUCH `task3-murdzek2` (that's me).

---

## CONTEXT for the agent (paste this first):

You are a new Claude Code agent on a 24h ML security hackathon. Your task: solve **Task 1 — DUCI** autonomously while another agent works on Task 3. You have full GPU access via Jülich SLURM. Read this playbook + `docs/tasks/task1_duci.md` to understand task spec.

**Stop conditions:**
- Achieve MAE ≤ 0.05 (excellent), or
- Submit at least 3 distinct approaches by 2026-05-10 10:00 UTC, or
- Run out of compute time

---

## 1. TASK SUMMARY

**Setup:** 9 ResNet checkpoints (3× ResNet18, 3× ResNet50, 3× ResNet152). Each was trained on a `(p · MIXED) + ((1-p) · held_out)` mixture. **Predict `p` for each.**

**Submission format:** `submission.csv` with columns `model_id,proportion`:
```
model_id,proportion
00,0.123
...
22,0.678
```
Validates: 9 unique rows, columns case-sensitive, values in [0,1].

**Public LB scored on 3 of 9 models, private on remaining 6.**

**Submit endpoint:** `POST http://35.192.205.84/submit/11-duci`
**Submit via:** `python3 scripts/submit.py task1 submissions/task1_duci.csv` (handles cooldown).

---

## 2. DATA & MODELS (already on Jülich)

```bash
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
# Mixed set + Population set (in-distribution, NOT seen by models)
ls $SCRATCH/DUCI/
# Model checkpoints (9 .pt files, names like model_00.pt etc)
ls $SCRATCH/DUCI/MODELS/  # or wherever checkpoints land
```

**Read first:**
```bash
scripts/juelich_exec.sh "head -50 $SCRATCH/DUCI/README.md"
scripts/juelich_exec.sh "ls -la $SCRATCH/DUCI/"
```

If MIXED is `X.npy + Y.npy`, load with `np.load`. Labels are likely CIFAR100-style (100 classes, 3×32×32) but **VERIFY** — task PDF only says "in-distribution".

---

## 3. METHODOLOGY (Tong 2025 ICLR paper, our `references/papers/05_*.txt`)

**Core algorithm (DUCI):**
1. **Train reference models** on KNOWN fractions (e.g., p=0.0, 0.5, 1.0) of MIXED on same architecture
2. For each model, compute **per-record MIA score** on MIXED (target audit set) and Population (non-member reference)
3. Set decision threshold at **TPR target on Population (non-members)** → FPR
4. Compute member rate `m_i` for record `i` across reference models, build calibration
5. **Equation 4**: `p̂_i = (m̂_i − FPR) / (TPR − FPR)`, then `p̂ = mean(p̂_i)`
6. TPR/FPR estimated GLOBALLY (one value across X), not per-i

**MIA backbone**: RMIA (Zarifzadeh 2024). Single-ref variant uses linear approx `a=0.3` for `Pr(x|θ)/Pr(x)`.

**Single reference model is enough** (max MAE ≈ 0.087 on CIFAR-100/WRN28-2 with 1 ref vs 0.034 with 42 refs).

---

## 4. EXISTING CODE (multan1's WIP, in `repo-multan1/code/attacks/task1/`)

```bash
scripts/juelich_exec.sh "ls $SCRATCH/repo-multan1/code/attacks/task1/"
# main.py, README.md, output/, refs/, refs_10ep/
```

If main.py is functional → adapt. If empty/broken → start fresh in `repo-${USER}/code/attacks/task1/`.

---

## 5. IMPLEMENTATION PLAN (priority order)

### Step 1: Verify data + models (15 min)
```bash
scripts/juelich_exec.sh "python3 -c \"
import numpy as np, torch
X = np.load('$SCRATCH/DUCI/X.npy')
Y = np.load('$SCRATCH/DUCI/Y.npy')
print(f'MIXED: X.shape={X.shape} Y.shape={Y.shape} num_classes={Y.max()+1}')
ckpt = torch.load('$SCRATCH/DUCI/MODELS/model_00.pt', map_location='cpu')
print(f'ckpt keys: {list(ckpt.keys())[:5] if isinstance(ckpt,dict) else type(ckpt)}')
\""
```

### Step 2: Train ONE reference ResNet18 at p=0.5 on MIXED+Population (1-1.5h on A800)
- Same hyperparams as target (likely SGD, lr=0.1, wd=5e-4, 50 epochs)
- For Population fill-in: take from `$SCRATCH/DUCI/Population.npy` (or whatever it's called)
- Save: `$SCRATCH/DUCI/refs/resnet18_p050.pt`

### Step 3: Compute MIA scores (RMIA single-ref) on MIXED + Population, for all 9 targets
For each target model `θ_t`:
- Forward pass MIXED + Population through `θ_t` and reference `θ_r`
- Per-record signal: `loss(θ_t) - loss(θ_r)` (or RMIA Eq.5: ratio of softmax)
- Determine threshold via TPR/FPR on Population reference

### Step 4: Apply DUCI debiasing (Equation 4) → estimate `p` for each model

### Step 5: Submit, iterate
- Sanity check: ResNet18 targets should be most reliable (matches ref arch)
- ResNet50/152: cross-architecture might fail — fallback: also train ResNet50 ref + ResNet152 ref

### Step 6 (if time): Train multiple refs (8 refs gives MAE ≈ 0.055 vs 0.087 with 1)
- Parallel train on 4× A800: 8 ResNet18 references × 50ep × 30min = 4h elapsed

---

## 6. CLUSTER WORKFLOW

```bash
# Local (laptop): edit code on branch, push to GitHub
git checkout -b task1-newuser   # or whatever branch user assigns
# ... edit files in code/attacks/task1/...
git commit -am "task1: ..."
git push origin task1-newuser

# On Jülich: pull and run
scripts/juelich_exec.sh "cd $SCRATCH/repo-${USER} && git pull origin task1-newuser"
scripts/juelich_exec.sh --force "cd $SCRATCH/repo-${USER} && sbatch code/attacks/task1/main_train_ref.sh"
```

**sbatch template:**
```bash
#!/bin/bash
#SBATCH --job-name=task1-train
#SBATCH --account=training2615
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j.out
#SBATCH --error=/p/scratch/training2615/kempinski1/Czumpers/repo-%u/output/%j.err
set -euo pipefail
jutil env activate -p training2615
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
source $SCRATCH/repo-${USER}/venv/bin/activate
cd $SCRATCH/repo-${USER}
python code/attacks/task1/train_ref.py ...
```

For 4× GPU parallel training: use `--gres=gpu:4` and split work across them.

---

## 7. SUBMISSION FLOW

```bash
# Pull CSV from cluster
scripts/juelich_exec.sh "cat $SCRATCH/DUCI/submission.csv" > submissions/task1_duci.csv

# Validate locally
python3 -c "
import csv
with open('submissions/task1_duci.csv') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 9, f'expected 9 rows, got {len(rows)}'
assert {r['model_id'] for r in rows} == {'00','01','02','10','11','12','20','21','22'}
for r in rows:
    p = float(r['proportion'])
    assert 0.0 <= p <= 1.0
print('OK')
"

# Submit (uses scripts/submit.py + .env API key)
python3 scripts/submit.py task1 submissions/task1_duci.csv

# Verify via leaderboard
curl -s http://35.192.205.84/leaderboard_page | grep '11_duci::Czumpers'
```

**Cooldown:** 5min on success / 2min on rejection (per Czumpers convention).

---

## 8. PITFALLS & TIPS

1. **CIFAR100 NOT confirmed in PDF** — only inferred from presentation. Verify with `Y.max()`. If it's CIFAR10 or different, adjust ResNet output layer (10/100 classes).
2. **Matched architecture for refs** — for ResNet50/152 targets, training a ResNet18 ref might miss signal. Train at least one ref per arch ideally.
3. **Public 3/9 modeli only** — DON'T overfit thresholds to public score. Cross-validate on architecture (train ref on R18, test on R50/152 → measure transfer).
4. **Held-out fill-in NOT given** — PDF distinguishes Population (we have) from Held-out (used for fill-in, internal). We only see Population as non-members. That's fine for RMIA.
5. **Special non-iid sampling** would penalize MAE (paper: 0.062 → 0.109). PDF says iid so we should be OK. Check empirically.
6. **Input normalization** — match training (mean/std). Wrong normalization gives garbage MIA.
7. **`task_template.py`** in cluster — uses different submit syntax than our `scripts/submit.py`. Use ours.

---

## 9. WHEN STUCK / IDEAS

- **Ensemble MIA scores**: avg of (loss diff, RMIA, Min-K%++)
- **Domain adapt**: if cross-arch fails, fit linear regressor `MIA_score → proportion` on per-arch ref data
- **Pseudo-label**: high/low confidence MIXED records → augment ref training
- **Read paper #5 (Tong 2025 ICLR) line by line** — especially Section 4 (algorithm) and Equation 4 (debiasing)
- **Read multan1's existing main.py** (in `repo-multan1/code/attacks/task1/`) for already-tried approaches

---

## 10. PROCESS LOG

Maintain `docs/tasks/task1_process_so_far.md` (per `feedback_process_log.md` convention):
- Score history (chronological)
- Approaches tried (success + failure)
- Pitfalls discovered
- Next steps queue
- Commit + push regularly

---

## 11. KEY FILES TO READ

1. `references/papers/txt/05_*.txt` — Tong 2025 DUCI ICLR paper (THE method)
2. `references/papers/txt/06_*.txt` — Maini DI 2021 (fallback)
3. `docs/tasks/task1_duci.md` — full task spec (this is the source of truth)
4. `references/papers/MAPPING.md` — entry for paper 05 (rich grep terms)
5. `references/papers/MAPPING_INDEX.md` — lean router (paper 05 should be #1 lookup)

**DO NOT:**
- Read all 25 papers
- Read full PDFs when .txt exists (saves 60% tokens)
- Modify task3 files (separate agent)
- Push to main (only to your task1 branch)

Good luck!
