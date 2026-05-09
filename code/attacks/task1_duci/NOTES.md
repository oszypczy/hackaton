# Task 1 (DUCI) — implementation notes

> Working notes for the implementation of Task 1.
> **Spec / ground truth** is in `docs/tasks/task1_duci.md` — DO NOT duplicate it here.
> This file = our discoveries + open questions + planning, alive on branch `task1`.
> Last updated: 2026-05-09

## Pointers

| What | Where |
|---|---|
| Task spec (organizer PDF) | `docs/tasks/task1_duci.md` |
| Deep-research prompt | `docs/tasks/task1_research_prompt.md` |
| Tong DUCI paper (THE one) | `references/papers/05_*.pdf` + `references/papers/txt/05_*.txt` |
| Maini DI fallback | `references/papers/06_*.pdf` |
| Dziedzic SSL DI | `references/papers/07_*.pdf` |
| Earlier research summary | `docs/STATUS.md` § "Task 1 (DUCI) — extracted facts" |
| Submission flow | `docs/SUBMISSION_FLOW.md` |
| Cluster workflow | `docs/CLUSTER_WORKFLOW.md` |

## Cluster layout (JURECA, shared `Czumpers/`)

Base: `/p/scratch/training2615/kempinski1/Czumpers/`

```
Czumpers/
├── DUCI/                       # task data + venv (shared, do NOT recreate)
│   ├── DATA/
│   │   ├── MIXED/              X.npy (5.9M) + y.npy (16K)
│   │   └── POPULATION/         X.npy (30M)  + y.npy (79K)
│   ├── MODELS/                 model_{00,01,02,10,11,12,20,21,22}.pkl
│   ├── .venv/                  uv venv (Python 3.12.13, torch 2.11.0+cu130)
│   ├── task_template.py        4.9 KB — submission API + model loader
│   ├── main.py                 boilerplate (ignore)
│   ├── pyproject.toml + uv.lock + requirements.txt
│   └── .git/                   independent HF clone (SprintML/DUCI), NOT our repo
└── repo-szypczyn1/             # OUR repo clone — kod taska tutaj
    └── code/attacks/task1_duci/  # ← this folder, after `git pull`
```

`DUCI/` is a separate HF git repo (SprintML/DUCI). **Never push to it, never pull.** Our code in
`repo-szypczyn1/code/attacks/task1_duci/` references DUCI/ files via absolute paths.

## Data — empirically confirmed (file sizes)

| Set | File | Shape | Dtype | Notes |
|---|---|---|---|---|
| MIXED X | `DUCI/DATA/MIXED/X.npy` | 2000 × 32×32×3 (inferred from bytes) | uint8 | CIFAR-style — sizes match |
| MIXED y | `DUCI/DATA/MIXED/y.npy` | 2000 × int64 | int64 | 100 classes ⇒ CIFAR-100 |
| POPULATION X | `DUCI/DATA/POPULATION/X.npy` | 10000 × 32×32×3 | uint8 | non-member reference, never seen by 9 models |
| POPULATION y | `DUCI/DATA/POPULATION/y.npy` | 10000 × int64 | int64 | 100 classes |

⚠ Shapes inferred from byte counts — not yet `np.load`'d successfully end-to-end. Verify
on first load (`assert X.shape == (2000, 32, 32, 3)` etc.).

## Models — files confirmed, format unknown

9 `.pkl` files in `DUCI/MODELS/`:

| ID prefix | Arch (per `task_template.py`) | Files | Size | Inferred params |
|---|---|---|---|---|
| `0?` | ResNet18 | `model_00/01/02.pkl` | 43 MB | ~11M |
| `1?` | ResNet50 | `model_10/11/12.pkl` | 91 MB | ~25M |
| `2?` | ResNet152 | `model_20/21/22.pkl` | 224 MB | ~60M |

`task_template.py` builds via `torchvision.models.resnetXX(weights=None, num_classes=100)`
(ImageNet-style head, `conv1=7×7`).

**⚠ Loading failed both ways with `torch.load`:**
- `weights_only=True` → "Unsupported operand 149"
- `weights_only=False` → "Invalid magic number; corrupt file?"

→ blocker — see Open question #1.

## Open questions (blocking code)

1. **`.pkl` load path** — what does the organizer use? Hypotheses:
   - (a) Plain Python binary dump of a `nn.Module` instance (not via `torch.save`) → load with stdlib `pickle`
   - (b) Lightning / dill / joblib serializer
   - (c) gzip-compressed Python binary dump → check magic bytes via `file MODELS/model_00.pkl` first
   - **Action:** read `task_template.py` end-to-end — it almost certainly ships the canonical loader call we need.
2. **Input resolution** — 32×32 or 224×224? Decide by `model.conv1.weight.shape` after a successful load.
3. **`state_dict` vs full `Module`?** — depends on (1).
4. **Normalization** — CIFAR-100 std mean=`[0.5071,0.4867,0.4408]`/std=`[0.2675,0.2565,0.2761]`, or something else? Check inside `task_template.py`.
5. **Public 3 / Private 6 split** — which `model_id`s are public? Spec doesn't reveal → treat all 9 as equally important. **Implication:** never tune thresholds against a subset of 9.

## Method recap (from STATUS.md research session)

- **Primary:** Tong et al. ICLR'25 — Eq. 4 debias: `p̂_i = (m̂_i − FPR) / (TPR − FPR)`, then mean over MIXED. Single reference model on CIFAR-100/ResNet-34 → MAE ≈ 0.053 (max).
- **MIA backbone:** RMIA (Zarifzadeh 2024). Single-ref linear approx `a=0.3`.
- **Reference impl:** `github.com/privacytrustlab/ml_privacy_meter` master branch — `run_duci.py`, `modules/duci/module_duci.py`, `configs/duci/cifar10.yaml`, `demo_duci.ipynb`.
- **Adaptation:** plug organizer checkpoints as "target" (no need to train them); we still train ref model(s).
- **Fallback:** Maini DI'21 — Blind Walk (black-box, k random directions) or MinGD (white-box). Shadow-free. 2-layer regressor + Welch t-test.

## Compute on JURECA

- **Login node** (where `juelich_exec.sh` runs commands by default): 2× Quadro RTX 8000, CUDA 13. OK for forward-pass debugging on 9 models.
- **Compute job** (sbatch `--partition=dc-gpu --reservation=cispahack`): 4× A800 per node. Use for ref-model training.
- Budget estimate (single A800, sequential): ResNet-18 50ep ~30 min, ResNet-50 ~1.5 h, ResNet-152 ~3-4 h. Parallel on 4× A800 → ~1.5 h elapsed for one ref per arch.

## Plan / next moves (rough order)

1. **Resolve Open Q #1** — read `task_template.py` for canonical loader; one experiment loading `model_00.pkl`. Confirm input resolution + state_dict vs Module.
2. **Smoke test** on login node — load all 9 models + `np.load` MIXED/POPULATION, run 1 forward pass per model, check accuracy on POPULATION (ground-truth labels exist so this is a sanity check).
3. **Implement RMIA single-ref** (Tong Eq. 4 debias) — `code/attacks/task1_duci/main.py`. Start with ResNet-18 ref only (smallest, cheapest).
4. **First submission** — even crude p̂ → CSV. We need the score signal early to calibrate.
5. **Train reference model(s)** if needed, sbatch on `dc-gpu`.
6. **Cross-arch validation** — does ResNet-18 ref work for ResNet-50/152 targets? If yes, big win.

## Things NOT to do

- Do not retrain the 9 organizer models — they are the targets.
- Do not tune anything against public 3 — generalization > leaderboard score (CLAUDE.md 🎯).
- Do not `pip install` in `DUCI/.venv/` without team OK (shared with 4 people).
- Do not push from cluster. Edits → laptop → push → cluster `git pull`.

## Decisions log

(append as we go — date + decision + reason)

- 2026-05-09 — Branch `task1` adopted; consolidated notes here. (`task1_duci.md` stays as spec.)
