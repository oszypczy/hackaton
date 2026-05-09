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

**Loader RESOLVED 2026-05-09:** `.pkl` files = stdlib `pickle.dump` of a raw `OrderedDict` state_dict (NOT via `torch.save`). Use:

```python
import pickle
with open(p, "rb") as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
```

`torch.load` fails because:
- `weights_only=True` (default torch ≥2.6) trips on opcode 149 = pickle proto-4 `FRAME` (weights-only unpickler doesn't support FRAME)
- `weights_only=False` falls through to `_legacy_load` which expects torch's magic-number header — file has none (no `torch.save` wrapper)

The shipped `task_template.py` uses `torch.load(...)` — it would fail in cluster's torch 2.11. We override with `pickle.load`. All 9 files verified compatible with this loader (2026-05-09).

**Architecture verified across all 9 checkpoints:**
| Files | Keys | conv1 | fc | → arch |
|---|---|---|---|---|
| `model_0?` | 122 | (64,3,7,7) | (100,512) | ResNet18 |
| `model_1?` | 320 | (64,3,7,7) | (100,2048) | ResNet50 |
| `model_2?` | 932 | (64,3,7,7) | (100,2048) | ResNet152 |

All ImageNet-style heads (7×7 conv1, NOT CIFAR-style 3×3). 100-class output ⇒ CIFAR-100.

## Open questions

1. ✅ **RESOLVED — `.pkl` load** = `pickle.load` (see Models section above).
2. ✅ **RESOLVED — `state_dict` vs `Module`** = raw `OrderedDict` state_dict.
3. **Input resolution** — 32×32 (raw CIFAR) vs 224×224 (upsampled)? `conv1=7×7` works on both, just downsamples differently. Decide by running fwd-pass on POPULATION at each resolution → pick whichever gives high accuracy.
4. **Normalization** — CIFAR-100 stats (mean=`[0.5071,0.4867,0.4408]`/std=`[0.2675,0.2565,0.2761]`) vs ImageNet (mean=`[0.485,0.456,0.406]`/std=`[0.229,0.224,0.225]`)? Try both during smoke test, pick higher acc.
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

1. ✅ Loader resolved (Open Q #1, #2 done).
2. **Smoke test** on login node — load all 9 models with `pickle.load`, `np.load` MIXED/POPULATION, fwd pass on POPULATION at 32×32 and 224×224 with both norm options → pick the (resolution, norm) combo that gives high acc on POPULATION. Resolves Open Q #3, #4.
3. **Implement RMIA single-ref** (Tong Eq. 4 debias) — `main.py`. Start with ResNet-18 ref only (smallest, cheapest). Compute m̂_i on MIXED + estimate (TPR, FPR) globally on POPULATION-vs-MIXED-not-in-train.
4. **First submission** — even crude p̂ → CSV. We need the score signal early to calibrate.
5. **Train reference model(s)** if needed, sbatch on `dc-gpu`. (Tong shows 1 ref already gives MAE ~0.087 on CIFAR-100/WRN28-2.)
6. **Cross-arch validation** — does ResNet-18 ref work for ResNet-50/152 targets? If yes, big win.

## Things NOT to do

- Do not retrain the 9 organizer models — they are the targets.
- Do not tune anything against public 3 — generalization > leaderboard score (CLAUDE.md 🎯).
- Do not `pip install` in `DUCI/.venv/` without team OK (shared with 4 people).
- Do not push from cluster. Edits → laptop → push → cluster `git pull`.

## Decisions log

(append as we go — date + decision + reason)

- 2026-05-09 — Branch `task1` adopted; consolidated notes here. (`task1_duci.md` stays as spec.)
- 2026-05-09 — Loader = stdlib `pickle.load` (NOT `torch.load`). All 9 .pkl = raw OrderedDict state_dicts. ImageNet-style heads (conv1=7×7) confirmed across all archs. Note: shipped `task_template.py` ships a `torch.load`-based loader that fails in cluster's torch 2.11 — we override it.
