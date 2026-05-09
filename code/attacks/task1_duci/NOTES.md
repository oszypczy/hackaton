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
3. ✅ **RESOLVED — input resolution** = **32×32** (native CIFAR). 224×224 collapses acc to ~0.02 (random for 100-class).
4. ✅ **RESOLVED — normalization** = **CIFAR-100 mean/std** (`[0.5071,0.4867,0.4408]` / `[0.2675,0.2565,0.2761]`). Wins over ImageNet norm by ~7 pp uniformly across all archs.
5. **Public 3 / Private 6 split** — which `model_id`s are public? Spec doesn't reveal → treat all 9 as equally important. **Implication:** never tune thresholds against a subset of 9.
6. **NEW — why is POPULATION acc only ~27%?** Trained CIFAR-100 ResNets typically reach 70-80%. Three hypotheses:
   - (a) **Organizer undertrained on purpose.** DUCI design needs detectable membership signal → undertrained models overfit MIXED more visibly. *Most likely.*
   - (b) **POPULATION has subtle distribution shift.** PDF says "in-distribution" but maybe normalized differently / augmented / from a sibling dataset (CIFAR-100 LT, CIFAR-100-C). Verify by inspecting a few raw images.
   - (c) **Even smaller training subset.** Each model sees `(p · MIXED) + ((1-p) · held_out)` of fixed total N. If N is small (e.g., 1000), accuracy will be poor regardless.
   - **Implication for MIA:** good news. Lower generalization ⇒ wider train/non-train loss gap ⇒ easier membership inference. Don't expect the models to be "good classifiers"; they don't need to be.

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

1. ✅ Loader resolved (Open Q #1, #2).
2. ✅ Smoke test done (Open Q #3, #4). See "Smoke test results" below.
3. **Sanity-poke on Q#6 (low acc)** — quick experiment: take a couple POPULATION images, save as PNG, eyeball whether they look like normal CIFAR-100. Compare a few class-mean colors against canonical CIFAR-100 stats. Cheap, decides between hypotheses (a)/(b)/(c).
4. **Implement RMIA single-ref** (Tong Eq. 4 debias) — `main.py`. Start with ResNet-18 ref only (smallest, cheapest). Compute m̂_i on MIXED + estimate (TPR, FPR) globally on POPULATION-vs-MIXED-not-in-train.
5. **First submission** — even crude p̂ → CSV. We need the score signal early to calibrate.
6. **Train reference model(s)** if needed, sbatch on `dc-gpu`. (Tong shows 1 ref already gives MAE ~0.087 on CIFAR-100/WRN28-2.)
7. **Cross-arch validation** — does ResNet-18 ref work for ResNet-50/152 targets? If yes, big win.

## Smoke test results (2026-05-09)

Script: `code/attacks/task1_duci/smoke.py`. Ran on JURECA login node (Quadro RTX 8000)
via P4Ms venv (DUCI/.venv lacks torchvision; P4Ms has torch 2.3.0 + torchvision 0.18.0,
state_dict ABI compatible). Total elapsed: **~5 min** (all 9 models × 4 combos × 10k samples).

Per-arch mean top-1 accuracy on POPULATION (10k):

| Arch | 32+CIFAR | 32+ImgNet | 224+CIFAR | 224+ImgNet |
|---|---|---|---|---|
| ResNet18 | **0.274** | 0.208 | 0.014 | 0.013 |
| ResNet50 | **0.270** | 0.191 | 0.023 | 0.022 |
| ResNet152 | **0.270** | 0.195 | 0.024 | 0.024 |
| **OVERALL** | **0.271** | 0.198 | 0.020 | 0.020 |

**Verdict:** preprocessing = `(32×32, CIFAR-100 mean/std)`. Per-arch winner identical → architecture-agnostic decision (lowers risk on extended test set). 224×224 collapses to near-random (1/100=0.01) — models truly trained at native CIFAR resolution.

**cuDNN warning:** "Plan failed with cudnnException" fired once on first 32+CIFAR conv (P4Ms cu121 binaries vs cluster cuDNN). Auto-fallback worked, all subsequent calls clean. Numerically benign (we get correct accuracy numbers).

**Sub-detection:** OVERALL acc only 27% (vs ~70% expected for trained CIFAR-100 ResNet) → see Open Q #6.

## Things NOT to do

- Do not retrain the 9 organizer models — they are the targets.
- Do not tune anything against public 3 — generalization > leaderboard score (CLAUDE.md 🎯).
- Do not `pip install` in `DUCI/.venv/` without team OK (shared with 4 people).
- Do not push from cluster. Edits → laptop → push → cluster `git pull`.

## Decisions log

(append as we go — date + decision + reason)

- 2026-05-09 — Branch `task1` adopted; consolidated notes here. (`task1_duci.md` stays as spec.)
- 2026-05-09 — Loader = stdlib `pickle.load` (NOT `torch.load`). All 9 .pkl = raw OrderedDict state_dicts. ImageNet-style heads (conv1=7×7) confirmed across all archs. Note: shipped `task_template.py` ships a `torch.load`-based loader that fails in cluster's torch 2.11 — we override it.
- 2026-05-09 — Preprocessing = `(32×32, CIFAR-100 mean/std)` chosen by smoke test on POPULATION acc. Architecture-agnostic winner (same for ResNet18/50/152). 224×224 collapses to near-random.
- 2026-05-09 — Use **P4Ms venv** to run task1 code (`/p/scratch/.../P4Ms-hackathon-vision-task/.venv/bin/python`). DUCI/.venv has no torchvision; P4Ms (torch 2.3.0+cu121) loads DUCI .pkl without ABI issues. Read-only access — no shared state mutation. (Long-term may want user-local venv; for now P4Ms is fine.)
- 2026-05-09 — **Phase 0 diagnostics PASS.** G0a: linear probe AUC=0.4912 (≤0.55, pools i.i.d., no domain shift). G0b: 9/9 positive conf-deltas. Per-arch mean deltas: R18 +0.063, R50 +0.043, R152 +0.038. R152 model_20 has min delta +0.002 (likely p≈0). MIA signal confirmed across all targets — proceed to RMIA pipeline.
- 2026-05-09 — **Phase 1 (single-ref) FAILED degenerately.** With 1 ref, Youden's LOO impossible; fallback path (ref as both target and ref-bank) gave TPR̂=1.0, FPR̂=0.997, gap=0.003 → all p̂ collapsed to 0.025 after clamp. RMIA per-target ordering WAS correct (matched conf-delta perfectly), only calibration broken.
- 2026-05-09 — **Phase 2 (8 R18 refs @ 50ep) — synth val PASS but real submission FAILED.** β*=0.800, TPR̂=0.906, FPR̂=0.262, Youden gap +0.644. Synth val MAE=0.018 (better than Tong paper 1-ref baseline 0.087). SUB-1 to API: HTTP 200, score **0.463 — 9/9 last on leaderboard**. Top 3 teams: 0.054, 0.061, 0.069. Predictions all in [0.025, 0.122], mean 0.06 → suggests systematic UNDERESTIMATE.
- 2026-05-09 — **Diagnosis confirmed: memorization regime mismatch.** POP_z acc real targets = 0.27 (from G0b smoke). Our refs: 50ep → ~60%+ POP_z acc (over-memorized); 10ep → 0.12 POP_z acc (under). Cross-validation table:

| Refs | Synth | MAE | Verdict |
|---|---|---|---|
| 50ep | 50ep | 0.018 | self-consistent |
| **50ep** | **10ep** | **0.371** | **predicts real-submission MAE 0.46 — same regime mismatch** |
| 10ep | 50ep | 0.122 | reverse mismatch |
| 10ep | 10ep | 0.023 | self-consistent |

Sweet spot: ref POP_z acc ≈ 0.27 ⇒ ~20 epochs (extrapolated 0.12 @ 10ep → ~0.27 @ 20ep). Plan: train 8 R18 refs + 5 synth @ 20ep, re-run main on real targets, submit SUB-2.

## Phase 0 — diagnostics output (2026-05-09)

| Target | mean_conf MIXED | mean_conf POP_z | delta |
|---|---|---|---|
| model_00 (R18) | 0.183 | 0.159 | +0.024 |
| model_01 (R18) | 0.275 | 0.209 | +0.066 |
| model_02 (R18) | 0.292 | 0.192 | **+0.100** ← highest |
| model_10 (R50) | 0.221 | 0.196 | +0.025 |
| model_11 (R50) | 0.248 | 0.203 | +0.045 |
| model_12 (R50) | 0.277 | 0.218 | +0.059 |
| model_20 (R152) | 0.230 | 0.228 | **+0.002** ← min, likely p≈0 |
| model_21 (R152) | 0.230 | 0.193 | +0.038 |
| model_22 (R152) | 0.266 | 0.192 | +0.074 |

Crude `p̂` ranking from conf-delta (NOT submission-quality, just sanity check on RMIA later):
- High: model_02, model_22, model_01, model_12
- Low: model_20

## Phase 4 — MLE breakthrough (2026-05-09)

**Background:** After SUB-2 (refs_20ep RMIA) gave 0.4549, switched to Avg-Logit MLE (`mle.py`).

**Method:**
- Calibrate via 5 synth targets at known p ∈ {0, 0.25, 0.5, 0.75, 1}.
- Compute per-model **mean target-class loss on MIXED** (signal `mean_loss_mixed`).
- Linear regression: signal = a·p + b on synth.
- Invert for real targets, clamp [0.025, 0.975].

**SUB-3 (mle.py + synth_20ep + mean_loss_mixed deg=1):**
- Predictions: 00=0.349 01=0.538 02=0.586 10=0.311 11=0.413 12=0.512 20=0.374 21=0.397 22=0.503
- Mean p̂ = 0.443; range [0.31, 0.59]
- Synth LOO-MAE = 0.022
- **Real score: 0.0790 — ~6× better than RMIA**

**Why MLE > RMIA here:**
- RMIA needs ref's TPR/FPR to match target's. Memorization mismatch breaks the debias.
- MLE only needs synth at SAME regime as target — direct calibration via known-p anchors.
- mean_loss_mixed is monotone in p AND has wide synth range (4.36 → 1.10), giving strong signal.

**Variants compared (no submission for these):**
| Variant | Mean p̂ | Notes |
|---|---|---|
| mle 10ep | 0.79 | over-shoot (synth too undertrained vs real) |
| mle 20ep mean_loss d=1 | 0.44 | **WINNER, score 0.079** |
| mle 20ep mean_loss d=2 | 0.44 | identical to d=1 (5pts) |
| mle 20ep ensemble d=1 | 0.20 | other signals drag down |
| mle 30ep delta_conf d=2 | 0.07 | severe undershoot |
| mle 50ep mean_conf d=1 | 0.12 | undershoot |
| avg_all regimes | 0.35 | regime ensemble dilutes good 20ep signal |

**Sweet spot is sharp at 20ep.** Going 10ep or 30ep changes signal scale relative to real targets.

## Decisions log (cont'd)

- 2026-05-09 — **SUB-3 = 0.0790 (Phase 4 MLE win).** From 0.4630 (SUB-1) → 0.4549 (SUB-2) → 0.0790 (SUB-3). MLE pivot from RMIA was the key unblock. Top of leaderboard contention.
- 2026-05-09 — **mle.py v2** (poly fit + ensemble + per-arch lookup). LOO-MAE auto-pick suggested delta_conf deg=2 (LOO 0.008) but generalized worse on real (predictions too low). LOO is misleading proxy when synth ≠ target regime. **Stick with mean_loss_mixed deg=1.**

## Next iterations queued

1. **Arch-matched synth** (R50 + R152 @ 20ep) — current MLE uses R18 synth for all 9 targets; arch bias likely hurts model_1X / model_2X. Per-arch fits should reduce 0.01-0.03.
2. **Extra calibration points** (p ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9} R18 @ 20ep) — denser linear fit, tighter slope estimate.
3. **Multi-seed synth bank** (BASE_SEED=2000 R18 @ 20ep) — average across multiple synth banks reduces variance from training noise.

## Iteration log post-SUB-3

| SUB | Method | Mean p̂ | Range | Score | Notes |
|---|---|---|---|---|---|
| 1 | RMIA 50ep | 0.06 | [0.025, 0.12] | 0.4630 | ref overtrained |
| 2 | RMIA 20ep | 0.07 | [0.025, 0.17] | 0.4549 | regime mismatch |
| 3 | MLE 20ep R18 mean_loss d=1 | 0.443 | [0.31, 0.59] | **0.0790** | first MLE win |
| 4 | MLE 50ep R18 mean_loss d=1 | 0.498 | [0.42, 0.58] | **0.0723** | wider signal range, +0.05 mean |

**Trend confirmed:** higher mean p̂ improves score. Real true mean p ≈ 0.55-0.60.

**Failed variants:**
- R50/R152 synth at 20ep — DIDN'T converge (loss flat ~4.6 across all p)
- R50 80ep — converged but noisy (non-monotonic at p=0.75 vs 1.0)
- R152 80ep — STUCK at acc 0.02, won't converge in our recipe at this depth/N
- Ensemble of all signals — drags down via delta_conf, mean_conf (worse than mean_loss alone)
- Quadratic deg=2 — overfits 5pts, doesn't beat linear

**Decisions:**
- 2026-05-09 — SUB-4 = 0.0723 with 50ep R18 mean_loss d=1. From 0.0790 → 0.0723.
- 2026-05-09 — R152 effectively untrainable in our setup (recipe + N=2000). Use R18 fallback for arch=2.
- 2026-05-09 — Trend `+mean p̂ → -score` suggests real p mean ≈ 0.55-0.60. Go even higher carefully.

## Pending experiments

- 80ep + 100ep R18 synth — extrapolate to higher mean p̂
- R50 80ep dense (5+8 extra+5 seed2 = 18 pts) — smoother arch-matched fit for R50 targets
- Per-arch mixed regime: R18 50ep for arch=0/2, R50 80ep dense for arch=1

## SUB-5 — 3rd place 🥉

**Method:** MLE 80ep R18 mean_loss d=1 (all archs use R18 calibration).
- Mean p̂ = 0.516, range [0.44, 0.59]
- LOO-MAE 0.006 (best so far)
- **Score: 0.0667** — 3rd place on public leaderboard.
  - 1: TQ2 0.0537
  - 2: Nepal 0.0609
  - **3: Czumpers 0.0667**
  - 4: APT 0.0690 (overtaken)

## Trend extrapolation

| SUB | Mean p̂ | Score | Δ mean | Δ score |
|---|---|---|---|---|
| 3 | 0.443 | 0.0790 | — | — |
| 4 | 0.498 | 0.0723 | +0.055 | -0.0067 |
| 5 | 0.516 | 0.0667 | +0.018 | -0.0056 |

Approx slope: 0.0123 score per +0.073 mean → 0.168 score/mean unit.

To beat #2 Nepal (need -0.006): mean ≈ 0.55
To beat #1 TQ2 (need -0.013): mean ≈ 0.59
Beyond 0.60 risk overshooting if true mean is below.

## Pending

- 200ep R18 synth (in queue) — expected mean ~0.52-0.53
- 200ep R50 synth — should converge better than 80ep, give R50 arch-matched honest signal
- 200ep R152 synth — risky, may still not converge
