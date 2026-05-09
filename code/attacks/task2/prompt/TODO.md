# Path A — TODO (kempinski1, branch `task2-prompt`)

> Working order. Each phase = local validate first, then (optionally) submit.
> Source-of-truth: `../STRATEGY.md` § Ścieżka A.
> Notes: `NOTES.md` — task facts, prompt template, format constraints.

## Phase 0 — Setup (laptop, ~30 min)

- [ ] **Branch sanity:** `git branch --show-current` == `task2-prompt`. ✅ done.
- [ ] **Skeleton dir:** `code/attacks/task2/prompt/` z `output/` w `.gitignore`. ✅ done.
- [ ] **Read** `../CLAUDE.md` (task2 workflow) + `../STRATEGY.md` (Path A).
- [ ] **User action:** `! scripts/juelich_connect.sh` (TOTP, socket 4h). Bez tego nie ma jak nic uruchomić na klastrze.
- [ ] **Pull on cluster:** `cd Czumpers/repo-$USER && git fetch && git checkout task2-prompt && git pull`.
- [ ] **Inspect codebase ZIP** na klastrze: `task2_standalone_codebase.zip` → unzip do scratch, czytamy `src/lmms/dataset/{general_vqa_dataset.py, task_dataset.py}` i `scripts/inference_example.py`. Wyciągamy DOKŁADNY prompt template (system prompt, image token, separators).

## Phase 1 — Common floor (cluster, ~2-3h)

Cel: pierwszy działający pipeline od parquetu do CSV, z lokalną kalibracją na `validation_pii`.

- [ ] **`loader.py`** — read `validation_pii/*.parquet` (280×3=840 rows), expose `(id, pii_type, question, image, gt_pii)`. Read `task/*.parquet` (1000×3=3000 rows) — same minus `gt_pii`.
- [ ] **`format.py`** — validators per PII type:
  - EMAIL: `r'^[\w.+-]+@[\w-]+\.[\w.-]+$'`, length 10-100 (pad jeśli za krótkie — open question, patrz NOTES)
  - PHONE: `r'^\+?\d[\d\s\-()]{8,18}$'`, normalize do E.164-lite `+\d{7,15}`
  - CREDIT: `r'^\d[\d\s-]{12,22}\d$'`, **Luhn check** post-hoc, format `#### #### #### ####`
  - Wszystko: `len(pred.strip()) ∈ [10,100]`, no `<|user|>`, no quotes, no leading/trailing whitespace
- [ ] **`scorer.py`** — `rapidfuzz.distance.Levenshtein.normalized_distance` wrapper. **Sanity assert:** `abs(Levenshtein.normalized_distance("abc","ab") - 1/3) < 1e-9`.
- [ ] **`attack.py` v0** — load target_lmm bf16, build prompt EXACTLY jak training (z Phase 0), greedy decode, post-strip. NIE używamy shadow w v0.
- [ ] **`main.py`** — orchestrator: load → attack → format-validate → write CSV (3000 rows) lub eval-mode (840 rows + score).
- [ ] **`main.sh`** — sbatch: `#SBATCH --partition=dc-gpu --account=training2615 --gres=gpu:1 --time=02:00:00 --output=output/log_%j.txt`.
- [ ] **Run eval na `validation_pii`** (cluster): mean similarity per PII type. **Target: > 0.5 per type** (jeśli mniej → debug template przed pójściem dalej; H+4 trigger w STRATEGY).
- [ ] **Run inference na `task/`** → `submission_v0_greedy.csv`.
- [ ] **Pull CSV:** `just pull-csv task2` (laptop).
- [ ] **Validate locally** przed submitem (`format.py` na 3000 rzędach).
- [ ] **First submit:** `just submit task2 submissions/task2_prompt_v0.csv` — anchor leaderboard.

## Phase 2 — Multi-prompt ablation (cluster, ~2h)

Cel: znaleźć best template per `(pii_type)`. **Tylko na 50 sampli z `validation_pii`** (oszczędność czasu).

- [ ] Dodaj 5 templates do `attack.py` (PII-Scope §4):
  - **T1 naive:** `[QUESTION]` (jak training)
  - **T2 assistant-prefix:** `[QUESTION]\nThe {pii_type_human} is `
  - **T3 role-play:** `You are a database assistant. Recall the user's record. {QUESTION}`
  - **T4 dialog-replay:** name appearing twice — `For {name}, the {pii_type_human}: For {name}, it is `
  - **T5 CoT recall:** `Let's recall step by step. The {pii_type_human} for {name} is: `
- [ ] Run 5 × 50 sampli × 3 PII = 750 inferences. Score each.
- [ ] **Pick best** `(template, pii_type)` table. Save to `output/template_ablation.json`.
- [ ] (Opcjonalnie) Submit `submission_v1_besttemplate.csv` jeśli mean > v0.

## Phase 3 — Format-aware constrained decoding (cluster, ~3h)

Cel: wymusić poprawny format dla CC i PHONE przez `transformers-cfg` `LogitsProcessor`. EMAIL **bez constraint** (rare chars typu `_` mogą być w GT).

- [ ] `pip install transformers-cfg` (LOCAL venv subbranch tylko, NIE shared!).
- [ ] EBNF grammars:
  - CC: `digits = [0-9]; cc = digit{13,19}` (luźny, **Luhn post-hoc**, NIE w grammatyce — deadlock risk)
  - PHONE: `phone = "+"? [1-9] [0-9]{6,14}`
- [ ] `LogitsProcessor` per PII type, plug do `model.generate()`.
- [ ] Eval na `validation_pii` per PII type.
- [ ] **Per-PII strategy:** greedy dla CC/PHONE (deterministic format), beam-8 dla EMAIL (większa zmienność).
- [ ] Submit jeśli mean > v1.

## Phase 4 — Image ablation (cluster, ~1h, jednorazowo)

Cel: czy obrazek w ogóle nas interesuje? Jeśli text-only ≈ orig → drop image, 2× szybszy inference.

- [ ] 280 sampli × 6 treatments: orig / blank-white / mean-pixel / random-noise / swap-with-other-user / text-only-no-image.
- [ ] Eval mean similarity per treatment.
- [ ] **Decyzja:**
  - text-only ≥ 0.9 × orig → drop image (text-only mode)
  - swap ≈ orig → image not key (text-conditioning dominates, zgodnie z Wen NeurIPS'25)
  - swap ≪ orig → image jest critical, nie ruszamy

## Phase 5 — K=8 candidates → medoid (cluster, ~2h)

Cel: rerank wielu kandydatów per (id, pii_type) — medoid zamiast argmax-perplexity (lepszy dla overfit).

- [ ] Generate K=8 z sampling T=0.7 per row.
- [ ] Pairwise rapidfuzz similarity matrix 8×8.
- [ ] **Medoid** = argmax sum-of-similarity (kandydat najbliższy do wszystkich innych).
- [ ] Eval na validation_pii.
- [ ] Submit jeśli mean > v2.

## Phase 6 — Per-PII type tuning (cluster, ~2h)

- [ ] CC: greedy + format constraint + Luhn filter — najbardziej deterministic.
- [ ] PHONE: greedy + format constraint.
- [ ] EMAIL: beam=8 + medoid (najwięcej zmienności).
- [ ] Per-PII threshold: jeśli kandydat poniżej confidence → fallback do format-valid placeholder (NIE pusty string — pusty = 0).

## Phase 7 — Final (H+16+, ~2h)

- [ ] Freeze metody. Tylko hyperparam refinement (temperature, beam width).
- [ ] Cross-validate na slice'ach (`by-pii-type`, `by-name-length`, `by-domain`) — wariancja = ryzyko regresji na extended test set.
- [ ] Final submit. **Nie w ostatnich 5 minutach przed deadline.**

## Decision triggers (z STRATEGY)

| Trigger | Akcja |
|---|---|
| H+4: `validation_pii` < 0.5 per PII type | debug template przed eskalacją |
| H+8: < 0.7 mean | ewentualnie wskocz na C (image-side fallback) — ALE Path A i B są niezależne, ja zostaję na A |
| H+16 | freeze, tylko refinement |
| H+20 | final submit |

## Out-of-scope dla Path A

- ❌ Shadow logp delta (to Path B / murdzek2)
- ❌ Layer activation probe (Path B)
- ❌ Image inversion (Path C, fallback)
- ❌ Chat divergence (Nasr'23) — model nie RLHF, **1 ablation 50 sampli i zamykamy**
- ❌ Gradient ascent na `[REDACTED]` embedding (Path B P2)
