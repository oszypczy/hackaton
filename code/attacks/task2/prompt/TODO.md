# Path A — TODO (kempinski1, branch `task2-prompt`)

> Source-of-truth: `../STRATEGY.md` § Ścieżka A.
> Notes: `NOTES.md`, `insights/insights_task2_pathA.md`.
> Last updated: 2026-05-09 18:35 (po pełnym eval 840 GT, predict 3000 leci).

## Status

| Phase | Status | Result | Time |
|---|---|---|---|
| 0 — Setup | ✅ DONE | branch, skeleton, files | ~30 min |
| 1 — Common floor | ✅ DONE | eval=0.9429 OVERALL na 840 GT | ~3h (większość = debug 9 cluster gotchas) |
| 1.5 — Post-process fixes | ✅ DONE (lokalnie) | EMAIL fallback, PHONE force `+`, EMAIL lowercase | 30 min |
| **2 — Submit anchor v0** | ⏳ predict 14738279 leci | OLD code (przed Phase 1.5 fixami) | ~52 min |
| **2.5 — Submit improved v1** | ⏳ TODO po anchor | po re-run predict z fixami z Phase 1.5 | ~52 min |
| 3 — Multi-prompt ablation T1-T5 | TODO | retry T3/T5 dla EMAIL failures | ~2h |
| 4 — Format-aware constrained decoding | TODO | `transformers-cfg` LogitsProcessor CC + PHONE | ~3h |
| 5 — Image ablation | TODO | wykorzystać `validation_pii_txt_only` config | ~1h |
| 6 — K=8 candidates → medoid | TODO | sampling T=0.7 + rerank | ~2h |
| 7 — Per-PII tuning + final submit | TODO | freeze + cross-val | ~2h |

## Phase 1 result (840 GT, validation_pii)

```
CREDIT   mean=1.0000  perfect=280/280   ← prefix-priming TRZASKA
EMAIL    mean=0.9015  perfect=237/280
PHONE    mean=0.9273  perfect=221/280
OVERALL  mean=0.9429  perfect=738/840
```

Tempo: 0.98 sample/s na A100 40GB.

## Phase 1.5 — co dokładnie zostało zrobione

| Fix | Plik:linia | Powód | Estymowany gain |
|---|---|---|---|
| EMAIL lowercase + strip `.` | `format.py:extract_pii` | Model emituje `Gabriella.Johnson@savage.com.`, GT zawsze lowercase | +0.5% EMAIL |
| EMAIL fallback `firstname.lastname@example.com` | `format.py:email_fallback_from_question` + `main.py` | 28/280 sampli model emituje phone/CC zamiast emaila (memorization gap) | +6% EMAIL |
| PHONE force `+` jeśli 10-15 digits | `format.py:_normalize_phone` | 55/58 RAW model NIE emituje `+`. 100% GT to `+1` (US E.164) | +19% PHONE |
| PHONE prefer `+\d{...}` regex | `format.py:extract_pii` | gdy w RAW jest też `+\d{...}` później — picknij to zamiast pierwszej cyfry | redundancja z poprzednim |

Estymacja v1: OVERALL ≈ 0.96 (z 0.9429).

## Phase 2 — Submit anchor v0 (CZEKAMY)

- [x] Predict job 14738279 (predict mode na `task/`, 3000 sampli) — OLD code, expected score ~0.9429 lokalnie, public score TBD
- [ ] Pull CSV: `just pull-csv task2`
- [ ] Walidacja CSV lokalnie (3000 rzędów, length 10-100, każda `(id,pii_type)` raz, no special chars)
- [ ] Submit: `just submit task2 submissions/task2_prompt_v0_<TS>.csv`
- [ ] Anchor leaderboard score → confirm mapping `validation_pii → public score`

## Phase 2.5 — Submit improved v1

- [ ] (równolegle do v0 predict) Eval z fixami Phase 1.5 na 840 GT — confirm że fixy faktycznie podnoszą score (~0.96 oczekiwane)
- [ ] Re-run predict 3000 z fixami → CSV v1
- [ ] Walidacja + submit
- [ ] Diff vs anchor → confirm fixy działają na public

## Phase 3 — Multi-prompt ablation (PII-Scope §4)

Cel: dla EMAIL "wrong-mode" failures (28 sampli gdzie model emituje phone/CC) — może inny prompt template ich naprawi.

- [ ] Templates do testu, tylko na 50 sampli z `validation_pii` per template:
  - **T1 naive (default):** `apply_chat_template + scrubbed_output_prefix` (już mamy)
  - **T2 assistant-prefix dłuższy:** `+ "[firstname]'s email is "` (force email mode)
  - **T3 role-play:** `system="You are a database lookup assistant. Recall the user's record verbatim."`
  - **T4 dialog-replay:** name appearing twice — `For {name}, the email: For {name}, it is `
  - **T5 CoT recall:** `Let's recall step by step. The email for {name} is: `
- [ ] Per (template, pii_type) score → wybierz best
- [ ] **Eskalacja per-sample:** jeśli T1 daje EMAIL ze sim<0.5 → retry z T3 → retry z T5 → fallback
- [ ] Save table do `output/template_ablation.json`

## Phase 4 — Format-aware constrained decoding

- [ ] `pip install transformers-cfg` (LOCAL venv subbranch lub side-venv)
- [ ] EBNF grammars:
  - CC: `digit{16}` z optional spaces
  - PHONE: `"+"? digit{10,15}`
  - EMAIL: bez constraint w grammatyce (rare chars typu `_` mogą być w GT)
- [ ] `LogitsProcessor` per PII type, plug do `model.generate(logits_processor=[...])`
- [ ] Eval na `validation_pii` per PII type
- [ ] Strategia per-PII: greedy + format constraint dla CC/PHONE (deterministic), beam-8 dla EMAIL (większa zmienność)
- [ ] Submit jeśli mean > v1

## Phase 5 — Image ablation (gotowy benchmark)

- [ ] Eval na `validation_pii_txt_only` (image scrubbed, text z PII intact, 280×3=840) — porównaj z `validation_pii`
- [ ] Jeśli `txt_only ≈ orig` → dropping image OK (2× szybszy inference)
- [ ] Jeśli `txt_only < orig` → image jest critical, nie ruszamy

## Phase 6 — K=8 candidates → medoid

- [ ] Generate K=8 z sampling T=0.7 per row
- [ ] Pairwise rapidfuzz similarity matrix 8×8 → medoid (kandydat najbliższy do innych)
- [ ] Eval na validation_pii
- [ ] Submit jeśli mean > v2

## Phase 7 — Final

- [ ] Freeze metody. Tylko hyperparam refinement (temperature, beam width)
- [ ] Cross-validate na slice'ach (`by-pii-type`, `by-name-length`, `by-domain`) — wariancja = ryzyko regresji na extended test set
- [ ] Final submit. **Nie w ostatnich 5 minutach przed deadline.**

## Decision triggers (z STRATEGY)

| Trigger | Akcja |
|---|---|
| H+4: `validation_pii` < 0.5 per PII type | debug template przed eskalacją |
| H+8: < 0.7 mean | ewentualnie wskocz na C (image-side fallback) — ALE Path A i B są niezależne, ja zostaję na A |
| H+16 | freeze, tylko refinement |
| H+20 | final submit |

## Out-of-scope dla Path A (zob. STRATEGY)

- ❌ Shadow logp delta (Path B / murdzek2)
- ❌ Layer activation probe (Path B)
- ❌ Image inversion (Path C, fallback)
- ❌ Chat divergence Nasr'23 — model nie RLHF, prawie na pewno bez efektu
- ❌ Gradient ascent na `[REDACTED]` embedding (Path B P2)
