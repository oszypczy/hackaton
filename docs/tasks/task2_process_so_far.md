# Task 2 — Process log (autonomous session, started 2026-05-10 ~22:50 UTC)

> Handoff log so anyone can resume. Newest at the bottom.
> Goal: defend `task2 = 0.387562 (#1)` and push higher via re-submitting unique CSVs we have lying around.

## Score history

| Time UTC | CSV | md5 | Result |
|---|---|---|---|
| 2026-05-09 18:36 | task2_shadow_baseline_20260509_203603.csv (full sentences) | e5d3c92 | FAILED (sentences vs raw PII) |
| 2026-05-09 18:50 | task2_shadow_pii_only_v2_204928.csv (raw PII) | 91de34b4 | success — leaderboard 0.381195 |
| 2026-05-09 19:18 | task2_shadow_hybrid_pii_only_205037.csv | 63f6bc39 | submitted, no improvement (leaderboard 0.381195) |
| 2026-05-09 ??:?? | (kempinski1 something) | ? | bumped leaderboard to 0.387562 |
| 2026-05-09 22:51 | task2_shadow_pii_only_v2_204928.csv (RE-submit) | 91de34b4 | success, no improvement (leaderboard 0.387562) |

## Key learnings

1. **Server scores against RAW PII** (not full sentences). Always pass through `extract_pii()` regex.
2. **Identical md5s discovered**:
   - `task2_shadow_pii_only_v2_204928.csv` == `extract_pii(task2_shadow_baseline_20260509_203603.csv)` (md5 91de34b4)
   - `task2_shadow_hybrid_pii_only_205037.csv` == `extract_pii(task2_shadow_hybrid_20260509_2050.csv)` (md5 63f6bc39)
3. **Cooldown observed:** 5 min on success, but server replied 429 with `cooldown=35` once — appears server holds extra time after rapid retries. Queue uses 305s between submits + 70s initial buffer.

## Cluster notes (from kempinski1 NOTES.md)

- Validation set scores (n=840):
  - Baseline blank: 0.316
  - direct_probe blank: 0.397 (+0.082)
  - role_play_dba blank: 0.399 (+0.084)
- Predict task/ score for direct_probe: ~0.43 (validation +0.031 task delta)
- kempinski1 has 2 active jobs (`14740302`, `14740626`) — DO NOT touch his runs.

## NEW unique sources discovered

Pulled from `repo-kempinski1-cd/code/attacks/task2/prompt/output/` (not in our local submissions before):
- `submission_v0_original_192543.csv` (md5 b1d90ab5) — image_mode=original baseline
- `submission_v0_original_direct_probe.csv` (md5 097045cc) — direct_probe variant

After `extract_pii()`:
- `task2_v0_original_192543_extracted.csv` (md5 f34a7647)
- `task2_v0_original_direct_probe_extracted.csv` (md5 490883ab)

## Ensembles built locally

| File | md5 | Sources | fit=2 | fit=1 | fit=0 |
|---|---|---|---|---|---|
| `task2_ensemble_4src.csv` | 00acb072 | v2_204928 + old_fixed_extracted + hybrid_pii_only + 204817 | 2145 | 388 | 467 |
| `task2_ensemble_6src.csv` | 915bfeda | + v0_original_192543_extracted + v0_original_direct_probe_extracted | **2745** | 248 | **7** |

Higher fit = more samples that look like well-formed PII per regex. ensemble_6src is dramatically better.

## Submission queue (in flight)

Started 23:01:38, finishing ~23:34:
1. `task2_ensemble_6src.csv` (best expected)
2. `task2_v0_original_192543_extracted.csv`
3. `task2_v0_original_direct_probe_extracted.csv`
4. `task2_ensemble_4src.csv`
5. `task2_old_fixed_extracted.csv`
6. `task2_shadow_pii_only_204817.csv`
7. `task2_shadow_hybrid_20260509_2050.csv` (full sentences — expected worst)

Verify pattern: BEFORE → submit → 40s wait → AFTER scrape. Logs to `/tmp/task2_tracker.log`.

## Files added this session

- `code/attacks/task2/extract_pii_from_sentences.py` — apply extract_pii regex to a sentence-form CSV
- `code/attacks/task2/ensemble.py` — regex-fitness voting across pii-only CSVs
- `scripts/submit_and_verify_task2.sh` — single submit with leaderboard verify
- `scripts/task2_submit_queue.sh` — sequential submit queue with 5min cooldown

## Open questions / next steps

- If queue exposes a new high score >0.387562 → identify which CSV did it (verify pattern guarantees attribution).
- If queue ends without improvement → kempinski1's active jobs may produce direct_probe predictions on full test set; pull and submit later.
- 213 non-redacted samples (visible PII) — kempinski1 already accounted for these in some form; not actioned this session.

## Round 1 results (2026-05-09 23:02 - 23:38 UTC)

7/7 zsubmitowane, **wszystkie no_improvement** (score zostaje 0.387562).

| # | Label | md5 | Result |
|---|---|---|---|
| 1 | ENSEMBLE_6SRC_regex_voting_with_kempinski_v0_original | 915bfeda | NO_IMPROVEMENT (0.387562) |
| 2 | V0_ORIGINAL_192543_kempinski_pulled | f34a7647 | NO_IMPROVEMENT |
| 3 | V0_ORIGINAL_DIRECT_PROBE_kempinski_pulled | 490883ab | NO_IMPROVEMENT |
| 4 | ENSEMBLE_4SRC_regex_voting | 00acb072 | NO_IMPROVEMENT |
| 5 | OLD_FIXED_EXTRACTED_pii_only | d8799014 | NO_IMPROVEMENT |
| 6 | PII_ONLY_204817_mixed_format | 4da513a3 | NO_IMPROVEMENT |
| 7 | HYBRID_FULL_SENTENCES_20260509_2050 | e7ede397 | NO_IMPROVEMENT |

Wszystkie nie pobiły aktualnego 0.387562. Ten score został ustawiony przez kempinski1 w różnym CSV (nie wiemy którym, między 19:18 a 22:22).

## Round 2 (started 23:41 UTC)

Pulled NEW CSV from cluster: `submission_v0_blank_question_repeat_20260510_002838.csv` (kempinski1 wygenerował 23:21Z).
- Eval na validation_pii (n=20 next n=840) dało 0.4008 — najlepszy wynik kolegi
- Predict task/ score: ~0.43

Built 2 new ensembles incorporating it:
- `ensemble_7src` (md5 dee691f9) — straight regex-fitness vote, 7 sources. fit=2: 3000/3000.
- `smart_ensemble_qr` (md5 2246649d) — question_repeat preferred + dummy detection (CREDIT all-zeros, EMAIL@example.com, PHONE 1234567890). Found **986/1000 CREDIT** in question_repeat were placeholder dummies; replaced with v2_204928 baseline.

### Round 2 results

| Submission | md5 | Score | Δ |
|---|---|---|---|
| QUESTION_REPEAT_kempinski_BLANK | f4d42dd3 | **0.39565** | +0.008083 ✅ |
| SMART_ENSEMBLE_QR_with_dummy_fallback | 2246649d | **0.39886** | +0.003216 ✅ |
| ENSEMBLE_7SRC_with_question_repeat | dee691f9 | 0.39886 | 0.0 |

## Round 3 (started 00:01 UTC, 2026-05-10)

Built 4 per-PII routing variants (smart_ensemble_v2 etc.):

| Submission | md5 | Score | Δ |
|---|---|---|---|
| SMART_V2_per_pii_routing_credit_voting | 2943dd8f | **0.40021** | +0.001353 ✅ |
| ROUTING_qr_v2baseline_qr | d8a491df | 0.40021 | 0.0 |
| ROUTING_qr_directprobe_qr | 1e096782 | 0.40021 | 0.0 |
| ROUTING_v0o_directprobe_qr | dd3b9052 | 0.40021 | 0.0 |

## Round 4 (started 00:24 UTC) — pulled new source `baseline_180723`

NEW source from kempinski1: `submission_v0_20260509_180723.csv` (3004 data rows, dedup'd to 3000, extracted to md5 87e53fee — distinct from v2_204928 — different baseline blank generation run).

| Submission | md5 | Score | Δ |
|---|---|---|---|
| SMART_V3_with_baseline_180723_extra | cf86c2ca | 0.40021 | 0.0 |
| SMART_V2_baseline_180723_as_BASE | 5f680003 | 0.40021 | 0.0 |
| ROUTING_qr_baseline180723_qr | f6440010 | 0.40021 | 0.0 |
| ROUTING_v0o_baseline180723_qr | bd3bf41d | 0.40021 | 0.0 |

## Round 5 (started 00:47 UTC) — majority-vote ensembles

| Submission | md5 | Score | Δ |
|---|---|---|---|
| MAJORITY_VOTE_qr_3x_weighted | c8c5fb2d | 0.40021 | 0.0 |
| MAJORITY_VOTE_all_equal | 3eb0c802 | 0.40021 | 0.0 |

## Final state (2026-05-10 00:58 UTC)

- **Score: 0.40021483** (#5, leader 0.4728 — APT)
- **Total improvement this session: +0.012653** (from 0.387562)
- **Best CSV: `task2_smart_ensemble_v2.csv`** — per-PII routing (EMAIL/PHONE = question_repeat preferred, CREDIT = non-dummy voting across base+extras)
- **Plateau**: 10 consecutive no_improvement after smart_v2 — exhausted local source combinations.

### Why we stalled

- `question_repeat` (best single source) is dummy on 98.6% of CREDIT samples. Smart_v2's CREDIT non-dummy voting fixed most of these.
- All other sources (baseline_v2, baseline_180723, v0_original, direct_probe) produce **highly correlated** CREDIT predictions for the non-dummy cases, so ensembling among them yields no diversity gain.
- EMAIL/PHONE: question_repeat dominates — no source consistently beats it.
- Need NEW signal (e.g., role_play_dba on full test set, per_pii_route on test set). kempinski1's job 14740302 has been running 3h+; if it produces a CSV, more lift possible.

### Files added this session

Code:
- `code/attacks/task2/extract_pii_from_sentences.py` — apply extract_pii regex (with newline strip)
- `code/attacks/task2/ensemble.py` — regex-fitness voting
- `code/attacks/task2/smart_ensemble.py` — question_repeat preferred + dummy detection
- `code/attacks/task2/smart_ensemble_v2.py` — per-PII routing + CREDIT non-dummy voting
- `code/attacks/task2/per_pii_routing.py` — pure per-PII source pick
- `code/attacks/task2/majority_vote.py` — per-row majority voting

Scripts:
- `scripts/submit_and_verify_task2.sh`
- `scripts/task2_submit_queue.sh`
