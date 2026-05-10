# Task 2 — Three ideas that lifted us 0.388 → 0.400 (pitch notes)

> Public LB confirms each lift was ours, not a teammate's, via a **verify pattern**:
> we scrape the leaderboard *before* and *after every submit*, with 5-min cooldown between.
> If `BEFORE = previous AFTER` and `AFTER > BEFORE` → that submit caused the change.
> Tracker log: `/tmp/task2_tracker.log`.

```
23:44:50 BEFORE_SUBMIT  QUESTION_REPEAT     czumpers=0.3875618              ← session start
23:45:37 AFTER_SUBMIT   QUESTION_REPEAT     czumpers=0.3956451  Δ=+0.008083 ✅ Idea #1
23:50:48 BEFORE_SUBMIT  SMART_ENSEMBLE_QR   czumpers=0.3956451
23:51:29 AFTER_SUBMIT   SMART_ENSEMBLE_QR   czumpers=0.3988615  Δ=+0.003216 ✅ Idea #2
00:04:05 BEFORE_SUBMIT  SMART_V2            czumpers=0.3988615
00:04:46 AFTER_SUBMIT   SMART_V2            czumpers=0.4002148  Δ=+0.001353 ✅ Idea #3 — przebicie 0.4
```

Total session lift: **+0.012653**.

---

## Idea #1 — Pull `question_repeat` CSV (Δ +0.008083)

**Observation**: Server scores against **raw PII** (e.g. `john.doe@peters.com`), not full sentences. Different prompts produce different memorization recall.

**What we did**: Teammate kempinski1 had been benchmarking 5 prompt strategies on `validation_pii` (n=840). The `question_repeat` strategy — re-asking the LMM the same question twice in the prompt — scored **0.4008 on val, the best of all variants**. Job `14740626` finished generating predictions on the full 3000-sample test set at 23:21Z — we pulled the CSV the moment it landed.

**Why it works**: Re-stating the question in the prompt acts like a memory probe. The model's first attention pass has already retrieved the relevant tokens; the repeat anchors the decoder to emit the recalled span as the answer rather than hedging.

**Pre-submit pipeline**:
```bash
# 1. Pull from cluster (no edit — kempinski1's raw output already in raw-PII form)
scripts/juelich_exec.sh "cat .../submission_v0_blank_question_repeat_*.csv" \
    > submissions/task2_v0_blank_question_repeat_pull.csv

# 2. Optional safety pass (idempotent on already-extracted CSVs — changed=0 here)
python3 code/attacks/task2/extract_pii_from_sentences.py <in> <out>

# 3. Verify-pattern submit
bash scripts/submit_and_verify_task2.sh QR submissions/task2_v0_blank_question_repeat_pull.csv
```

**Slide bullet**: "Pulled `question_repeat` CSV — different prompt → different memorized facts surfaced. **+0.008**."

---

## Idea #2 — Smart ensemble with dummy fallback (Δ +0.003216)

**Observation**: After Idea #1 we noticed `question_repeat` outputs `0000 0000 0000 0000` for **986 / 1000 CREDIT samples** — when the card image is scrubbed, the model has no visual anchor and emits a placeholder.

**What we did** (`code/attacks/task2/smart_ensemble.py`):

For each `(id, pii_type)` row, with QR as primary and `v2_204928` baseline as fallback + 4 extras:
1. If QR's value is **well-formed** (regex_fit ≥ 2) **AND not dummy** → keep it.
2. Otherwise, walk strict candidates (fit≥2, non-dummy) from baseline + extras → take the first.
3. Otherwise, walk loose candidates (fit≥1) → take the first.
4. Last resort: baseline value.

**Dummy detection rules**:
- `CREDIT` — 16 digits where the digit-set has size ≤ 2 (`0000…`, `1111…`).
- `EMAIL` — throwaway domains (`example.com`, `test.com`, …) or single-token locals (`test@anything`).
- `PHONE` — 8+ digits with ≤ 2 unique digits, or `1234567890` family.

**Effect**: 4 unique CSVs picked from the fallback path for the 986 dummied CREDIT — most landed on the baseline's actual memorized 16-digit numbers. EMAIL/PHONE: still ~99% from QR (almost never dummy there).

**Slide bullet**: "QR was placeholder on 986/1000 CREDIT samples → dummy-detection + fallback to baseline lift CREDIT only. **+0.003**."

---

## Idea #3 — Per-PII routing + CREDIT non-dummy voting (Δ +0.001353 → break 0.4)

**Observation**: Idea #2 picks the *first* non-dummy fallback candidate for CREDIT — which is arbitrary, since multiple sources may have equally well-formed but different predictions for the same `(id, CREDIT)`.

**What we did** (`code/attacks/task2/smart_ensemble_v2.py`):

Split the algorithm by `pii_type`:

- **EMAIL & PHONE** → `qr_or_base` (same logic as Idea #2 — QR almost always wins here).
- **CREDIT** → `non_dummy_vote`:
  1. Collect every non-dummy + fit≥1 candidate from baseline + extras (skip QR unless non-dummy).
  2. **Normalize** each to digit-only form (strip whitespace/dashes).
  3. Count how many sources voted for each normalized digit string.
  4. Pick the **highest-voted** value; tie-break = source order (baseline first).

**Why per-PII matters**: EMAIL/PHONE are *memorization-dominated* — one good source wins. CREDIT is *noise-dominated* — averaging over agreeing sources removes hallucination tail.

**Why voting matters for CREDIT**: when `v2_204928` and `v0_original_direct_probe` both emit `1429 3116 3052 4685`, that's a 2-source agreement → high confidence. Lone outlier "1234 5678 9012 3456" loses the vote.

**Final source pick distribution** (3000 rows):
```
question_repeat:                          2000   ← all EMAIL + PHONE
v2_204928 (baseline) — voting winner:      978   ← almost all CREDIT
v0_original_direct_probe — minority:        12
v0_original_192543      — minority:          7
hybrid_pii_only         — minority:          3
```

**Slide bullet**: "Routed by PII type — EMAIL/PHONE from QR, CREDIT from non-dummy voting across sources. **+0.001 → broke 0.4**."

---

## Verify pattern (the meta-idea — also worth a slide)

We never trusted "score went up after we submitted" without proving the change was ours. Each submit is wrapped:

```bash
before=$(scrape_leaderboard "Czumpers" "27_p4ms")
python3 scripts/submit.py task2 "$CSV"
sleep 40                              # server eval is async
after=$(scrape_leaderboard "Czumpers" "27_p4ms")
delta=$(after - before)
log "BEFORE=$before AFTER=$after Δ=$delta CSV=$CSV md5=$md5"
```

Without this, with 4 teammates submitting concurrently, you can't tell whether `0.387 → 0.395` was your CSV or a colleague's.

**Slide bullet**: "Every submit logs scrape-before / scrape-after — proves attribution and avoids racing teammates."

---

## Cluster job context (background — useful in Q&A)

- 7 source CSVs total. 5 came from teammate kempinski1's `repo-kempinski1-cd/code/attacks/task2/prompt/` — different prompt strategies (baseline, direct_probe, role_play_dba, per_pii_route, oneshot_demo, cd_decoding, question_repeat).
- Validation eval (`run_log.csv`, n=840 from validation_pii):
  - baseline_blank: 0.316
  - direct_probe: 0.397
  - role_play_dba: 0.399
  - per_pii_route: 0.393
  - oneshot_demo: 0.349
  - cd_decoding: 0.298
  - **question_repeat: 0.401 ← best**
- Predicted task/ score for QR was ~0.43; actual lift on test was 0.396 → 0.400 with all 3 ideas combined. Lower than predicted because QR's CREDIT predictions degenerated on the scrubbed test set (ideas #2/#3 partially recovered this).

---

## What plateaued us at 0.400 (open in Q&A if asked)

10 further ensemble variants (with extra source `baseline_180723`, equal-weight majority vote, qr-3x-weighted vote, different CREDIT routing) all returned `delta = 0`. Reason: our 7 sources are highly correlated for non-dummy CREDIT predictions, so further ensembling doesn't add new signal. Breaking through 0.4xx needs **new prompt strategies on the full test set** (kempinski1's `role_play_dba` and `per_pii_route` exist only on val so far).
