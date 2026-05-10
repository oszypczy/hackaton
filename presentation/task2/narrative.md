# Task 2 — PII extraction — 1-min narrative

> Method-first framing. Audience: hackathon participants + CISPA SprintML +
> 2 CS professors. Target language: English. Total spoken time ≈ 60 s
> (≈ 130 words at 130 wpm). Speaker = task 2 owner (kempinski1).

---

## 0:00 – 0:10  Opening (≈ 25 words)

> Task 2: a multimodal LLM was fine-tuned on synthetic user records —
> emails, credit cards, phones. Our job: extract that PII back from the
> model, given only the user's name and a scrubbed image.

[on screen: Slide enters — split-screen V1 vs V2 (`figures/01_split_screen.png`)]

## 0:10 – 0:30  Baseline (≈ 45 words)

> We started by taking the training-time answer template, cutting it
> right before the redacted PII, and pasting it as the *start* of the
> assistant's reply. The model thinks it has already begun answering and
> just continues. That gave us our baseline — and a fair comparison
> point for everything we tried next.

[gesture to LEFT half of slide — the assistant-prefix injection]

## 0:30 – 0:45  Strategy comparison (≈ 35 words)

> We then benchmarked five different prompt strategies on local
> validation — naive, role-play, system-override, completion format, and
> a direct memory probe. The direct probe won: ask the model what it
> remembers, output only that.

[gesture to RIGHT half of slide — direct probe + small "1 of 5 strategies" tag]

## 0:45 – 1:00  Fallbacks + result (≈ 30 words)

> When the model fails to emit a valid email or phone — wrong format,
> wrong type — we fall back: synthesize `firstname.lastname@example.com`
> from the question, force the E.164 plus prefix, normalize card
> formatting. End result: +0.07 over the baseline.

[gesture to FALLBACK panel + score delta on slide]

---

## Speaker cheat-sheet (≤ 5 bullets to memorize)

1. **Setup:** multimodal LLM fine-tuned on user records with PII; we get
   scrubbed images + names, must recover the PII.
2. **Baseline (V1):** prefix-priming — paste the training answer template
   into the model's reply so it auto-completes the redacted PII.
3. **Comparison:** 5 prompt strategies tested on validation; *direct
   probe* (ask the model what it remembers) won.
4. **Fallbacks:** when output isn't a valid email/phone — synthesize
   `firstname.lastname@example.com`, force `+` prefix, format-normalize.
5. **Closer:** +0.07 over baseline. EMAIL and PHONE recovery jumped;
   credit-card numbers stayed at the floor.

## Self-check timing

- 130 spoken words ≈ 60 s at 130 wpm. Buffer ≈ 5 s.
- If running long: drop the explicit list of 5 strategy names — just say
  "five prompt strategies, the direct memory probe won".
- If running short: add — *"per-PII the lift was concentrated on EMAIL
  and PHONE, where the model genuinely memorized the patterns."*

## Hard "do-not-say" list

- ❌ "OCR" / "we discovered the validation set wasn't scrubbed". The
  organizers told us validation has visible PII — it's not a discovery.
- ❌ Specific scoreboard ranks or comparisons to other teams.
- ❌ Numbers from teammate's downstream ensemble (0.40+) — that belongs
  to Paweł's slot.
- ❌ "Hallucination" without context — say "the model fills in plausible
  values when it lacks a visual anchor".
- ❌ Internal job IDs, slurm specifics, commit hashes, file paths.
- ❌ "We failed at first" / "we got lucky". The progression is presented
  as systematic comparison.
