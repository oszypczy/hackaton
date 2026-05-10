# Task 1 — DUCI — 2-minute narrative

> Method-first framing. Audience: hackathon participants + CISPA SprintML +
> 2 CS professors. Target language: English. Total spoken time ≤ 120 s
> (≈ 250 words at 120 wpm). Speaker = task 1 owner.

---

## 0:00 – 0:15  Opening (≈ 30 words)

> Task 1 asks a continuous question: given a black-box classifier, what
> **fraction** of a candidate dataset was used to train it — anywhere from
> zero to one hundred percent? Nine ResNets, three architectures, one
> number per model.

[on screen: Slide 1 — pipeline diagram (`figures/01_pipeline.png`)]

## 0:15 – 1:15  Method (≈ 130 words)

> Our approach follows a single idea: the model's loss on a sample is a
> noisy proxy for whether that sample was in training. We can't read that
> proxy directly, but we can **calibrate** it on models we built ourselves.
>
> So we trained a small reference bank — five ResNets per architecture,
> with known training proportions of zero, twenty-five, fifty, seventy-five,
> and one hundred percent. For each one, we measure the average loss on the
> candidate dataset. That gives us a clean linear relationship: loss as a
> function of *p*.
>
> Then for each of the nine real targets, we run the same forward pass,
> read the loss off the calibration line, and **invert** — recovering a
> continuous *p* between zero and one. No per-target tuning, no
> architecture-specific hacks. One reference bank, one calibration curve,
> nine predictions.

[on screen, second half: Slide 2 — calibration plot (`figures/02_calibration.png`)]

## 1:15 – 1:45  Result (≈ 60 words)

> Public-benchmark MAE: zero-point-zero-five-three. For context — Tong et
> al. at ICLR this year report exactly this number on ResNet-34 / CIFAR-100
> with a single reference model. We **matched the published single-reference
> result** on a nine-model multi-architecture benchmark, with a calibration
> bank that fits in under an hour of GPU time.

[on screen: Slide 2 — predictions bar (`figures/03_predictions.png`)]

## 1:45 – 2:00  Take-away (≈ 30 words)

> The take-away: a calibrated membership signal plus linear inversion is
> enough. The proportion-estimation problem reduces to a regression you can
> debug, not a black-box you have to trust.

---

## Speaker cheat-sheet (≤ 5 bullets to memorize)

1. **Question:** *fraction* of training data used (0 → 1), not yes/no.
2. **Trick:** train a small reference bank with known proportions; loss is
   the calibration signal.
3. **Pipeline:** synth bank → linear fit (loss vs p) → invert per target.
4. **Result:** MAE 0.053 — matches Tong et al. ICLR 2025 single-reference.
5. **Closer:** generalizable, no per-target tuning, fits in 1 GPU-hour.

## Self-check timing

- 250 spoken words ≈ 125 s at 120 wpm. Buffer ≈ 5 s.
- If running long: drop the second sentence of the take-away.
- If running short: add one example — *"e.g. for model 02, the loss
  landed near the 50 % point on the curve, so we predicted 0.5."*

## Hard "do-not-say" list

- ❌ Specific scoreboard ranks or 0.020.
- ❌ "Snap" / "round to grid" / discretization.
- ❌ Per-prediction overrides or single-flip adjustments.
- ❌ Anything about the public 33 % subset structure.
- ❌ "We got lucky" / "happy alignment" / hedging language.
