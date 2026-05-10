# Task 3 (BEST3) — full solution logic in plain English

This is a **linear** walkthrough from raw data to the submission file: what you compute, in what order, and why the pieces fit together.

---

## 1. What you are solving

The organizer gives you short texts. Some are **watermarked** (generated under a watermarking scheme); some are **clean**. For each test text you must output a number in \((0,1)\): higher means stronger belief the text is watermarked. They evaluate you at **TPR when FPR on clean texts is 1%**. In practice that is almost entirely a **ranking** problem: you need the watermarked texts to sort **above** most clean texts once the threshold is set to allow only 1% false alarms on clean. Absolute values of scores matter far less than **order** among samples.

The training data is **multi-scheme** in spirit (several watermark families are possible). A detector tuned to only one algorithm is brittle, so the approach stacks **many weak, diverse signals** and lets a simple classifier learn weights.

---

## 2. What raw data you start from

You have a **labeled** set (on the order of a few hundred texts: traditionally train plus validation merged, ~540 samples) with binary labels: clean vs watermarked. You also have an **unlabeled** test set of **2250** texts with ids. **Only** the labeled data is used to fit the scoring models. The test set is only fed forward to produce scores.

For every feature pipeline you use **one fixed ordering**: all labeled texts first (in a consistent train+val order), then all test texts. That way row \(1\) to row \(540\) in the feature matrix always corresponds to supervised data and the remaining rows always correspond to test predictions.

---

## 3. The shared “general” feature layer

Before any split into heads, each text is turned into a **wide vector of numbers** by **many independent extractors**. This is **not** “six methods” in the sense of six extractors: there are **several** families, each emitting **many** scalars. Together they answer: “How statistically odd is this text under different language models and geometry?” without assuming a single watermark API.

Conceptually the block includes: cheap statistics under a **small GPT-2–class** causal LM (log-probs, rank buckets, n-gram diversity, burstiness, compression-style signals); several **Binoculars-style** observer–performer pairs (e.g. **GPT-2** and **Pythia** at different sizes) comparing perplexity / log-prob behavior; **aggregate** sequence scores under **OLMo-7B-Instruct** and **OLMo-1B-Instruct** (mean log-prob, spread, perplexity); **token-level curvature** under a causal LM (how surprising each realized token is relative to the predicted softmax); **semantic** features from sentence embeddings (e.g. similarity of adjacent sentences, coarse histograms in embedding space—useful if a watermark acts in meaning space); and **optionally** train-derived token priors (green-list–style preferences estimated from labels—useful for unigram-style schemes but partly “looking at the answers” on the training fold).

Both final scoring heads must see the **exact same** general layer (same caches / same extraction run). If head A and head B used different baselines, the two outputs would not be comparable for fusion.

---

## 4. Cross-model discrepancy features

On top of the general layer you add a **small fixed set of derived features**: differences and one ratio of log-prob / perplexity summaries between **OLMo-7B-Instruct** and **GPT-2 medium**, between **OLMo-7B** and **Pythia** at two scales, between two **Pythia** sizes, and a perplexity ratio. These are **algebraic combinations** of columns already computed—no new neural forward passes. Intuition: **relative** strangeness across model families often separates watermarked and clean better than a single absolute perplexity.

---

## 5. Where the two heads diverge: Head A vs Head B

Everything above is **identical** for both runs. You then run **two separate training jobs** that differ only in **one extra specialist block** concatenated to the vector.

### Head A (distribution / entropy under OLMo-7B-Instruct)

You run **OLMo-2-1124-7B-Instruct** in teacher-forcing mode. At each position you have the **full next-token softmax**, not only the log-prob of the true token. Per step you derive **entropy**, **rank** of the realized token, gap to the argmax log-prob, margin between top-1 and top-2 log-probs, and a uniform-distribution proximity measure. You then **aggregate** over the sequence: moments and percentiles of entropy, rank statistics, fractions of very likely tokens, and counts of **runs** of consecutive low-entropy steps (bursts). Watermarks often **sharpen** or **regularize** the conditional distribution; that shows up in entropy and rank paths even when average log-prob alone barely moves. Short or malformed texts zero out this block while the general layer may still carry signal.

### Head B (multi-variant green-list z-tests, Kirchenbauer-style)

You **do not** run a large LM forward here—only **tokenizers** and loops over token ids. For several combinations of tokenizer (e.g. **Llama-2–style**, **OLMo-7B–style**, **GPT-2**), green fraction \(\gamma \in \{0.25, 0.5\}\), and context width for the hash seed (previous token only vs a wider XOR-style mix of previous tokens), you mark each token as pseudo-“green” or not via a **fast hash** and count how green the sequence is versus the \(\gamma\) null. Each configuration yields a **z-score**; you also store aggregates such as max absolute z, sum of absolute z, and how many configurations exceed a significance-style cutoff. Even if the competition watermark is **not** exactly this hash, these numbers can still correlate with regularity; **logistic regression** learns how much to trust them. Very short texts may zero this block.

---

## 6. Single vector → single linear scoring model (done twice)

For Head A you concatenate **[general features | cross-model discrepancies | Head-A specialist block]**. For Head B you concatenate **[general features | cross-model discrepancies | Head-B specialist block]**. Infinities and missing values become safe zeros.

On the ~540 labeled rows you **standardize** features (per training fold or globally as designed), fit a **strongly regularized logistic regression**, optionally measure out-of-fold **TPR at 1% FPR** for internal comparison (it may not match the public leaderboard). Then you **refit** on **all** labeled rows and apply the model to the 2250 test rows. You get **one probability per test text** per head; values are clipped slightly away from exactly 0 or 1.

---

## 7. Rank fusion on the test set

You now have **two** score vectors on the same 2250 examples. You **do not** average them directly. For each head separately you convert scores to **ranks** across the test batch (ties averaged as usual). Then for each text you take the **mean of the two ranks** (equal weights by default). You **linearly rescale** that combined rank score to a submission-friendly interval and clip again. Intuition: the two heads may have very different **calibration**; rank averaging aligns them with the competition metric, which is effectively about **ordering**.

---

## 8. Final artifact

You write a CSV: **id, score** for 2250 lines. The file name (e.g. a “BEST3” label) is just the team’s chosen alias for the same pipeline: dual head + rank fusion on shared general features.

---

## 9. Why this structure

- **Shared wide layer**: captures many watermark-agnostic and scheme-flavored hints in one place.
- **Head A**: exploits **full conditional distributions** under a strong **OLMo-7B-Instruct** model—orthogonal to “PPL-only” baselines from the same family.
- **Head B**: adds **list-style** statistics cheaply, complementary to LM forward passes.
- **Rank fusion**: matches the **FPR-fixed** evaluation without hand-tuning scale between heads.

English counterpart of the Polish walkthrough: see [task3_logika_rozwiazania.md](task3_logika_rozwiazania.md) for the same story with the hackathon step numbering and naming used in-repo.
