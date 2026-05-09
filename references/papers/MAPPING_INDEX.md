# Papers — lean index (read first)

**Use:** route to ONE paper, then open `MAPPING.md` for grep terms + key sections, then `txt/NN_*.txt` for surgical Read.

Format: `NN — Title (Venue) [SprintML?] — Use for: …`

## 1. Required (organizers' email)
- **01** — Carlini 2023, *Extracting Training Data from DMs* (USENIX 2023) [**TASK2-PDF cited as ref [2]**]. Use for: Task 2 (PII) baseline (generative-model extraction analogue).
- **02** — Maini, Jia, Papernot, Dziedzic 2024, *LLM Dataset Inference* (NeurIPS 2024) [SprintML] [**TASK1-PDF cited**]. Use for: Task 1 (DUCI) — pipeline.
- **03** — Zawalski 2025, *CoDeC: Data Contamination via ICL* (NeurIPS Workshop 2025). Use for: novel angle for Task 1.
- **04** — Kirchenbauer 2023, *A Watermark for LLMs* (ICML 2023) [**TASK3-PDF cited**]. Use for: Task 3 primary.

## 1b. Task PDF references (cited by organizers in revealed task PDFs — high authority)
- **05** — Tong, Ye, Zarifzadeh, Shokri 2025, **"How Much of My Dataset Did You Use?" — Quantitative DUCI** (ICLR 2025) [**TASK1-PDF**]. Use for: **Task 1 PRIMARY** — proposes the exact DUCI method.
- **06** — Maini, Yaghini, Papernot 2021, *Dataset Inference: Ownership Resolution in ML* (ICLR 2021) [**TASK1-PDF**]. Use for: Task 1 — original DI paper, methodological foundation.
- **07** — Dziedzic, Duan, Kaleem, Dhawan, Guan, Cattan, Boenisch, Papernot 2022, *Dataset Inference for Self-Supervised Models* (NeurIPS 2022) [**SprintML — Dziedzic+Boenisch**] [**TASK1-PDF**]. Use for: Task 1 — DI for encoders; **organizers' own paper**.
- **11** — Carlini, Tramèr, Wallace et al. 2021, *Extracting Training Data from Large Language Models* (USENIX 2021) [**TASK2-PDF cited as ref [1]**]. Use for: Task 2 PII foundational — six MIA features; **context-dependency (Sec 6.5)**; insertion-frequency threshold (Sec 7). Added 2026-05-09.

## 2. Supplementary
- **08** — *Watermarks Provably Removable* (NeurIPS 2024). Use for: Task 3 watermark robustness.

## 3. Hidden (SprintML 2022–2026)
- **09** — Dubiński et al. 2025, **CDI** (CVPR 2025) [SprintML]. Use for: Challenge C **hard mode** — top hackathon target.
- **10** — Kowalczuk et al. 2025, *Privacy Attacks on IARs* (ICML 2025) [SprintML]. Use for: if hackathon uses IAR (VAR/MUSE/RAR).
- **12** — Hayes et al. 2025, *Strong MIAs on LLMs* (NeurIPS 2025) [SprintML co-auth]. Use for: Task 1 DUCI hard mode MIA.
- **13** — Hintersdorf et al. 2024, *NeMo* (NeurIPS 2024) [SprintML]. Use for: Task 2 PII — memorization neuron analysis.
- **15** — Carlini et al. 2024, **Stealing Part of a Production LM** (ICML 2024 Best Paper). Use for: Task 2 PII extraction from LLM.
- **18** — Carlini et al. 2022, *MIA From First Principles* (S&P 2022) [LiRA]. Use for: Task 1 DUCI hard mode; shadow model backbone.

## 4. Competition-ready tools
- **20** — Zhang et al. 2024, *Min-K%++* (ICLR 2025 spotlight). Use for: Challenge A — upgrade primary feature.
- **21** — Jovanović et al. 2024, **Watermark Stealing** (ICML 2024). Use for: Challenge B advanced — KGW spoof + scrub.
- **22** — Krishna et al. 2023, **DIPPER** (NeurIPS 2023). Use for: Challenge B B2 removal — needs 45 GB GPU.
- **23** — Sadasivan et al. 2024, *Recursive Paraphrasing* (ICLR 2024). Use for: Challenge B B2 max mode + theory.
- **24** — An et al. 2024, **WAVES** (ICML 2024). Use for: Challenge B image-side reference + benchmark protocol.
- **25** — Nasr et al. 2023, *Extracting Training Data from ChatGPT*. Use for: any LLM verbatim extraction.

## Mapping by task (Warsaw 2026)

| Task | Primary (TASK-PDF cited) | Methodological | Hard mode | Tools |
|---|---|---|---|---|
| **1 — DUCI (ResNet MIA)** | **05** (THE paper), 02, 06, 07 | — | 18, 12 | 20 |
| **2 — PII Extraction (LMM)** | **11** (Carlini'21), 01 (Carlini DM), 25 (Nasr divergence) | 10, 15 | 09, 13 | 25 |
| **3 — Watermark Detection** | **04** (Kirchenbauer baseline) | + Liu 2024 + Zhao 2024 (PDF cited, NOT in repo) | 08 | 21, 22, 23, 24 |

## Next step
Once you've picked a paper number from above:
- Open `references/papers/MAPPING.md` for that paper's grep terms + key sections
- Then Grep on `references/papers/txt/NN_*.txt`
- Then offset-Read with `limit` for surgical content

`MAPPING.md` is the rich form (~3.1k words). `MAPPING_INDEX.md` (this file) is the lean router (~500 words).
