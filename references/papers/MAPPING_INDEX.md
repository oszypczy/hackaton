# Papers — lean index (read first)

**Use:** route to ONE paper, then open `MAPPING.md` for grep terms + key sections, then `txt/NN_*.txt` for surgical Read.

Format: `NN — Title (Venue) [SprintML?] — Use for: …`

## 1. Required (organizers' email + task PDFs)
- **01** — Carlini 2023, *Extracting Training Data from DMs* (USENIX 2023). Use for: Task 2 PII baseline (generative-model extraction analogue); Carlini 2023 ref [2] from task 2 PDF.
- **02** — Maini, Jia, Papernot, Dziedzic 2024, *LLM Dataset Inference* (NeurIPS 2024) [SprintML]. Use for: Task 1 DUCI primary; SprintML eval template.
- **03** — Zawalski 2025, *CoDeC: Data Contamination via ICL* (NeurIPS Workshop 2025). Use for: "is this benchmark contaminated?"; novel angle for Task 1.
- **04** — Kirchenbauer 2023, *A Watermark for LLMs* (ICML 2023). Use for: Task 3 watermark detection primary.
- **05** — Carlini, Tramèr, Wallace et al. 2021, *Extracting Training Data from LLMs* (USENIX 2021). Use for: Task 2 PII — six MIA features; **context-dependency (Sec 6.5)**; insertion-frequency threshold (Sec 7); task 2 PDF ref [1]. Added 2026-05-09.

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

| Task | Primary | Hard mode | Tools |
|---|---|---|---|
| **1 — DUCI (ResNet MIA)** | 02 | 18, 12 | 20 |
| **2 — PII Extraction (LMM)** | 01, 10 | 09, 13, 15 | 25 |
| **3 — Watermark Detection** | 04 | 08 | 21, 22, 23, 24 |

## Next step
Once you've picked a paper number from above:
- Open `references/papers/MAPPING.md` for that paper's grep terms + key sections
- Then Grep on `references/papers/txt/NN_*.txt`
- Then offset-Read with `limit` for surgical content

`MAPPING.md` is the rich form (~3.1k words). `MAPPING_INDEX.md` (this file) is the lean router (~500 words).
