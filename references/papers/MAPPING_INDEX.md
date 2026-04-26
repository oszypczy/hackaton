# Papers — lean index (read first)

**Use:** route to ONE paper, then open `MAPPING.md` for grep terms + key sections, then `txt/NN_*.txt` for surgical Read.

Format: `NN — Title (Venue) [SprintML?] — Use for: …`

## 1. Required (organizers' email)
- **01** — Carlini 2023, *Extracting Training Data from DMs* (USENIX 2023). Use for: Challenge C primary baseline.
- **02** — Maini, Jia, Papernot, Dziedzic 2024, *LLM Dataset Inference* (NeurIPS 2024) [SprintML]. Use for: Challenge A primary; SprintML eval template.
- **03** — Zawalski 2025, *CoDeC: Data Contamination via ICL* (NeurIPS Workshop 2025). Use for: "is this benchmark contaminated?"; novel angle for A.
- **04** — Kirchenbauer 2023, *A Watermark for LLMs* (ICML 2023). Use for: Challenge B primary.

## 2. Supplementary
- **05** — Survey: Model Inversion Attacks. Use for: taxonomy lookup only.
- **06** — Survey: Model Extraction Attacks. Use for: Challenge E orientation.
- **07** — FGSM/PGD Defense Strategies. Use for: adversarial-example baseline (low priority).
- **08** — *Watermarks Provably Removable* (NeurIPS 2024). Use for: Challenge B B2 (image-side removal).

## 3. Hidden (SprintML 2022–2026)
- **09** — Dubiński et al. 2025, **CDI** (CVPR 2025) [SprintML]. Use for: Challenge C **hard mode** — top hackathon target.
- **10** — Kowalczuk et al. 2025, *Privacy Attacks on IARs* (ICML 2025) [SprintML]. Use for: if hackathon uses IAR (VAR/MUSE/RAR).
- **11** — Dubiński et al. 2023, **B4B** (NeurIPS 2023) [SprintML]. Use for: Challenge E — likeliest SprintML defense template.
- **12** — Hayes et al. 2025, *Strong MIAs on LLMs* (NeurIPS 2025) [SprintML co-auth]. Use for: Challenge A hard mode.
- **13** — Hintersdorf et al. 2024, *NeMo* (NeurIPS 2024) [SprintML]. Use for: Challenge C alternate framing; mitigation via neuron ablation.
- **14** — Dziedzic et al. 2022, *Calibrated PoW* (ICLR 2022 Spotlight) [SprintML]. Use for: Challenge E if defense involves PoW.
- **15** — Carlini et al. 2024, **Stealing Part of a Production LM** (ICML 2024 Best Paper). Use for: Challenge E LLM variant; embedding extraction primer.
- **16** — Podhajski et al. 2024, *Efficient GNN Stealing* (ECAI 2024 / AAAI 2026 Oral) [SprintML]. Use for: GNN challenge.
- **17** — Xu, Boenisch, Dziedzic 2025, *ADAGE* [SprintML]. Use for: GNN challenge defense.
- **18** — Carlini et al. 2022, *MIA From First Principles* (S&P 2022) [LiRA]. Use for: Challenge A and D hard modes; MIA backbone.
- **19** — Orekondy et al. 2019, **Knockoff Nets** (CVPR 2019). Use for: Challenge E primary baseline.

## 4. Competition-ready tools
- **20** — Zhang et al. 2024, *Min-K%++* (ICLR 2025 spotlight). Use for: Challenge A — upgrade primary feature.
- **21** — Jovanović et al. 2024, **Watermark Stealing** (ICML 2024). Use for: Challenge B advanced — KGW spoof + scrub.
- **22** — Krishna et al. 2023, **DIPPER** (NeurIPS 2023). Use for: Challenge B B2 removal — needs 45 GB GPU.
- **23** — Sadasivan et al. 2024, *Recursive Paraphrasing* (ICLR 2024). Use for: Challenge B B2 max mode + theory.
- **24** — An et al. 2024, **WAVES** (ICML 2024). Use for: Challenge B image-side reference + benchmark protocol.
- **25** — Nasr et al. 2023, *Extracting Training Data from ChatGPT*. Use for: any LLM verbatim extraction.

## Mapping by challenge (1-line per challenge)

| Challenge | Primary | Hard mode | Tools |
|---|---|---|---|
| **A — LLM Dataset Inference** | 02 | 18, 12, 09 | 20 |
| **B — LLM Watermark** | 04 | 08 | 21, 22, 23, 24 |
| **C — Diffusion Memorization** | 01 | 09, 13, 10 | 24 |
| **D — Property Inference** | research 06 | 18, 12, 09 | — |
| **E — Model Stealing** | 19 | 11, 14, 15, 16/17 | 25 |

## Highest-probability templates (Dubiński + Boenisch + Dziedzic concentration)
**09, 10, 11, 16** — these four most likely templates for Warsaw 2026.

## Next step
Once you've picked a paper number from above:
- Open `references/papers/MAPPING.md` for that paper's grep terms + key sections
- Then Grep on `references/papers/txt/NN_*.txt`
- Then offset-Read with `limit` for surgical content

`MAPPING.md` is the rich form (~3.1k words). `MAPPING_INDEX.md` (this file) is the lean router (~500 words).
