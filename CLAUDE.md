# CISPA European Cybersecurity & AI Hackathon Championship — Warsaw

## Context
3-person team preparing for a 24h hackathon (**2026-05-09/10**) at Warsaw University of Technology.
Organized by SprintML Lab (CISPA) — Adam Dziedzic & Franziska Boenisch.
AI tools are explicitly allowed during the competition.

## Reading order for Claude (token-aware)

When user asks about state, next steps, papers, or attack techniques — read in this order, stop as soon as you have enough:

1. **`TODO.md`** (root, ~5 KB) — current state of prep, what's blocked, what's next
2. **`references/papers/MAPPING.md`** (~12 KB) — paper lookup table; one entry per paper with arXiv, code repo, core idea, key result, "use for which challenge"
3. **`docs/practice/QUICKSTART.md`** (~3 KB) — "scenario → tool → numbers" cheat sheet; covers all 7 attack categories (text/image watermark, MIA, diffusion mem, model stealing, adversarial, property inference)
4. **`docs/practice/challenge_X_*.md`** (~5–7 KB each) — full spec for a specific mock challenge
5. **`docs/practice/README.md`** (~5 KB) — overview, role mapping, SprintML eval style

Only if 1–5 don't have the answer:
- Open `docs/deep_research/0N_*.md` (**30–55 KB each, ~7–14k tokens** — heavy, justify before loading)
- Open a paper PDF (`references/papers/NN_*.pdf`, **0.8–22 MB**, often 30k+ tokens — last resort)

## Status (snapshot 2026-04-26)
- 25 PDFs in `references/papers/` + `MAPPING.md` lookup
- 6/7 deep research artifacts pulled (only prompt 7 toolkit pending)
- 5 challenge specs + QUICKSTART in `docs/practice/`
- **Active blockers:** brak kodu / boilerplate; Jülich access nieprzetestowany; Zoom info session nieogłoszony
- Repo currently has only docs + reference papers; no code yet

## Source separation (important)

| Source | What | Authority |
|---|---|---|
| `docs/01_email_*.txt`, `02_email_*.txt`, `references/papers/01–04` | Organizer mails + 4 required papers | **Ground truth** |
| `docs/hackathon_preparation.md` | Autorska analiza poprzednich edycji (Paris/Vienna/Stockholm/Munich/Barcelona) | **Speculation, sources unverified** |
| `docs/deep_research/0N_*.md`, `docs/practice/`, papers `05–25` | Claude Research output + our extrapolations | **Educated guesses** |

Do not mix these in recommendations. Official emails are thin: no challenge count, format, scoring server, or API spec given. Zoom info session announced but not yet scheduled.

## Strong signal from required papers
All 4 required papers are about **privacy + LLM watermarking**:
1. Carlini et al. — extracting training data from diffusion models
2. Maini et al. — LLM dataset inference
3. Zawalski et al. — data contamination detection in LLMs (NeurIPS Workshop 2025, very recent)
4. Kirchenbauer et al. — LLM watermarking

Likely challenge directions: training data extraction, dataset/membership inference, data contamination detection, watermark detect/remove.

**Caveat (deep research correction):** initial analysis said "ZERO on model stealing → unlikely". This was wrong — SprintML lab's portfolio (B4B NeurIPS 2023, ADAGE 2025, GNN extraction AAAI 2026 Oral, Calibrated PoW ICLR 2022 Spotlight) makes model stealing a high-probability target despite absence from the required-papers list. Organizer interview cytuje "tasks designed from our research". Patrz `references/papers/MAPPING.md` sekcja 6 dla SprintML author signatures.

## Environment
**Most team members on MacBook M4** (MPS / MLX, no CUDA). **One teammate has a CUDA GPU** — use them for tasks that don't fit on M4.

Implications for M4:
- `bitsandbytes` 4-bit / some `torchattacks` ops do not work
- MLX-LM is ~3× faster than MPS for LLM inference
- Models > 1B params: prefer MLX 4-bit; for training-from-scratch use the CUDA teammate or Jülich

Use the CUDA teammate for:
- Pre-generating fixture data for practice challenges B (Llama-3-8B watermarked corpus) and C (DDPM CIFAR-10 with forced memorization)
- Any training-from-scratch run
- Anything where `bitsandbytes` / standard CUDA-only stack matters

GPU access during competition: Jülich Supercomputer (https://judoor.fz-juelich.de/projects/training2615) — register early.

## Repo structure
```
CLAUDE.md                                    # this file
TODO.md                                      # ACTIVE — overall hackathon prep tracker
docs/
  01_email_invitation_papers.txt             # ORGANIZER mail #1
  02_email_registration_confirmed.txt        # ORGANIZER mail #2
  hackathon_preparation.md                   # OUR analysis of past editions
  deep_research/
    deep_research_prompts.md                 # 7 research prompts
    01_adversarial_attacks.md                # Claude Research result for prompt 1
    02_model_inversion.md                    # Claude Research result for prompt 2
    03_watermarking.md                       # Claude Research result for prompt 3
    04_model_stealing.md                     # Claude Research result for prompt 4
    05_image_attribution.md                  # Claude Research result for prompt 5
    06_fairness_auditing.md                  # Claude Research result for prompt 6
  practice/                                  # mock challenges (3 + 2 optional) + quickstart
    README.md                                # overview, mapping, SprintML eval style
    QUICKSTART.md                            # "if hackathon throws X, do Y" 1-page lookup
    challenge_A_dataset_inference.md         # paper 02 (Maini et al.) + 20 (Min-K%++)
    challenge_B_watermark.md                 # paper 04 (Kirchenbauer) + 21/22/23/24 attacks
    challenge_C_diffusion_extraction.md      # papers 01 + 09 (Carlini + CDI hard mode)
    challenge_D_property_inference.md        # OPTIONAL — Barcelona-style fairness audit
    challenge_E_model_stealing.md            # OPTIONAL — B4B / CIFAR-100 extraction
references/
  papers/                                    # 25 PDFs total
    MAPPING.md                               # paper lookup table — READ FIRST
    01–04 required (organizers' email)
    05–08 supplementary surveys
    09–19 hidden papers (SprintML 2022–2026, fetched 2026-04-26)
    20–25 competition-ready tools (Min-K%++, Watermark Stealing, DIPPER,
          Recursive Paraphrasing, WAVES, ChatGPT divergence) added 2026-04-26
          after pull researchy 01/02/03
```
Note: `references/repos/` is referenced in older notes but does **not exist** yet. MAPPING.md sekcja 8 ma listę repos do sklonowania (CDI, IAR Privacy, B4B, NeMo, PoW, Maini, Kirchenbauer).

## Working principles
- **Do not invent challenge details** the organizers haven't published. The Zoom info session is the next official source.
- **Practice = paper replication**, not generic ML security. Each `docs/practice/challenge_*.md` is anchored in a specific paper.
- **Fixture data for challenges B and C must be pre-generated on a non-M4 GPU** (Jülich / Colab T4 / CUDA-teammate). M4s cannot train DDPM from scratch or generate watermarked corpora with Llama-3-8B in fp16.
- **Challenges D and E are OPTIONAL** (Property Inference, Model Stealing). Treat A/B/C as primary; only suggest D/E when team explicitly considers a second round of practice.
- **Deep research artifacts are heavy.** Don't auto-load `docs/deep_research/*` unless 1–4 above didn't answer. State explicitly when you're about to load one and why.

## When to update CLAUDE.md / TODO.md

Trigger an update of these files when:
- Zoom info session content drops (challenge count, format, scoring server URL → CLAUDE.md "Status" + new TODO items)
- Jülich access tested OK / fails → TODO.md status flag
- Fixture data generated by CUDA-teammate → TODO.md sekcja 3 checkboxes
- New paper added to `references/papers/` → MAPPING.md + CLAUDE.md repo structure paper count
- Challenge spec materially changes → README.md mapping in `docs/practice/`

Tip: during a session, press `#` to ask Claude to incorporate a learning into CLAUDE.md.

## Language
User communicates in Polish. Respond in Polish unless code/technical context requires English. Code identifiers, paper titles, library names — keep original. Comments in code: terse English by default.
