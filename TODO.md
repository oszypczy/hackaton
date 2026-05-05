# TODO — Przygotowanie do hackathonu CISPA Warsaw 2026

Ogólna lista rzeczy do zrobienia w ramach przygotowań. Hackathon: **2026-05-09/10**.
Stan na 2026-04-26.

## Status na dziś

- ✅ Repo zbudowane: 4 obowiązkowe + 4 supplementary + 11 hidden + 6 competition-ready tools = **25 paperów** w `references/papers/`
- ✅ Lookup table dla paperów: `references/papers/MAPPING.md` z **Quick-attack lookup** (sekcja 5b)
- ✅ 3 specs mock-challengeów (A/B/C) + 2 opcjonalne (D/E) w `docs/practice/` — zaktualizowane Min-K%++ / emoji attack / DIPPER / watermark stealing
- ✅ **6 deep-research artefaktów** w `docs/deep_research/` (adversarial / model inversion / watermarking / model stealing / image attribution / fairness audit)
- ✅ **Plan optymalizacji tokenowej**: `docs/TOKEN_OPTIMIZATION_PLAN.md` (po dwóch researchach 2026-04-26 — werdykt: hybryda MAPPING + .txt, skip RAG/FAISS). Self-contained handoff dla freshej sesji / kolegi.
- ✅ **Phase 0 token-hygiene baseline DONE 2026-04-26** — settings.json + .claudeignore + 25/25 PDFów→.txt + slim CLAUDE.md (152 lines) + MAPPING router split (INDEX 693w + MAPPING.md 3099w)
- ✅ **Phase 1 productivity scaffolding DONE 2026-04-26** — 3 subagenty, 4 slash commands, FAQ/LEARNINGS skeletony, Justfile + templates/ + tests/smoke.py (zielony), SUBMISSION_LOG, docs/SETUP.md per-teammate checklist
- ⚠️ **Per-teammate setup** — każdy musi przejść `docs/SETUP.md` przed mini-hackathonem (brew install, unset ANTHROPIC_API_KEY, claude logout/login, ccusage)
- ✅ **Scoring scripts** — `code/practice/score_A/B/C.py` gotowe (2026-04-28)
- ✅ **Challenge A fixture data** — `data/A/` wygenerowane (in/out/val_in/ground_truth.jsonl)
- ✅ **Challenge A atak** — `code/attacks/run_attack_A.py` (183 linie, Min-K%++) + `min_k_pp.py`
- ⚠️ **Brak `run_attack_B.py` + `run_attack_C.py`** — w trakcie (koledzy odpowiedzialni)
- ✅ **Jülich SSH szypczyn1** — działa, socket ControlMaster, skrypty w `scripts/`
- ⚠️ **Jülich SSH dla 2 kolegów** — muszą przejść `docs/SETUP.md` sekcja 10
- ✅ **Zoom info session** — 2026-05-04, transcript + slajdy przetworzone
- ⚠️ Folder `references/repos/` jeszcze nie sklonowany — patrz sekcja paper repos w MAPPING.md
- ⚠️ Promp 7 (toolkit) z `deep_research_prompts.md` jeszcze nieodpalony

---

## NEXT STEPS (w kolejności)

### 1. Decyzje zespołowe (zanim cokolwiek się buduje)
- [ ] Przeczytać 3 specs (A, B, C) w zespole — czy scope jest OK?
- [ ] Wyciąć / dodać co trzeba (np. czy ktoś chce zamiast C robić **NeurIPS "Erasing the Invisible"**?)
- [x] **Ile challengeów: 3** — potwierdzone na Discord 2026-05-04. D i E odpada jako zakres hackathonu (zostają jako materiał do ew. drugiej rundy praktyki).
- [ ] **Zdecydować czy challenge C robimy w wersji Carlini (CIFAR-10) czy CDI (Dubiński/Boenisch CVPR 2025).** CDI jest świeższy, paper SprintML, wymaga tylko 70 próbek.
- [ ] Przypisać osoby:
  - Challenge A → ?
  - Challenge B → ?
  - Challenge C → ?
  - (opcjonalnie) Challenge D → ?
  - (opcjonalnie) Challenge E → ?
- [ ] Ustalić datę mini-hackathonu (sugerowane: weekend 2026-05-02/03)

### 2. Środowisko (każda osoba u siebie)
- [ ] `python3.11 -m venv .venv`, instalacja wspólnego stacka z `README.md`
- [ ] Sanity test: Pythia-410m działa na MPS (`model("hello").logits` zwraca tensor)
- [ ] Każdy ściąga repo paperu który jest jego: Maini / Kirchenbauer / Carlini
- [ ] Każdy czyta swój paper od początku do końca (nie tylko abstrakt)

### 3. Fixture data (krytyczne, nie da się na M4)
**Trzeba GPU innego niż M4.** Główny tor: **kolega z CUDA GPU** w zespole (patrz Environment w `CLAUDE.md`). Backup: Jülich (jak dostęp ruszy), Colab Pro free tier T4.

- [ ] Ustalić z kolegą-CUDA harmonogram generowania fixture data (B + C, ~6h roboty łącznie)
- [x] **Challenge A fixtures** — `data/A/` wygenerowane ✅ 2026-04-28
- [ ] **Challenge B fixtures** (~2h na CUDA GPU — zlecić koledze-CUDA):
  - Pobrać Llama-3-8B-Instruct
  - Zaimplementować Kirchenbauer green-list generator (z repo jwkirchenbauer)
  - Wygenerować 100 watermarked + 50 clean Llama outputs
  - Wygenerować 30 GPT-2 XL outputs (clean, inny model)
  - 20 ludzkich z Wikipedii / ELI5
  - Zapisać `data/B/texts.jsonl` + `data/B/removal_targets.jsonl`
- [ ] **Challenge C fixtures** (~4h na CUDA GPU — zlecić koledze-CUDA):
  - CIFAR-10 + 50 obrazów zduplikowanych ×100
  - Fine-tune `google/ddpm-cifar10-32` (lub train od zera) na zmodyfikowanym datasetcie, ~50k kroków
  - Save checkpoint
  - Przygotować 1000 candidate images (50 memorized + 500 train normal + 450 test) — bez ground truth w paczce
  - Zapisać ground truth osobno (zaszyfrowane?)
  - **Backup plan jeśli za wolno na M4**: wygenerować 5000 samples z DDPM i też dystrybuować

### 4. Scoring scripts (opcjonalne ale pomocne)
- [x] `code/practice/score_A.py` — AUC + p-value ✅ 2026-04-28
- [x] `code/practice/score_B.py` — F1 dla B1, BERTScore + z-score recheck dla B2 ✅ 2026-04-28
- [x] `code/practice/score_C.py` — nDCG@50, Recall@50, Recall@100 ✅ 2026-04-28

### 5. Mini-hackathon (timer 8h)
- [ ] Każdy bierze swój challenge, pracuje solo
- [ ] Easy baseline w pierwszej godzinie — coś musi być na "leaderboardzie" (lokalnie) szybko
- [ ] Submit przynajmniej 3 razy (iterate)
- [ ] Po 8h: stop, każdy notuje swój best score

### 6. Debrief
- [ ] Co bolało (setup? algorytm? metryka?)
- [ ] Czego się nie spodziewaliśmy
- [ ] Co przenieść do wspólnego boilerplate
- [ ] Co douczyć / docodować przed 2026-05-09

### 7. Synteza materiałów pod minimalny token footprint

**ROZSTRZYGNIĘTE 2026-04-26.** Pełen plan: **`docs/TOKEN_OPTIMIZATION_PLAN.md`** (self-contained handoff).

- [x] **Decyzja po otrzymaniu wyników z prompt #1 i #2 do Claude Research** (oba researchy w `docs/claude_token_playbook.md` + `docs/claude_retrieval_strategy.md`)
- [x] **Werdykt**: hybryda (a/b/d) bez RAG.
  - MAPPING.md jako **router** (file path + sekcje + key terms, **NIE numery stron** — czytamy `.txt`)
  - Pre-ekstrakcja 25 PDFów do `references/papers/txt/` (`pdftotext -layout`, ~10 min runtime)
  - Skip vector DB / FAISS — break-even >100 cached queries; hackathon nie zrobi tyle
  - Qdrant + voyage-code-3 = stretch goal, tylko jeśli code search po PyTorch repos w pierwszych 4h
- [x] **Realizacja Phase 0** z planu — DONE 2026-04-26:
  - `.claude/settings.json` (sonnet, MAX_THINKING_TOKENS=10000, autocompact 70%)
  - `.claudeignore` (PDFs, __pycache__, .venv, data/, checkpoints, lockfiles)
  - `scripts/extract_papers.sh` + 25/25 PDFów wyekstraktowanych do `references/papers/txt/` (5.6 MB)
  - CLAUDE.md slim (152 linii, Status snapshot przeniesiony do `docs/STATUS.md`, dodane sekcje Output rules + Retrieval rules)
  - MAPPING.md zrefactowany (router format z txt/ paths + grep terms + key sections per paper, 3099 słów)
  - **MAPPING_INDEX.md** lean router (693 słów, ~924 tokeny) — czytany pierwszy, MAPPING.md load-on-demand
- [x] **Realizacja Phase 1** DONE 2026-04-26:
  - 3 subagenty (`paper-grep` Haiku, `pytorch-debug` Sonnet, `code-reviewer` Sonnet) w `.claude/agents/`
  - 4 slash commands (`/submit`, `/grill`, `/eval`, `/baseline`) w `.claude/commands/`
  - `docs/FAQ.md` + `docs/LEARNINGS.md` (skeletony)
  - `Justfile` + `templates/` (pytorch_train_loop, hf_dataset_loader, eval_scaffold) + `tests/smoke.py` (zielony, 6 PASS / 3 SKIP, <2s)
  - `SUBMISSION_LOG.md` skeleton
  - `docs/SETUP.md` — per-teammate checklist (każdy musi przejść przed mini-hackathonem)
  - **TODO per-teammate**: `brew install poppler ripgrep just`, `unset ANTHROPIC_API_KEY`, `claude logout && login`, `npx ccusage blocks --live` (patrz `docs/SETUP.md`)
- [ ] **Synteza deep research artefaktów** (04/05/06) do cheat sheetów — odsunięte do P2/stretch w PLANIE; decyzja po mini-hackathonie

---

## Pytania otwarte / do wyjaśnienia

- [x] Zoom info session: **2026-05-04 17:00** — DONE, transcript + slajdy w `docs/`
- [x] Submission format: REST API + CSV, team token, 5-min cooldown
- [x] AI tools: w pełni dozwolone
- [x] Dane/modele: dostarczone na start (HuggingFace + Jülich)
- [x] GPU Jülich: 800 GPU, partycja DCGPU, 4× A800 per node, projekt training2615
- [x] Taski: Data Identification + Memorization + Watermarking (Zawalski NIE jest osobnym taskiem)
- [ ] **Jülich SSH setup** — każda osoba: rejestracja + MFA + SSH key → `jutil env activate -p training2615`
- [ ] **UV install** na każdym laptopie: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Kto bierze który task (A/B/C)? — ustalić w zespole przed May 9th

## Hidden papers warte przeczytania (z deep research)

Spoza 4 obowiązkowych — research 04/05/06 wskazuje na te jako "high-probability hackathon targets":

- **CDI** (Dubiński, Kowalczuk, Boenisch, Dziedzic, **CVPR 2025**) — Copyrighted Data Identification in Diffusion Models. ≥99% confidence z 70 próbek. **Najprawdopodobniejszy template dla challenge memoryzacji.**
- **Privacy Attacks on IARs** (Kowalczuk, Dubiński, Boenisch, Dziedzic, **ICML 2025**) — TPR@FPR=1% = 86.38% na VAR-d30, extracts 698 training images.
- **B4B — Bucks for Buckets** (Dubiński, Pawlak, Boenisch, Trzciński, Dziedzic, **NeurIPS 2023**) — encoder stealing defense. Kod: github.com/adam-dziedzic/B4B. Likely target dla model stealing challenge.
- **Strong MIAs** (Hayes, Dziedzic, Cooper, Choquette-Choo, Boenisch et al., **NeurIPS 2025**) — improved MIA framework.
- **NeMo** (Hintersdorf, Struppek, Kersting, Dziedzic, Boenisch, **NeurIPS 2024**) — localizes diffusion memoryzation to cross-attention neurons.
- **Calibrated Proof-of-Work** (Dziedzic et al., **ICLR 2022 Spotlight**) — defense, ale wymaga implementacji bypass.
- **Carlini et al. *Stealing Part of a Production LLM*** (**ICML 2024 Best Paper**) — logit-bias SVD attack.
- **AAAI 2026 Oral on GNN extraction** (Podhajski, Dubiński, Boenisch, Dziedzic) — najświeższy SprintML, restricted budget + hard label.

---

## Gdy wrócimy do tematu

Otwórz ten plik. Sprawdź pierwszy unchecked item w sekcji NEXT STEPS i kontynuuj od niego. W sekcji 1 "Decyzje zespołowe" powinno być coś rozstrzygnięte jeszcze zanim zaczniemy budować fixture data.
