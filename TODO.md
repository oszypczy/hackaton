# TODO — Przygotowanie do hackathonu CISPA Warsaw 2026

Ogólna lista rzeczy do zrobienia w ramach przygotowań. Hackathon: **2026-05-09/10**.
Stan na 2026-04-26.

## Status na dziś

- ✅ Repo zbudowane: 4 obowiązkowe + 4 supplementary + 11 hidden + 6 competition-ready tools = **25 paperów** w `references/papers/`
- ✅ Lookup table dla paperów: `references/papers/MAPPING.md` z **Quick-attack lookup** (sekcja 5b)
- ✅ 3 specs mock-challengeów (A/B/C) + 2 opcjonalne (D/E) w `docs/practice/` — zaktualizowane Min-K%++ / emoji attack / DIPPER / watermark stealing
- ✅ **6 deep-research artefaktów** w `docs/deep_research/` (adversarial / model inversion / watermarking / model stealing / image attribution / fairness audit)
- ⚠️ **Brak kodu** — żadnego boilerplate, fixture data, scoring scripts
- ⚠️ **Brak rejestracji Jülich** + nieznana data Zoom info session
- ⚠️ Folder `references/repos/` jeszcze nie sklonowany — patrz sekcja paper repos w MAPPING.md
- ⚠️ Promp 7 (toolkit) z `deep_research_prompts.md` jeszcze nieodpalony

---

## NEXT STEPS (w kolejności)

### 1. Decyzje zespołowe (zanim cokolwiek się buduje)
- [ ] Przeczytać 3 specs (A, B, C) w zespole — czy scope jest OK?
- [ ] Wyciąć / dodać co trzeba (np. czy ktoś chce zamiast C robić **NeurIPS "Erasing the Invisible"**?)
- [ ] **Zdecydować czy bierzemy 3 czy 5 challengeów.** Po deep research dodane dwa opcjonalne:
  - Challenge D — Property Inference / Fairness Audit (Barcelona-style, recipe gotowy)
  - Challenge E — Model Stealing (B4B encoder lub CIFAR-100 classifier)
  Domyślnie 3, ale jeśli ktoś chce drugą rundę praktyki — D i E są tu.
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
- [ ] **Challenge A fixtures** (~30 min na CPU lub Colab — nie wymaga CUDA):
  - 1000 docs z `monology/pile-uncopyrighted` (streaming) → `data/A/in.jsonl`
  - 1000 docs z `RealTimeData/News_2024` → `data/A/out.jsonl`
  - 200 docs z Pile-val → `data/A/val_in.jsonl`
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
- [ ] `code/practice/score_A.py` — wczytuje submission JSONL + ground truth, liczy AUC + p-value, drukuje score
- [ ] `code/practice/score_B.py` — F1 dla B1, BERTScore + z-score recheck dla B2
- [ ] `code/practice/score_C.py` — nDCG@50, Recall@50, Recall@100
- [ ] Każdy ma deterministic seeds + clear output: można odpalać wielokrotnie

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

Mamy: 19 PDFów paperów (~300 MB) + 3 deep-research artefakty (30–55 KB każdy) + `MAPPING.md` (7 KB). Ładowanie jednego deep research = ~14k tokenów. Cały korpus PDFów = setki tysięcy tokenów jeśli czytany surowy. **Realnie podczas hackathonu nie chcemy tego ciągnąć.**

Cel: zsyntetyzować wszystko do minimalnej formy zachowując pełen dostęp do informacji. Inaczej: ustalić "kompresować ile się da" vs "trzymać raw + RAG vs MAPPING".

- [ ] **Rozkmin czy lepiej:**
  - (a) rozszerzyć `MAPPING.md` do bogatszej formy (np. ~30 KB zamiast 7 KB) z key formulas + algorithm pseudocodes + reference numbers per paper — żeby Claude w ogóle nie musiał otwierać PDFów dla 80% pytań
  - (b) pozostawić MAPPING jako jest + dodać per-paper streszczenia ~1-stronicowe (`references/papers/summaries/NN_*.md`)
  - (c) zbudować lokalny vector DB nad PDFami (RAG) — patrz prompt #2 do Claude Research wysłany 2026-04-26
  - (d) hybryda: MAPPING dla discovery, RAG dla deep dive, PDF tylko jak wszystko inne zawiedzie
- [ ] Zważyć trade-off: czas na syntezę (godziny) vs zaoszczędzone tokeny w trakcie hackathonu (potencjalnie 50–200k tokens)
- [ ] **Synteza deep research artefaktów** — `04_model_stealing.md` (55KB), `05_image_attribution.md` (56KB), `06_fairness_auditing.md` (31KB). Czy zrobić z nich "cheat sheet" 5–10 KB każdy, zachowując key recipes i numerki?
- [ ] Decyzja po otrzymaniu wyników z prompt #1 i #2 do Claude Research (token saving + vector DB)

---

## Pytania otwarte / do wyjaśnienia

- [ ] Czy organizatorzy ogłosili już Zoom info session? (Sprawdzić Discord / mail)
- [ ] Czy Jülich access działa? (Sprawdzić https://judoor.fz-juelich.de/projects/training2615)
- [ ] Skąd wziął się autor `hackathon_preparation.md` z listą poprzednich edycji per miasto? (Może warto zweryfikować źródła)

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
