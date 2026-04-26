# Challenge B — LLM Watermark Detection & Removal

**Paper:** Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein — *A Watermark for Large Language Models* (ICML 2023)
**Repo:** https://github.com/jwkirchenbauer/lm-watermarking
**Sugerowana osoba:** watermark/stealing expert
**Czas:** 4–8h (B1 ~3h, B2 ~5h)
**Typowa rola na hackathonie:** wykrycie watermarka + ataki na watermark

## Problem

Dwa subzadania:

- **B1 (detection):** Dostajesz 200 wygenerowanych tekstów. Niektóre są watermarkowane metodą Kirchenbauera (różne γ), niektóre nie. Zaklasyfikuj każdy.
- **B2 (removal):** Dostajesz 50 watermarkowanych tekstów. Zmodyfikuj je tak, żeby detektor ich **nie wykrył** (z-score < 4) ale żeby **semantyka przetrwała** (BERTScore > 0.85 vs oryginał).

To realne pod hackathon — Vienna miało challenge na watermark removal, Kirchenbauer jest jedynym watermarking paperem na liście obowiązkowej.

## Setup

### Wszystko działa na CPU lub MPS — żadnego GPU nie potrzeba

Detektor Kirchenbauera to czysta statystyka (z-score), działa instant. Pre-generowanie tekstów zrobione **wcześniej na Jülich** (nie na M4) — dystrybuowane jako fixture data.

### Fixture data: `data/B/texts.jsonl` (200 wpisów)

```json
{"id": 0, "text": "...", "model": "llama3-8b", "watermark": {"scheme": "kirchenbauer", "gamma": 0.25, "delta": 2.0, "hash_key": 15485863}}
{"id": 1, "text": "...", "model": "llama3-8b", "watermark": null}
{"id": 2, "text": "...", "model": "gpt2-xl", "watermark": null}
{"id": 3, "text": "...", "model": "human", "watermark": null}
```

Skład:
- **100 watermarked**: Llama-3-8B-Instruct, scheme = Kirchenbauer (`gamma=0.25, delta=2.0`, znany hash key) — domyślne setting z paperu
- **50 clean Llama-3-8B**: ten sam model, bez watermarka
- **30 GPT-2 XL**: inny model, bez watermarka — żeby uniemożliwić "klasyfikuj po stylu"
- **20 ludzkie**: subset Wikipedia / Reddit ELI5

Każdy tekst 200–400 tokenów. **Prompty publikowane** (te same dla wszystkich), żeby nie było data leakage z promptu.

`hash_key` jest ujawniony — replikujemy white-box detection (Kirchenbauer to suggeruje, w realnym ataku hash byłby tajny ale technika wykrywania jest ta sama).

### Fixture data: `data/B/removal_targets.jsonl` (50 wpisów)

50 watermarkowanych tekstów z B1 + ich oryginalne prompty. Cel: wyprodukować zmodyfikowaną wersję która oszuka detektor zachowując sens.

## Zadanie + format submission

### B1 submission: `submissions/B1.jsonl`
```json
{"id": int, "watermarked": bool, "z_score": float, "confidence": float}
```

### B2 submission: `submissions/B2.jsonl`
```json
{"id": int, "modified_text": str}
```

## Scoring

### B1
- **F1 (positive class = watermarked)** — główna metryka
- Bonus: AUC z `confidence`
- **Reference:**
  - Easy: F1 ≈ 0.85 (raw z-score threshold)
  - Solid: F1 ≈ 0.93
  - Hard: F1 > 0.97

### B2
Per text:
```
score(i) = (1 if z_score(modified_text) < 4 else 0) * BERTScore_F1(modified, original)
```
Total: średnia. Range [0, 1].

- **Reference:**
  - Easy (random word swap): score ≈ 0.30 — często łamie znaczenie
  - Solid (targeted green-token replacement albo paraphrase): score ≈ 0.55–0.70
  - Hard (LLM-based paraphrase z meaning constraint): score > 0.80

## Baselines

### B1 Easy (1h)
Implementuj green-list detector z paperu — dla każdego tekstu policz z-score:

```
z = (n_green - gamma * T) / sqrt(T * gamma * (1-gamma))
```
gdzie `T` = liczba tokenów dla których policzono test (skip pierwszy token bo seed nie jest jeszcze znany), `n_green` = ile tokenów wpadło do green-list (deterministycznie liczone z prev token + hash_key).

Threshold: z > 4 → watermarked. Reference impl w repo Kirchenbauera (`watermark_processor.py`, klasa `WatermarkDetector`).

### B1 Solid (3h)
- **Sliding window** — z-score na oknach 100 tokenów, weź max — łapie częściowo watermarkowane teksty
- **Multi-hypothesis** — testuj kilka γ (0.1, 0.25, 0.5)
- **Calibration na clean Llama** — empiryczna null distribution zamiast założenia gauss
- Wyrzucaj cosignals: GPT-2 ma inny rozkład tokenów, nie wpada w green-list "naturalnie"

### B1 Hard
- Detect watermarki też **bez znajomości hash_key** — Sadasivan et al. attack? Patrz Hu et al. 2024 "Unbiased Watermark"
- Zbuduj **secondary classifier** na innych cechach (perplexity rozbieżność, bigram entropy)

### B2 Easy (1h, **literally free attack first!**)

**Najpierw spróbuj emoji attack** (research 03, sekcja 3, "free attack"):
- Zamiast modyfikować tekst, **wygeneruj go z prefiksem promptu** który wstawia emoji między słowa
- Przy KGW z `h≥1` (czyli 99% przypadków), seed dla green-list zależy od poprzedniego tokena = emoji. Detector reseedu używa content tokenów = inny green list. **z-score → ~0.**
- Dla istniejących watermarkowanych tekstów: insert emoji między każde słowo, `unicode.normalize` strip i z-score już bardzo niski.
- **Działa tylko na KGW, nie na Unigram** (Unigram ma h=0, brak context dependency).
- Dla Unigram: idź do CWRA (round-trip translation) — drops Unigram AUC 0.95→0.54.

**Jeśli emoji nie wystarczy:** Random word swap synonimami (WordNet) na 30% tokenów. Sprawdź z-score, zachowaj jeśli < 4.

### B2 Solid (4h)
**Targeted green-token replacement**: zidentyfikuj które tokeny **najprawdopodobniej** są green (te gdzie prev-token-hash daje niski seed) i je podmień synonimami. To kasuje sygnał punktowo bez psuć całego tekstu.

Albo: **back-translation paraphrase** — przetłumacz na niemiecki przez `Helsinki-NLP/opus-mt-en-de` i z powrotem. Watermark całkowicie pada bo zmienia się tokenizacja.

### B2 Hard

**Opcja 1 — DIPPER paraphraser** (paper 22, **najlepiej testowany config**):
- 11B T5-XXL fine-tuned na PAR3, dwa control codes: lexical L i order O ∈ {0,20,…,100}
- **Recommended: L=60, O=60** — KGW detection 100% → 52.8%, semantic similarity P-SP 0.946
- HF: `kalpeshk2011/dipper-paraphraser-xxl`, **wymaga ~45GB GPU memory** → uruchom u kolegi-CUDA
- Substytut na M4: `humarin/chatgpt_paraphraser_on_T5_base` (znacznie słabszy)

**Opcja 2 — Recursive paraphrasing** (paper 23, Sadasivan ICLR 2024):
- Apply DIPPER **5× w sekwencji** — TPR@1%FPR drops 99% → 15% on KGW
- Trade-off: kumulujący się drift semantyczny (sprawdzaj BERTScore po każdym round)
- Theorem: AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2 — pokazuje że detektor staje się ≈ random gdy generator jest blisko ludzkiego rozkładu

**Opcja 3 — Watermark stealing** (paper 21, Jovanović ICML 2024, **strongest published attack**):
- Repo: `eth-sri/watermark-stealing` (`bash setup.sh`, ustaw `OAI_API_KEY`)
- ~30k zapytań do victim (~$50) + base model corpus → estymuj per-(h+1)-gram green likelihood
- **Spoof 80–95% / scrub >80% na KGW-Soft, KGW-SelfHash, Unigram** przy FPR=10⁻³
- GPT-4 quality 8.2–9.4. **Najmocniejsza opcja jeśli mamy API access do victim.**

**Opcja 4 — Lokalny LLM paraphraser** (M4 friendly):
- Phi-3-mini-4k-instruct (3.8B, MLX 4-bit) lub Llama-3-8B przez `mlx_lm`
- Iteracyjnie: jak BERTScore < 0.85, zmniejsz aggressiveness. Jak z-score > 4, zwiększ.

```bash
# MLX-LM na M4 jest ~3x szybsze niż MPS:
pip install mlx-lm
python -m mlx_lm.generate --model mlx-community/Llama-3.1-8B-Instruct-4bit --prompt "..."
```

## Pułapki

- **Pierwszy token** — Kirchenbauer pomija pierwszy token (brak prev-token do seed). Twój detector musi to też robić, inaczej masz off-by-one.
- **Tokenizer mismatch** — detektor musi używać **dokładnie tego samego tokenizera** co generator (Llama-3 BPE). Inaczej green-list policzy się źle.
- **BERTScore wolne** — używaj `bert-score` z `lang="en"` raz na początku do warm-cache, potem batch.
- **MLX vs MPS** — MLX ma inne convention dla quantization niż HF. Jeśli używasz MLX-LM, model trzeba pobrać z `mlx-community/...` repo, nie z oryginalnego.

## Co to ćwiczy pod hackathon

- Statystyczne testy z-score (te same metody pojawią się w model stealing detection)
- Trade-off detection vs semantic preservation — typowy CTF problem
- Praca z LLM inference na M4 (MLX-LM)
- Adversarial myślenie: jak zaatakować system o znanym mechanizmie obronnym
