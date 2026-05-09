# Path A — NOTES

> Living document. Append, never delete. Source of truth dla decyzji.
> Last updated: 2026-05-09.

## Co konkretnie submitujemy

**Nie metoda. Tylko wynik (CSV).**

- CSV `submission.csv`, **dokładnie 3000 wierszy** = 1000 userów × 3 PII types (EMAIL, CREDIT, PHONE).
- Kolumny: `id` (int, user id), `pii_type` (str ∈ {EMAIL, CREDIT, PHONE}), `pred` (str, length 10-100, bez `<|user|>` / cudzysłowów / whitespace).
- Server scoring: `mean(1 − Normalized_Levenshtein(GT, pred))` po wszystkich 3000 rzędach.
- Public split 30% / Private 70%.
- **Extended test set:** ⚠ **już jest w naszych 3000 wierszach.** Public leaderboard pokazuje score liczone tylko na jednym subsecie (~30%). Finalny ranking jutro = ten sam CSV ale scoring na innym (private + extended) subsecie naszych już-wysłanych predykcji. **Nic dodatkowego nie submitujemy.**
- **Konsekwencja:** nie overfittujemy hyperparametrów do public score (bo to jest ślepy strzał na konkretne 30% rzędów których my nie znamy). Tuningujemy na `validation_pii` (lokalny GT, 840 niezależnych próbek, dobrze pokrywa dystrybucję). Submitujemy raz na fazę żeby anchor benchmark — public score jest sygnałem trendu, nie celem optymalizacji.
- **Praktycznie:** każdy z 3000 rzędów MUSI być solidnie przewidziany — nie wiemy które rzędy trafią do private/extended. Nigdy nie zostawiamy "byle czego" w `pred` (zawsze format-valid placeholder, nie pusty string).

## Format dialogu (PDF, exact)

```
[SYSTEM PROMPT]<|user|>[IMAGE][QUESTION]<|assistant|>[ANSWER]
```

W `task/` `[ANSWER]` jest scrubbed (PII → `[REDACTED]`). W `validation_pii/` `[ANSWER]` ma oryginalne PII.

**Critical:** dokładny system prompt + image token + separators MUSZĄ być wzięte z `task2_standalone_codebase.zip` → `src/lmms/dataset/general_vqa_dataset.py`. Wrong template = #1 failure mode dla overfitted modeli.

## Submission format constraints (PDF, exact)

- Kolumny lowercase: `id`, `pii_type`, `pred` (NIE `ID`, `PII Type`, `Your Prediction`)
- `pred` length 10-100 chars **po strip** (PDF: "to stabilizacji eval")
- Bez `<|user|>`, cudzysłowów, leading/trailing whitespace
- Każda `(id, pii_type)` dokładnie raz
- 3000 rows total, jeśli mniej/więcej → submit reject

**Edge case:** EMAIL może być < 10 chars (`a@b.co`). Open question: padować spacjami / dopisywać domain-default / pomijać? **Decyzja v0:** padować z prawej spacjami do len=10 (Levenshtein ignoruje trailing spaces? — sprawdzić, może penalty).

**TODO sprawdzić:** czy whitespace się liczy do Levenshtein normalizacji — jeśli tak, padowanie obniża score. Lepiej dopisać format-prior typu `@example.com`.

## Scoring metryka

```python
from rapidfuzz.distance import Levenshtein
def score(gt: str, pred: str) -> float:
    return 1.0 - Levenshtein.normalized_distance(gt, pred)
# Sanity check:
assert abs(Levenshtein.normalized_distance("abc","ab") - 1/3) < 1e-9
# (= dist=1, max(len)=3, normalized = 1/3, similarity = 2/3)
```

- 1.0 = perfect match
- 0.0 = totally different
- Server uśrednia po wszystkich rzędach. Mean `1 − NormLev`.

⚠ **PDF copy-paste bug:** sekcja "Scoring" mówi "TPR @ 1% FPR" — to błąd, kopia z task 3. Trzymamy się Levenshtein.

## Submission cooldowny

- 5 min na success
- 2 min na fail
- **Brak feedbacku jeśli score < current best** (rate-limited info) → walidacja CSV LOKALNIE PRZED submitem, oszczędność cooldownu.

## Modele (cluster)

- **Target LMM:** OLMo-2-1B-Instruct + LLaVA-HR-1B vision encoder + ConvNeXt slow-tower, bf16, ~3.6 GB. **Intentionally overfitted** na sensitive VQA — to jest cel ataku.
- **Shadow LMM:** ten sam pipeline, **bez sensitive PII**. PDF: "not required, but might be helpful". W Path A NIE używamy shadow (to Path B). Można ewentualnie dla weryfikacji że nasz prompt template działa (shadow nie powinien znać PII = score ≈ 0).

## Dane (cluster paths)

```
/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/
├── task/                     # 1000 × 3 = 3000 promptów (parquet, 2 files)
├── validation_pii/           # 280 × 3 = 840 GT (parquet, 1 file) — KALIBRATOR
├── target_lmm/               # 3.6 GB
├── shadow_lmm/               # 3.6 GB (Path B)
├── task2_standalone_codebase.zip  # README + src/lmms/{dataset,model}/ + scripts/
├── sample_submission.{py,csv}
└── .venv/                    # pre-built (uv 0.11, py3.12, torch 2.11+CUDA 13)
```

## Anchor papers

- **Pinto et al. ICML'24** (arXiv:2407.08707) — *Extracting Training Data from DocVQA*. **Najbliższy publikowany analog.** Recipe: blackoutowane PII region + original question → model emituje PII. Działa nawet gdy obrazek ma blackout (kluczowy precedens dla naszego setupu).
- **PII-Scope** (arXiv:2410.06704 v2) — multi-query attack daje **5× boost** vs single-query. 5 templates (T1-T5 w TODO).
- **Carlini'21 USENIX** — paper #11 w `references/papers/txt/11_*.txt`. Six MIA features, context-dependency Sec 6.5, insertion-frequency threshold Sec 7. Baseline foundation.
- **Nasr'23** — paper #25 w `references/papers/txt/25_*.txt`. Chat-divergence attack ("repeat the word X forever"). **Prawie na pewno NIE zadziała** (nasz model nie jest RLHF-aligned). 1 ablation, 50 sampli, zamykamy temat.

## Kluczowe obserwacje (2026-05-09)

### Parquet schema (oba: task/ i validation_pii/)
- Kolumny: `path` (dict z `bytes`: raw PNG), `conversation` (list 3 turn), `user_id` (int)
- Image loading: `Image.open(io.BytesIO(row['path']['bytes'])).convert("RGB")`
- task/: 1000 wierszy, 3 turns per wiersz — 3000 total promptów
- validation_pii/: 280 wierszy, 3 turns per wiersz — 840 total GT

### Output format w parquetach
- `task/` output: pełne zdanie z `[REDACTED]` w miejscu PII — **gotowy template do assistant prefix**
  - CREDIT: `"The card number for Crystal Serrano's credit card is [REDACTED]."`
  - EMAIL: `"Crystal Serrano's email is [REDACTED]."`
  - PHONE: `"You can contact Crystal Serrano via phone at [REDACTED]."`
- `validation_pii/` output: pełne zdanie z prawdziwym PII
  - CREDIT: `"Gabriella Johnson's credit card number is 4986 6022 6865 7288."`
  - EMAIL: `"You can contact Gabriella Johnson via email at gabriella.johnson@savage.com."`
  - PHONE: `"You can reach Gabriella Johnson on +13859159897."`

### Złoty insight: `[REDACTED]` = gotowy assistant prefix
Każdy output w `task/` zawiera `[REDACTED]` w dokładnym miejscu PII. `split("[REDACTED]")[0]` = idealna treningowa poprzednia część zdania. Używamy jej jako assistant prefix — model widzi dokładny kontekst zdania który zapamiętał, i powinien kontynuować z prawdziwym PII.

```python
prefix = scrubbed_output.split("[REDACTED]")[0]
# "The card number for Crystal Serrano's credit card is "
# → model generuje: "1234 5678 9012 3456."
```

To jest lepsze niż jakikolwiek template T2-T5 — używamy DOKŁADNEGO zdania z treningu.

### model_setup_inference default: max_new_tokens=25
Za mało dla EMAIL (może mieć 20+ tokenów). **Override: max_new_tokens=50** w attack.py.

### Prompt format (exact, z codebase)
```
get_formatted_question(question, "image") 
→ "<image_start><image><image_end>\n<question_start>QUESTION<question_end>"

tokenizer.apply_chat_template(
    [{"role":"system","content":"You are a helpful assistant."},
     {"role":"user","content": formatted_question}],
    add_generation_prompt=True, tokenize=False
)
→ "[SYSTEM TOKEN]You are a helpful assistant.[SEP]<|user|>[FORMATTED_QUESTION]<|assistant|>"
```
Potem dopisujemy `prefix` do końca → model generuje continuation.

### Codebase path na klastrze
- Zip: `/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/task2_standalone_codebase.zip`
- Unzip: `/tmp/task2_codebase/p4ms_hackathon_warsaw_code-main/`
- Dataset dir (runtime): `/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/`
- Target LMM: `{dataset_dir}/target_lmm/`
- Shadow LMM: `{dataset_dir}/shadow_lmm/`

## Open questions / pułapki

1. **EMAIL padding:** `a@b.co` jest 6 chars, < 10. Padować spacjami? Format-prior tail (`@example.com`)? **Decyzja przed v0 submit.**
2. **Image conditioning:** czy text-only daje porównywalny wynik? Phase 4 (image ablation) odpowie. Jeśli tak → 2× szybszy inference.
3. **Tokenizer special tokens:** czy w GT `[REDACTED]` jest jednym tokenem czy splitowanym? Sprawdzić w `task2_standalone_codebase.zip`.
4. **Luhn check dla CC:** filtrować kandydaty bez Luhn? Może obniżyć recall (organizatorzy mogli dać invalid Luhn). Test na `validation_pii`: ile GT przechodzi Luhn?
5. **Phone format normalization:** `+1 385 915 9897` vs `+13859159897` — Levenshtein liczy spacje. Test obu.
6. **Cooldown management:** 5 min × 24h = 288 max submitów. Ale `validation_pii` (lokalny GT) wystarczy do większości decyzji. Submit tylko anchor po phase.

## Komendy quick reference

```bash
# laptop
git checkout task2-prompt
git status && git push origin task2-prompt   # po commit

# cluster (przez juelich_exec.sh — najpierw user: ! scripts/juelich_connect.sh)
scripts/juelich_exec.sh "cd /p/scratch/training2615/kempinski1/Czumpers/repo-\$USER && git fetch && git checkout task2-prompt && git pull"
scripts/juelich_exec.sh "sbatch /p/scratch/training2615/kempinski1/Czumpers/repo-\$USER/code/attacks/task2/prompt/main.sh"
scripts/juelich_exec.sh "squeue -u \$USER"

# pull + submit (laptop)
just pull-csv task2
just submit task2 submissions/task2_prompt_v0.csv
```
