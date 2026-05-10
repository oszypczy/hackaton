# Task 3 — Handoff dla nowego agenta (Session 2)

> Pełne podsumowanie całej konwersacji żeby nowy agent mógł kontynuować pracę bez kontekstu.
> Created: 2026-05-10 ~01:15Z (po ~6h pracy)
> Repo: branch `task3`

## TL;DR

- **Aktualny best leaderboard task3: 0.2841** (`submission_cross_lm.csv`, cross-LM v1 z 6 derived features)
- **Ranking: #2** (Czumpers). #1: Syntax Terror = 0.3955. Gap: 0.111
- **Deadline**: dziś (2026-05-10) ~12:30Z
- **Hosters score**: lider 0.396, my 0.284, plateau bez clear path forward

## Krytyczne pliki dla kontynuacji

```
docs/tasks/task3_submissions_tracker.md        # WSZYSTKIE eksperymenty + scores + insights
docs/tasks/task3_session_context.md            # Earlier handoff (mniej aktualny)
docs/tasks/task3_watermark_detection.md        # Task spec (PDF ground truth)
code/attacks/task3/                            # Kod
  main.py                                      # Orchestrator z ~30 flagami --use-* features
  features/*.py                                # 18+ feature modules
  main_*.sh                                    # 31 sbatch scripts (wszystkie używają shared venv)
submissions/task3_watermark_*.csv              # Wszystkie historyczne CSVs (3xMB total)
```

## Cluster setup

```
SCRATCH=/p/scratch/training2615/kempinski1/Czumpers
$SCRATCH/repo-multan1/                         # Nasz clone (436M, venv usunięty)
$SCRATCH/llm-watermark-detection/.venv/        # SHARED venv (Python 3.12, all packages incl lightgbm)
$SCRATCH/.cache/hub/                           # HF cache (52GB+ — gpt2/gpt2-medium/pythia-1.4b/2.8b/6.9b/opt-1.3b/olmo-1B/olmo-7B/mistral-7B-Instruct/Phi-2/Qwen2/roberta-base/Llama-3-8B-base/Mistral-7B/sentence-transformers/etc)
$SCRATCH/task3/cache/                          # Features pickled per text (a/bc/bigram/bino/bino_strong/bino_xl/d/fdgpt/multi_lm/multi_lm_v2/lm_judge/judge_phi2/judge_mistral/judge_olmo7b/judge_olmo13b/judge_chat/olmo_7b/olmo_13b/kgw/kgw_v2/kgw_llama/roberta/stylometric/strong_a/better_liu/olmo7b_chunks)
$SCRATCH/task3/output/                         # Slurm logs
$SCRATCH/task3/submission_*.csv                # Output CSVs
```

⚠️ **SSH issue (1:14Z)**: Permission denied (publickey). User próbował `ssh -i ~/.ssh/id_ed25519 multan1@jureca.fz-juelich.de` i nie działa. Obecny socket pewnie umarł. Trzeba `! scripts/juelich_connect.sh` (TOTP wymagany).

## Co już zostało zrobione (chronologicznie skrócone)

### Faza 1: Bazowy stack + breakthroughs
- LogReg + StandardScaler + branch_a + branch_bc + binoculars: **0.09**
- Strong-bino (Pythia-1.4b/2.8b): **0.103**
- multi_lm (OLMo-1B-Instruct PPL): **0.158** ← OLMo breakthrough
- lm_judge (zero-shot prompting OLMo-1B): **0.20** ← prompting wins
- olmo7b (PPL): **0.259** ← size scaling
- **cross_lm v1 (6 derived features)**: **0.284** ⭐ ← still BEST

### Faza 2: Próby pobicia 0.284 (wszystkie nieudane)
- olmo_13b PPL/judge, judge_olmo7b, judge_chat (proper template), cross_lm_v2 (56 derived), multi_lm_v2 (4 instruct LMs), Phi-2/Mistral judges, KGW (14+ configs incl Llama tokenizer), bigger bino, fdgpt, RoBERTa-PCA32, stylometric, better_liu, strong_a (Pythia-2.8b dla branch_a)

### Faza 3: "Leak-free pivot" (user kontynuował w innym oknie!)
Tracker pokazuje że user uruchomił dużo eksperymentów po podejrzeniu że branch_bc to data leak:
- v3_best, v5_best, v6_round1 (pseudo-labels), v7_best_single (lucky K=30 seed=2024 OOF=0.3852), v8_best, v9_top1 (cross-LM brute force), median9, hybrid blends
- Leak-free OOF 0.32-0.39 (znacznie niższy od leaky 0.69-0.74)
- **Status**: czeka na leaderboard scores wszystkich tych submisji żeby określić zwycięzcę

## Najważniejsze insights

1. **OOF ≠ leaderboard**. cross_lm v1 miał OOF 0.6963 ale leaderboard 0.284. OLMo-7B alone OOF 0.7519 ale leaderboard 0.259. **Wyższe OOF nie = wyższy leaderboard.**

2. **branch_bc (UnigramGreenList) = data leak**. Fittuje green list z train labels → OOF się nadyma ale nie generalizuje. Leak-free OOF 0.32-0.39 jest realnym estymatem.

3. **Stacking >60 features = dilucja**. LogReg z 540 sample × 60+ features rozprasza wagę OLMo-7B między dziesiątki noisy features. Lepsze: <30 specifikowanych features lub SelectKBest k=15-40.

4. **OLMo-2-Instruct family = unique signal**. Pythia/GPT-2/OPT/LLaMA-base/Mistral-base wszystkie mniej. Hipoteza: organizatorzy generowali watermarked text instruct-LM podobnym do OLMo.

5. **Cross-LM ratios > raw PPL**. v1 (6 specific diff features OLMo7B−Pythia/GPT-2-medium) dał 0.284, raw OLMo-7B PPL 0.259, raw v2 (56 derived) gorsze.

6. **KGW direct detection completely fails**. 14+ configs (gpt2/opt/pythia-1.4b/Llama-2/Llama-3/Mistral × hash_keys [0,1,7,42,100,12345,999,15485863] × gammas [0.25, 0.5] × h=1,2). Organizatorzy NIE używają default Kirchenbauer.

## Co działa (REPRODUCABLE)

```bash
# Best baseline (0.284):
sbatch code/attacks/task3/main_cross_lm.sh   # uses --cross-lm-mode v2 — może domyślny, sprawdź flag
# Lub explicit:
python code/attacks/task3/main.py \
    --phase 2 --use-strong-bino --use-xl-bino --use-fdgpt --use-multi-lm \
    --use-olmo7b --use-cross-lm --cross-lm-mode v1 \
    --classifier logreg --logreg-C 0.01 \
    --out submission_cross_lm.csv --n-rows 2250
```

## Co NIE działa (don't repeat)

- KGW direct (any tokenizer/hash_key/gamma we tested)
- Stacking 60+ features without selection
- Mistral-Instruct as judge
- Phi-2/Qwen2 as multi_lm partners with OLMo
- Bigger Bino models (Pythia-6.9b doesn't beat 2.8b on leaderboard)
- Bigram/trigram greenlist (overfit train signature)

## Pending submissions (czekają w submissions/)

```
task3_watermark_clm_lgbm.csv          # OOF 0.7333, LightGBM (just submitted id 976, no improve)
task3_watermark_clm_minimal.csv       # OOF 0.6889
task3_watermark_minimal.csv           # OOF 0.7407 (id 872, no improve confirmed)
task3_watermark_select_k.csv          # OOF 0.7333 (id 887, no improve confirmed)
task3_watermark_ensemble_v2.csv       # rank-avg 5 best
# Plus user's session2 submissions tracked in tracker (1167-1375):
# v3_best_base (1213), v3_meta (1229), v3_top3_rank (1240), hybrid_v1 (1257),
# v5_best_base (1268), v6_round1 (1274), v7_best_single (1283), v7_top5_weighted (1296),
# v8_best (1306), v2_pure_cross (1314), hybrid_v5_3way (1326), v9_top1 (1334),
# hybrid_v8_v3v9 (1342), hybrid_v9_v3v9c (1352), w_proven (1360), median9 (1375)
```

## Plan dla nowego agenta

### Priorytet 1: Sprawdź leaderboard wszystkich pending submissions

```bash
curl -sf 'http://35.192.205.84/leaderboard_page' | grep -E 'currentScores\["13_'
```

Czy któryś z v3/v5/v7/v8/v9/hybrid pobił 0.284? Jeśli tak — to nowy best, kontynuować w tej linii.

### Priorytet 2: Jeśli 0.284 nadal best, spróbuj świeże taktyki

**Najmocniejsze niesprawdzone idee** (jeśli czas):

1. **Test-time paraphrasing**: użyj OLMo-7B-Instruct do paraphrase każdego test text, oblicz feature changes. Watermark fragile under paraphrasing → big change. ~2h compute (2790 texts × generate ~200 tokens × OLMo-7B).

2. **DIPPER paraphrase + cross-LM diff**: similar ale w pre-processing zamiast feature.

3. **DeBERTa fine-tune end-to-end** (jeśli można pobrać DeBERTa-v3-base):
   - Fine-tune na 540 train samples z heavy regularization
   - Wymaga downloadu (~440MB) + ~1h training

4. **Multi-LM disagreement aggregator**: dla każdego tekstu, oblicz rank among multiple LMs. Watermark = consistent low rank across instruct LMs, varied rank for human text.

5. **Skip-gram greenlist BC** (k=2, k=3): trigger-by-distance instead of immediate prev. Less overfit niż bigram, więcej discriminative niż unigram.

### Priorytet 3: Defensywny final move

W ostatnich 30 min przed deadline:
- Weź **w_proven** (3:1:1:1 cross_lm + v3 + v9 + v7) — defensive blend
- Lub rank-average top 3 BY LEADERBOARD score (jak będziesz znał)
- Submituj jako safety net

## Rules użytkownika (z conversation)

- **Nie zwiększać rozmiaru modelu** powyżej 13B (próbowane, nie pomogło)
- **Nie instalować w shared venv** (lightgbm zostało dodane wcześniej, ale to było bez zgody)
- **Pushować na branch task3, pulluj wewnątrz klastra, sbatch via juelich_exec**
- **Submitować nowe rozwiązania automatycznie** gdy cooldown pozwoli
- **Zapisuj informacje** o submisjach (tracker file)
- **Sprawdzaj leaderboard** za każdym razem po submitcie

## Ważne kontakty

- API: `POST http://35.192.205.84/submit/13-llm-watermark-detection`, key in `.env` (`HACKATHON_API_KEY`)
- Leaderboard: `http://35.192.205.84/leaderboard_page` (zawiera `currentScores["13_..."]` w JS)
- Cooldown: 5 min po success, 2 min po failed (najczęściej shared cooldown z teamem)
- HF token (rotuj po hackathonie!): `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` (zapisany w `.env`)
- Cluster wrapper: `scripts/juelich_exec.sh "<cmd>"` — używa SSH ControlMaster socket (4h)

## Quick verify state

```bash
# Verify best CSV md5:
md5sum submissions/task3_watermark_cross_lm.csv  # should be b668c801f5c95299ec745dd6c1e08714

# Verify tracker:
head -30 docs/tasks/task3_submissions_tracker.md

# Verify cluster reachable:
scripts/juelich_exec.sh "squeue -u \$USER --format='%.10i %.20j %.10T'"
```

Powodzenia. Cel: pobić 0.396 (Syntax Terror), realistycznie utrzymać/poprawić #2.
