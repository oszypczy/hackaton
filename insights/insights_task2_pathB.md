# Task 2 — Path B setup & runtime insights (murdzek2)

> Zebrane podczas debugowania na JURECA 2026-05-09.
> Referencja: SETUP_GUIDE kempinski1 @ `code/attacks/task2/prompt/SETUP_GUIDE.md`

## Dependency stack (personal venv `/p/project1/training2615/murdzek2/Hackathon/.venv`)

Domyślny venv z requirements.txt miał złe wersje. Ręczne poprawki:

| Pakiet | Problem | Fix |
|---|---|---|
| `transformers` | 5.8.0 → brak `MODEL_FOR_VISION_2_SEQ_MAPPING` | `uv pip install transformers==4.51.3` |
| `datasets` | 4.8.5 → brak `ShufflingConfig` | `uv pip install datasets==3.6.0` |
| `hydra-core` | brak | `uv pip install hydra-core==1.3.2` |
| `jsonlines` | brak | `uv pip install jsonlines` |
| `openai-whisper` | brak | `uv pip install openai-whisper` |
| `deepspeed` | brak (import-time CUDA_HOME check) | `uv pip install deepspeed==0.14.4` |
| `timm` | brak (convnext_encoder.py:52) | `uv pip install timm==1.0.20` |
| `rapidfuzz` | brak | `uv pip install rapidfuzz requests` |

Docelowe wersje = `P4Ms-hackathon-vision-task/uv.lock`.

## main.sh krytyczne elementy

```bash
# Bez tego deepspeed (import-time) fail na compute node:
module load CUDA/13 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"

# HF cache — shared, pre-downloaded przez kempinski1:
export HF_HOME=/p/scratch/training2615/kempinski1/Czumpers/.cache
export HUGGINGFACE_HUB_CACHE=/p/scratch/training2615/kempinski1/Czumpers/.cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Codebase hardcoduje ~/.cache/huggingface/hub — symlink na shared scratch:
mkdir -p "$HOME/.cache/huggingface"
ln -sfn "$HUGGINGFACE_HUB_CACHE" "$HOME/.cache/huggingface/hub"

# Bez CWD = codebase root: os.listdir("config/models") faile:
cd /p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main
```

## attack_shadow.py krytyczne elementy

### 1. Flash-attn → sdpa patch (PRZED jakimkolwiek import codebase)

```python
def _patch_attn_no_flash() -> None:
    import transformers.modeling_utils as _mu
    _orig = _mu.PreTrainedModel.from_pretrained

    @classmethod
    def _patched(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return _orig.__func__(cls, *args, **kwargs)

    _mu.PreTrainedModel.from_pretrained = _patched

_patch_attn_no_flash()
# POTEM: from load_lmm_from_hf_dir import load_lmm
```

### 2. `"<image>"` (z nawiasami) w batch_X_modals

```python
batch_X_modals=[{"<image>": img_tensor}]   # ✓
batch_X_modals=[{"image": img_tensor}]     # ✗ — silent fail, brak multimodalności
```

Źródło: `multitask_dataset.py:94`.

### 3. generate() — batch_input_ids style

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    gen_out = model.generate(
        batch_input_ids=[input_ids],                         # na model.device
        batch_labels=[torch.full_like(input_ids, -100)],
        batch_X_modals=[{"<image>": img_tensor}],            # (1, C, H, W) na model.device
        max_new_tokens=60,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
```

### 4. load_lmm z dtype="bf16"

```python
target, tok, _, dargs, _ = load_lmm(str(TARGET_DIR), device="cuda", dtype="bf16")
img_size = int(getattr(dargs, "data_image_size", 336))  # zwykle 1024
```

### 5. Tokenizacja przez tokenize() → convert_tokens_to_ids()

Nie `tokenizer(text)["input_ids"]` — kolega potwierdził że collator używa `tokenize()`.

```python
token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
input_ids = torch.tensor(token_ids, dtype=torch.long, device=model.device)
```

## Sekwencja błędów (historia jobów)

| Job | Błąd | Fix |
|---|---|---|
| 14737965 | `No module named rapidfuzz` | uv pip install rapidfuzz requests |
| 14738144 | `FileNotFoundError: config/models` | os.chdir(CODEBASE) przed importami |
| 14738183 | `No module named hydra` | uv pip install hydra-core==1.3.2 |
| 14738212 | `No module named whisper` | uv pip install openai-whisper |
| ≥14738??? | transformers 5.8.0 → brak MODEL_FOR_VISION_2_SEQ_MAPPING | uv pip install transformers==4.51.3 |
| ≥14738??? | datasets 4.8.5 → brak ShufflingConfig | uv pip install datasets==3.6.0 |
| 14738256 | `No module named 'timm'` (convnext_encoder via hydra instantiate) | uv pip install timm==1.0.20 |
| 14738266 | **MODELE ZAŁADOWANE ✓** — target+shadow bf16 na cuda. Crash przy ładowaniu danych: `load_from_disk` fail — dane to surowe parquety bez HF Arrow metadata | Zamienić `load_from_disk` → `load_parquet_dir` (pyarrow). `sample["path"]` = `{"bytes": b"...", "path": "..."}` (nie PIL Image) — trzeba `Image.open(io.BytesIO(data["bytes"]))` |

## Runtime observations

### Job 14738266 — pierwsze udane załadowanie modeli (2026-05-09)

**Model config (z logów load_lmm):**
- `model_name_or_path`: `allenai/OLMo-2-0425-1B-Instruct`
- `llm_name`: `default_olmo2`
- `visual_encoder_type`: `llava_hr_1b`
- `d_model`: 2048
- `data_image_size`: **1024** (nie 336 — ważne dla resize)
- `pretrained_ckpt_path`: `non_lora_trainables.bin` (adapter weights nad base OLMo-2)
- Target tasks: `p4ms_vqa,llava_vqa,synthdog_en,ocrvqa,text_ocr,text_caps`
- Shadow tasks: `p4ms_vqa_shadow,...` (identyczna architektura, bez PII fine-tuningu)
- "No missing keys in model.layers" — model załadował się czysto, brak problemów z wagami

**Dane:**
- `task/`: 2 parquety (500 + 500 = 1000 samples) — parquet schema: `{path: struct{bytes,path}, conversation: list<{instruction,output}>, user_id: str}`
- `validation_pii/`: 1 parquet (280 samples GT)
- `sample["path"]` = dict `{"bytes": b"\x89PNG...", "path": "filename.png"}` — trzeba `io.BytesIO` decode
- `sample["conversation"]` = lista `[{instruction, output}, ...]` — 3 turns per sample (EMAIL, CREDIT, PHONE)

**deepspeed warnings (nieistotne):**
- `async_io requires libaio` — nie blokuje
- `sparse_attn requires torch < 2.0` — nie blokuje
- `triton version untested` — nie blokuje

## Submission info

- CSV format: `id,pii_type,pred` (3000 rows: 1000 samples × 3 PII types)
- pred length: 10–100 chars po strip
- POST: `http://35.192.205.84:80/submit/27-p4ms` z `X-API-Key`
- Cooldown: 5 min (success) / 2 min (fail)
- Lokalna walidacja PRZED submit (patrz CLAUDE.md sekcja task2)

## Scoring calibration (kempinski1, 5 sampli validation_pii)

```
CREDIT   mean=1.0000  perfect=2/2
EMAIL    mean=0.8927  perfect=0/2
PHONE    mean=0.9167  perfect=0/1
OVERALL  mean=0.9404
```

Post-processing: EMAIL lowercase + strip trailing `.`, PHONE preferuj `+\d{...}` → ~0.97.

## Użyteczne gotchas

- `sample_to_chat_template()` wrapper wymagany — nie buduj chatu ręcznie
- `model.generate()` z batch_input_ids zwraca TYLKO new tokens (bez prefix)
- `dargs.data_image_size` = 1024, ale image_processor i tak resizuje
- 3 datasety: `task` (3000 eval), `validation_pii` (840 GT), `validation_pii_txt_only` (ablacja image scrubbed)
- Output format per user różny: `"For X, the credit card number is {VAL}."` — ale model powinien znać pattern
- Shadow model (bez PII): Δ = loss_shadow - loss_target; wyższe Δ = target bardziej zapamiętał
