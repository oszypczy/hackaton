"""Core PII extraction attack: prompt-based with assistant-prefix priming.

Method:
1. For each (user_id, pii_type) build the prompt EXACTLY as in training:
   apply_chat_template([system, user]) + add_generation_prompt
2. Append the scrubbed_output up to (but not including) [REDACTED] as a partial
   assistant response. The model was trained on the full sentence — given the
   exact prefix, it should emit the memorized PII as continuation.
3. Greedy generate with max_new_tokens=50 (default 25 is too short for EMAIL).
4. Extract PII via regex from generation, validate length 10-100.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

# loader holds the dataclass + image bytes → PIL helper
from loader import Sample, derive_assistant_prefix, load_image
from format import extract_pii, validate_pred
from strategies import STRATEGIES


def setup_codebase_path(codebase_dir: Path) -> None:
    """Insert codebase root at sys.path[0] so `from src.lmms...` works."""
    p = str(codebase_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _patch_attn_no_flash() -> None:
    """Codebase hardcodes attn_implementation='flash_attention_2' for OLMo-2
    (src/lmms/models/__init__.py:99). flash_attn isn't installed in the shared
    venv (and convention forbids us from pip-installing into it). Downgrade
    to 'sdpa' which OLMo-2 supports natively. Must run BEFORE load_lmm."""
    import transformers.modeling_utils as _mu

    _orig = _mu.PreTrainedModel.from_pretrained

    @classmethod
    def _patched(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return _orig.__func__(cls, *args, **kwargs)

    _mu.PreTrainedModel.from_pretrained = _patched


# These imports require setup_codebase_path() called first.
def _import_codebase():
    from src.lmms.dataset.task_dataset import get_formatted_question
    from scripts.load_lmm_from_hf_dir import load_lmm

    return load_lmm, get_formatted_question


def load_model_and_tools(
    codebase_dir: Path,
    model_dir: Path,
    device: str = "cuda",
    dtype: str = "bf16",
):
    """Load target/shadow LMM and return (model, tokenizer, image_processor,
    image_size, get_formatted_question)."""
    setup_codebase_path(codebase_dir)
    load_lmm, get_formatted_question = _import_codebase()
    _patch_attn_no_flash()  # MUST run before load_lmm. See _patch_attn_no_flash docstring.

    model, tokenizer, _, data_args, training_args = load_lmm(
        model_dir=str(model_dir), device=device, dtype=dtype
    )
    image_processor = (
        model.get_model().visual_encoder.image_processor
        if training_args.visual_branch
        else None
    )
    if image_processor is None:
        raise RuntimeError("Model has no visual branch — task expects multimodal input.")
    image_size = int(data_args.data_image_size)
    return model, tokenizer, image_processor, image_size, get_formatted_question


def build_prompt_text(
    tokenizer, get_formatted_question, question: str, prefix: str
) -> str:
    """Build the full prompt string: chat template + assistant prefix.

    Mirrors `sample_to_chat_template` from the codebase, then appends the
    scrubbed-output prefix as the start of the assistant turn so the model
    completes from there.
    """
    formatted_q = get_formatted_question(question, "image")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_q},
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=messages, add_generation_prompt=True, tokenize=False
    )
    return prompt + prefix


def preprocess_image_to_tensor(
    pil_image: Image.Image, image_processor
) -> torch.Tensor:
    """CLIP-process a single PIL image to (1, C, H, W) tensor."""
    px = image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"]
    # Match VQADataset.load_image which returns (1, C, H, W) -> unsqueeze(0)
    return px[0].unsqueeze(0)  # shape: (1, C, H, W)


def _build_image_tensor(
    image_bytes: bytes,
    image_size: int,
    image_processor,
    mode: str,
    user_id: str | None = None,
    scrubbed_image_dir: Path | None = None,
) -> torch.Tensor:
    """Build image input tensor under one of:
    - 'original': decode PNG bytes from parquet, resize, CLIP-process
    - 'blank':    mid-gray placeholder (no visual signal)
    - 'noise':    random uniform pixels (sanity baseline)
    - 'scrubbed': load pre-scrubbed PNG from `scrubbed_image_dir/<user_id>.png`
                  (PII values masked offline by code/attacks/task2/prompt/scrub_image.py).
                  Better task/ proxy than 'blank' — preserves Name + caption + layout.
    """
    if mode == "original":
        pil_img = load_image(image_bytes, image_size)
    elif mode == "blank":
        pil_img = Image.new("RGB", (image_size, image_size), color=(127, 127, 127))
    elif mode == "noise":
        import numpy as np
        arr = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
    elif mode == "scrubbed":
        if not user_id or scrubbed_image_dir is None:
            raise ValueError("scrubbed mode requires user_id + scrubbed_image_dir")
        path = Path(scrubbed_image_dir) / f"{user_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"Scrubbed image missing: {path}")
        pil_img = Image.open(path).convert("RGB")
        pil_img = pil_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    else:
        raise ValueError(f"unknown image_mode: {mode!r}")
    return preprocess_image_to_tensor(pil_img, image_processor)


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    image_processor,
    image_size: int,
    get_formatted_question,
    sample: Sample,
    max_new_tokens: int = 50,
    use_prefix: bool = True,
    image_mode: str = "original",
    scrubbed_image_dir: Path | None = None,
    strategy: str = "baseline",
) -> str:
    """Generate one PII prediction. Returns the raw (post-prefix) text.

    `strategy` selects prompt construction from `strategies.STRATEGIES`.
    For 'baseline' the legacy `use_prefix` flag still controls whether the
    scrubbed-output prefix is appended (kept for --no_prefix smoke runs).
    For other strategies the prompt structure is fully dictated by the
    strategy function; use_prefix is ignored.
    """
    if strategy == "baseline":
        prefix = derive_assistant_prefix(sample.scrubbed_output) if use_prefix else ""
        prompt_text = build_prompt_text(
            tokenizer, get_formatted_question, sample.question, prefix
        )
    else:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy!r}. Available: {list(STRATEGIES)}")
        prompt_text = STRATEGIES[strategy](sample, get_formatted_question, tokenizer)

    # Tokenize like the codebase collator does (tokenize then convert_to_ids).
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=model.device)

    # Image preprocessing — supports image_mode for ablation
    image_tensor = _build_image_tensor(
        sample.image_bytes, image_size, image_processor, image_mode,
        user_id=sample.user_id, scrubbed_image_dir=scrubbed_image_dir,
    ).to(model.device)

    # Use the codebase's overridden `generate`. It accepts the unified-arch
    # batched inputs directly and calls `prepare_multimodal_inputs` internally
    # (see src/lmms/models/unified_mllm.py:99). Per src/README.md.
    # autocast: matches inference_example.py — projector outputs FP32 but LLM
    # weights are bf16; without autocast we hit "mat1 float != mat2 bf16".
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gen_out = model.generate(
            batch_input_ids=[input_ids],
            batch_labels=[torch.full_like(input_ids, -100)],
            batch_X_modals=[{"<image>": image_tensor}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # When inputs_embeds-based, generate returns only new tokens (no input prepended).
    new_tokens = gen_out[0]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded


def predict_one(
    model,
    tokenizer,
    image_processor,
    image_size: int,
    get_formatted_question,
    sample: Sample,
    max_new_tokens: int = 50,
    use_prefix: bool = True,
) -> str:
    """End-to-end: generate → extract PII → validate format. Returns final pred."""
    raw = generate_one(
        model,
        tokenizer,
        image_processor,
        image_size,
        get_formatted_question,
        sample,
        max_new_tokens=max_new_tokens,
        use_prefix=use_prefix,
    )
    extracted = extract_pii(raw, sample.pii_type)
    validated = validate_pred(extracted, sample.pii_type)
    return validated
