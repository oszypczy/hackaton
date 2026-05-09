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


def setup_codebase_path(codebase_dir: Path) -> None:
    """Insert codebase root at sys.path[0] so `from src.lmms...` works."""
    p = str(codebase_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


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
) -> str:
    """Generate one PII prediction. Returns the raw (post-prefix) text."""
    prefix = derive_assistant_prefix(sample.scrubbed_output) if use_prefix else ""
    prompt_text = build_prompt_text(
        tokenizer, get_formatted_question, sample.question, prefix
    )

    # Tokenize like the codebase collator does (tokenize then convert_to_ids).
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
    input_ids = torch.tensor(token_ids, dtype=torch.long)

    # Image preprocessing
    pil_img = load_image(sample.image_bytes, image_size)
    image_tensor = preprocess_image_to_tensor(pil_img, image_processor)

    # Build batch of size 1
    batch_input_ids = [input_ids]
    batch_labels = [torch.full_like(input_ids, -100)]  # all masked, generation only
    batch_X_modals = [{"<image>": image_tensor.to(model.device)}]

    # Embed + interleave images
    inputs = model.prepare_multimodal_inputs(
        batch_input_ids=batch_input_ids,
        batch_labels=batch_labels,
        batch_X_modals=batch_X_modals,
    )

    # The custom `generate` in src/lmms/models/utils/generation_utils.py:1548
    # expects inputs_embeds as either a tensor or a list
    # [embeds, mask_text, mask_video, mask_audio, mask_question]. The
    # codebase's `prepare_multimodal_inputs` already returns the right shape
    # depending on the model's `inputs_embeds_with_mmask` flag — pass through.
    inputs_embeds = inputs["inputs_embeds"]
    attention_mask = inputs["attention_mask"]

    # Generate
    gen_out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # When using inputs_embeds, generate returns only new tokens (no input prepended).
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
