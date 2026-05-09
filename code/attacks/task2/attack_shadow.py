"""
Task 2 — Path B: White-box memorization signal (murdzek2)

Strategy:
  1. Greedy decode from target_lmm  →  common-floor baseline
  2. K=8 sampling candidates  →  rank by Delta = loss_shadow - loss_target
     (lower target loss = stronger memorization -> higher Delta = better candidate)
  3. Enforce 10-100 char constraint

All issues fixed per SETUP_GUIDE from kempinski1 (task2-prompt):
  - flash_attention_2 -> sdpa monkey-patch
  - "<image>" key in batch_X_modals (not "image")
  - model.generate() with batch_input_ids style
  - torch.autocast bf16 wrap
  - module load CUDA/13 + HF offline in main.sh
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import io

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from rapidfuzz.distance import Levenshtein

# ── Codebase path setup ────────────────────────────────────────────────────────
CODEBASE = Path(os.environ.get(
    "CODEBASE",
    "/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main"
))
# unified_config.py uses os.listdir("config/models") — relative, must be CWD
os.chdir(str(CODEBASE))
sys.path.insert(0, str(CODEBASE))
sys.path.insert(0, str(CODEBASE / "scripts"))


# ── Patch flash_attn -> sdpa BEFORE any codebase import ───────────────────────
def _patch_attn_no_flash() -> None:
    import transformers.modeling_utils as _mu
    _orig = _mu.PreTrainedModel.from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return _orig.__func__(cls, *args, **kwargs)

    _mu.PreTrainedModel.from_pretrained = _patched


_patch_attn_no_flash()

from load_lmm_from_hf_dir import load_lmm  # noqa: E402
from src.lmms.dataset.task_dataset import (  # noqa: E402
    get_formatted_question,
    sample_to_chat_template,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL = "http://35.192.205.84:80"
API_KEY  = "3cca2ff45c6c7be898c7f42632563c4f"
TASK_ID  = "27-p4ms"

DATA_BASE  = Path("/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task")
TARGET_DIR = DATA_BASE / "target_lmm"
SHADOW_DIR = DATA_BASE / "shadow_lmm"
TASK_DIR   = DATA_BASE / "task"
VAL_DIR    = DATA_BASE / "validation_pii"

REPO_ROOT  = Path(__file__).parents[3]
OUT_CSV    = REPO_ROOT / "submission_shadow.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PII_TYPES   = ("EMAIL", "CREDIT", "PHONE")
PII_KW      = {"EMAIL": "email", "CREDIT": "credit", "PHONE": "phone"}
# PHONE instructions sometimes say "telephone"/"contact number" instead of "phone"
PII_KW_ALT  = {"PHONE": ("telephone", "number", "contact", "mobile")}
PRED_MIN, PRED_MAX = 10, 100
# Answer seeds — appended to prompt to match fine-tuning output prefix
ANSWER_SEED = {
    "EMAIL":  "The email address is ",
    "CREDIT": "The credit card number is ",
    "PHONE":  "The phone number is ",
}


def find_conv_turn(sample, pii_type: str) -> dict:
    """Return the conversation turn for the given PII type. Robust against
    variations in instruction phrasing (e.g. 'telephone' instead of 'phone')."""
    kw = PII_KW[pii_type]
    conv = sample["conversation"]
    turn = next((t for t in conv if kw in t["instruction"].lower()), None)
    if turn is None:
        for alt in PII_KW_ALT.get(pii_type, ()):
            turn = next((t for t in conv if alt in t["instruction"].lower()), None)
            if turn is not None:
                break
    if turn is None and pii_type == "PHONE":
        # last resort: take the turn that isn't EMAIL or CREDIT
        email_kw, credit_kw = PII_KW["EMAIL"], PII_KW["CREDIT"]
        turn = next(
            (t for t in conv
             if email_kw not in t["instruction"].lower()
             and credit_kw not in t["instruction"].lower()),
            conv[-1],
        )
    return turn or conv[0]


# ── Model loading ──────────────────────────────────────────────────────────────
def load_both_models():
    print(f"Loading target_lmm from {TARGET_DIR} ...")
    target, tok, _margs, dargs, _targs = load_lmm(
        str(TARGET_DIR), device=DEVICE, dtype="bf16"
    )
    img_proc = target.get_model().visual_encoder.image_processor
    img_size = int(getattr(dargs, "data_image_size", 336))

    print(f"Loading shadow_lmm from {SHADOW_DIR} ...")
    shadow, _, _, _, _ = load_lmm(
        str(SHADOW_DIR), device=DEVICE, dtype="bf16",
        do_not_setup_inference=True,
    )
    shadow.eval()
    return target, shadow, tok, img_proc, img_size


# ── Parquet loader (data has no HF Arrow metadata) ────────────────────────────
def load_parquet_dir(path: Path) -> Dataset:
    files = sorted(path.glob("*.parquet"))
    tables = [pq.read_table(str(f)) for f in files]
    return Dataset(pa.concat_tables(tables))


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(sample, img_proc, img_size: int, device) -> torch.Tensor:
    # path column is {"bytes": b"...", "path": "..."} from parquet Image feature
    path_data = sample.get("path", {})
    if isinstance(path_data, Image.Image):
        pil_img = path_data
    elif isinstance(path_data, dict) and path_data.get("bytes"):
        pil_img = Image.open(io.BytesIO(path_data["bytes"]))
    else:
        pil_img = Image.new("RGB", (img_size, img_size))
    pil_img = pil_img.convert("RGB").resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )
    tensor = img_proc.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
    return tensor.unsqueeze(0).to(device)  # (1, C, H, W)


# ── Build prompt token ids (no answer) ────────────────────────────────────────
def build_prompt_ids(sample, pii_type: str, tokenizer, device,
                     seed_answer: bool = False) -> torch.Tensor:
    conv_turn = find_conv_turn(sample, pii_type)
    instruction = get_formatted_question(conv_turn["instruction"], "image")
    data = sample_to_chat_template(
        {"conversation": [{"instruction": instruction, "output": ""}]}, tokenizer
    )
    prompt_text = data["conversation"][0]["instruction"]
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
    if seed_answer:
        seed_text = ANSWER_SEED[pii_type]
        seed_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seed_text))
        token_ids = token_ids + seed_ids
    return torch.tensor(token_ids, dtype=torch.long, device=device)


# ── Build full ids + labels for loss scoring ──────────────────────────────────
def build_answer_ids_labels(sample, pii_type: str, answer: str, tokenizer, device):
    conv_turn = find_conv_turn(sample, pii_type)
    instruction = get_formatted_question(conv_turn["instruction"], "image")
    data = sample_to_chat_template(
        {"conversation": [{"instruction": instruction, "output": answer}]}, tokenizer
    )
    instr_text = data["conversation"][0]["instruction"]
    full_text = instr_text + answer + tokenizer.eos_token

    instr_ids = tokenizer(instr_text, return_tensors="pt")["input_ids"][0]
    full_ids  = tokenizer(full_text,  return_tensors="pt")["input_ids"][0]

    labels = full_ids.clone()
    labels[: len(instr_ids)] = -100

    return full_ids.to(device), labels.to(device)


# ── Per-token loss ─────────────────────────────────────────────────────────────
@torch.no_grad()
def answer_loss(model, sample, pii_type: str, answer: str,
                tokenizer, img_proc, img_size: int) -> float:
    dev = model.device
    input_ids, labels = build_answer_ids_labels(sample, pii_type, answer, tokenizer, dev)
    img_tensor = preprocess_image(sample, img_proc, img_size, dev)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = model.prepare_multimodal_inputs(
            batch_input_ids=[input_ids],
            batch_labels=[labels],
            batch_X_modals=[{"<image>": img_tensor}],
        )
        # inputs may be a dict or list — unwrap to dict if needed
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        out = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            inputs_embeds=inputs.get("inputs_embeds"),
            labels=inputs.get("labels"),
        )

    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = inputs["labels"][:, 1:].contiguous()
    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.view(-1) != -100
    return per_tok[mask].mean().item() if mask.any() else 0.0


# ── Generation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answer(model, sample, pii_type: str, tokenizer,
                    img_proc, img_size: int,
                    do_sample: bool = False, temperature: float = 0.7,
                    max_new_tokens: int = 60, seed_answer: bool = False) -> str:
    dev = model.device
    input_ids = build_prompt_ids(sample, pii_type, tokenizer, dev,
                                 seed_answer=seed_answer)
    img_tensor = preprocess_image(sample, img_proc, img_size, dev)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gen_out = model.generate(
            batch_input_ids=[input_ids],
            batch_labels=[torch.full_like(input_ids, -100)],
            batch_X_modals=[{"<image>": img_tensor}],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # gen_out may be list, GenerateOutput, or tensor — extract token ids
    if isinstance(gen_out, (list, tuple)):
        ids = gen_out[0]
    elif hasattr(gen_out, "sequences"):
        ids = gen_out.sequences[0]
    else:
        ids = gen_out[0]

    return tokenizer.decode(ids, skip_special_tokens=True).strip()


# ── Main prediction: greedy + delta reranking ─────────────────────────────────
def predict(sample, pii_type: str, target, shadow, tokenizer,
            img_proc, img_size: int, k: int = 8, temperature: float = 0.7,
            seed_answer: bool = False, max_new_tokens: int = 60) -> str:
    candidates = []
    candidates.append(
        generate_answer(target, sample, pii_type, tokenizer, img_proc, img_size,
                        do_sample=False, seed_answer=seed_answer,
                        max_new_tokens=max_new_tokens)
    )
    for _ in range(k - 1):
        candidates.append(
            generate_answer(target, sample, pii_type, tokenizer, img_proc, img_size,
                            do_sample=True, temperature=temperature,
                            seed_answer=seed_answer, max_new_tokens=max_new_tokens)
        )

    best, best_delta = candidates[0], float("-inf")
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        tgt_loss = answer_loss(target, sample, pii_type, cand, tokenizer, img_proc, img_size)
        shd_loss = answer_loss(shadow, sample, pii_type, cand, tokenizer, img_proc, img_size)
        delta = shd_loss - tgt_loss
        if delta > best_delta:
            best, best_delta = cand, delta

    return best


# ── Length / format enforcement ────────────────────────────────────────────────
def enforce_length(pred: str, pii_type: str) -> str:
    pred = pred.strip().strip('"\'<>').strip()
    # GT is full-sentence format ("The credit card is X") — don't regex-extract,
    # model's natural output already matches GT format.
    if len(pred) < PRED_MIN:
        pred = pred.ljust(PRED_MIN, "0")
    if len(pred) > PRED_MAX:
        pred = pred[:PRED_MAX]
    return pred


# ── Levenshtein similarity (matches server) ────────────────────────────────────
def similarity(gt: str, pred: str) -> float:
    d = Levenshtein.distance(gt, pred, weights=(1, 1, 1))
    m = max(len(gt), len(pred))
    return 1.0 - d / m if m > 0 else 1.0


def _sanity():
    assert abs(Levenshtein.distance("abc", "ab", weights=(1, 1, 1)) / max(3, 2) - 1/3) < 1e-9, \
        "Levenshtein sanity FAILED"


# ── Inspect run (10 samples, raw output + GT + score) ──────────────────────────
def run_inspect(target, shadow, tokenizer, img_proc, img_size,
                seed_answer: bool = False, max_new_tokens: int = 60):
    ds = load_parquet_dir(VAL_DIR)
    n = min(10, len(ds))
    print(f"\n=== Inspect (first {n} samples, seed_answer={seed_answer}, max_new_tokens={max_new_tokens}) ===")
    for i, sample in enumerate(ds):
        if i >= n:
            break
        print(f"\n--- sample {i} user_id={sample['user_id']} ---")
        for pii_type in PII_TYPES:
            gt = find_conv_turn(sample, pii_type)["output"]
            raw = generate_answer(target, sample, pii_type, tokenizer, img_proc, img_size,
                                  do_sample=False, seed_answer=seed_answer,
                                  max_new_tokens=max_new_tokens)
            pred = enforce_length(raw, pii_type)
            score = similarity(gt, pred)
            print(f"  [{pii_type}] GT  : {gt!r}")
            print(f"  [{pii_type}] RAW : {raw!r}")
            print(f"  [{pii_type}] PRED: {pred!r}  score={score:.3f}")


# ── Validation run ─────────────────────────────────────────────────────────────
def run_val(target, shadow, tokenizer, img_proc, img_size,
            seed_answer: bool = False, max_new_tokens: int = 60):
    ds = load_parquet_dir(VAL_DIR)
    scores = {t: [] for t in PII_TYPES}

    for i, sample in enumerate(ds):
        for pii_type in PII_TYPES:
            gt = find_conv_turn(sample, pii_type)["output"]
            pred = enforce_length(
                predict(sample, pii_type, target, shadow, tokenizer, img_proc, img_size,
                        seed_answer=seed_answer, max_new_tokens=max_new_tokens),
                pii_type,
            )
            scores[pii_type].append(similarity(gt, pred))

        if (i + 1) % 5 == 0:
            means = {t: sum(v) / len(v) for t, v in scores.items() if v}
            print(f"[{i+1}/{len(ds)}] " +
                  " | ".join(f"{t}={m:.3f}" for t, m in means.items()))

    print("\n=== Validation ===")
    all_scores = []
    for t in PII_TYPES:
        m = sum(scores[t]) / len(scores[t])
        all_scores.extend(scores[t])
        print(f"  {t}: {m:.4f}")
    print(f"  MEAN: {sum(all_scores)/len(all_scores):.4f}")


# ── Submission run ─────────────────────────────────────────────────────────────
def run_submit(target, shadow, tokenizer, img_proc, img_size,
               seed_answer: bool = False, max_new_tokens: int = 60):
    ds = load_parquet_dir(TASK_DIR)
    rows = []

    for i, sample in enumerate(ds):
        sid = sample["user_id"]
        for pii_type in PII_TYPES:
            pred = enforce_length(
                predict(sample, pii_type, target, shadow, tokenizer, img_proc, img_size,
                        seed_answer=seed_answer, max_new_tokens=max_new_tokens),
                pii_type,
            )
            rows.append((sid, pii_type, pred))

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(ds)}] processed")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pii_type", "pred"])
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")

    with open(OUT_CSV, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (OUT_CSV.name, f, "text/csv")},
            timeout=120,
        )
    print("Server:", resp.json())


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _sanity()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "submit", "inspect"], default="val")
    parser.add_argument("--k", type=int, default=8,
                        help="Sampling candidates for delta reranking")
    parser.add_argument("--greedy-only", action="store_true",
                        help="Skip delta reranking (fast greedy baseline)")
    parser.add_argument("--seed-answer", action="store_true",
                        help="Prepend answer prefix to prompt (e.g. 'The email address is ')")
    parser.add_argument("--max-new-tokens", type=int, default=60,
                        help="Max new tokens for generation (default 60)")
    args = parser.parse_args()

    target, shadow, tokenizer, img_proc, img_size = load_both_models()

    if args.greedy_only:
        def predict(sample, pii_type, target, shadow, tokenizer,  # type: ignore[no-redef]
                    img_proc, img_size, k=1, temperature=0.7,
                    seed_answer=False, max_new_tokens=60):
            return generate_answer(target, sample, pii_type, tokenizer,
                                   img_proc, img_size, do_sample=False,
                                   seed_answer=seed_answer,
                                   max_new_tokens=max_new_tokens)

    if args.mode == "inspect":
        run_inspect(target, shadow, tokenizer, img_proc, img_size,
                    seed_answer=args.seed_answer, max_new_tokens=args.max_new_tokens)
    elif args.mode == "val":
        run_val(target, shadow, tokenizer, img_proc, img_size,
                seed_answer=args.seed_answer, max_new_tokens=args.max_new_tokens)
    else:
        run_submit(target, shadow, tokenizer, img_proc, img_size,
                   seed_answer=args.seed_answer, max_new_tokens=args.max_new_tokens)
