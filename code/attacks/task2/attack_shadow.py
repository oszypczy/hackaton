"""
Task 2 — Path B: White-box memorization signal (murdzek2)

Strategy:
  1. Greedy decode from target_lmm  →  common-floor baseline
  2. K=8 sampling candidates  →  rank by Δ = loss_shadow - loss_target
     (lower target loss = stronger memorization → higher Δ = better candidate)
  3. Enforce 10-100 char constraint

Paths on Jülich (run from codebase root):
  CODEBASE = /p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main
  DATA     = /p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task

Run:
  cd $CODEBASE
  source /p/project1/training2615/murdzek2/Hackathon/.venv/bin/activate
  python /p/project1/training2615/murdzek2/Hackathon/code/attacks/task2/attack_shadow.py --mode val
  python /p/project1/training2615/murdzek2/Hackathon/code/attacks/task2/attack_shadow.py --mode submit
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image
from rapidfuzz.distance import Levenshtein

# ── Resolve codebase (must run from codebase dir OR set CODEBASE env var) ─────
CODEBASE = Path(os.environ.get(
    "CODEBASE",
    "/p/scratch/training2615/kempinski1/Czumpers/p4ms_codebase/p4ms_hackathon_warsaw_code-main"
))
sys.path.insert(0, str(CODEBASE))
sys.path.insert(0, str(CODEBASE / "scripts"))

from load_lmm_from_hf_dir import load_lmm, model_setup_inference  # noqa: E402
from src.lmms.dataset.task_dataset import (  # noqa: E402
    get_formatted_question,
    sample_to_chat_template,
)
from src.lmms.utils.util import prepare_sample  # noqa: E402

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
IMG_SIZE = 336   # matches training config

PII_TYPES   = ("EMAIL", "CREDIT", "PHONE")
PII_KW      = {"EMAIL": "email", "CREDIT": "credit", "PHONE": "phone"}
PRED_MIN, PRED_MAX = 10, 100

# ── Model loading ──────────────────────────────────────────────────────────────
def load_both_models():
    print(f"Loading target_lmm from {TARGET_DIR} ...")
    target, tok, margs, dargs, targs = load_lmm(str(TARGET_DIR), device=DEVICE)
    img_proc = target.get_model().visual_encoder.image_processor

    print(f"Loading shadow_lmm from {SHADOW_DIR} ...")
    shadow, _, _, _, _ = load_lmm(str(SHADOW_DIR), device=DEVICE,
                                   do_not_setup_inference=True)
    shadow.eval()
    return target, shadow, tok, img_proc, dargs


# ── Build single-turn batch from raw HF sample + pii_type ─────────────────────
def make_batch(sample, pii_type: str, tokenizer, img_proc,
               include_answer: bool = True):
    """
    Returns a collated batch dict ready for prepare_multimodal_inputs.
    include_answer=False → generation mode (no label tokens).
    include_answer=True  → loss scoring mode.
    """
    kw = PII_KW[pii_type]
    conv_turn = next(
        (t for t in sample["conversation"] if kw in t["instruction"].lower()),
        sample["conversation"][0],
    )

    instruction = get_formatted_question(conv_turn["instruction"], "image")
    output = conv_turn["output"] if include_answer else ""

    fake_sample = {
        "conversation": [{"instruction": instruction, "output": output}],
    }
    data = sample_to_chat_template(fake_sample, tokenizer)

    # Tokenize
    full_text = data["conversation"][0]["instruction"]
    if include_answer:
        full_text += data["conversation"][0]["output"] + tokenizer.eos_token

    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    # Labels: -100 for instruction part, token ids for answer part
    if include_answer:
        instr_ids = tokenizer(
            data["conversation"][0]["instruction"], return_tensors="pt"
        )["input_ids"]
        n_instr = instr_ids.shape[1]
        labels = input_ids.clone()
        labels[0, :n_instr] = -100
    else:
        labels = torch.full_like(input_ids, -100)

    # Image
    pil_img = sample["path"]
    if isinstance(pil_img, Image.Image):
        pil_img = pil_img.convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        image_tensor = img_proc.preprocess(pil_img, return_tensors="pt")[
            "pixel_values"
        ]  # (1, C, H, W)
    else:
        image_tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    batch = {
        "batch_input_ids": input_ids,
        "batch_labels": labels,
        "batch_X_modals": [{"image": image_tensor}],
    }
    return batch


# ── Per-token loss → mean answer loss ─────────────────────────────────────────
@torch.no_grad()
def answer_loss(model, batch, device: torch.device) -> float:
    s = prepare_sample(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = model.prepare_multimodal_inputs(
            batch_input_ids=s["batch_input_ids"],
            batch_labels=s["batch_labels"],
            batch_X_modals=s["batch_X_modals"],
        )
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            labels=inputs["labels"],
        )
    # Return mean cross-entropy over labeled (answer) tokens
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = inputs["labels"][:, 1:].contiguous()
    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = (shift_labels.view(-1) != -100)
    if mask.sum() == 0:
        return 0.0
    return per_tok[mask].mean().item()


# ── Greedy generation ──────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answer(model, batch, tokenizer, device: torch.device,
                    do_sample: bool = False, temperature: float = 0.7,
                    max_new_tokens: int = 50) -> str:
    s = prepare_sample(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = model.prepare_multimodal_inputs(
            batch_input_ids=s["batch_input_ids"],
            batch_labels=s["batch_labels"],
            batch_X_modals=s["batch_X_modals"],
        )
        gen_ids = model.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return text


# ── Main prediction: greedy baseline + delta reranking ────────────────────────
def predict(sample, pii_type: str, target, shadow, tokenizer, img_proc,
            k: int = 8, temperature: float = 0.7) -> str:
    dev = torch.device(DEVICE)

    # Generate K candidates from target
    batch_gen = make_batch(sample, pii_type, tokenizer, img_proc, include_answer=False)
    candidates = []
    # greedy first
    candidates.append(generate_answer(target, batch_gen, tokenizer, dev, do_sample=False))
    for _ in range(k - 1):
        candidates.append(
            generate_answer(target, batch_gen, tokenizer, dev,
                            do_sample=True, temperature=temperature)
        )

    # Rank by Δ = loss_shadow - loss_target  (higher = target memorized it more)
    best, best_delta = candidates[0], float("-inf")
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        batch_score = make_batch(
            {**sample, "conversation": [
                {"instruction": next(
                    t["instruction"] for t in sample["conversation"]
                    if PII_KW[pii_type] in t["instruction"].lower()
                ),
                 "output": cand}
            ]},
            pii_type, tokenizer, img_proc, include_answer=True,
        )
        tgt_loss = answer_loss(target, batch_score, dev)
        shd_loss = answer_loss(shadow, batch_score, dev)
        delta = shd_loss - tgt_loss
        if delta > best_delta:
            best, best_delta = cand, delta

    return best


# ── Length enforcement ─────────────────────────────────────────────────────────
def enforce_length(pred: str, pii_type: str) -> str:
    pred = pred.strip().strip('"\'<>').strip()
    if pii_type == "EMAIL" and "@" not in pred and len(pred) < PRED_MIN:
        pred = pred + "@example.com"
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


# ── Sanity check ───────────────────────────────────────────────────────────────
def _sanity():
    assert abs(Levenshtein.distance("abc", "ab", weights=(1, 1, 1)) / max(3, 2) - 1/3) < 1e-9, \
        "Levenshtein sanity FAILED — wrong variant"


# ── Validation run ─────────────────────────────────────────────────────────────
def run_val(target, shadow, tokenizer, img_proc):
    ds = load_from_disk(str(VAL_DIR))["train"]
    scores = {t: [] for t in PII_TYPES}

    for i, sample in enumerate(ds):
        for pii_type in PII_TYPES:
            gt_turn = next(
                (t for t in sample["conversation"]
                 if PII_KW[pii_type] in t["instruction"].lower()),
                sample["conversation"][0],
            )
            gt = gt_turn["output"]
            pred = enforce_length(
                predict(sample, pii_type, target, shadow, tokenizer, img_proc),
                pii_type,
            )
            scores[pii_type].append(similarity(gt, pred))

        if (i + 1) % 10 == 0:
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
def run_submit(target, shadow, tokenizer, img_proc):
    ds = load_from_disk(str(TASK_DIR))["train"]
    rows = []

    for i, sample in enumerate(ds):
        sid = sample["user_id"]
        for pii_type in PII_TYPES:
            pred = enforce_length(
                predict(sample, pii_type, target, shadow, tokenizer, img_proc),
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
    parser.add_argument("--mode", choices=["val", "submit"], default="val")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of sampling candidates for delta reranking")
    parser.add_argument("--greedy-only", action="store_true",
                        help="Skip delta reranking (fast baseline, no shadow model)")
    args = parser.parse_args()

    target, shadow, tokenizer, img_proc, _ = load_both_models()

    if args.greedy_only:
        # Override predict to greedy-only (skips shadow load cost in testing)
        def predict(sample, pii_type, target, shadow, tokenizer, img_proc, k=1, temperature=0.7):
            dev = torch.device(DEVICE)
            batch = make_batch(sample, pii_type, tokenizer, img_proc, include_answer=False)
            return generate_answer(target, batch, tokenizer, dev, do_sample=False)

    if args.mode == "val":
        run_val(target, shadow, tokenizer, img_proc)
    else:
        run_submit(target, shadow, tokenizer, img_proc)
