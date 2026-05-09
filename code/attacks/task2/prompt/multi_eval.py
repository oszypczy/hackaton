"""Multi-strategy parallel evaluation.

For each sample in (a stratified subset of) validation_pii, run N prompt
strategies through the model and score each. Cheaper than N full evals
because the model is loaded once and image preprocessing is shared.

Calibrator: use --image_mode blank — task/ leaderboard score (0.31)
matches val_pii blank-mode (0.30). Lifts above the blank-baseline are
real memorization signal.

Output JSON schema:
    {
      "config": {...},
      "per_strategy": {
        "<name>": {
            "scores": {"CREDIT": {mean, n, perfect}, "EMAIL": ..., "PHONE": ..., "OVERALL": ...},
            "rows": [{user_id, pii_type, gt, pred, raw}, ...]
        }, ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from attack import (
    _build_image_tensor,
    load_model_and_tools,
)
from format import extract_pii, validate_pred, email_fallback_from_question, looks_like_phone, PHONE_FALLBACK
from loader import load_parquets
from scorer import score_batch, _sanity
from strategies import STRATEGIES


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--codebase_dir", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--output_log", type=Path, required=True)
    p.add_argument(
        "--image_mode",
        choices=["original", "blank", "noise", "scrubbed"],
        default="blank",
        help="blank is the calibrator for task/ conditions; scrubbed loads "
             "pre-scrubbed PNGs from --scrubbed_image_dir.",
    )
    p.add_argument(
        "--scrubbed_image_dir",
        type=Path,
        default=None,
        help="Required for image_mode=scrubbed. Folder with <user_id>.png files.",
    )
    p.add_argument(
        "--strategies",
        type=str,
        default=",".join(STRATEGIES.keys()),
        help="Comma-separated subset of strategy names.",
    )
    p.add_argument(
        "--per_type",
        type=int,
        default=50,
        help="Samples per pii_type (stratified). 50/type × 3 types = 150 total.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16")
    return p.parse_args()


@torch.no_grad()
def _generate_with_prompt(
    model, tokenizer, prompt_text: str, image_tensor: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Tokenize prompt_text exactly like attack.generate_one, run model.generate."""
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=model.device)
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
    return tokenizer.decode(gen_out[0], skip_special_tokens=True)


def _stratified_subset(samples, per_type: int, seed: int):
    rng = random.Random(seed)
    by_type: dict[str, list] = {"CREDIT": [], "EMAIL": [], "PHONE": []}
    for s in samples:
        if s.pii_type in by_type:
            by_type[s.pii_type].append(s)
    out = []
    for t, lst in by_type.items():
        if len(lst) <= per_type:
            out.extend(lst)
        else:
            out.extend(rng.sample(lst, per_type))
    rng.shuffle(out)
    return out


def _post_process(pred_raw: str, sample) -> str:
    """Mirror main.py post-processing: EMAIL fallback, PHONE force '+', PHONE
    16-digit fallback, then validate_pred. Identical to production pipeline."""
    extracted = extract_pii(pred_raw, sample.pii_type)
    if sample.pii_type == "EMAIL" and "@" not in extracted:
        extracted = email_fallback_from_question(sample.question)
    elif sample.pii_type == "PHONE" and not looks_like_phone(extracted):
        extracted = PHONE_FALLBACK
    return validate_pred(extracted, sample.pii_type)


def main() -> None:
    args = cli()
    _sanity()

    if args.model_dir is None:
        args.model_dir = args.data_dir / "target_lmm"

    chosen = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for name in chosen:
        if name not in STRATEGIES:
            raise SystemExit(f"unknown strategy: {name!r}. Available: {list(STRATEGIES)}")

    if args.image_mode == "scrubbed" and args.scrubbed_image_dir is None:
        raise SystemExit("--image_mode scrubbed requires --scrubbed_image_dir")

    print(f"[multi_eval] strategies={chosen}")
    print(f"[multi_eval] image_mode={args.image_mode}  per_type={args.per_type}")
    if args.scrubbed_image_dir:
        print(f"[multi_eval] scrubbed_image_dir={args.scrubbed_image_dir}")

    samples = load_parquets(args.data_dir / "validation_pii", with_gt=True)
    samples = _stratified_subset(samples, args.per_type, args.seed)
    print(f"[multi_eval] subset size={len(samples)} (stratified, seed={args.seed})")

    model, tokenizer, image_processor, image_size, get_fmt_q = load_model_and_tools(
        codebase_dir=args.codebase_dir,
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"[multi_eval] model loaded. image_size={image_size}")

    # Pre-build image tensors per sample to avoid redundant CLIP preprocessing
    # across strategies. blank/noise modes share a single tensor.
    print(f"[multi_eval] preprocessing {len(samples)} images ({args.image_mode})...")
    image_tensors: list[torch.Tensor] = []
    for s in samples:
        t = _build_image_tensor(
            s.image_bytes, image_size, image_processor, args.image_mode,
            user_id=s.user_id, scrubbed_image_dir=args.scrubbed_image_dir,
        ).to(model.device)
        image_tensors.append(t)

    per_strategy: dict[str, dict] = {name: {"rows": []} for name in chosen}
    total = len(samples) * len(chosen)
    done = 0
    t0 = time.time()

    for i, (sample, img_t) in enumerate(zip(samples, image_tensors)):
        for name in chosen:
            prompt = STRATEGIES[name](sample, get_fmt_q, tokenizer)
            try:
                raw = _generate_with_prompt(
                    model, tokenizer, prompt, img_t, args.max_new_tokens,
                )
            except Exception as e:
                raw = f"<ERROR: {type(e).__name__}: {e}>"
            pred = _post_process(raw, sample)
            per_strategy[name]["rows"].append({
                "user_id": sample.user_id,
                "pii_type": sample.pii_type,
                "gt": sample.gt_pii,
                "pred": pred,
                "raw": raw[:200],
            })
            done += 1
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            rate = done / dt
            eta = (total - done) / rate
            print(f"[multi_eval] sample {i+1}/{len(samples)}  {done}/{total} forwards  "
                  f"{rate:.2f}/s  ETA {eta/60:.1f} min")

    # Score per strategy
    for name in chosen:
        items = [{"pii_type": r["pii_type"], "gt": r["gt"], "pred": r["pred"]}
                 for r in per_strategy[name]["rows"]]
        per_strategy[name]["scores"] = score_batch(items)

    # Print summary table
    print("\n[multi_eval] === SUMMARY (image_mode=" + args.image_mode + ") ===")
    print(f"{'strategy':22s} {'CREDIT':>8s} {'EMAIL':>8s} {'PHONE':>8s} {'OVERALL':>9s}")
    for name in chosen:
        sc = per_strategy[name]["scores"]
        c = sc.get("CREDIT", {}).get("mean", 0)
        e = sc.get("EMAIL", {}).get("mean", 0)
        p = sc.get("PHONE", {}).get("mean", 0)
        o = sc["OVERALL"]["mean"]
        print(f"{name:22s} {c:8.4f} {e:8.4f} {p:8.4f} {o:9.4f}")

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "strategies": chosen,
            "image_mode": args.image_mode,
            "scrubbed_image_dir": str(args.scrubbed_image_dir) if args.scrubbed_image_dir else None,
            "per_type": args.per_type,
            "seed": args.seed,
            "n_samples": len(samples),
            "max_new_tokens": args.max_new_tokens,
        },
        "per_strategy": per_strategy,
    }
    with open(args.output_log, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"[multi_eval] log → {args.output_log}")
    print(f"[multi_eval] total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
