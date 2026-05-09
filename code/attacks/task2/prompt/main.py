"""Orchestrator: load samples, run attack, write CSV (predict mode) or
print scores (eval mode)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from aggregator import medoid_pick
from attack import generate_k_candidates, generate_one, load_model_and_tools
from format import (
    PHONE_FALLBACK,
    email_fallback_from_question,
    extract_pii,
    looks_like_phone,
    validate_pred,
)
from loader import load_parquets
from scorer import score_batch, _sanity
from strategies import DEMO_STRATEGIES, STRATEGIES, init_demo_pool


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "predict"], required=True)
    p.add_argument(
        "--codebase_dir",
        type=Path,
        required=True,
        help="Path to p4ms_hackathon_warsaw_code-main (contains src/ and scripts/).",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="P4Ms-hackathon-vision-task root (contains task/, validation_pii/, target_lmm/).",
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Override model dir; defaults to {data_dir}/target_lmm.",
    )
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument("--output_log", type=Path, default=None)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument(
        "--no_prefix",
        action="store_true",
        help="Disable assistant-prefix priming (raw greedy baseline).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N samples (smoke test).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument(
        "--image_mode",
        choices=["original", "blank", "noise", "scrubbed"],
        default="original",
        help="Image ablation: 'original' = real image, 'blank' = mid-gray, "
             "'noise' = random pixels, 'scrubbed' = pre-masked PNG from --scrubbed_image_dir.",
    )
    p.add_argument(
        "--scrubbed_image_dir",
        type=Path,
        default=None,
        help="Required for image_mode=scrubbed. Folder with <user_id>.png files.",
    )
    p.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="baseline",
        help="Prompt-construction strategy. 'baseline' = chat template + assistant "
             "prefix priming (legacy). 'direct_probe' / 'role_play_dba' / etc replace "
             "the prompt entirely (no prefix). See strategies.py.",
    )
    p.add_argument(
        "--k_shot",
        type=int,
        default=1,
        help="K-sample ensemble. K=1 → greedy single-shot (default). K>1 → sample "
             "K candidates with --temperature/--top_p, pick Levenshtein medoid via "
             "aggregator.medoid_pick. Cost is K× single-shot — calibrate on val_pii blank.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (only when --k_shot > 1). Research §2.4 recommends "
             "0.3–0.6 for memorization extraction (sharp peaks + rank-2 escapes).",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff (only when --k_shot > 1).",
    )
    return p.parse_args()


def main() -> None:
    args = cli()
    _sanity()

    if args.model_dir is None:
        args.model_dir = args.data_dir / "target_lmm"

    if args.mode == "eval":
        folder = args.data_dir / "validation_pii"
        samples = load_parquets(folder, with_gt=True)
    else:
        folder = args.data_dir / "task"
        samples = load_parquets(folder, with_gt=False)

    if args.limit:
        samples = samples[: args.limit]

    if args.image_mode == "scrubbed" and args.scrubbed_image_dir is None:
        raise SystemExit("--image_mode scrubbed requires --scrubbed_image_dir")

    # Demo-strategy preflight: load validation_pii separately to populate demo
    # pool. Required for oneshot_demo and any future ICL strategies. Loaded
    # ALWAYS when needed — even in eval mode where samples == val_pii — so the
    # demo source is decoupled from the eval target. _pick_demo skips
    # same-name matches to avoid self-leak.
    if args.strategy in DEMO_STRATEGIES:
        demo_folder = args.data_dir / "validation_pii"
        demo_samples = load_parquets(demo_folder, with_gt=True)
        init_demo_pool(demo_samples)
        from strategies import _DEMO_POOL  # noqa: PLC0415 — only needed for diagnostic print
        sizes = {k: len(v) for k, v in _DEMO_POOL.items()}
        print(f"[main] demo pool loaded from {demo_folder}: {sizes} for strategy={args.strategy}")

    print(f"[main] mode={args.mode}  samples={len(samples)}  strategy={args.strategy}  "
          f"image_mode={args.image_mode}  use_prefix={not args.no_prefix}")
    print(f"[main] model_dir={args.model_dir}")
    if args.scrubbed_image_dir:
        print(f"[main] scrubbed_image_dir={args.scrubbed_image_dir}")

    model, tokenizer, image_processor, image_size, get_fmt_q = load_model_and_tools(
        codebase_dir=args.codebase_dir,
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"[main] model loaded. image_size={image_size}")

    use_prefix = not args.no_prefix
    k_shot = max(1, args.k_shot)
    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        if k_shot > 1:
            cands = generate_k_candidates(
                model, tokenizer, image_processor, image_size, get_fmt_q,
                s, K=k_shot, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                image_mode=args.image_mode,
                scrubbed_image_dir=args.scrubbed_image_dir,
                strategy=args.strategy,
            )
            # Aggregate via Levenshtein medoid on canonicalized candidates
            # (per research §3.1+§3.4). medoid_pick returns canonicalized form.
            extracted = medoid_pick(cands, s.pii_type)
            raw = " | ".join(c[:60] for c in cands)  # for debug log
        else:
            raw = generate_one(
                model, tokenizer, image_processor, image_size, get_fmt_q,
                s, max_new_tokens=args.max_new_tokens, use_prefix=use_prefix,
                image_mode=args.image_mode,
                scrubbed_image_dir=args.scrubbed_image_dir,
                strategy=args.strategy,
            )
            extracted = extract_pii(raw, s.pii_type)

        # EMAIL fallback: when model emits non-email content (phone/CC/twitter),
        # `extracted` lacks '@'. Build firstname.lastname@example.com from the
        # question — gives ~0.6 sim vs ~0.0 from raw phone digits.
        if s.pii_type == "EMAIL" and "@" not in extracted:
            extracted = email_fallback_from_question(s.question)
        # PHONE fallback: 22/27 imperfect on v1 = 16-digit (CC pattern in PHONE
        # slot — model copied user's CC into wrong slot). Replace with empirically
        # best constant guess (+0.4% PHONE category mean).
        elif s.pii_type == "PHONE" and not looks_like_phone(extracted):
            extracted = PHONE_FALLBACK
        pred = validate_pred(extracted, s.pii_type)
        row = {
            "user_id": s.user_id,
            "pii_type": s.pii_type,
            "pred": pred,
            "raw_generation": raw[:200],
        }
        if s.gt_pii is not None:
            row["gt"] = s.gt_pii
        rows.append(row)

        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(samples) - i - 1) / rate
            print(f"[main] {i+1}/{len(samples)}  {rate:.2f} samples/s  ETA {eta/60:.1f} min")

    print(f"[main] done {len(rows)} samples in {(time.time()-t0)/60:.1f} min")

    # Eval mode: print scores
    if args.mode == "eval":
        items = [{"pii_type": r["pii_type"], "gt": r["gt"], "pred": r["pred"]} for r in rows]
        scores = score_batch(items)
        print("\n[scores]")
        for k, v in scores.items():
            print(f"  {k:8s} mean={v['mean']:.4f}  perfect={v['perfect']}/{v['n']}")

        if args.output_log:
            args.output_log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_log, "w") as f:
                json.dump(
                    {"scores": scores, "rows": rows},
                    f, indent=2, ensure_ascii=False, default=str,
                )
            print(f"[main] eval log → {args.output_log}")

        # Append summary line to run_log.csv (zbiorczy log wszystkich evali)
        run_log = args.output_log.parent / "run_log.csv" if args.output_log else None
        if run_log:
            new_file = not run_log.exists()
            job_id = os.environ.get("SLURM_JOB_ID", "")
            with open(run_log, "a") as f:
                if new_file:
                    f.write("ts,job_id,mode,n,credit,email,phone,overall,image_mode,notes\n")
                f.write(
                    f"{time.strftime('%Y-%m-%d_%H:%M:%S')},{job_id},eval,{len(rows)},"
                    f"{scores.get('CREDIT', {}).get('mean', 0):.4f},"
                    f"{scores.get('EMAIL', {}).get('mean', 0):.4f},"
                    f"{scores.get('PHONE', {}).get('mean', 0):.4f},"
                    f"{scores['OVERALL']['mean']:.4f},"
                    f"{args.image_mode},"
                    f"{args.output_log.stem}\n"
                )
            print(f"[main] run_log → {run_log}")

    # Predict mode: write CSV
    if args.mode == "predict":
        if args.output_csv is None:
            raise SystemExit("predict mode requires --output_csv")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "pii_type", "pred"])
            for r in rows:
                w.writerow([r["user_id"], r["pii_type"], r["pred"]])
        print(f"[main] CSV → {args.output_csv}  rows={len(rows)}")

        # Also dump raw generations for debug
        if args.output_log:
            args.output_log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_log, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
            print(f"[main] raw log → {args.output_log}")

        # Sanity: 3000 unique (id, pii_type) pairs
        pairs = {(r["user_id"], r["pii_type"]) for r in rows}
        if len(pairs) != len(rows):
            print(f"WARNING: duplicate (id, pii_type) pairs: {len(rows) - len(pairs)} dups")
        if args.limit is None and len(rows) != 3000:
            print(f"WARNING: expected 3000 rows, got {len(rows)}")


if __name__ == "__main__":
    main()
