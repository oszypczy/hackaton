"""Orchestrator: load samples, run attack, write CSV (predict mode) or
print scores (eval mode)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from attack import generate_one, load_model_and_tools
from format import extract_pii, validate_pred
from loader import load_parquets
from scorer import score_batch, _sanity


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

    print(f"[main] mode={args.mode}  samples={len(samples)}  use_prefix={not args.no_prefix}")
    print(f"[main] model_dir={args.model_dir}")

    model, tokenizer, image_processor, image_size, get_fmt_q = load_model_and_tools(
        codebase_dir=args.codebase_dir,
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"[main] model loaded. image_size={image_size}")

    use_prefix = not args.no_prefix
    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        raw = generate_one(
            model, tokenizer, image_processor, image_size, get_fmt_q,
            s, max_new_tokens=args.max_new_tokens, use_prefix=use_prefix,
        )
        extracted = extract_pii(raw, s.pii_type)
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
