from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FULL_VARIANT_NAME = "FULL_A+BC+D+Binoculars"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick best task3 variant and optionally submit")
    p.add_argument(
        "--report",
        default="code/attacks/task3/results/ablation_report.json",
        help="Path to ablation report JSON",
    )
    p.add_argument(
        "--full-csv",
        default="submissions/task3_watermark_detection_full.csv",
        help="CSV produced by full variant",
    )
    p.add_argument(
        "--conservative-csv",
        default="submissions/task3_watermark_detection_conservative.csv",
        help="CSV produced by conservative variant",
    )
    p.add_argument("--task", default="task3", help="Task name for just submit")
    p.add_argument(
        "--yes",
        action="store_true",
        help="Submit without interactive confirmation",
    )
    return p.parse_args()


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected report JSON list")
    rows = [r for r in data if r.get("trunc_words") is None]
    if not rows:
        raise ValueError("No rows for trunc_words=None in report")
    return rows


def _score_rows(rows: list[dict]) -> list[dict]:
    scored = []
    for r in rows:
        cv_mean = float(r.get("cv_mean", 0.0))
        cv_std = float(r.get("cv_std", 0.0))
        q5 = float(r.get("bootstrap_tpr_q5", 0.0))
        robust = min(cv_mean - cv_std, q5)
        out = dict(r)
        out["robust"] = robust
        scored.append(out)
    scored.sort(key=lambda x: x["robust"], reverse=True)
    return scored


def _print_table(rows: list[dict], top_k: int = 6) -> None:
    print("Top variants (trunc_words=None):")
    for i, r in enumerate(rows[:top_k], start=1):
        print(
            f"{i:>2}. {r.get('variant','?'):<28} "
            f"robust={r['robust']:.4f}  "
            f"cv={float(r.get('cv_mean',0)):.4f}±{float(r.get('cv_std',0)):.4f}  "
            f"q5={float(r.get('bootstrap_tpr_q5',0)):.4f}  "
            f"q1={float(r.get('tpr_q1',0)):.4f} q4={float(r.get('tpr_q4',0)):.4f}"
        )


def _pick_csv(best_variant: str, full_csv: Path, conservative_csv: Path) -> Path:
    if best_variant == FULL_VARIANT_NAME:
        return full_csv
    return conservative_csv


def main() -> int:
    args = parse_args()
    rows = _load_rows(Path(args.report))
    scored = _score_rows(rows)
    _print_table(scored)

    best = scored[0]
    chosen_csv = _pick_csv(
        best_variant=str(best.get("variant", "")),
        full_csv=Path(args.full_csv),
        conservative_csv=Path(args.conservative_csv),
    )
    print("\nSelected variant:")
    print(
        f"- variant={best.get('variant')} robust={best['robust']:.4f} "
        f"cv={float(best.get('cv_mean',0)):.4f}±{float(best.get('cv_std',0)):.4f} "
        f"q5={float(best.get('bootstrap_tpr_q5',0)):.4f}"
    )
    print(f"- chosen CSV: {chosen_csv}")

    if not chosen_csv.exists():
        print(f"ERROR: chosen CSV does not exist: {chosen_csv}", file=sys.stderr)
        return 1

    cmd = ["just", "submit", args.task, str(chosen_csv)]
    print(f"\nPlanned submit command:\n{' '.join(cmd)}")
    if args.yes:
        submit = True
    else:
        answer = input("Submit now? [y/N]: ").strip().lower()
        submit = answer in {"y", "yes"}

    if not submit:
        print("Submission skipped.")
        return 0

    result = subprocess.run(cmd)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
