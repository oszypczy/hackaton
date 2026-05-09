#!/usr/bin/env python3
"""Pull submission CSV from Jülich Czumpers/<task>/<file> to local submissions/.

Usage: python scripts/pull_csv.py <task> [remote_filename]
       task: task1 | task2 | task3
       remote_filename: defaults to "submission.csv"
                        e.g. "submission_kgw.csv", "submission_strong_bino.csv"

Local file gets a name derived from remote_filename:
  submission.csv             -> task3_watermark.csv (default)
  submission_kgw.csv         -> task3_watermark_kgw.csv
  submission_strong_bino.csv -> task3_watermark_strong_bino.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_MAP: dict[str, dict[str, str]] = {
    "task1": {"cluster_dir": "DUCI", "local_prefix": "task1_duci"},
    "task2": {"cluster_dir": "P4Ms-hackathon-vision-task", "local_prefix": "task2_pii"},
    "task3": {"cluster_dir": "task3", "local_prefix": "task3_watermark"},
}

CLUSTER_BASE = "/p/scratch/training2615/kempinski1/Czumpers"


def _local_name(local_prefix: str, remote_filename: str) -> str:
    """Derive local CSV name from remote filename + task prefix."""
    stem = Path(remote_filename).stem  # "submission" or "submission_kgw"
    if stem == "submission":
        return f"{local_prefix}.csv"
    # "submission_kgw" -> "_kgw"
    suffix = stem[len("submission"):] if stem.startswith("submission") else f"_{stem}"
    return f"{local_prefix}{suffix}.csv"


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("Usage: pull_csv.py <task1|task2|task3> [remote_filename]", file=sys.stderr)
        return 2
    task_name = argv[1]
    remote_filename = argv[2] if len(argv) == 3 else "submission.csv"

    if task_name not in TASK_MAP:
        print(f"Unknown task {task_name!r}. Use task1, task2, or task3.", file=sys.stderr)
        return 2

    task = TASK_MAP[task_name]
    remote = f"{CLUSTER_BASE}/{task['cluster_dir']}/{remote_filename}"
    local_dir = REPO_ROOT / "submissions"
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / _local_name(task["local_prefix"], remote_filename)

    print(f"Fetching {remote} → {local_path}")
    proc = subprocess.run(
        [str(REPO_ROOT / "scripts" / "juelich_exec.sh"), f"cat {remote}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"juelich_exec failed (exit {proc.returncode}):", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    local_path.write_text(proc.stdout)
    size = local_path.stat().st_size
    rows = proc.stdout.count("\n")
    print(f"OK: {size} bytes, {rows} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
