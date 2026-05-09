#!/usr/bin/env python3
"""Pull submission.csv from Jülich Czumpers/<task>/submission.csv to local submissions/.

Usage: python scripts/pull_csv.py <task>
       task: task1 | task2 | task3
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_MAP: dict[str, dict[str, str]] = {
    "task1": {"remote_path": "DUCI/submission.csv", "local_name": "task1_duci.csv"},
    "task2": {"remote_path": "P4Ms-hackathon-vision-task/submission.csv", "local_name": "task2_pii.csv"},
    "task3": {
        "remote_path": "repo-multan1/submissions/task3_watermark_detection_full.csv",
        "local_name": "task3_watermark.csv",
    },
}

CLUSTER_BASE = "/p/scratch/training2615/kempinski1/Czumpers"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: pull_csv.py <task1|task2|task3> [remote_filename]", file=sys.stderr)
        return 2
    task_name = argv[1]
    if task_name not in TASK_MAP:
        print(f"Unknown task {task_name!r}. Use task1, task2, or task3.", file=sys.stderr)
        return 2

    task = TASK_MAP[task_name]
    remote_path = argv[2] if len(argv) >= 3 else task["remote_path"]
    remote = f"{CLUSTER_BASE}/{remote_path}"
    local_dir = REPO_ROOT / "submissions"
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / task["local_name"]

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
