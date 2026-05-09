#!/usr/bin/env python3
"""POST submission.csv to organizer API. Logs to SUBMISSION_LOG.md.

Usage: python scripts/submit.py <task> <csv_path>
       task: task1 | task2 | task3
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

BASE_URL = "http://35.192.205.84"


class TaskSpec(TypedDict):
    id: str
    expected_cols: list[str]
    expected_rows: int


TASK_MAP: dict[str, TaskSpec] = {
    "task1": {
        "id": "11-duci",
        "expected_cols": ["model_id", "proportion"],
        "expected_rows": 9,
    },
    "task2": {
        "id": "27-p4ms",
        "expected_cols": ["id", "pii_type", "pred"],
        "expected_rows": 3000,
    },
    "task3": {
        "id": "13-llm-watermark-detection",
        "expected_cols": ["id", "score"],
        "expected_rows": 2250,  # PDF says 2250; submission_template has copy-paste bug claiming 2400
    },
}

MAX_BYTES = 10 * 1024 * 1024


def _validate_csv(path: Path, task: TaskSpec) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"CSV {size} bytes > 10 MB limit")
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        rows = sum(1 for _ in f)
    expected_header = ",".join(task["expected_cols"])
    if header != expected_header:
        raise ValueError(f"Header mismatch: got {header!r}, expected {expected_header!r}")
    if rows != task["expected_rows"]:
        raise ValueError(f"Row count mismatch: got {rows}, expected {task['expected_rows']}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(task_name: str, csv_path: Path, csv_md5: str, response: dict | str,
         http_ok: bool) -> None:
    log_path = REPO_ROOT / "SUBMISSION_LOG.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    if isinstance(response, dict):
        score = response.get("score")
        sid = response.get("submission_id")
        status = response.get("status")
        if score is not None:
            tag = f"score={score}"
        elif http_ok and (status == "success" or sid is not None):
            tag = f"submitted sid={sid}"
        else:
            tag = "FAILED"
    else:
        tag = "FAILED" if not http_ok else "submitted"
    line = f"- {ts} {task_name} {tag} csv-md5={csv_md5} ({csv_path.name})\n"
    with log_path.open("a") as f:
        f.write(line)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: submit.py <task1|task2|task3> <csv_path>", file=sys.stderr)
        return 2
    task_name, csv_arg = argv[1], argv[2]
    if task_name not in TASK_MAP:
        print(f"Unknown task {task_name!r}. Use task1, task2, or task3.", file=sys.stderr)
        return 2
    task = TASK_MAP[task_name]
    csv_path = Path(csv_arg).resolve()

    api_key = os.environ.get("HACKATHON_API_KEY")
    if not api_key:
        print("HACKATHON_API_KEY not set. Add it to .env (gitignored).", file=sys.stderr)
        return 2

    _validate_csv(csv_path, task)
    csv_md5 = _md5(csv_path)
    print(f"CSV validated. md5={csv_md5} rows={task['expected_rows']}")

    url = f"{BASE_URL}/submit/{task['id']}"
    print(f"POST {url}")
    with csv_path.open("rb") as f:
        resp = requests.post(
            url,
            headers={"X-API-Key": api_key},
            files={"file": (csv_path.name, f, "text/csv")},
            timeout=120,
        )
    try:
        body: dict | str = resp.json()
    except json.JSONDecodeError:
        body = resp.text
    print(f"HTTP {resp.status_code}")
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    _log(task_name, csv_path, csv_md5, body, resp.ok)
    return 0 if resp.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
