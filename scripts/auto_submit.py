#!/usr/bin/env python3
"""
Autonomous submission orchestrator for Task 1.

Reads submissions/auto_queue.json (priority list), submits one CSV at a time
respecting cooldowns, scrapes the leaderboard after each submission, and logs
the score progression to submissions/auto_log.md + state in submissions/auto_state.json.

Usage:
    # one tick: dispatch the next submission, wait for cooldown, then exit
    python scripts/auto_submit.py --task task1 --tick

    # status only (no submission)
    python scripts/auto_submit.py --status

    # dry-run (validate next CSV without submitting)
    python scripts/auto_submit.py --task task1 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO_ROOT / "submissions" / "auto_queue.json"
STATE_PATH = REPO_ROOT / "submissions" / "auto_state.json"
LOG_PATH = REPO_ROOT / "submissions" / "auto_log.md"
COOLDOWN_OK_S = 305
COOLDOWN_FAIL_S = 125
LB_URL = "http://35.192.205.84/leaderboard_page"
LB_PATTERN = re.compile(r'currentScores\["([^"]+)::([^"]+)"\]\s*=\s*([0-9.eE+-]+);')


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_ts() -> float:
    return time.time()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "submitted_md5s": [],
        "events": [],
        "last_submit_ts": 0.0,
        "last_submit_status": "none",
        "best_public_score": None,
        "best_public_csv": None,
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def submitted_md5s_from_global_log() -> set[str]:
    """Cross-check against SUBMISSION_LOG.md so we never re-submit a CSV from prior sessions."""
    log_path = REPO_ROOT / "SUBMISSION_LOG.md"
    if not log_path.exists():
        return set()
    out = set()
    for line in log_path.read_text().splitlines():
        m = re.search(r"csv-md5=([0-9a-f]{32})", line)
        if m and ("submitted" in line or "score=" in line or "SUB-" in line):
            out.add(m.group(1))
    return out


def fetch_leaderboard() -> dict[str, dict[str, float]]:
    try:
        with urllib.request.urlopen(LB_URL, timeout=20) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    for m in LB_PATTERN.finditer(html):
        out.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return out


def lb_summary(scores_for_task: dict[str, float], team: str = "Czumpers") -> dict:
    if not scores_for_task:
        return {"rank": None, "score": None, "n_teams": 0, "top": []}
    items = sorted(scores_for_task.items(), key=lambda kv: kv[1])
    rank = next((i + 1 for i, (t, _) in enumerate(items) if t == team), None)
    score = scores_for_task.get(team)
    top = [{"rank": i + 1, "team": t, "score": s} for i, (t, s) in enumerate(items[:5])]
    return {"rank": rank, "score": score, "n_teams": len(items), "top": top}


def cooldown_remaining(state: dict) -> float:
    last = state.get("last_submit_ts", 0.0)
    last_status = state.get("last_submit_status", "none")
    if last == 0:
        return 0.0
    delta = now_ts() - last
    needed = COOLDOWN_OK_S if last_status == "ok" else COOLDOWN_FAIL_S
    return max(0.0, needed - delta)


def pick_next(queue: list, submitted: set[str]) -> dict | None:
    for item in queue:
        if item["md5"] in submitted:
            continue
        return item
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--tick", action="store_true",
                    help="submit one CSV (or report cooldown remaining), then exit")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def cmd_status(state: dict, queue: list):
    submitted_global = submitted_md5s_from_global_log()
    submitted_local = set(state.get("submitted_md5s", []))
    submitted = submitted_global | submitted_local
    cd = cooldown_remaining(state)
    nxt = pick_next(queue, submitted)
    print(f"=== auto_submit status @ {now_iso()} ===")
    print(f"queue size:        {len(queue)}")
    print(f"submitted (any):   {len(submitted)}")
    print(f"cooldown:          {cd:.0f}s remaining (last: {state.get('last_submit_status')})")
    if state.get("best_public_score") is not None:
        print(f"best public score: {state['best_public_score']:.6f}  ({state.get('best_public_csv')})")
    print()
    if nxt:
        preds = " ".join(f"{k}={v:.3f}" for k, v in sorted(nxt["predictions"].items()))
        print(f"next CSV: {nxt['csv']}")
        print(f"  {nxt['signal']}/{nxt['method']}/{nxt['post']}  loo={nxt['mean_loo']:.4f}")
        print(f"  preds: {preds}")
    else:
        print("queue empty — all CSVs submitted.")
    lb = fetch_leaderboard()
    if lb:
        s = lb_summary(lb.get("11_duci", {}))
        print(f"\nleaderboard 11_duci ({s['n_teams']} teams):")
        you_score = f"{s['score']:.6f}" if s['score'] is not None else "N/A"
        you_rank = f"#{s['rank']}" if s['rank'] is not None else "N/A"
        print(f"  Czumpers: {you_rank} {you_score}")
        for row in s["top"]:
            mark = " <- you" if row["team"] == "Czumpers" else ""
            print(f"  {row['rank']}. {row['team']:30s} {row['score']:.6f}{mark}")


def cmd_tick(args, state: dict, queue: list):
    submitted_global = submitted_md5s_from_global_log()
    submitted_local = set(state.get("submitted_md5s", []))
    submitted = submitted_global | submitted_local
    cd = cooldown_remaining(state)
    if cd > 0:
        print(f"COOLDOWN_WAIT remaining={cd:.0f}s")
        return 10  # special exit code: still cooling down
    nxt = pick_next(queue, submitted)
    if nxt is None:
        print("QUEUE_EMPTY")
        return 11
    csv_path = Path(nxt["csv"])
    if not csv_path.exists():
        print(f"CSV_MISSING {csv_path}", file=sys.stderr)
        return 2
    print(f"SUBMITTING rank={nxt['rank']} csv={csv_path.name} loo={nxt['mean_loo']:.4f}")
    if args.dry_run:
        print("DRY_RUN — not submitting")
        return 0
    rc = subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "submit.py"),
                          args.task, str(csv_path)])
    state["last_submit_ts"] = now_ts()
    state["last_submit_status"] = "ok" if rc == 0 else "fail"
    if rc == 0:
        state.setdefault("submitted_md5s", []).append(nxt["md5"])
    # Wait briefly for leaderboard to update
    time.sleep(8)
    lb = fetch_leaderboard()
    s = lb_summary(lb.get("11_duci", {})) if lb else {"rank": None, "score": None}
    pub_score = s.get("score")
    if pub_score is not None:
        prev_best = state.get("best_public_score")
        if prev_best is None or pub_score < prev_best:
            state["best_public_score"] = pub_score
            state["best_public_csv"] = csv_path.name
    state.setdefault("events", []).append({
        "ts": now_iso(), "csv": csv_path.name, "md5": nxt["md5"],
        "rc": rc, "rank_in_queue": nxt["rank"],
        "signal": nxt["signal"], "method": nxt["method"], "post": nxt["post"],
        "loo": nxt["mean_loo"], "public_score_after": pub_score,
        "leaderboard_rank_after": s.get("rank"),
    })
    save_state(state)
    you_rank = f"#{s['rank']}" if s.get('rank') is not None else "N/A"
    pub_score_str = f"{pub_score:.6f}" if pub_score is not None else "N/A"
    append_log(f"- {now_iso()} {args.task} csv={csv_path.name} md5={nxt['md5']} "
               f"rc={rc} loo={nxt['mean_loo']:.4f} → public={pub_score_str} ({you_rank})")
    print(f"DONE rc={rc} public_score={pub_score_str} rank={you_rank}")
    return 0 if rc == 0 else 1


def main():
    args = parse_args()
    if not QUEUE_PATH.exists():
        print(f"QUEUE_MISSING {QUEUE_PATH}", file=sys.stderr)
        sys.exit(3)
    queue_blob = json.loads(QUEUE_PATH.read_text())
    queue = queue_blob.get("queue", [])
    state = load_state()

    if args.status:
        cmd_status(state, queue)
        return
    if args.tick:
        rc = cmd_tick(args, state, queue)
        sys.exit(rc)
    cmd_status(state, queue)


if __name__ == "__main__":
    main()
