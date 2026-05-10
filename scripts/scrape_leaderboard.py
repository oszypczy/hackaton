#!/usr/bin/env python3
"""
Scrape http://35.192.205.84/leaderboard_page and emit Czumpers + top-5 per task as JSON.

Usage:
    python scripts/scrape_leaderboard.py                # pretty-print
    python scripts/scrape_leaderboard.py --json         # raw JSON
    python scripts/scrape_leaderboard.py --task 11_duci # only DUCI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

URL = "http://35.192.205.84/leaderboard_page"
PATTERN = re.compile(r'currentScores\["([^"]+)::([^"]+)"\]\s*=\s*([0-9.eE+-]+);')


def fetch() -> str:
    with urllib.request.urlopen(URL, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse(html: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in PATTERN.finditer(html):
        task, team, score = m.group(1), m.group(2), float(m.group(3))
        out.setdefault(task, {})[team] = score
    return out


def rank(scores: dict[str, float], lower_better: bool = True) -> list[tuple[int, str, float]]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=not lower_better)
    return [(i + 1, team, score) for i, (team, score) in enumerate(items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--task", default="")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--team", default="Czumpers")
    args = ap.parse_args()

    try:
        html = fetch()
    except Exception as e:
        print(f"FETCH_FAILED: {e}", file=sys.stderr)
        sys.exit(2)
    data = parse(html)
    if not data:
        print("NO_SCORES_PARSED", file=sys.stderr)
        sys.exit(3)

    result = {"tasks": {}}
    for task, scores in data.items():
        if args.task and task != args.task:
            continue
        ranked = rank(scores, lower_better=True)
        team_entry = next(((r, t, s) for r, t, s in ranked if t == args.team), None)
        result["tasks"][task] = {
            "n_teams": len(scores),
            "team_rank": team_entry[0] if team_entry else None,
            "team_score": team_entry[2] if team_entry else None,
            "top": [{"rank": r, "team": t, "score": s} for r, t, s in ranked[:args.top_n]],
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return
    for task, info in result["tasks"].items():
        you = (f"#{info['team_rank']} {args.team}={info['team_score']:.6f}"
               if info['team_rank'] else f"{args.team}=NOT_FOUND")
        print(f"\n=== {task} ({info['n_teams']} teams) — {you} ===")
        for row in info["top"]:
            mark = " <- you" if row["team"] == args.team else ""
            print(f"  {row['rank']:2d}. {row['team']:30s} {row['score']:.6f}{mark}")


if __name__ == "__main__":
    main()
