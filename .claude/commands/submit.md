---
description: Submit a CSV to the organizer API for $ARGUMENTS (task1|task2|task3)
---
Submit submission CSV to organizer API and log result.

Steps:
1. If user didn't specify a task in $ARGUMENTS, ask which task (task1/task2/task3).
2. Check that `submissions/<task>_*.csv` exists locally. If not, run `just pull-csv <task>` to fetch from cluster.
3. Validate CSV format locally (header, rows, scores in [0,1] etc) per docs/SUBMISSION_FLOW.md.
4. Run `just submit <task> <csv_path>`.
5. Show response. Append outcome line to SUBMISSION_LOG.md (script does this).
6. If failed: tell user what server said. Note 2-min cooldown before retry.
7. If success: note 5-min cooldown before next submit; check leaderboard at http://35.192.205.84/leaderboard_page.

Refuse if HACKATHON_API_KEY is missing from .env — point user at .env.example.
