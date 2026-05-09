default:
    @just --list

eval:
    python tests/smoke.py

baseline:
    @tail -5 SUBMISSION_LOG.md

extract-papers:
    bash scripts/extract_papers.sh

# Submit a CSV to organizer API. Loads HACKATHON_API_KEY from .env.
# Usage: just submit task1 submissions/task1_duci.csv
submit task csv:
    @python scripts/submit.py {{task}} {{csv}}

# Pull submission.csv from cluster (Czumpers/<task>/submission.csv) to local.
# Usage: just pull-csv task1   # writes submissions/task1_duci.csv
pull-csv task:
    @python scripts/pull_csv.py {{task}}
