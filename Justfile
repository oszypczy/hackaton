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

# Pull submission CSV from cluster. Default remote file: submission.csv.
# Usage: just pull-csv task1                            # → submissions/task1_duci.csv
#        just pull-csv task3 submission_kgw.csv         # → submissions/task3_watermark_kgw.csv
pull-csv task remote_file="submission.csv":
    @python scripts/pull_csv.py {{task}} {{remote_file}}
