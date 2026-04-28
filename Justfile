default:
    @just --list

eval:
    python tests/smoke.py

score CHALLENGE:
    python code/practice/score_{{CHALLENGE}}.py

submit:
    @echo "Submission packaging — TBD per challenge"

baseline:
    @tail -5 SUBMISSION_LOG.md

extract-papers:
    bash scripts/extract_papers.sh

gen-B:
    python scripts/generate_B_fixtures.py

gen-C:
    python scripts/generate_C_fixtures.py

gen-B-dry:
    python scripts/generate_B_fixtures.py --dry-run

gen-C-dry:
    python scripts/generate_C_fixtures.py --dry-run
