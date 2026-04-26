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
