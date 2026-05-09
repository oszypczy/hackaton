default:
    @just --list

eval:
    python tests/smoke.py

baseline:
    @tail -5 SUBMISSION_LOG.md

extract-papers:
    bash scripts/extract_papers.sh
