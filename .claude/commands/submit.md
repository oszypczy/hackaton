---
description: Run eval, submit if score > current best
---
!just eval
Read SUBMISSION_LOG.md for current best.
If new score is better, run `just submit` and append "<timestamp> <score> <method>" to SUBMISSION_LOG.md.
Otherwise print comparison and stop.
