#!/bin/bash
# Run a queue of Task 2 submissions with verify pattern + 5min cooldown.
# Logs everything to /tmp/task2_tracker.log.
#
# Usage: task2_submit_queue.sh
# Edit the QUEUE array below to choose CSVs.

set +e
cd /mnt/c/projekty/hackaton-cispa-2026
LOG=/tmp/task2_tracker.log

# Queue: "label::csv_path"
# Round 5: majority-vote ensembles (different from smart_v2 strategy).
QUEUE=(
    "MAJORITY_VOTE_qr_3x_weighted::submissions/task2_majority_vote_qr_weighted.csv"
    "MAJORITY_VOTE_all_equal::submissions/task2_majority_vote_all.csv"
)

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "$(ts) | QUEUE_START | items=${#QUEUE[@]}" | tee -a "$LOG"

# Wait for any in-flight cooldown from earlier submission to expire.
# Round 5 launching at ~00:46Z; last POST 00:44:47, cooldown ends ~00:49:47.
echo "$(ts) | INITIAL_COOLDOWN | sleep=240s (round 5)" | tee -a "$LOG"
sleep 240

for i in "${!QUEUE[@]}"; do
    entry="${QUEUE[$i]}"
    label="${entry%%::*}"
    csv="${entry#*::}"

    # Wait cooldown if not first
    if [[ $i -gt 0 ]]; then
        echo "$(ts) | COOLDOWN_WAIT | next=$label sleep=305s" | tee -a "$LOG"
        sleep 305
    fi

    echo "$(ts) | QUEUE_NEXT | idx=$i label=$label csv=$csv" | tee -a "$LOG"
    bash scripts/submit_and_verify_task2.sh "$label" "$csv" 2>&1 | tee -a "$LOG"

    # Inner script may exit 1 on REJECTED. Retry once after 35s if so.
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "$(ts) | RETRY_AFTER_REJECT | label=$label sleep=35s" | tee -a "$LOG"
        sleep 35
        bash scripts/submit_and_verify_task2.sh "${label}_retry" "$csv" 2>&1 | tee -a "$LOG"
    fi
done

echo "$(ts) | QUEUE_DONE" | tee -a "$LOG"
