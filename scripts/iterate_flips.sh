#!/bin/bash
# Auto-submit a queue of CSVs with 10-min cooldown between successful submits.
# If HTTP 429, parses Wait N and waits that long + 60s buffer.
# Stops if score improves below threshold.
set +e

QUEUE=(
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip21_06.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip12_05.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip11_04.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip11_06.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_flip00_06.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_snap_025.csv"
)

LOG=/tmp/task1_iterate.log
THRESHOLD=0.053  # below this we stop

scrape() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["11_duci::Czumpers"\] = [0-9.]+' \
        | head -1 \
        | grep -oE '[0-9.]+$'
}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

for CSV in "${QUEUE[@]}"; do
    LABEL=$(basename "$CSV" .csv)
    echo "$(ts) | START | $LABEL" | tee -a "$LOG"
    out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task1 "$CSV" 2>&1)
    http=$(echo "$out" | grep -oP 'HTTP \K\d+' | head -1)
    echo "$(ts) | HTTP=$http | $LABEL" | tee -a "$LOG"

    if [[ "$http" == "429" ]]; then
        wait_secs=$(echo "$out" | grep -oP 'Wait \K\d+' | head -1)
        echo "$(ts) | RETRY_WAIT | $LABEL | wait=$wait_secs" | tee -a "$LOG"
        sleep $((wait_secs + 60))
        # Re-submit after wait
        out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task1 "$CSV" 2>&1)
        http=$(echo "$out" | grep -oP 'HTTP \K\d+' | head -1)
        echo "$(ts) | RETRY_HTTP=$http | $LABEL" | tee -a "$LOG"
    fi

    if [[ "$http" == "200" ]]; then
        sleep 30
        score=$(scrape)
        echo "$(ts) | AFTER | $LABEL | score=$score" | tee -a "$LOG"
        # Check if improved
        if python3 -c "import sys; sys.exit(0 if float('$score') < $THRESHOLD else 1)"; then
            echo "$(ts) | IMPROVED! | $LABEL | score=$score" | tee -a "$LOG"
            break
        fi
        # Wait 9 more min for next slot (10 min total since submit)
        sleep 540
    else
        echo "$(ts) | FAILED | $LABEL | http=$http" | tee -a "$LOG"
        sleep 120  # short wait on failure
    fi
done
echo "$(ts) | DONE" | tee -a "$LOG"
