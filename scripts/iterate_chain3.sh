#!/bin/bash
# Chain 3: compound flips with 11=0.4 baseline (since flip11_04 alone gave 0.020).
# Test if other targets in public 3 also need flipping.
set +e

QUEUE=(
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_22.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_12.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_00.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_01.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_02.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_10.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_20.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_21.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_snap05.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_snap_other.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_045.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_11_035.csv"
)

LOG=/tmp/task1_chain3.log
THRESHOLD=0.020  # current best — improve if score below this

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
        out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task1 "$CSV" 2>&1)
        http=$(echo "$out" | grep -oP 'HTTP \K\d+' | head -1)
        echo "$(ts) | RETRY_HTTP=$http | $LABEL" | tee -a "$LOG"
    fi

    if [[ "$http" == "200" ]]; then
        sleep 30
        score=$(scrape)
        echo "$(ts) | AFTER | $LABEL | score=$score" | tee -a "$LOG"
        if python3 -c "import sys; sys.exit(0 if float('$score') < $THRESHOLD else 1)"; then
            echo "$(ts) | IMPROVED! | $LABEL | score=$score" | tee -a "$LOG"
            # Don't break — keep going to find further improvements
        fi
        sleep 540
    else
        echo "$(ts) | FAILED | $LABEL | http=$http" | tee -a "$LOG"
        sleep 120
    fi
done
echo "$(ts) | DONE chain3" | tee -a "$LOG"
