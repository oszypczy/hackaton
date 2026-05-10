#!/bin/bash
# Second chain: mean-shifted variants (broad recalibrations).
set +e

QUEUE=(
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_rank_uniform09.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_rank_uniform08.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_compromise_avg.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_swap_22u_11d.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_swap_22u_21d.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_swap_12d_10u.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_adam_calib_snap10.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_wd0_calib.csv"
    "/mnt/c/projekty/hackaton-cispa-2026/submissions/task1_duci_dense_R18all_snap10.csv"
)

LOG=/tmp/task1_chain2.log
THRESHOLD=0.053

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
            break
        fi
        sleep 540
    else
        echo "$(ts) | FAILED | $LABEL | http=$http" | tee -a "$LOG"
        sleep 120
    fi
done
echo "$(ts) | DONE chain2" | tee -a "$LOG"
