#!/bin/bash
# Scrape leaderboard before + submit task1 + scrape after.
# Logs to /tmp/task1_tracker.log
# Usage: submit_and_verify_task1.sh <label> <csv_path>

set +e
LABEL="$1"
CSV="$2"
LOG=/tmp/task1_tracker.log

scrape() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["11_duci::Czumpers"\] = [0-9.]+' \
        | head -1 \
        | grep -oE '[0-9.]+$'
}

scrape_leader() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["11_duci::[^"]+"\] = [0-9.]+' \
        | sort -t'=' -k2 -n \
        | head -1
}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

before=$(scrape)
leader_before=$(scrape_leader)
echo "$(ts) | BEFORE_SUBMIT | $LABEL | czumpers=$before | leader=$leader_before" | tee -a "$LOG"

out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task1 "$CSV" 2>&1)
http=$(echo "$out" | grep -oP 'HTTP \K\d+' | head -1)
md5=$(echo "$out" | grep -oP 'md5=\K\w+' | head -1)
echo "$(ts) | POST_RESPONSE | $LABEL | http=$http md5=$md5" | tee -a "$LOG"
echo "$out" | tail -10 | tee -a "$LOG"

if [[ "$http" != "200" ]]; then
    cooldown=$(echo "$out" | grep -oP 'Wait \K\d+')
    echo "$(ts) | REJECTED | $LABEL | cooldown=$cooldown" | tee -a "$LOG"
    exit 1
fi

# Wait for evaluation (server processes async)
sleep 40

after=$(scrape)
delta=$(python3 -c "print(round(float('$after')-float('$before'),6))" 2>/dev/null)
echo "$(ts) | AFTER_SUBMIT | $LABEL | czumpers=$after delta=$delta" | tee -a "$LOG"

if [[ "$after" == "$before" ]]; then
    echo "$(ts) | NO_IMPROVEMENT | $LABEL (score didn't update)" | tee -a "$LOG"
fi
