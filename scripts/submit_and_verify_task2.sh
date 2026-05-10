#!/bin/bash
# Scrape leaderboard before + submit task2 + scrape after.
# Logs to /tmp/task2_tracker.log AND SUBMISSION_LOG.md (durable).
# Usage: submit_and_verify_task2.sh <label> <csv_path>

set +e
LABEL="$1"
CSV="$2"
LOG=/tmp/task2_tracker.log
SUB_LOG=/mnt/c/projekty/hackaton-cispa-2026/SUBMISSION_LOG.md

scrape() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["27_p4ms::Czumpers"\] = [0-9.]+' \
        | head -1 \
        | grep -oE '[0-9.]+$'
}

scrape_leader() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["27_p4ms::[^"]+"\] = [0-9.]+' \
        | sort -t'"' -k2 -k4 \
        | head -5
}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

before=$(scrape)
echo "$(ts) | BEFORE_SUBMIT | $LABEL | czumpers=$before" | tee -a "$LOG"
echo "- $(ts) task2 BEFORE_SUBMIT label=$LABEL czumpers=$before csv=$(basename "$CSV")" >> "$SUB_LOG"

out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task2 "$CSV" 2>&1)
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
verdict="OK"; [[ "$after" == "$before" ]] && verdict="NO_IMPROVEMENT"
echo "- $(ts) task2 AFTER_SUBMIT label=$LABEL czumpers=$after delta=$delta verdict=$verdict csv=$(basename "$CSV") md5=$md5" >> "$SUB_LOG"

if [[ "$after" == "$before" ]]; then
    echo "$(ts) | NO_IMPROVEMENT | $LABEL (score didn't update)" | tee -a "$LOG"
fi
