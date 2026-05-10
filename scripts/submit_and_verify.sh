#!/bin/bash
# Scrape leaderboard before + submit + scrape after.
# Logs to /tmp/task3_tracker.log
# Usage: submit_and_verify.sh <task_label> <csv_path>

set +e
LABEL="$1"
CSV="$2"
LOG=/tmp/task3_tracker.log

scrape() {
    curl -s http://35.192.205.84/leaderboard_page 2>/dev/null \
        | grep -oE 'currentScores\["13_llm_watermark_detection::Czumpers"\] = [0-9.]+' \
        | head -1 \
        | grep -oE '[0-9.]+$'
}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

before=$(scrape)
echo "$(ts) | BEFORE_SUBMIT | $LABEL | score=$before" | tee -a "$LOG"

out=$(python3 /mnt/c/projekty/hackaton-cispa-2026/scripts/submit.py task3 "$CSV" 2>&1)
http=$(echo "$out" | grep -oP 'HTTP \K\d+' | head -1)
md5=$(echo "$out" | grep -oP 'md5=\K\w+' | head -1)
echo "$(ts) | POST_RESPONSE | $LABEL | http=$http md5=$md5" | tee -a "$LOG"

if [[ "$http" != "200" ]]; then
    cooldown=$(echo "$out" | grep -oP 'Wait \K\d+')
    echo "$(ts) | REJECTED | $LABEL | cooldown=$cooldown" | tee -a "$LOG"
    exit 1
fi

# Wait for evaluation (server processes async)
sleep 40

after=$(scrape)
delta=$(python3 -c "print(round(float('$after')-float('$before'),6))" 2>/dev/null)
echo "$(ts) | AFTER_SUBMIT | $LABEL | score=$after delta=$delta" | tee -a "$LOG"

if [[ "$after" == "$before" ]]; then
    echo "$(ts) | NO_IMPROVEMENT | $LABEL" | tee -a "$LOG"
fi
