#!/usr/bin/env bash
# Auto-submit queue of CSVs to task1, retrying on cooldown.
#
# Usage: scripts/auto_submit_queue.sh <csv1> [csv2] [csv3] ...
#
# Strategy: try submit; if HTTP 429 (cooldown) detected, sleep 60s and retry.
# Between successful submits, wait 320s (5 min cooldown + 20s buffer).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <csv1> [csv2] ..."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUCCESS_SLEEP=320
RETRY_SLEEP=60
MAX_RETRIES=20

submit_with_retry() {
    local csv="$1"
    local retries=0
    while [[ $retries -lt $MAX_RETRIES ]]; do
        local out
        out=$(just submit task1 "$csv" 2>&1) || true
        echo "$out" | tail -10
        if echo "$out" | grep -q '"status": "success"'; then
            return 0
        fi
        if echo "$out" | grep -q "HTTP 429"; then
            local wait_s
            wait_s=$(echo "$out" | grep -oE 'Wait [0-9]+ seconds' | grep -oE '[0-9]+' || echo "$RETRY_SLEEP")
            wait_s=$(( wait_s + 5 ))
            echo "  [cooldown] waiting ${wait_s}s and retrying ($((retries+1))/$MAX_RETRIES)..."
            sleep "$wait_s"
            retries=$((retries + 1))
            continue
        fi
        echo "  [error] non-cooldown failure; aborting this CSV"
        return 1
    done
    echo "  [error] exceeded retries"
    return 1
}

i=0
total=$#
for csv in "$@"; do
    i=$((i + 1))
    echo "============================================================"
    echo "[$i/$total] $csv at $(date -u +%H:%M:%S)"
    echo "============================================================"
    if [[ ! -f "$csv" ]]; then
        echo "  [skip] file does not exist"
        continue
    fi
    submit_with_retry "$csv" || echo "  [warn] failed; continuing"
    if [[ $i -lt $total ]]; then
        echo "  Waiting ${SUCCESS_SLEEP}s before next..."
        sleep "$SUCCESS_SLEEP"
    fi
done

echo "============================================================"
echo "[done] all $total processed at $(date -u +%H:%M:%S)"
