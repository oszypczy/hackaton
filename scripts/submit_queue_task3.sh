#!/usr/bin/env bash
# Sequential task3 submits with 429 retry and post-success cooldown (~5m team limit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

queue=(
  "submissions/task3_watermark_clm_mega.csv"
  "submissions/task3_watermark_clm_para.csv"
  "submissions/task3_watermark_blend_para_kgwo7be.csv"
  "submissions/task3_watermark_triblend_para.csv"
  "submissions/task3_watermark_clm_llama2.csv"
  "submissions/task3_watermark_blend_kgwx_o7be_BEST3.csv"
  "submissions/task3_watermark_triblend_4way.csv"
  "submissions/task3_watermark_blend_mega_blend.csv"
  "submissions/task3_watermark_triblend_mega.csv"
  "submissions/task3_watermark_blend_kgwx_o7be_clm.csv"
  "submissions/task3_watermark_blend_kgwx_heavy.csv"
  "submissions/task3_watermark_blend_o7be_heavy2.csv"
  "submissions/task3_watermark_blend_median.csv"
  "submissions/task3_watermark_blend_geomean.csv"
)

submit_one() {
  local csv="$1"
  if [[ ! -f "$csv" ]]; then
    echo "[SKIP] missing $csv"
    return 0
  fi
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "=== $(date -Is) === $csv (attempt $attempt)"
    local out
    out=$(python scripts/submit.py task3 "$csv" 2>&1) || true
    echo "$out"
    if echo "$out" | grep -q "HTTP 200"; then
      echo "[OK] $csv — cooldown 310s before next"
      sleep 310
      return 0
    fi
    if echo "$out" | grep -q "HTTP 429"; then
      local w
      w=$(echo "$out" | grep -oE '[0-9]+ seconds' | head -1 | grep -oE '[0-9]+' || true)
      w=${w:-300}
      echo "[429] sleep $((w + 15))s then retry $csv"
      sleep "$((w + 15))"
      continue
    fi
    echo "[DONE] $csv — non-200/429 or error, moving on"
    sleep 130
    return 0
  done
}

for csv in "${queue[@]}"; do
  submit_one "$csv"
done
echo "=== queue finished $(date -Is) ==="
