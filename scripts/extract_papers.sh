#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p references/papers/txt

extract_one() {
  local pdf="$1"
  local out="references/papers/txt/$(basename "$pdf" .pdf).txt"
  if pdftotext -layout "$pdf" "$out" 2>/dev/null; then
    echo "OK: $out"
  else
    echo "FAIL: $pdf" >&2
  fi
}
export -f extract_one

ls references/papers/*.pdf | xargs -n 1 -P 8 -I {} bash -c 'extract_one "$@"' _ {}

echo "---"
echo "Extracted: $(ls references/papers/txt/*.txt 2>/dev/null | wc -l) / $(ls references/papers/*.pdf | wc -l) PDFs"
echo "Total size:"
du -sh references/papers/txt
