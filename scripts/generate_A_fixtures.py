#!/usr/bin/env python3
"""
Generate Challenge A fixture data.

IN set  — openwebtext (in The Pile; Pythia trained on OpenWebText2)
OUT set — cnn_dailymail (not in The Pile; genuinely unseen by Pythia)
VAL set — wikitext-103 test split (same Pile distribution; for threshold calibration)

All documents: 200–500 tokens, approximated as 800–2000 characters.
Long docs are truncated at the last sentence boundary ≤ 2000 chars.

Outputs (JSONL, one JSON object per line):
  data/A/in.jsonl            {"id": int, "text": str}   1000 docs
  data/A/out.jsonl           {"id": int, "text": str}   1000 docs
  data/A/val_in.jsonl        {"id": int, "text": str}    200 docs
  data/A/ground_truth.jsonl  {"id": int, "is_member": bool}  2200 docs
    IDs 0–999    → in.jsonl      (is_member = true)
    IDs 1000–1999 → out.jsonl   (is_member = false)
    IDs 2000–2199 → val_in.jsonl (is_member = true; used for calibration only)

Usage:
  pip install datasets
  python scripts/generate_A_fixtures.py [--n-in 1000] [--n-out 1000] [--n-val 200]
"""

import argparse
import json
import re
import sys
from pathlib import Path

MIN_CHARS = 800    # ≈ 200 tokens
MAX_CHARS = 2000   # ≈ 500 tokens


def clean_and_truncate(text: str) -> str | None:
    """Return cleaned text in [MIN_CHARS, MAX_CHARS], or None if too short."""
    text = text.strip()
    # wikitext section headers (` = Title = `) — skip those lines
    text = re.sub(r"^\s*=+[^=]+=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < MIN_CHARS:
        return None

    if len(text) <= MAX_CHARS:
        return text

    # Truncate at the last sentence boundary ≤ MAX_CHARS
    chunk = text[:MAX_CHARS]
    last_period = max(chunk.rfind(". "), chunk.rfind(".\n"))
    if last_period > MIN_CHARS:
        chunk = chunk[: last_period + 1].strip()
    else:
        chunk = chunk.strip()

    if len(chunk) < MIN_CHARS:
        return None
    return chunk


def stream_openwebtext(n: int, skip: int = 0) -> list[dict]:
    """Stream n cleaned docs from openwebtext, skipping first `skip` docs."""
    from datasets import load_dataset

    ds = load_dataset("openwebtext", split="train", streaming=True)
    docs = []
    skipped = 0

    for row in ds:
        raw = row.get("text", "")
        if len(raw) < MIN_CHARS:
            continue
        if skipped < skip:
            skipped += 1
            continue
        cleaned = clean_and_truncate(raw)
        if cleaned:
            docs.append(cleaned)
        if len(docs) >= n:
            break

    return docs


def stream_cnn_dailymail(n: int) -> list[str]:
    """Stream n cleaned articles from cnn_dailymail 3.0.0."""
    from datasets import load_dataset

    ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    docs = []

    for row in ds:
        raw = row.get("article", "")
        cleaned = clean_and_truncate(raw)
        if cleaned:
            docs.append(cleaned)
        if len(docs) >= n:
            break

    return docs


def stream_wikipedia_recent(n: int, skip: int = 0) -> list[str]:
    """Stream n cleaned articles from Wikipedia 2023 dump (post-Pythia cutoff)."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    docs = []
    skipped = 0

    for row in ds:
        raw = row.get("text", "")
        if len(raw) < MIN_CHARS:
            continue
        if skipped < skip:
            skipped += 1
            continue
        cleaned = clean_and_truncate(raw)
        if cleaned:
            docs.append(cleaned)
        if len(docs) >= n:
            break

    return docs


def stream_wikitext_test(n: int) -> list[dict]:
    """Stream n cleaned docs from wikitext-103 test split (Pile-like distribution)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)
    docs = []
    buffer = ""  # wikitext is line-by-line; accumulate into articles

    for row in ds:
        line = row.get("text", "")
        if re.match(r"^\s*=\s+[^=]", line):
            # new article header — flush buffer
            cleaned = clean_and_truncate(buffer)
            if cleaned:
                docs.append(cleaned)
            buffer = ""
            if len(docs) >= n:
                break
        else:
            buffer += line + "\n"

    if len(docs) < n and buffer:
        cleaned = clean_and_truncate(buffer)
        if cleaned:
            docs.append(cleaned)

    return docs[:n]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} docs → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Challenge A fixture data")
    ap.add_argument("--n-in",  type=int, default=1000, help="IN set size (default 1000)")
    ap.add_argument("--n-out", type=int, default=1000, help="OUT set size (default 1000)")
    ap.add_argument("--n-val", type=int, default=200,  help="VAL set size (default 200)")
    ap.add_argument("--out-dir", default="data/A", help="Output directory (default data/A)")
    ap.add_argument("--seed",  type=int, default=42,   help="Not used — streaming is deterministic")
    ap.add_argument("--out-source", choices=["cnn", "wikipedia"], default="cnn",
                    help="OUT set source: cnn=cnn_dailymail (default), wikipedia=wikimedia 2023")
    ap.add_argument("--out-only", action="store_true",
                    help="Regenerate only out.jsonl (skip IN and VAL)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    n_in, n_out, n_val = args.n_in, args.n_out, args.n_val

    out_source_label = {"cnn": "cnn_dailymail", "wikipedia": "wikimedia/wikipedia 20231101.en"}[args.out_source]
    print(f"Generating Challenge A fixtures → {out_dir}/")
    print(f"  IN={n_in}  OUT={n_out} ({out_source_label})  VAL={n_val}\n")

    # ── IN set ────────────────────────────────────────────────────────────────
    if not args.out_only:
        print(f"[1/3] Streaming IN set from openwebtext (n={n_in})...")
        in_texts = stream_openwebtext(n_in, skip=0)
        if len(in_texts) < n_in:
            print(f"  [WARN] Only got {len(in_texts)} docs (wanted {n_in})", file=sys.stderr)
        in_records = [{"id": i, "text": t} for i, t in enumerate(in_texts)]
        write_jsonl(out_dir / "in.jsonl", in_records)
    else:
        print("[1/3] Skipping IN set (--out-only)")
        in_records = []

    # ── OUT set ───────────────────────────────────────────────────────────────
    print(f"\n[2/3] Streaming OUT set from {out_source_label} (n={n_out})...")
    if args.out_source == "wikipedia":
        out_texts = stream_wikipedia_recent(n_out)
    else:
        out_texts = stream_cnn_dailymail(n_out)
    if len(out_texts) < n_out:
        print(f"  [WARN] Only got {len(out_texts)} docs (wanted {n_out})", file=sys.stderr)
    out_records = [{"id": n_in + i, "text": t} for i, t in enumerate(out_texts)]
    write_jsonl(out_dir / "out.jsonl", out_records)

    # ── VAL set ───────────────────────────────────────────────────────────────
    if args.out_only:
        print("\n[3/3] Skipping VAL set (--out-only)")
        print(f"\nDone. Regenerated only out.jsonl ({len(out_records)} docs, source={args.out_source})")
        return

    print(f"\n[3/3] Streaming VAL set from wikitext-103 test (n={n_val})...")
    val_texts = stream_wikitext_test(n_val)
    shortage = n_val - len(val_texts)
    if shortage > 0:
        print(f"  wikitext test gave {len(val_texts)} docs; supplementing {shortage} from openwebtext (offset {n_in})...")
        extra = stream_openwebtext(shortage, skip=n_in)
        val_texts.extend(extra)
    if len(val_texts) < n_val:
        print(f"  [WARN] Only got {len(val_texts)} docs (wanted {n_val})", file=sys.stderr)
    val_records = [{"id": n_in + n_out + i, "text": t} for i, t in enumerate(val_texts)]
    write_jsonl(out_dir / "val_in.jsonl", val_records)

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_records: list[dict] = (
        [{"id": r["id"], "is_member": True}  for r in in_records]
        + [{"id": r["id"], "is_member": False} for r in out_records]
        + [{"id": r["id"], "is_member": True}  for r in val_records]
    )
    write_jsonl(out_dir / "ground_truth.jsonl", gt_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_chars_in  = sum(len(r["text"]) for r in in_records)
    total_chars_out = sum(len(r["text"]) for r in out_records)
    total_chars_val = sum(len(r["text"]) for r in val_records)

    print(f"""
Summary
  IN  : {len(in_records):4d} docs  avg {total_chars_in // max(len(in_records),1)} chars
  OUT : {len(out_records):4d} docs  avg {total_chars_out // max(len(out_records),1)} chars
  VAL : {len(val_records):4d} docs  avg {total_chars_val // max(len(val_records),1)} chars

IDs:
  0     – {len(in_records)-1}          → in.jsonl      (is_member=true)
  {len(in_records)} – {len(in_records)+len(out_records)-1}  → out.jsonl     (is_member=false)
  {len(in_records)+len(out_records)} – {len(in_records)+len(out_records)+len(val_records)-1}  → val_in.jsonl  (is_member=true)

Note: IN = openwebtext (in The Pile).
      OUT = {out_source_label} (not in Pythia training; post-cutoff).
      VAL = wikitext-103 test (same distribution as Pile Wikipedia).
""")


if __name__ == "__main__":
    main()
