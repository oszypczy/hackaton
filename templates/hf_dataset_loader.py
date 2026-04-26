"""HuggingFace streaming dataset loader. Saves to JSONL for offline use.

Streaming avoids the multi-GB Pile download. CIFAR-10 fits in memory; load it
non-streaming. Pile and News_2024 must stream.

Usage:
    python templates/hf_dataset_loader.py monology/pile-uncopyrighted train 1000 data/A/in.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def stream_to_jsonl(repo: str, split: str, n: int, out: Path, text_key: str = "text") -> None:
    ds = load_dataset(repo, split=split, streaming=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as f:
        for row in ds:
            if text_key not in row:
                raise KeyError(f"row missing '{text_key}': keys={list(row)}")
            f.write(json.dumps({"text": row[text_key]}) + "\n")
            written += 1
            if written >= n:
                break
    print(f"wrote {written} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("split", default="train")
    p.add_argument("n", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--text-key", default="text")
    args = p.parse_args()
    stream_to_jsonl(args.repo, args.split, args.n, args.out, args.text_key)


if __name__ == "__main__":
    main()
