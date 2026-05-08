"""
Challenge B2 — watermark removal via zero-width-space insertion ("emoji attack" variant).

Strategy:
  - Insert zero-width space (U+200B) every N tokens. The detector tokenises the
    text with Llama-3 BPE which encodes ZWSP as its own token. Since the green-list
    seed is derived from the *previous* token, content tokens following ZWSP get
    a green-list keyed off the ZWSP token id — a list disjoint from the one used
    during generation. Result: green-token rate collapses to ~γ (≈ random).

  - Visually invisible → BERTScore ≈ 1.0.

Usage:
    python code/attacks/run_attack_B2.py
Output:
    submissions/B2.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB_DIR     = ROOT / "submissions"
DATA_DIR    = ROOT / "data" / "B"
TARGETS     = DATA_DIR / "removal_targets.jsonl"
OUT         = SUB_DIR / "B2.jsonl"

ZWSP        = "​"
INSERT_EVERY = 1   # every word — every content token gets ZWSP as prev → green-list reseeded


def perturb(text: str, every: int = INSERT_EVERY) -> str:
    parts = text.split(" ")
    out: list[str] = []
    for i, w in enumerate(parts):
        out.append(w)
        if (i + 1) % every == 0 and i < len(parts) - 1:
            out.append(ZWSP)
    return " ".join(out)


def main() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in TARGETS.open()]
    print(f"Perturbing {len(rows)} texts (ZWSP every {INSERT_EVERY} words)...")

    with OUT.open("w") as f:
        for r in rows:
            mod = perturb(r["text"])
            f.write(json.dumps({"id": r["id"], "modified_text": mod}) + "\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
