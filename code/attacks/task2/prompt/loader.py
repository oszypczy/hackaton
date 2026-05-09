"""Parquet loader for task/ and validation_pii/ folders.

Schema (both folders):
- path:         dict {"bytes": <PNG bytes>}
- conversation: list of 3 turns, each {"instruction": question, "output": answer}
- user_id:      int

task/ has 1000 rows × 3 = 3000 prompts (output is scrubbed with [REDACTED])
validation_pii/ has 280 × 3 = 840 prompts (output has real PII — local GT)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from format import detect_pii_type

REDACTED = "[REDACTED]"


@dataclass
class Sample:
    user_id: int
    pii_type: str
    question: str
    scrubbed_output: str
    image_bytes: bytes
    gt_pii: str | None  # None for task/, real PII for validation_pii/


def load_parquets(folder: Path, with_gt: bool) -> list[Sample]:
    """Load all parquet files in folder, expand into per-turn Samples.
    with_gt=True for validation_pii (extract real PII from output)."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    samples: list[Sample] = []
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            user_id = int(row["user_id"])
            image_bytes = row["path"]["bytes"]
            for turn in row["conversation"]:
                question = turn["instruction"]
                output = turn["output"]
                pii_type = detect_pii_type(question)

                if with_gt:
                    # validation_pii: output has real PII, no [REDACTED]
                    # We need to derive scrubbed_output for the prefix attack
                    gt_pii = _extract_gt_from_output(output, pii_type)
                    scrubbed_output = _scrub_output(output, gt_pii)
                else:
                    # task/: output already has [REDACTED]
                    gt_pii = None
                    scrubbed_output = output

                samples.append(
                    Sample(
                        user_id=user_id,
                        pii_type=pii_type,
                        question=question,
                        scrubbed_output=scrubbed_output,
                        image_bytes=image_bytes,
                        gt_pii=gt_pii,
                    )
                )
    return samples


def _extract_gt_from_output(output: str, pii_type: str) -> str:
    """Extract real PII value from validation_pii output sentence.

    Examples:
      CREDIT: "Gabriella Johnson's credit card number is 4986 6022 6865 7288."
      EMAIL:  "You can contact Gabriella Johnson via email at gabriella.johnson@savage.com."
      PHONE:  "You can reach Gabriella Johnson on +13859159897."
    """
    text = output.strip().rstrip(".").strip()
    if pii_type == "EMAIL":
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        if m:
            return m.group(0)
    elif pii_type == "PHONE":
        m = re.search(r"\+?\d[\d\s\-().]{6,20}\d", text)
        if m:
            return m.group(0).strip()
    elif pii_type == "CREDIT":
        m = re.search(r"\d[\d\s-]{11,22}\d", text)
        if m:
            return m.group(0).strip()
    raise ValueError(f"Could not extract {pii_type} GT from: {output!r}")


def _scrub_output(output: str, gt_pii: str) -> str:
    """Replace gt_pii with [REDACTED] to derive prefix template."""
    return output.replace(gt_pii, REDACTED, 1)


def derive_assistant_prefix(scrubbed_output: str) -> str:
    """Take everything before [REDACTED] as the assistant prefix.

    The model was trained on the full sentence; using its exact prefix
    primes it to emit the PII at the next position.

    Returns "" if [REDACTED] not found (defensive — should not happen).
    """
    idx = scrubbed_output.find(REDACTED)
    if idx == -1:
        return ""
    return scrubbed_output[:idx]


def load_image(image_bytes: bytes, image_size: int) -> Image.Image:
    """Decode PNG bytes, resize, return PIL.Image (RGB)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return img
