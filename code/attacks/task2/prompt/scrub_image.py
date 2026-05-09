"""Mask EMAIL/CREDIT/PHONE values from val_pii images, producing task/-like appearance.

Public:
    scrub_pii(img, gt_email, gt_phone, gt_card) -> PIL.Image.Image

CLI:
    python scrub_image.py --demo                 # 5 val PNGs → side-by-side
    python scrub_image.py --all  /path/to/in/ /path/to/out/   # batch over dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

try:
    import pytesseract
except ImportError as e:
    raise SystemExit("pytesseract missing — pip install pytesseract (and brew install tesseract)") from e

LANCZOS = Image.Resampling.LANCZOS


class Box(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


# ── panel geometry ─────────────────────────────────────────────────────────────

def _detect_panel(arr: np.ndarray) -> tuple[int, int, str]:
    """Find the info panel via color uniformity at the image edges.

    Strategy: median color per column over middle 60% of rows; the panel side
    is whichever edge has a long run of columns that match the edge bg color.
    """
    H, W = arr.shape[:2]
    rows = arr[int(H * 0.20):int(H * 0.80), :, :3].astype(np.float32)
    col_med = np.median(rows, axis=0)  # (W, 3)

    left_ref = np.median(col_med[5:25], axis=0)
    right_ref = np.median(col_med[W - 25:W - 5], axis=0)

    left_dist = np.linalg.norm(col_med - left_ref, axis=1)
    right_dist = np.linalg.norm(col_med - right_ref, axis=1)

    # Walk inward; first column where the distance exceeds threshold is the edge
    threshold = 28
    left_edge = 0
    for x in range(W):
        if left_dist[x] < threshold:
            left_edge = x
        else:
            break

    right_edge = W
    for x in range(W - 1, -1, -1):
        if right_dist[x] < threshold:
            right_edge = x
        else:
            break

    left_run = left_edge
    right_run = W - right_edge

    # Add small margin into the panel to be safe
    if left_run > right_run:
        return 0, min(W, left_edge + 12), "left"
    return max(0, right_edge - 12), W, "right"


def _sample_bg(arr: np.ndarray, x0: int, x1: int) -> tuple[int, int, int]:
    """Median color of the panel — robust to text pixels."""
    band = arr[:, x0:x1, :3].astype(np.float32).reshape(-1, 3)
    return tuple(int(np.median(band[:, c])) for c in range(3))  # type: ignore[return-value]


def _is_dark(rgb: tuple[int, int, int]) -> bool:
    return sum(rgb) / 3 < 128


# ── OCR ─────────────────────────────────────────────────────────────────────────

def _ocr_panel(panel_img: Image.Image, dark_bg: bool, scale: int = 2) -> list[dict]:
    """Run OCR on the panel; return word records in original-scale coords."""
    big = panel_img.resize((panel_img.width * scale, panel_img.height * scale), LANCZOS)
    if dark_bg:
        big = Image.fromarray(255 - np.array(big))
    data = pytesseract.image_to_data(big, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)
    out: list[dict] = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        out.append({
            "word": word,
            "x": data["left"][i] // scale,
            "y": data["top"][i] // scale,
            "w": data["width"][i] // scale,
            "h": data["height"][i] // scale,
            "conf": data["conf"][i],
            "block": data["block_num"][i],
            "line": data["line_num"][i],
        })
    return out


def _by_line(words: list[dict]) -> list[list[dict]]:
    """Group words by visual Y-coordinate (tesseract's line_num is unreliable
    when the panel mixes top-aligned menu items with bottom-aligned PII text)."""
    if not words:
        return []
    # Sort by y (top), then cluster: word belongs to current line if its
    # vertical center is within (current_line_height * 0.7) of the line's center.
    sorted_w = sorted(words, key=lambda w: w["y"])
    lines: list[list[dict]] = []
    for w in sorted_w:
        wc = w["y"] + w["h"] / 2
        placed = False
        for line in lines:
            lc = sum(ww["y"] + ww["h"] / 2 for ww in line) / len(line)
            lh = sum(ww["h"] for ww in line) / len(line)
            if abs(wc - lc) <= lh * 0.6:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    # Sort within each line by x
    for line in lines:
        line.sort(key=lambda w: w["x"])
    # Sort lines by y
    lines.sort(key=lambda line: min(w["y"] for w in line))
    return lines


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _line_bbox(words: list[dict]) -> Box:
    x0 = min(w["x"] for w in words)
    y0 = min(w["y"] for w in words)
    x1 = max(w["x"] + w["w"] for w in words)
    y1 = max(w["y"] + w["h"] for w in words)
    return Box(x0, y0, x1, y1)


# ── value-region matching ──────────────────────────────────────────────────────

LABEL_PAT = re.compile(r"^(email|e-mail|tel|phone|mobile|card|credit)\s*:?$", re.I)


def _value_words(line: list[dict]) -> list[dict]:
    """Drop leading label tokens like 'email:' or 'Tel:'."""
    out = list(line)
    while out and (LABEL_PAT.match(out[0]["word"].rstrip(":.")) or
                   out[0]["word"].rstrip(":.").lower() in {"email", "tel", "card"} or
                   out[0]["word"].strip() in {":", ";"}):
        out = out[1:]
    return out


def _find_email_box(lines: list[list[dict]], gt_email: str) -> Box | None:
    target = gt_email.lower().strip()
    target_local = target.split("@")[0]
    target_dom = target.split("@")[1] if "@" in target else ""
    best = None
    best_score = 0.0
    for line in lines:
        text = " ".join(w["word"] for w in line).lower()
        if "@" not in text:
            continue
        # similarity proxy: count of target-local + domain chars present
        score = (sum(c in text for c in target_local) + sum(c in text for c in target_dom)) / max(1, len(target_local) + len(target_dom))
        if score > best_score:
            best_score = score
            best = line
    if best is None or best_score < 0.4:
        return None
    val = _value_words(best)
    if not val:
        # Fall back to first @-bearing word and everything after
        idx = next((i for i, w in enumerate(best) if "@" in w["word"]), None)
        if idx is None:
            return None
        val = best[idx:]
    return _line_bbox(val)


def _find_phone_box(lines: list[list[dict]], gt_phone: str) -> Box | None:
    target = _digits(gt_phone)[-10:]  # last 10 digits (E.164 country-code agnostic)
    best = None
    best_match = 0
    for line in lines:
        text = " ".join(w["word"] for w in line)
        digs = _digits(text)
        if not (8 <= len(digs) <= 14):
            continue
        # require at least 6 of the last 10 GT digits to appear in line digits
        match = sum(1 for c in target if c in digs)
        # and avoid matching the credit row by skipping 16-digit lines
        if len(digs) == 16:
            continue
        if match > best_match:
            best_match = match
            best = line
    if best is None or best_match < 6:
        return None
    val = _value_words(best)
    if not val:
        # phone value is the line itself
        val = best
    return _line_bbox(val)


def _find_card_box(lines: list[list[dict]], gt_card: str) -> Box | None:
    target = _digits(gt_card)[:16]
    best = None
    best_match = 0
    for line in lines:
        text = " ".join(w["word"] for w in line)
        digs = _digits(text)[:16]
        if len(digs) < 12:
            continue
        match = sum(a == b for a, b in zip(digs, target))
        if match > best_match:
            best_match = match
            best = line
    if best is None or best_match < 6:
        return None
    val = _value_words(best)
    if not val:
        # First word that is purely digits
        idx = next((i for i, w in enumerate(best) if w["word"].strip(" -").isdigit() and len(_digits(w["word"])) >= 3), 0)
        val = best[idx:]
    return _line_bbox(val)


# ── public API ──────────────────────────────────────────────────────────────────

def scrub_pii(img: Image.Image, gt_email: str, gt_phone: str, gt_card: str) -> Image.Image:
    """Return a copy of img with email/phone/card values masked, preserving
    Name, caption, profile photo, layout."""
    arr = np.array(img.convert("RGB")).copy()
    H, W = arr.shape[:2]

    panel_x0, panel_x1, _ = _detect_panel(arr)
    bg = _sample_bg(arr, panel_x0, panel_x1)

    panel_img = Image.fromarray(arr[:, panel_x0:panel_x1, :])
    words = _ocr_panel(panel_img, dark_bg=_is_dark(bg))
    lines = _by_line(words)

    boxes: list[Box] = []
    for finder, gt in (
        (_find_email_box, gt_email),
        (_find_phone_box, gt_phone),
        (_find_card_box, gt_card),
    ):
        if gt:
            b = finder(lines, gt)  # type: ignore[arg-type]
            if b:
                boxes.append(b)

    pad_x, pad_y = 4, 3
    for b in boxes:
        ax0 = max(0, panel_x0 + b.x0 - pad_x)
        ax1 = min(W, panel_x0 + b.x1 + pad_x)
        ay0 = max(0, b.y0 - pad_y)
        ay1 = min(H, b.y1 + pad_y)
        arr[ay0:ay1, ax0:ax1, :3] = bg

    return Image.fromarray(arr)


# ── demo ───────────────────────────────────────────────────────────────────────

def _load_gt(path: str = "/tmp/cmp_gt.json") -> dict[str, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"GT cache not found at {p}. Pull it from cluster (see SCRUB_RESEARCH_PROMPT.md).")
    return json.loads(p.read_text())


def _demo(n: int = 5) -> None:
    gt_map = _load_gt()
    val_dir = Path("/tmp/cmp_local/cmp_val")
    out_dir = Path(__file__).parent / "output" / "scrub_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(val_dir.glob("*.png"))[:n]
    for png in pngs:
        uid = re.sub(r"^\d+_", "", png.stem)
        gt = gt_map.get(uid, {})
        img = Image.open(png).convert("RGB")
        scrubbed = scrub_pii(
            img,
            gt.get("EMAIL", ""),
            gt.get("PHONE", ""),
            gt.get("CREDIT", ""),
        )
        combined = Image.new("RGB", (img.width * 2 + 4, img.height), (40, 40, 40))
        combined.paste(img, (0, 0))
        combined.paste(scrubbed, (img.width + 4, 0))
        out = out_dir / f"{png.stem}_compare.png"
        combined.save(out)
        print(f"saved {out.name} | uid={uid} email={gt.get('EMAIL','-')} phone={gt.get('PHONE','-')} card={gt.get('CREDIT','-')}")
    print(f"\n→ {out_dir}")


def _all(in_dir: str, out_dir: str) -> None:
    """Apply scrub to every val_pii PNG in in_dir, write to out_dir."""
    gt_map = _load_gt()
    src = Path(in_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    pngs = sorted(src.glob("*.png"))
    print(f"scrubbing {len(pngs)} images → {dst}")
    for i, png in enumerate(pngs):
        uid = re.sub(r"^\d+_", "", png.stem)
        gt = gt_map.get(uid, {})
        if not gt:
            print(f"[skip] no GT for {uid}")
            continue
        img = Image.open(png).convert("RGB")
        scrubbed = scrub_pii(
            img,
            gt.get("EMAIL", ""),
            gt.get("PHONE", ""),
            gt.get("CREDIT", ""),
        )
        scrubbed.save(dst / png.name)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pngs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--all", nargs=2, metavar=("IN_DIR", "OUT_DIR"))
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()
    if args.demo:
        _demo(args.n)
    elif args.all:
        _all(*args.all)
    else:
        parser.print_help()
        sys.exit(1)
