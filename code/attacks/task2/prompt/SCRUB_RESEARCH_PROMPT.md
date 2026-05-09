# Prompt: realistic image-scrub PoC for Task 2 calibration

> Paste into a fresh Claude Code session at `/Users/arturkempinski/hackaton` on branch `task2-prompt`.
> Previous agent attempt failed because the sandbox blocked `/tmp` access — make sure your tools can `Read` PNGs from `/tmp/cmp_local/...` and run Bash freely.

## Context (read this first)

We're at the CISPA hackathon, Task 2: extract PII (EMAIL/CREDIT/PHONE) from a multimodal LMM (OLMo-2-1B + LLaVA-HR vision). The model was intentionally OVERFITTED on a sensitive VQA dataset.

**Two relevant data folders on the JURECA cluster:**

- `validation_pii/` — 280 users, screenshot images with **PII values filled in** on the right-side panel (`email: gabriella.johnson@savage.com`, `Card: 4986 6022 6865 7288`, `Tel: +13859159897`). We have ground-truth PII strings (extractable from the conversation field). Our prefix-attack hits **0.96 OVERALL** here.

- `task/` — 1000 users, **same screenshot template, but PII fields are scrubbed/blank**. Only `Name:` is preserved. No GT available (this is the eval set).

`task/` ∩ `validation_pii/` user_ids = **0** (disjoint sets).

**Phase 5 ablation finding:**
- val_pii eval with `image_mode=original`: 0.96 OVERALL
- val_pii eval with `image_mode=blank` (all gray): 0.31 OVERALL
- task/ leaderboard score (real, with our best CSV): 0.31 OVERALL

So `blank` is already a near-perfect proxy for task/ conditions. **Open question: can we do better?** Our hypothesis is that task/ images preserve _some_ context (Name, profile photo, caption like "Edmonton wanderlust fulfilled" — see `docs/tasks/task2_pii_extraction.md` line 137), and blanking everything throws away that context. A more realistic scrub (mask only PII values, keep everything else) would give us a tighter calibrator for testing memorization-extraction prompt strategies.

## Layout variants observed (40-image visual sample)

Right-side info panel varies a lot:
- Panel can be on **right** OR **left** (e.g. val/12 has it left)
- Values block can be at **top** OR **bottom** of the panel; vertical menu (Home/Explore/Notifications/...) sometimes interleaves
- Panel bg colors observed: dark navy `#0E1A26`, mid-purple `#3B1F4F`, coral `#D86A7B`, royal-blue `#1F77B4`, mustard-yellow `#E8D81A`, forest-green `#0F6E3F`, charcoal `#1B1B1B`. Per-image bg sampling needed.
- Image canvas: 1024×1024 PNG (some preview tools downscale to 512×512)
- `Name:` field is **adjacent** to values block — risky to mask blindly

Caption on the left side and the profile photo also vary; both should be PRESERVED.

## What you have to work with

```
/Users/arturkempinski/hackaton/code/attacks/task2/prompt/
├── attack.py            # has image_mode='original'|'blank'|'noise', easy to add 'scrubbed'
├── loader.py            # parquet loader; Sample dataclass has image_bytes, gt_pii, scrubbed_output, question
├── format.py            # GT extraction regex etc.
├── multi_eval.py        # already runs N strategies × M samples, picks up image_mode flag
└── output/
    └── scrub_research_verdict.md   # prior attempt's verdict (skim, don't trust without verifying images)
```

Helpful local samples (already pulled, no internet/cluster needed):
```
/tmp/cmp_local/cmp_task/   # 20 task/ PNGs (PII-blanked, reference target appearance)
/tmp/cmp_local/cmp_val/    # 20 val_pii PNGs (PII-filled, source we want to scrub)
```

If `/tmp/cmp_local/` is missing, regenerate via:
```bash
scripts/juelich_exec.sh "source /p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/.venv/bin/activate && python3 -c \"
import pandas as pd, glob, os, random
random.seed(7)
def dump(folder, prefix, n):
    files = sorted(glob.glob(os.path.join(folder, '*.parquet')))
    df = pd.concat([pd.read_parquet(f, columns=['user_id', 'path']) for f in files], ignore_index=True)
    out_dir = f'/tmp/cmp_{prefix}'; os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(random.sample(range(len(df)), n)):
        with open(f'{out_dir}/{i:02d}_{str(df.iloc[idx][\\\"user_id\\\"]).strip()}.png', 'wb') as f:
            f.write(df.iloc[idx]['path']['bytes'])
dump('/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/task', 'task', 30)
dump('/p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/validation_pii', 'val', 30)
\" && tar -C /tmp -czf /tmp/cmp_images.tar.gz cmp_task cmp_val"
scripts/juelich_exec.sh "cat /tmp/cmp_images.tar.gz" > /tmp/cmp_images.tar.gz
mkdir -p /tmp/cmp_local && tar -C /tmp/cmp_local -xzf /tmp/cmp_images.tar.gz
```

To get GT PII strings for the 20 val samples (so you can test "GT-string locate + mask" approach), run a similar parquet pull but include the `conversation` column and parse `[ANSWER]` strings — see `loader._extract_gt_from_output` for the regex.

## Your task

Open the actual PNGs with the `Read` tool (it renders images). **Look at minimum 15 val_pii images and 15 task/ images, paired by index.** Then deliver a working PoC.

**Deliverable:**

1. A Python script `code/attacks/task2/prompt/scrub_image.py` exposing one function:
   ```python
   def scrub_pii(img: PIL.Image.Image, gt_email: str, gt_phone: str, gt_card: str) -> PIL.Image.Image:
       """Return a copy of img with email/phone/card values masked,
       preserving Name field, caption, profile photo, layout."""
   ```
   Pure PIL+numpy preferred. If you need OCR, **install locally on M4 only** (`brew install tesseract` then `pip install pytesseract` in our `.venv`). Cluster venv is shared — do NOT install there.

2. A driver `scrub_image.py --demo` that loads 5 val_pii images, scrubs them, saves before/after side-by-side PNGs to `code/attacks/task2/prompt/output/scrub_demo/` so we can eyeball quality.

3. Quality bar:
   - All 3 PII values masked (no readable digits or `@` characters left from the original PII text)
   - `Name:` field still readable
   - Profile photo, caption, menu, layout preserved
   - Works across 4-6 layout variants (panel left/right, top/bottom values block, varied bg colors)
   - Per-image bg color sampling (no single hardcoded color)

4. A short `output/scrub_design_notes.md` (≤300 words) explaining: chosen approach, what fails, recall on the 5 demo images.

**Constraints:**
- ≤2 hours implementation budget
- Do NOT modify shared cluster venv (`/p/scratch/.../.venv/`)
- Do NOT introduce heavy ML deps (no torch, no transformers — just PIL + numpy + optionally pytesseract)
- Do NOT over-engineer: failure on 1-2 of 20 layout variants is OK if dominant variants work

## Verification path (after PoC works)

Once your `scrub_pii(img, ...)` works on 5 demo images, the next step (which I'll do, NOT you) is:
1. Add `image_mode='scrubbed'` to `attack._build_image_tensor` that calls `scrub_pii`
2. Run `multi_eval.sh 50 baseline scrubbed` — compares baseline-with-blank vs baseline-with-scrubbed on val_pii
3. If scrubbed-baseline scores noticeably ABOVE 0.31, scrub preserves useful signal → better calibrator
4. If scrubbed-baseline scores ≈ blank (0.31), scrub doesn't help, blank is enough

Report at the end: top recommendation in 3 lines + path to demo PNGs.

## What NOT to do

- Don't write tests, READMEs, abstractions, or class hierarchies — single function, single driver, single notes file
- Don't change `loader.py`, `attack.py`, `multi_eval.py` — just create the new file
- Don't try to handle every edge case; document what fails
- Don't skip looking at the actual images — that was the prior agent's failure mode
