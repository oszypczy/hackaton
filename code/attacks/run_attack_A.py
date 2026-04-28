"""
Run Challenge A attack: Min-K%++ + reference model + Welch t-test.

Usage:
    python code/attacks/run_attack_A.py

Outputs:
    submissions/A_docs.jsonl     — doc-level scores (IDs 0-1999)
    submissions/A_datasets.jsonl — dataset-level verdicts (set1, set2)

Runtime: ~30-60 min on M4 MPS (first run); <5s from cache.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path so local `code/` package shadows stdlib `code` module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
from tqdm import tqdm

from code.attacks.min_k_pp import (
    TARGET_MODEL, REF_MODEL, MAX_LENGTH,
    pick_device, load_model, compute_doc_features, dataset_level_test,
)

ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data" / "A"
SUB_DIR      = ROOT / "submissions"
CACHE_DIR    = ROOT / ".cache"
CACHE_TARGET = CACHE_DIR / "A_target_feats.pkl"
CACHE_REF    = CACHE_DIR / "A_ref_feats.pkl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_inference(docs: list[dict], model_id: str, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"  Loading cache: {cache_path.name}")
        return joblib.load(cache_path)

    device = pick_device()
    print(f"  Device: {device}")
    print(f"  Loading {model_id} ...", flush=True)
    tokenizer, model = load_model(model_id, device)
    print(f"  Model loaded.", flush=True)

    feats = {}
    for doc in tqdm(docs, desc=f"Scoring {model_id}"):
        feats[doc["id"]] = compute_doc_features(
            doc["text"], tokenizer, model, device, MAX_LENGTH
        )

    CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(feats, cache_path)
    print(f"  Cached to {cache_path.name}")
    return feats


def main() -> None:
    in_docs  = load_jsonl(DATA_DIR / "in.jsonl")
    out_docs = load_jsonl(DATA_DIR / "out.jsonl")
    val_docs = load_jsonl(DATA_DIR / "val_in.jsonl")
    all_docs = in_docs + out_docs + val_docs
    print(f"Loaded {len(in_docs)} IN + {len(out_docs)} OUT + {len(val_docs)} VAL docs")

    print("\n[1/2] Target model inference")
    target_feats = run_inference(all_docs, TARGET_MODEL, CACHE_TARGET)

    print("\n[2/2] Reference model inference")
    ref_feats = run_inference(all_docs, REF_MODEL, CACHE_REF)

    # Two separate scores:
    # - doc_score: used for doc-level AUC/TPR. Combines zlib_ratio + raw loss
    #   (both point IN > OUT for this fixture: openwebtext is more complex than news).
    # - ds_score: used for dataset-level t-test. Pure zlib_ratio only, because
    #   raw loss adds noise when comparing same-distribution IN vs VAL.
    # NOTE: on hackathon fixtures with proper same-distribution splits, replace
    #   doc_score with minkpp/-loss and ds_score with minkpp/-loss.
    def doc_score(doc_id: int) -> float:
        f = target_feats[doc_id]
        if math.isnan(f["zlib_ratio"]) or math.isnan(f["loss"]):
            return float("nan")
        # Normalize loss to ~[0,1] range; both zlib and loss point same direction here
        return 0.50 * f["zlib_ratio"] + 0.50 * (f["loss"] / 10.0)

    def ds_score(doc_id: int) -> float:
        f = target_feats[doc_id]
        return f["zlib_ratio"] if not math.isnan(f["zlib_ratio"]) else float("nan")

    scores    = {doc["id"]: doc_score(doc["id"]) for doc in all_docs}
    ds_scores = {doc["id"]: ds_score(doc["id"])  for doc in all_docs}

    # Calibrate threshold on val_in (doc-level score)
    val_scores = [scores[d["id"]] for d in val_docs if not math.isnan(scores[d["id"]])]
    threshold  = float(np.median(val_scores)) if val_scores else 0.0

    # Write doc-level submission (in + out only, NOT val_in)
    SUB_DIR.mkdir(exist_ok=True)
    sub_docs_path = SUB_DIR / "A_docs.jsonl"
    with open(sub_docs_path, "w") as f:
        for doc in in_docs + out_docs:
            s = scores[doc["id"]]
            row = {
                "id":        doc["id"],
                "score":     0.0 if math.isnan(s) else s,
                "is_member": bool(s >= threshold) if not math.isnan(s) else False,
            }
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {sub_docs_path}")

    # Dataset-level Welch t-test (uses ds_score = pure zlib_ratio)
    in_scores  = [scores[d["id"]]    for d in in_docs  if not math.isnan(scores[d["id"]])]
    out_scores = [scores[d["id"]]    for d in out_docs  if not math.isnan(scores[d["id"]])]
    val_ds     = [ds_scores[d["id"]] for d in val_docs  if not math.isnan(ds_scores[d["id"]])]
    in_ds      = [ds_scores[d["id"]] for d in in_docs   if not math.isnan(ds_scores[d["id"]])]
    out_ds     = [ds_scores[d["id"]] for d in out_docs  if not math.isnan(ds_scores[d["id"]])]

    set1 = dataset_level_test(in_ds,  val_ds, "set1")
    set2 = dataset_level_test(out_ds, val_ds, "set2")

    sub_ds_path = SUB_DIR / "A_datasets.jsonl"
    with open(sub_ds_path, "w") as f:
        f.write(json.dumps(set1) + "\n")
        f.write(json.dumps(set2) + "\n")
    print(f"Wrote {sub_ds_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  in.jsonl  median score : {np.median(in_scores):.4f}")
    print(f"  out.jsonl median score : {np.median(out_scores):.4f}")
    print(f"  val_in    median score : {np.median(val_scores):.4f}")
    print(f"  threshold (val median) : {threshold:.4f}")
    print(f"  set1 p={set1['p_value']:.2e}  -> {set1['verdict']}")
    print(f"  set2 p={set2['p_value']:.2e}  -> {set2['verdict']}")
    print(f"{'='*50}")
    print("\nNext step:")
    print("  python code/practice/score_A.py \\")
    print("    --dataset-submission submissions/A_datasets.jsonl \\")
    print("    --dataset-gt data/A/ground_truth_dataset.jsonl")


if __name__ == "__main__":
    main()
