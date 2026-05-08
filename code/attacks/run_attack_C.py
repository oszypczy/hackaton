"""
Run Challenge C attack: Diffusion memorization discovery (Carlini et al. pipeline).

Steps:
  1. Build candidate images from CIFAR-10 raw (if candidates/ dir missing)
  2. Sample images from DDPM checkpoint (or public google/ddpm-cifar10-32 as fallback)
  3. For each candidate compute min L2 distance to generated samples (Easy baseline)
  4. Optionally upgrade to CLIP/DINO embedding similarity (Solid)
  5. Output ranked submission

Usage:
    # First run: build candidates from CIFAR-10 raw + generate DDPM samples
    python code/attacks/run_attack_C.py --build-candidates --n-samples 5000

    # Subsequent runs: reuse cached samples (fast)
    python code/attacks/run_attack_C.py

    # Solid mode: CLIP embeddings instead of pixel L2
    python code/attacks/run_attack_C.py --mode clip

Outputs:
    submissions/C.jsonl   {"id", "memorization_score", "rank"}

Runtime:
    Easy (pixel L2, MPS): ~15-30 min (sampling 5000 CIFAR-10 32x32)
    Solid (CLIP):         +2 min (CLIP encode 6000 images on MPS)

NOTE: Requires model weights in data/C/ddpm_cifar10_memorized/unet/diffusion_pytorch_model.bin
      or diffusion_pytorch_model.safetensors.
      Fallback: google/ddpm-cifar10-32 (public, not memorized — pipeline runs but scores random).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT         = Path(__file__).resolve().parents[2]
DATA_C       = ROOT / "data" / "C"
CANDIDATES   = DATA_C / "candidates"
CACHE_DIR    = ROOT / ".cache"
SAMPLES_NPY  = CACHE_DIR / "C_ddpm_samples.npy"
CLIP_EMB_NPY = CACHE_DIR / "C_clip_embeddings.npy"
SUB_DIR      = ROOT / "submissions"

MODEL_DIR    = DATA_C / "ddpm_cifar10_memorized"
FALLBACK_ID  = "google/ddpm-cifar10-32"
CIFAR_RAW    = DATA_C / "cifar10_raw" / "cifar-10-batches-py"


# ── CIFAR-10 loader ───────────────────────────────────────────────────────────

def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10_train() -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels): images shape (50000, 32, 32, 3) uint8."""
    batches = [_unpickle(CIFAR_RAW / f"data_batch_{i}") for i in range(1, 6)]
    imgs    = np.concatenate([b[b"data"] for b in batches], axis=0)
    labels  = np.concatenate([b[b"labels"] for b in batches], axis=0)
    imgs    = imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW→NHWC
    return imgs, labels


def load_cifar10_test() -> tuple[np.ndarray, np.ndarray]:
    batch  = _unpickle(CIFAR_RAW / "test_batch")
    imgs   = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"])
    return imgs, labels


# ── Build candidate images ────────────────────────────────────────────────────

def build_candidates(meta_path: Path = DATA_C / "candidates_meta.jsonl") -> None:
    """
    Reconstruct candidate images from CIFAR-10 raw.

    Candidate layout (from challenge spec):
      - 50 memorized  (CIFAR-10 train, indices from memorized_indices.json)
      - 500 train non-memorized (next 500 train images not in memorized set)
      - 450 test images (first 450 from test set)

    Image IDs 0-999 match candidates_meta.jsonl order.
    Ground truth is in ground_truth.jsonl (is_memorized flag).
    """
    from PIL import Image

    CANDIDATES.mkdir(exist_ok=True)

    memorized_idx = set(json.loads((DATA_C / "memorized_indices.json").read_text()))
    gt_rows       = {r["id"]: r for r in _load_jsonl(DATA_C / "ground_truth.jsonl")}

    train_imgs, _ = load_cifar10_train()
    test_imgs,  _ = load_cifar10_test()

    # Memorized candidates: CIFAR-10 train images at memorized indices
    mem_indices = sorted(memorized_idx)[:50]
    # Non-memorized train: first 500 train images NOT in memorized set
    non_mem_train = [i for i in range(len(train_imgs)) if i not in memorized_idx][:500]
    # Test images: first 450
    test_indices = list(range(450))

    candidate_sources = (
        [("train", i) for i in mem_indices]
        + [("train", i) for i in non_mem_train]
        + [("test",  i) for i in test_indices]
    )

    meta = _load_jsonl(meta_path) if meta_path.exists() else []
    meta_ids = [r["id"] for r in meta]

    for cand_id, (src, idx) in enumerate(tqdm(candidate_sources, desc="Building candidates")):
        img_arr = train_imgs[idx] if src == "train" else test_imgs[idx]
        img     = Image.fromarray(img_arr.astype(np.uint8))
        fname   = f"img_{cand_id:04d}.png"
        img.save(CANDIDATES / fname)

    print(f"Built {len(candidate_sources)} candidate images → {CANDIDATES}")


# ── DDPM sampling ─────────────────────────────────────────────────────────────

def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_ddpm_pipeline():
    from diffusers import DDPMPipeline, DDIMScheduler

    unet_bin = MODEL_DIR / "unet" / "diffusion_pytorch_model.bin"
    unet_sf  = MODEL_DIR / "unet" / "diffusion_pytorch_model.safetensors"

    if unet_bin.exists() or unet_sf.exists():
        model_id = str(MODEL_DIR)
        print(f"  Loading local DDPM: {model_id}")
    else:
        model_id = FALLBACK_ID
        print(f"  [WARN] No local weights — using public {FALLBACK_ID} (not memorized, scores random)")

    pipe = DDPMPipeline.from_pretrained(model_id, use_safetensors=unet_sf.exists())
    # Use DDIM scheduler for speed (50 steps vs 1000 original)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate_samples(n_samples: int = 5000, batch_size: int = 100) -> np.ndarray:
    """Generate DDPM samples, shape (N, 32, 32, 3) uint8. Cached to .cache/."""
    CACHE_DIR.mkdir(exist_ok=True)
    if SAMPLES_NPY.exists():
        print(f"  Cache hit: {SAMPLES_NPY} ({n_samples} samples)")
        return np.load(SAMPLES_NPY)

    device = _pick_device()
    print(f"  Generating {n_samples} DDPM samples on {device}...")
    pipe = load_ddpm_pipeline()
    pipe.to(device)

    all_imgs = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches), desc="DDPM sampling"):
        out = pipe(batch_size=min(batch_size, n_samples - len(all_imgs)), num_inference_steps=50)
        all_imgs.extend([np.array(img) for img in out.images])
        if len(all_imgs) >= n_samples:
            break

    samples = np.stack(all_imgs[:n_samples])
    np.save(SAMPLES_NPY, samples)
    print(f"  Saved {SAMPLES_NPY}")
    return samples


# ── Scoring: pixel L2 (Easy) ──────────────────────────────────────────────────

def pixel_l2_scores(candidates: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    For each candidate, find min L2 distance to any generated sample.
    Score = 1 / (1 + min_dist) — high means likely memorized.

    candidates: (N_cand, 32, 32, 3) float32
    samples:    (N_samp, 32, 32, 3) float32
    """
    cands_flat = candidates.reshape(len(candidates), -1).astype(np.float32) / 255.0
    samps_flat = samples.reshape(len(samples),    -1).astype(np.float32) / 255.0

    scores = []
    chunk  = 256  # process in chunks to avoid OOM
    for i in tqdm(range(0, len(cands_flat), chunk), desc="Pixel L2 scoring"):
        c     = cands_flat[i : i + chunk]          # (chunk, D)
        dists = np.sqrt(((c[:, None, :] - samps_flat[None, :, :]) ** 2).sum(-1))  # (chunk, N_samp)
        scores.append(1.0 / (1.0 + dists.min(axis=1)))

    return np.concatenate(scores)


# ── Scoring: CLIP embeddings (Solid) ─────────────────────────────────────────

def clip_scores(candidates: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    CLIP ViT-B/32 cosine similarity — top-3 mean per candidate.
    Requires: pip install open_clip_torch
    """
    import torch
    try:
        import open_clip
    except ImportError:
        print("  [WARN] open_clip not installed (`pip install open_clip_torch`). Falling back to pixel L2.")
        return pixel_l2_scores(candidates, samples)

    device = _pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    from PIL import Image

    def encode(imgs: np.ndarray) -> np.ndarray:
        tensors = torch.stack([
            preprocess(Image.fromarray(img.astype(np.uint8))) for img in imgs
        ]).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    print("  Encoding candidates with CLIP...")
    cand_emb = encode(candidates)  # (N_cand, D)
    print("  Encoding samples with CLIP...")
    samp_emb = encode(samples)     # (N_samp, D)

    np.save(CLIP_EMB_NPY, np.concatenate([cand_emb, samp_emb]))

    # For each candidate: mean cosine sim to top-3 nearest samples
    scores = []
    for i in tqdm(range(len(cand_emb)), desc="CLIP top-3 similarity"):
        sims = cand_emb[i] @ samp_emb.T  # (N_samp,)
        scores.append(float(np.sort(sims)[-3:].mean()))

    return np.array(scores)


# ── Load candidate images ─────────────────────────────────────────────────────

def load_candidates() -> tuple[np.ndarray, list[dict]]:
    """Returns (images_uint8, meta_list). Builds candidates if dir missing."""
    meta = _load_jsonl(DATA_C / "candidates_meta.jsonl")

    if not CANDIDATES.exists() or not any(CANDIDATES.iterdir()):
        print("  candidates/ missing — building from CIFAR-10 raw...")
        build_candidates()

    from PIL import Image
    imgs = []
    for row in tqdm(meta, desc="Loading candidates"):
        p = CANDIDATES / row["filename"]
        imgs.append(np.array(Image.open(p)))
    return np.stack(imgs), meta


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Challenge C: diffusion memorization ranking")
    ap.add_argument("--mode",             choices=["pixel", "clip"], default="pixel",
                    help="Scoring mode: pixel=L2 (easy, fast), clip=CLIP sim (solid)")
    ap.add_argument("--n-samples",        type=int, default=5000,
                    help="Number of DDPM samples to generate (more = better, but slower)")
    ap.add_argument("--build-candidates", action="store_true",
                    help="Force rebuild candidates/ from CIFAR-10 raw")
    args = ap.parse_args()

    if args.build_candidates and CANDIDATES.exists():
        import shutil
        shutil.rmtree(CANDIDATES)

    # 1. Load / build candidates
    candidates, meta = load_candidates()
    print(f"  {len(candidates)} candidates loaded")

    # 2. Generate or load cached DDPM samples
    samples = generate_samples(n_samples=args.n_samples)
    print(f"  {len(samples)} DDPM samples")

    # 3. Score
    if args.mode == "clip":
        scores = clip_scores(candidates, samples)
    else:
        scores = pixel_l2_scores(candidates, samples)

    # 4. Rank descending (higher score = more likely memorized)
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    # 5. Write submission
    SUB_DIR.mkdir(exist_ok=True)
    out_path = SUB_DIR / "C.jsonl"
    with open(out_path, "w") as f:
        for row, score, rank in zip(meta, scores, ranks):
            f.write(json.dumps({
                "id":                row["id"],
                "memorization_score": round(float(score), 6),
                "rank":              int(rank),
            }) + "\n")

    print(f"\nWrote {out_path}")
    print(f"  Top-10 candidates by score: ids {[meta[i]['id'] for i in order[:10]]}")
    print(f"\nScore: python code/practice/score_C.py --submission submissions/C.jsonl")


if __name__ == "__main__":
    main()
