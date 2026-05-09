"""
Smoke test for Task 1 (DUCI).

Loads all 9 organizer checkpoints + POPULATION (10k labeled, never seen by
any model). For each combo of (input resolution, normalization), measures
top-1 accuracy on POPULATION.

Goal: pick the (resolution, norm) the organizer used during training. The
combo with highest mean accuracy is our preprocessing for downstream MIA.

Run on JURECA login node:
    cd /p/scratch/training2615/kempinski1/Czumpers/DUCI
    .venv/bin/python /p/scratch/training2615/kempinski1/Czumpers/repo-$USER/code/attacks/task1_duci/smoke.py
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm

DUCI_ROOT = Path(os.environ.get(
    "DUCI_ROOT",
    "/p/scratch/training2615/kempinski1/Czumpers/DUCI",
))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 128

CIFAR_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
CIFAR_STD  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
IMG_MEAN   = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMG_STD    = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ARCH = {"0": tvm.resnet18, "1": tvm.resnet50, "2": tvm.resnet152}


def load_model(model_id: str) -> torch.nn.Module:
    arch_digit = model_id.removeprefix("model_")[0]
    model = ARCH[arch_digit](weights=None, num_classes=100)
    with open(DUCI_ROOT / "MODELS" / f"{model_id}.pkl", "rb") as f:
        sd = pickle.load(f)
    model.load_state_dict(sd)
    model.train(False)
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, X, y, resolution: int, mean: torch.Tensor, std: torch.Tensor) -> float:
    mean, std = mean.to(DEVICE), std.to(DEVICE)
    correct, total = 0, 0
    for i in range(0, len(X), BATCH):
        xb = torch.from_numpy(X[i:i + BATCH]).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        if resolution != 32:
            xb = F.interpolate(xb, size=resolution, mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        pred = model(xb).argmax(dim=1)
        correct += (pred.cpu().numpy() == y[i:i + BATCH]).sum()
        total += pred.shape[0]
    return correct / total


def main() -> None:
    print(f"Device: {DEVICE}, DUCI_ROOT: {DUCI_ROOT}")

    X = np.load(DUCI_ROOT / "DATA" / "POPULATION" / "X.npy")
    y = np.load(DUCI_ROOT / "DATA" / "POPULATION" / "y.npy")
    if y.ndim == 2:
        y = y.argmax(1)
    print(f"POPULATION: X={X.shape} {X.dtype}, y={y.shape} {y.dtype}, classes={int(y.max()) + 1}")

    combos = [
        ("32+CIFAR",   32,  CIFAR_MEAN, CIFAR_STD),
        ("32+ImgNet",  32,  IMG_MEAN,   IMG_STD),
        ("224+CIFAR",  224, CIFAR_MEAN, CIFAR_STD),
        ("224+ImgNet", 224, IMG_MEAN,   IMG_STD),
    ]
    combo_names = [c[0] for c in combos]

    model_ids = [f"model_{a}{i}" for a in (0, 1, 2) for i in (0, 1, 2)]
    results: dict[str, list[float]] = {}

    for mid in model_ids:
        t0 = time.time()
        model = load_model(mid)
        accs = [evaluate(model, X, y, res, mean, std) for _, res, mean, std in combos]
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        results[mid] = accs
        print(f"{mid}  " + "  ".join(f"{n}={a:.3f}" for n, a in zip(combo_names, accs))
              + f"  ({time.time() - t0:.1f}s)")

    print("\n--- per-arch mean acc on POPULATION ---")
    archs = {
        "resnet18":  ["model_00", "model_01", "model_02"],
        "resnet50":  ["model_10", "model_11", "model_12"],
        "resnet152": ["model_20", "model_21", "model_22"],
    }
    overall_means = np.mean([results[m] for m in model_ids], axis=0)
    for arch_name, mids in archs.items():
        means = np.mean([results[m] for m in mids], axis=0)
        best = combo_names[int(np.argmax(means))]
        line = "  ".join(f"{n}={m:.3f}" for n, m in zip(combo_names, means))
        print(f"  {arch_name:10}: {line}  -> best: {best}")

    overall_best = combo_names[int(np.argmax(overall_means))]
    print(f"\n  OVERALL    : "
          + "  ".join(f"{n}={m:.3f}" for n, m in zip(combo_names, overall_means))
          + f"  -> best: {overall_best}")


if __name__ == "__main__":
    main()
