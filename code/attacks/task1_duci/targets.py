"""
Load organizer's 9 .pkl checkpoints into torchvision ResNets.

The .pkl files are stdlib-pickle dumps of raw OrderedDict state_dicts (verified
in smoke.py for all 9 files). torch.load fails because there's no torch wrapper
magic header; use stdlib loader directly.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torchvision.models as tvm

from .data import DUCI_ROOT

ARCH = {"0": tvm.resnet18, "1": tvm.resnet50, "2": tvm.resnet152}


def load_target(model_id: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    arch_digit = model_id.removeprefix("model_")[0]
    if arch_digit not in ARCH:
        raise ValueError(f"unknown arch digit {arch_digit!r} in {model_id!r}")
    model = ARCH[arch_digit](weights=None, num_classes=100)
    path = DUCI_ROOT / "MODELS" / f"{model_id}.pkl"
    with open(path, "rb") as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    model.train(False)
    return model.to(device)


def build_resnet18_for_ref(device: str | torch.device = "cpu", num_classes: int = 100) -> torch.nn.Module:
    """Same arch convention as organizer's targets (ImageNet head, conv1=7x7)."""
    model = tvm.resnet18(weights=None, num_classes=num_classes)
    return model.to(device)


def build_resnet(arch_digit: str, device: str | torch.device = "cpu",
                 num_classes: int = 100) -> torch.nn.Module:
    if arch_digit not in ARCH:
        raise ValueError(f"unknown arch digit {arch_digit!r}")
    return ARCH[arch_digit](weights=None, num_classes=num_classes).to(device)
