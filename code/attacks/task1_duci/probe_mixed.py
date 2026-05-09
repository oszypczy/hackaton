"""Quick diagnostic: target / ref / synth accuracy + confidence on MIXED.

Check if regime match on POP_z carries over to MIXED.

Trust boundary note: ref/synth checkpoints are stdlib-pickle dumps written by
our own train_ref.py / train_synth.py.
"""
from __future__ import annotations

import argparse
import json
import pickle as _pkl  # internal trust: own outputs
from pathlib import Path

import numpy as np
import torch

from .data import BATCH, MODEL_IDS, load_mixed, preprocess_batch
from .targets import build_resnet, load_target


def load_state_internal(path):
    with open(path, "rb") as f:
        return _pkl.load(f)


@torch.no_grad()
def stats_on(model, X, y, device):
    correct = 0
    sum_loss = 0.0
    sum_conf = 0.0
    total = 0
    for i in range(0, len(X), BATCH):
        xb = preprocess_batch(X[i:i + BATCH], device)
        yb = torch.from_numpy(y[i:i + BATCH]).long().to(device)
        logits = model(xb)
        sm = torch.softmax(logits, dim=1)
        tc = sm.gather(1, yb.unsqueeze(1)).squeeze(1)
        sum_conf += tc.sum().item()
        sum_loss += -torch.log(torch.clamp(tc, min=1e-12)).sum().item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.shape[0]
    return correct/total, sum_loss/total, sum_conf/total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", type=str, default="")
    ap.add_argument("--synth-dir", type=str, default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_m, y_m = load_mixed()
    print(f"[probe_mixed] device={device} MIXED N={len(X_m)}")
    print(f"\n{'kind':6s} {'name':24s} {'arch':4s} {'p':>6s} {'acc_mixed':>10s} {'loss_mixed':>10s} {'conf_mixed':>10s}")
    print("-" * 80)

    for mid in MODEL_IDS:
        arch = mid.removeprefix("model_")[0]
        model = load_target(mid, device=device)
        a, l, c = stats_on(model, X_m, y_m, device)
        print(f"{'target':6s} {mid:24s} {arch:>4s} {'-':>6s} {a:>10.4f} {l:>10.4f} {c:>10.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.refs_dir:
        d = Path(args.refs_dir)
        for jp in sorted(d.glob("manifest_*.json"))[:3]:
            with open(jp) as f:
                m = json.load(f)
            model = build_resnet(m["arch_digit"], device=device, num_classes=100)
            model.load_state_dict(load_state_internal(m["checkpoint"]))
            model.train(False)
            a, l, c = stats_on(model, X_m, y_m, device)
            print(f"{'ref':6s} {jp.stem:24s} {m['arch_digit']:>4s} {'B':>6s} {a:>10.4f} {l:>10.4f} {c:>10.4f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    if args.synth_dir:
        d = Path(args.synth_dir)
        for jp in sorted(d.glob("synth_*.json")):
            with open(jp) as f:
                m = json.load(f)
            model = build_resnet(m["arch_digit"], device=device, num_classes=100)
            model.load_state_dict(load_state_internal(m["checkpoint"]))
            model.train(False)
            a, l, c = stats_on(model, X_m, y_m, device)
            print(f"{'synth':6s} {jp.stem:24s} {m['arch_digit']:>4s} {m['true_p']:>6.2f} {a:>10.4f} {l:>10.4f} {c:>10.4f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
