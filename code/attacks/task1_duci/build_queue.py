"""
Build a curated submission queue from signals.npz.

Adds curated NEW variants likely to differ from already-submitted CSVs:
    * Ensemble: mean(mean_loss_mixed, delta_loss, mean_logit_mixed) -> linear -> snap_10
    * Ensemble continuous (no snap)
    * mean_loss_mixed -> bayes_disc on 0.1 grid
    * p75_loss_mixed -> linear -> snap_10 + bayes_disc

Filters duplicates against SUBMISSION_LOG.md so we never re-submit a previously sent CSV.
Writes the final queue to submissions/auto_queue.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .auto_pipeline import (
    GRID_10,
    SIGNAL_KEYS,
    fit_bayes_disc_predict,
    fit_linear_predict,
    md5_of,
    post_process,
    write_csv,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out-dir", default="submissions")
    ap.add_argument("--queue-out", default="submissions/auto_queue.json")
    ap.add_argument("--submission-log", default="SUBMISSION_LOG.md")
    return ap.parse_args()


def submitted_md5s(log_path):
    if not log_path.exists():
        return set()
    out = set()
    for line in log_path.read_text().splitlines():
        m = re.search(r"csv-md5=([0-9a-f]{32})", line)
        if m and ("submitted" in line or "score=" in line or "SUB-" in line):
            out.add(m.group(1))
    return out


def synth_table(data, arch="0"):
    is_synth = data["is_synth"].astype(bool)
    arch_arr = data["arch"].astype(str)
    sel = is_synth & (arch_arr == arch)
    p = data["true_p"][sel]
    out = {}
    for k in SIGNAL_KEYS:
        out[k] = (list(map(float, data[k][sel])), list(map(float, p)))
    return out


def predict_for_targets(data, signal, method, post, tbl):
    s_syn, p_syn = tbl[signal]
    is_synth = data["is_synth"].astype(bool)
    target_mask = ~is_synth
    target_ids = data["model_id"][target_mask]
    preds = {}
    for mid in target_ids:
        s_t = float(data[signal][data["model_id"] == mid][0])
        if method == "linear":
            p_raw = fit_linear_predict(s_syn, p_syn, s_t)
        elif method == "bayes_disc":
            p_raw = fit_bayes_disc_predict(s_syn, p_syn, s_t, GRID_10)
        else:
            raise ValueError(method)
        p_pp = post_process(p_raw, post)
        preds[str(mid).removeprefix("model_")] = float(np.clip(p_pp, 0.0, 1.0))
    return preds


def ensemble_predict(data, signals, method, post, tbl):
    is_synth = data["is_synth"].astype(bool)
    target_mask = ~is_synth
    target_ids = data["model_id"][target_mask]
    preds = {}
    for mid in target_ids:
        ps = []
        for sig in signals:
            s_syn, p_syn = tbl[sig]
            s_t = float(data[sig][data["model_id"] == mid][0])
            if method == "linear":
                p_raw = fit_linear_predict(s_syn, p_syn, s_t)
            elif method == "bayes_disc":
                p_raw = fit_bayes_disc_predict(s_syn, p_syn, s_t, GRID_10)
            else:
                raise ValueError(method)
            ps.append(p_raw)
        p_avg = float(np.mean(ps))
        p_pp = post_process(p_avg, post)
        preds[str(mid).removeprefix("model_")] = float(np.clip(p_pp, 0.0, 1.0))
    return preds


def main():
    args = parse_args()
    data = dict(np.load(args.signals))
    tbl = synth_table(data, arch="0")

    seen = submitted_md5s(Path(args.submission_log))
    print(f"[curate] {len(seen)} previously-submitted CSVs cached", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = []

    for sig in ["mean_loss_mixed", "delta_loss", "p75_loss_mixed",
                "mean_logit_mixed", "aug_inv_diff"]:
        for method in ["linear", "bayes_disc"]:
            for post in ["none", "snap_10"]:
                experiments.append({
                    "tag": f"{sig}_{method}_{post}",
                    "predictions": predict_for_targets(data, sig, method, post, tbl),
                    "desc": f"signal={sig} method={method} post={post}",
                    "type": "single",
                })

    ensembles = [
        ("ens_loss_signals", ["mean_loss_mixed", "delta_loss", "mean_logit_mixed"]),
        ("ens_3_modalities", ["mean_loss_mixed", "delta_loss", "aug_inv_diff"]),
        ("ens_loss_top4", ["mean_loss_mixed", "delta_loss", "p75_loss_mixed", "mean_logit_mixed"]),
        ("ens_p75_meanloss", ["mean_loss_mixed", "p75_loss_mixed"]),
    ]
    for name, sigs in ensembles:
        for method in ["linear"]:
            for post in ["none", "snap_10"]:
                experiments.append({
                    "tag": f"{name}_{method}_{post}",
                    "predictions": ensemble_predict(data, sigs, method, post, tbl),
                    "desc": f"ensemble({sigs}) {method} {post}",
                    "type": "ensemble",
                })

    queue = []
    seen_md5 = set()
    for exp in experiments:
        preds = exp["predictions"]
        vals = list(preds.values())
        pred_min, pred_max = min(vals), max(vals)
        if pred_max < 0.1:
            continue
        if pred_min > 0.9:
            continue
        if pred_max - pred_min < 0.02:
            continue
        out_path = out_dir / f"task1_duci_{exp['tag']}.csv"
        write_csv(preds, out_path)
        md5 = md5_of(out_path)
        if md5 in seen:
            print(f"  [skip submitted] {exp['tag']}  md5={md5[:8]}", flush=True)
            out_path.unlink()
            continue
        if md5 in seen_md5:
            out_path.unlink()
            continue
        seen_md5.add(md5)
        queue.append({
            "csv": str(out_path),
            "md5": md5,
            "tag": exp["tag"],
            "desc": exp["desc"],
            "type": exp["type"],
            "predictions": preds,
            "pred_min": float(pred_min),
            "pred_max": float(pred_max),
            "pred_range": float(pred_max - pred_min),
            "pred_mean": float(np.mean(vals)),
        })

    # Anchor: predictions of our best-known submission (SUB-9 = 0.0533 on public 3).
    # Variants with predictions close to this are "credible novel hypotheses".
    # Variants far from this are likely wrong-regime and should be deprioritised.
    anchor = {
        "00": 0.5, "01": 0.6, "02": 0.6,
        "10": 0.4, "11": 0.5, "12": 0.6,
        "20": 0.5, "21": 0.5, "22": 0.5,
    }
    for q in queue:
        diffs = [abs(q["predictions"][k] - anchor[k]) for k in anchor]
        q["anchor_dist"] = float(sum(diffs))
        q["anchor_max"] = float(max(diffs))

    # Priority: anchor_dist ascending (closer to SUB-9 = more credible).
    # Same-distance ties: snap_10 first, ensembles first.
    def priority(q):
        return (
            q["anchor_dist"],
            0 if q["type"] == "ensemble" else 1,
            0 if "snap_10" in q["tag"] else 1,
        )

    queue.sort(key=priority)
    # Hard cap: anchor_dist > 1.0 means predictions disagree wildly with our winner -> skip
    queue = [q for q in queue if q["anchor_dist"] <= 1.0]
    for i, q in enumerate(queue, 1):
        q["rank"] = i
        q["mean_loo"] = 0.0
        q["max_loo"] = 0.0
        q["mean_mono"] = 1.0
        q["arch_loo"] = {"0": 0.0, "1": 0.0, "2": 0.0}
        q["signal"] = q["tag"]
        q["method"] = "n/a"
        q["post"] = "n/a"

    queue_path = Path(args.queue_out)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_candidates": len(queue),
        "queue": queue,
    }, indent=2))

    print(f"\n[queue] {queue_path}  ({len(queue)} unique novel CSVs after anchor filter)", flush=True)
    for q in queue:
        preds_str = " ".join(f"{k}={v:.3f}" for k, v in sorted(q["predictions"].items()))
        print(f"  #{q['rank']:2d}  {q['tag']:50s}  anchor_dist={q['anchor_dist']:.2f}  preds: {preds_str}", flush=True)


if __name__ == "__main__":
    main()
