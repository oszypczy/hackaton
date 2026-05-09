"""
Read signals.npz produced by extract_signals, evaluate every (signal, method, post)
combination via LOO-MAE on synth, and emit the top-K candidate CSVs.

Methods
-------
linear    : analytical inverse of degree-1 fit on synth, full float64 precision
bayes_disc: posterior p_hat = argmax_q  P(observed | p=q)  with q in {0.0, 0.1, ..., 1.0}.
            Likelihood = Gaussian over signal-vs-p curve, sigma from synth residuals.

Post-processing
---------------
none      : keep continuous prediction
snap_10   : round to nearest 0.1
snap_05   : round to nearest 0.05

Per-arch routing
----------------
arch=0 (R18) -> uses arch=0 synth (we have 13 points)
arch=1 (R50) -> uses arch=1 synth if available, else arch=0
arch=2 (R152)-> uses arch=2 synth if available, else arch=0 (R152 doesn't converge)

Output
------
submissions/task1_duci_auto_<rank>_<signal>_<method>_<post>.csv
auto_queue.json    : ordered priority list with metadata
auto_report.md     : human-readable LOO-MAE table
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

GRID_10 = np.round(np.arange(0.0, 1.001, 0.1), 6)
GRID_05 = np.round(np.arange(0.0, 1.001, 0.05), 6)

SIGNAL_KEYS = [
    "mean_loss_mixed", "mean_loss_pop", "delta_loss", "loss_ratio",
    "p25_loss_mixed", "p10_loss_mixed", "p75_loss_mixed",
    "aug_invariance_mixed", "aug_invariance_pop", "aug_inv_diff",
    "mean_logit_mixed", "mean_conf_mixed", "mean_conf_pop", "delta_conf",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out-dir", default="submissions")
    ap.add_argument("--queue-out", default="submissions/auto_queue.json")
    ap.add_argument("--report-out", default="submissions/auto_report.md")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--clamp-lo", type=float, default=0.0)
    ap.add_argument("--clamp-hi", type=float, default=1.0)
    ap.add_argument("--unified-arch", default="0",
                    help="if set (e.g. '0'), use this arch's synth for ALL targets "
                         "(default: R18=arch 0; matches our winning submission). "
                         "Pass empty string to enable per-arch routing.")
    return ap.parse_args()


def fit_linear_predict(s_synth, p_synth, s_target):
    s_arr = np.asarray(s_synth, dtype=np.float64)
    p_arr = np.asarray(p_synth, dtype=np.float64)
    a, b = np.polyfit(p_arr, s_arr, 1)
    if abs(a) < 1e-12:
        return float(np.mean(p_arr))
    return float((s_target - b) / a)


def loo_mae_linear(s_synth, p_synth):
    n = len(s_synth)
    if n < 3:
        return float("inf")
    errs = []
    for i in range(n):
        s_tr = [s for j, s in enumerate(s_synth) if j != i]
        p_tr = [p for j, p in enumerate(p_synth) if j != i]
        p_pred = fit_linear_predict(s_tr, p_tr, s_synth[i])
        errs.append(abs(p_pred - p_synth[i]))
    return float(np.mean(errs))


def fit_bayes_disc_predict(s_synth, p_synth, s_target, grid=GRID_10):
    s_arr = np.asarray(s_synth, dtype=np.float64)
    p_arr = np.asarray(p_synth, dtype=np.float64)
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]
    s_sorted = s_arr[order]
    a, b = np.polyfit(p_sorted, s_sorted, 1)
    residuals = s_sorted - (a * p_sorted + b)
    sigma = max(float(np.std(residuals)), 1e-4)
    mu_grid = np.interp(grid, p_sorted, s_sorted)
    log_likelihood = -0.5 * ((s_target - mu_grid) / sigma) ** 2
    return float(grid[int(np.argmax(log_likelihood))])


def loo_mae_bayes(s_synth, p_synth, grid=GRID_10):
    n = len(s_synth)
    if n < 3:
        return float("inf")
    errs = []
    for i in range(n):
        s_tr = [s for j, s in enumerate(s_synth) if j != i]
        p_tr = [p for j, p in enumerate(p_synth) if j != i]
        p_pred = fit_bayes_disc_predict(s_tr, p_tr, s_synth[i], grid)
        errs.append(abs(p_pred - p_synth[i]))
    return float(np.mean(errs))


def signal_monotone_score(s_synth, p_synth):
    s = np.asarray(s_synth)
    p = np.asarray(p_synth)
    rs = np.argsort(np.argsort(s))
    rp = np.argsort(np.argsort(p))
    return float(abs(np.corrcoef(rs, rp)[0, 1]))


def post_process(p_raw, mode):
    if mode == "snap_10":
        return float(round(p_raw / 0.1) * 0.1)
    if mode == "snap_05":
        return float(round(p_raw / 0.05) * 0.05)
    return float(p_raw)


def build_arch_table(data, arch, fallback_arch=None):
    is_synth = data["is_synth"].astype(bool)
    arch_arr = data["arch"].astype(str)
    selected = is_synth & (arch_arr == arch)
    if not selected.any() and fallback_arch is not None:
        selected = is_synth & (arch_arr == fallback_arch)
    p = data["true_p"][selected]
    out = {}
    for key in SIGNAL_KEYS:
        out[key] = (list(map(float, data[key][selected])), list(map(float, p)))
    return out


def predict_target(method, sigs_synth, ps_synth, sig_target, grid):
    if method == "linear":
        return fit_linear_predict(sigs_synth, ps_synth, sig_target)
    if method == "bayes_disc":
        return fit_bayes_disc_predict(sigs_synth, ps_synth, sig_target, grid)
    raise ValueError(method)


def loo_mae(method, sigs_synth, ps_synth, grid):
    if method == "linear":
        return loo_mae_linear(sigs_synth, ps_synth)
    if method == "bayes_disc":
        return loo_mae_bayes(sigs_synth, ps_synth, grid)
    raise ValueError(method)


def write_csv(predictions, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model_id", "proportion"])
        for mid in sorted(predictions):
            w.writerow([mid, f"{predictions[mid]:.6f}"])


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    data = dict(np.load(args.signals, allow_pickle=False))
    arch_arr = data["arch"].astype(str)
    is_synth = data["is_synth"].astype(bool)
    model_id_arr = data["model_id"]

    arch_tables = {}
    if args.unified_arch:
        unified_tbl = build_arch_table(data, args.unified_arch, fallback_arch=None)
        n_pts = len(unified_tbl["mean_loss_mixed"][1])
        for arch in ("0", "1", "2"):
            arch_tables[arch] = unified_tbl
        print(f"[unified arch={args.unified_arch}] {n_pts} synth points used for all targets",
              flush=True)
    else:
        for arch in ("0", "1", "2"):
            fallback = None if arch == "0" else "0"
            tbl = build_arch_table(data, arch, fallback_arch=fallback)
            n_pts = len(tbl["mean_loss_mixed"][1])
            arch_tables[arch] = tbl
            print(f"[arch {arch}] {n_pts} synth points", flush=True)

    methods = ["linear", "bayes_disc"]
    posts = ["none", "snap_10"]
    candidates = []
    arches_to_score = ["0"] if args.unified_arch else list(arch_tables.keys())
    for sig in SIGNAL_KEYS:
        for method in methods:
            arch_loo = {}
            arch_mono = {}
            for arch in arches_to_score:
                tbl = arch_tables[arch]
                s_syn, p_syn = tbl[sig]
                if len(s_syn) < args.min_points:
                    arch_loo[arch] = float("inf")
                    arch_mono[arch] = 0.0
                    continue
                arch_loo[arch] = loo_mae(method, s_syn, p_syn, GRID_10)
                arch_mono[arch] = signal_monotone_score(s_syn, p_syn)
            for arch in ("0", "1", "2"):
                arch_loo.setdefault(arch, arch_loo[arches_to_score[0]])
                arch_mono.setdefault(arch, arch_mono[arches_to_score[0]])
            mean_loo = float(np.mean([arch_loo[a] for a in arches_to_score]))
            max_loo = float(np.max([arch_loo[a] for a in arches_to_score]))
            mean_mono = float(np.mean([arch_mono[a] for a in arches_to_score]))
            for post in posts:
                candidates.append({
                    "signal": sig, "method": method, "post": post,
                    "arch_loo": arch_loo, "mean_loo": mean_loo, "max_loo": max_loo,
                    "mean_mono": mean_mono,
                })

    # ---- Prediction sanity: compute predictions for each candidate before ranking
    target_mask = ~is_synth
    target_ids = model_id_arr[target_mask]
    target_archs = arch_arr[target_mask]

    def candidate_predictions(c):
        preds = {}
        for mid, arch in zip(target_ids, target_archs):
            tbl = arch_tables[str(arch)]
            s_syn, p_syn = tbl[c["signal"]]
            if len(s_syn) < args.min_points:
                preds[str(mid).removeprefix("model_")] = 0.5
                continue
            sig_target = float(data[c["signal"]][model_id_arr == mid][0])
            p_raw = predict_target(c["method"], s_syn, p_syn, sig_target, GRID_10)
            p_post = post_process(p_raw, c["post"])
            preds[str(mid).removeprefix("model_")] = float(np.clip(p_post, args.clamp_lo, args.clamp_hi))
        return preds

    for c in candidates:
        c["predictions"] = candidate_predictions(c)
        vals = list(c["predictions"].values())
        c["pred_min"] = float(min(vals))
        c["pred_max"] = float(max(vals))
        c["pred_range"] = c["pred_max"] - c["pred_min"]
        c["pred_mean"] = float(np.mean(vals))
        # Sanity: penalize degenerate predictions outside plausible [0.05, 0.95] band
        c["sanity_penalty"] = 0.0
        if c["pred_min"] < 0.05 and c["pred_max"] < 0.3:
            c["sanity_penalty"] += 1.0  # all-low corner
        if c["pred_max"] > 0.95 and c["pred_min"] > 0.7:
            c["sanity_penalty"] += 1.0
        if c["pred_range"] < 0.05:
            c["sanity_penalty"] += 0.5

    # Rank: penalty first, then LOO ascending, then mono descending
    candidates.sort(key=lambda c: (c["sanity_penalty"], c["mean_loo"], -c["mean_mono"]))

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-pipeline LOO-MAE report",
        f"_generated {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}_",
        "",
        f"Signals input: `{args.signals}`",
        "",
        "| rank | signal | method | post | mean_loo | max_loo | arch_0 | arch_1 | arch_2 | mono |",
        "|------|--------|--------|------|----------|---------|--------|--------|--------|------|",
    ]
    for i, c in enumerate(candidates):
        lines.append(
            f"| {i + 1} | {c['signal']} | {c['method']} | {c['post']} | "
            f"{c['mean_loo']:.4f} | {c['max_loo']:.4f} | "
            f"{c['arch_loo']['0']:.4f} | {c['arch_loo']['1']:.4f} | {c['arch_loo']['2']:.4f} | "
            f"{c['mean_mono']:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[report] {report_path}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = []

    seen_csv_md5 = set()
    rank_count = 0
    idx = 0
    while rank_count < args.top_k and idx < len(candidates):
        c = candidates[idx]
        idx += 1
        if not np.isfinite(c["mean_loo"]):
            continue
        if c["sanity_penalty"] >= 1.0:
            continue  # skip degenerate predictions
        predictions = c["predictions"]
        tag = f"auto_{rank_count + 1:02d}_{c['signal']}_{c['method']}_{c['post']}"
        out_path = out_dir / f"task1_duci_{tag}.csv"
        write_csv(predictions, out_path)
        md5 = md5_of(out_path)
        if md5 in seen_csv_md5:
            out_path.unlink()
            continue
        seen_csv_md5.add(md5)
        queue.append({
            "rank": rank_count + 1,
            "csv": str(out_path),
            "md5": md5,
            "signal": c["signal"],
            "method": c["method"],
            "post": c["post"],
            "mean_loo": c["mean_loo"],
            "max_loo": c["max_loo"],
            "arch_loo": c["arch_loo"],
            "mean_mono": c["mean_mono"],
            "pred_range": c["pred_range"],
            "pred_min": c["pred_min"],
            "pred_max": c["pred_max"],
            "predictions": predictions,
        })
        rank_count += 1

    queue_path = Path(args.queue_out)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "signals_path": args.signals,
            "n_candidates": len(queue),
            "queue": queue,
        }, f, indent=2)
    print(f"[queue] {queue_path}  ({len(queue)} unique CSVs)", flush=True)
    for q in queue[:5]:
        preds_str = " ".join(f"{k}={v:.3f}" for k, v in sorted(q["predictions"].items()))
        print(f"  #{q['rank']:2d}  {q['signal']}/{q['method']}/{q['post']:8s}  "
              f"loo={q['mean_loo']:.4f}  preds: {preds_str}", flush=True)


if __name__ == "__main__":
    main()
