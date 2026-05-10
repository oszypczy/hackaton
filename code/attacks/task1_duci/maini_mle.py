"""
Maini MLE: per-arch synth-bank × signal selection + curve inversion.

Reads pre-extracted signal JSONs from `maini_blind_walk` runs:
    <signals_root>/targets/target_<model_id>.json
    <signals_root>/synth_2k_r18/synth_<arch>_<p>.json
    <signals_root>/synth_7k_r18/synth_<arch>_<p>.json
    <signals_root>/synth_7k_r50/synth_<arch>_<p>.json
    <signals_root>/synth_7k_r152/synth_<arch>_<p>.json

For each arch ∈ {0, 1, 2}:
    For each synth bank candidate (R18: both 2k+7k; R50/R152: 7k only):
        For each signal key in payload["signals"]:
            For each polynomial degree ∈ {1, 2, 3}:
                LOO-MAE on synth (≥3 points required)
        Pick best (bank, signal, degree) by minimum LOO-MAE.

Predict 9 real targets via curve inversion (numpy roots / dense grid),
clamp to [0, 1] (no further; organizer says p ∈ [0, 1] but NOT quantized).

Writes submissions/<out>.csv. Reuses debias.write_submission_csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import MODEL_IDS
from .debias import write_submission_csv
from .mle import fit_predict_poly, loo_mae_poly


SYNTH_BANK_DIRS = {
    "synth_2k_r18": ["0"],          # only R18 in legacy bank
    "synth_7k_r18": ["0"],
    "synth_7k_r50": ["1"],
    "synth_7k_r152": ["2"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-root", type=str, required=True,
                    help="Root dir holding targets/ and synth_*/ subdirs")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--degrees", type=str, default="1,2,3")
    ap.add_argument("--use-bank-r18", type=str, default="auto",
                    choices=["auto", "synth_2k_r18", "synth_7k_r18"],
                    help="auto = pick by LOO-MAE; force a specific bank for R18")
    ap.add_argument("--use-signal", type=str, default="",
                    help="force a single signal across all archs (else auto)")
    ap.add_argument("--ensemble-banks-r18", action="store_true",
                    help="If set, average R18 predictions across both banks")
    ap.add_argument("--clamp-lo", type=float, default=0.0)
    ap.add_argument("--clamp-hi", type=float, default=1.0)
    ap.add_argument("--write-decisions", type=str, default="",
                    help="optional path to dump full decision table JSON")
    return ap.parse_args()


def _load_targets(targets_dir: Path) -> dict[str, dict]:
    out = {}
    for jp in sorted(targets_dir.glob("target_model_*.json")):
        with open(jp) as f:
            payload = json.load(f)
        mid = payload["model_id"]
        out[mid] = payload
    return out


def _load_synth_bank(bank_dir: Path, expect_arch: str) -> list[dict]:
    if not bank_dir.exists():
        return []
    out = []
    for jp in sorted(bank_dir.glob("synth_*.json")):
        with open(jp) as f:
            payload = json.load(f)
        if payload.get("arch_digit") != expect_arch:
            # legacy bank may not match — skip silently
            continue
        out.append(payload)
    return out


def _all_signals(payload: dict) -> list[str]:
    return list(payload["signals"].keys())


def _grid_search_bank(synth_payloads: list[dict], degrees: list[int]) -> list[dict]:
    """Returns list of decision dicts: {signal, degree, loo_mae, sigs, ps}."""
    if len(synth_payloads) < 3:
        return []
    ps = [float(p["true_p"]) for p in synth_payloads]
    keys = _all_signals(synth_payloads[0])
    decisions = []
    for k in keys:
        sigs = [float(p["signals"][k]) for p in synth_payloads]
        for d in degrees:
            try:
                m = loo_mae_poly(sigs, ps, d)
            except Exception:
                m = float("inf")
            decisions.append({
                "signal": k, "degree": d, "loo_mae": float(m),
                "sigs": sigs, "ps": ps,
            })
    return sorted(decisions, key=lambda r: r["loo_mae"])


def predict_curve(decision: dict, s_target: float) -> float:
    return fit_predict_poly(decision["sigs"], decision["ps"], s_target, decision["degree"])


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    degrees = [int(x) for x in args.degrees.split(",")]

    sig_root = Path(args.signals_root)
    targets_dir = sig_root / "targets"
    if not targets_dir.exists():
        raise SystemExit(f"missing {targets_dir}")
    targets = _load_targets(targets_dir)
    if len(targets) != 9:
        print(f"[maini_mle] WARNING: expected 9 targets, got {len(targets)}", flush=True)

    # Build per-arch bank candidates
    arch_to_banks: dict[str, list[str]] = {"0": [], "1": [], "2": []}
    for bank_name, archs in SYNTH_BANK_DIRS.items():
        for a in archs:
            arch_to_banks[a].append(bank_name)

    arch_decisions: dict[str, dict] = {}
    full_table: dict[str, list] = {}
    for arch in ("0", "1", "2"):
        print(f"\n[maini_mle] === arch={arch} ===", flush=True)
        bank_options = arch_to_banks[arch]
        if arch == "0" and args.use_bank_r18 != "auto":
            bank_options = [args.use_bank_r18]
        per_bank_best: list[tuple[str, dict]] = []
        for bank_name in bank_options:
            bank_dir = sig_root / bank_name
            synths = _load_synth_bank(bank_dir, expect_arch=arch)
            if len(synths) < 3:
                print(f"  [skip] {bank_name}: {len(synths)} synth payloads (need ≥3)",
                      flush=True)
                continue
            print(f"  [{bank_name}] {len(synths)} synth, p values: "
                  f"{sorted(set(round(s['true_p'], 2) for s in synths))}", flush=True)
            decs = _grid_search_bank(synths, degrees)
            if not decs:
                continue
            full_table[f"{arch}_{bank_name}"] = [
                {"signal": d["signal"], "degree": d["degree"], "loo_mae": d["loo_mae"]}
                for d in decs[:10]
            ]
            top3 = decs[:3]
            for d in top3:
                print(f"    LOO-MAE={d['loo_mae']:.4f}  signal={d['signal']:24s}  deg={d['degree']}",
                      flush=True)
            per_bank_best.append((bank_name, decs[0]))

        if not per_bank_best:
            print(f"  [arch={arch}] no usable bank!", flush=True)
            continue

        # Pick globally best bank
        if args.use_signal:
            for bank_name, dec in per_bank_best:
                bank_dir = sig_root / bank_name
                synths = _load_synth_bank(bank_dir, expect_arch=arch)
                ps = [float(p["true_p"]) for p in synths]
                k = args.use_signal
                if k not in synths[0]["signals"]:
                    continue
                sigs = [float(p["signals"][k]) for p in synths]
                best_d, best_m = None, float("inf")
                for d in degrees:
                    try:
                        m = loo_mae_poly(sigs, ps, d)
                    except Exception:
                        m = float("inf")
                    if m < best_m:
                        best_m, best_d = m, d
                dec["signal"] = k
                dec["degree"] = best_d
                dec["loo_mae"] = best_m
                dec["sigs"] = sigs
                dec["ps"] = ps

        bank_choice, decision = min(per_bank_best, key=lambda kv: kv[1]["loo_mae"])
        print(f"  [arch={arch}] BEST bank={bank_choice}  signal={decision['signal']}  "
              f"deg={decision['degree']}  LOO-MAE={decision['loo_mae']:.4f}", flush=True)
        arch_decisions[arch] = {"bank": bank_choice, **decision,
                                  "all_banks": per_bank_best}

    # Decision summary + sanity gate
    print(f"\n[maini_mle] === arch decisions summary ===", flush=True)
    for a, dec in arch_decisions.items():
        risk = ("PASS" if dec["loo_mae"] < 0.04 else
                "RISKY" if dec["loo_mae"] < 0.07 else "FAIL")
        print(f"  arch={a}  bank={dec['bank']:18s}  signal={dec['signal']:24s}  "
              f"deg={dec['degree']}  LOO-MAE={dec['loo_mae']:.4f}  [{risk}]", flush=True)

    # Predict 9 real targets
    print(f"\n[maini_mle] === predicting 9 real targets ===", flush=True)
    predictions: dict[str, float] = {}
    for mid in MODEL_IDS:
        arch = mid.removeprefix("model_")[0]
        if arch not in arch_decisions:
            print(f"  {mid}  arch={arch}  [SKIP — no decision]", flush=True)
            continue
        dec = arch_decisions[arch]
        signal = dec["signal"]
        if mid not in targets:
            print(f"  {mid}  [SKIP — target signal missing]", flush=True)
            continue
        s_target = float(targets[mid]["signals"][signal])

        if arch == "0" and args.ensemble_banks_r18:
            # Ensemble across all R18 banks for this signal
            preds = []
            for bank_name, dec_b in dec.get("all_banks", []):
                p_b = predict_curve(dec_b, s_target)
                preds.append(p_b)
                print(f"    {mid} bank={bank_name:18s}  s={s_target:+.5f}  p={p_b:+.4f}",
                      flush=True)
            p_raw = float(np.mean(preds))
        else:
            p_raw = predict_curve(dec, s_target)

        p_final = float(np.clip(p_raw, args.clamp_lo, args.clamp_hi))
        key = mid.removeprefix("model_")
        predictions[key] = p_final
        print(f"  {mid}  arch={arch}  signal={signal:24s}  "
              f"s={s_target:+.5f}  p_raw={p_raw:+.4f}  p_final={p_final:.4f}",
              flush=True)

    # Sanity gates (print only)
    if predictions:
        ps_arr = np.array(list(predictions.values()))
        print(f"\n[maini_mle] sanity:  N={len(ps_arr)}  mean={ps_arr.mean():.4f}  "
              f"min={ps_arr.min():.4f}  max={ps_arr.max():.4f}  std={ps_arr.std():.4f}",
              flush=True)
        if not (0.20 <= ps_arr.mean() <= 0.65):
            print(f"  [WARN] mean p̂ out of expected range [0.20, 0.65]", flush=True)
        if (ps_arr.max() - ps_arr.min()) < 0.05:
            print(f"  [WARN] very narrow spread — likely flat signal", flush=True)

    if args.write_decisions:
        with open(args.write_decisions, "w") as f:
            json.dump({
                "arch_decisions": {a: {"bank": d["bank"], "signal": d["signal"],
                                       "degree": d["degree"], "loo_mae": d["loo_mae"]}
                                   for a, d in arch_decisions.items()},
                "full_table": full_table,
            }, f, indent=2)
        print(f"[maini_mle] wrote decisions → {args.write_decisions}", flush=True)

    if not predictions:
        raise SystemExit("[maini_mle] no predictions; aborting CSV write")
    write_submission_csv(predictions, str(out_path))
    print(f"\n[maini_mle] wrote {out_path}", flush=True)


if __name__ == "__main__":
    run()
