"""
evaluate_model.py — Trustworthy evaluation of a trained model on its held-out test set.

Loads the exact spatially-disjoint test split saved by pipeline/train.py
(X_test.npy / y_test.npy in the results dir), runs the 5-fold ensemble, fits
temperature scaling on a calibration split, and reports:

  accuracy, macro-F1, weighted-F1, Cohen's kappa,
  ECE + Brier (uncalibrated and calibrated),
  AURC + accuracy@80%-coverage (selective prediction).

Also writes reliability-diagram and risk-coverage figures, and per-sample eval
predictions/confidence (so run_benchmark.py can run paired McNemar tests).

Usage:
    python -m evaluation.evaluate_model --season kharif \
        --model-dir models --results-dir results
"""

import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from inference.predict import CropPredictor
from evaluation.calibrate import fit_temperature, apply_temperature, softmax_np
from evaluation.metrics import (
    expected_calibration_error, brier_score, risk_coverage_curve,
    accuracy_at_coverage,
)


def _split_calib_eval(y, calib_frac, seed):
    """Stratified calib/eval split; falls back to random if stratification fails."""
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - calib_frac,
                                     random_state=seed)
        calib_idx, eval_idx = next(sss.split(np.zeros_like(y), y))
    except ValueError:
        ss = ShuffleSplit(n_splits=1, test_size=1 - calib_frac, random_state=seed)
        calib_idx, eval_idx = next(ss.split(np.zeros_like(y)))
    return calib_idx, eval_idx


def _plot_reliability(rows_uncal, rows_cal, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for rows, lab, col in [(rows_uncal, "uncalibrated", "#d62728"),
                           (rows_cal, "calibrated (T-scaled)", "#1f77b4")]:
        if rows:
            xs = [r["confidence"] for r in rows]
            ys = [r["accuracy"] for r in rows]
            ax.plot(xs, ys, "o-", color=col, label=lab)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def _plot_risk_coverage(coverage, risk, aurc, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coverage, risk, color="#2ca02c")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error rate)")
    ax.set_title(f"Risk-coverage curve (AURC={aurc:.4f})")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Trustworthy evaluation on held-out test set.")
    ap.add_argument("--season", required=True, choices=["rabi", "kharif"])
    ap.add_argument("--model-dir", default="models", help="Parent models dir.")
    ap.add_argument("--results-dir", default="results",
                    help="Dir with <season>/X_test.npy & y_test.npy (from train.py).")
    ap.add_argument("--out-dir", default=None, help="Where to write eval artefacts "
                    "(default: <results-dir>/<season>).")
    ap.add_argument("--calib-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    res_dir = Path(args.results_dir) / args.season
    out_dir = Path(args.out_dir) if args.out_dir else res_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    xt, yt = res_dir / "X_test.npy", res_dir / "y_test.npy"
    if not xt.exists() or not yt.exists():
        sys.exit(f"ERROR: {xt} / {yt} not found. Train with --test-size > 0 first.")
    X_test = np.load(xt)
    y_test = np.load(yt)

    predictor = CropPredictor(season=args.season, model_dir=args.model_dir)
    res = predictor.predict(X_test, return_attention=False)
    logits = res["logits"]
    num_classes = predictor.num_classes

    # ---- Calibration split + temperature scaling ----
    calib_idx, eval_idx = _split_calib_eval(y_test, args.calib_frac, args.seed)
    T = fit_temperature(logits[calib_idx], y_test[calib_idx])

    y_eval = y_test[eval_idx]
    probs_uncal = softmax_np(logits[eval_idx])
    probs_cal = apply_temperature(logits[eval_idx], T)
    preds = probs_uncal.argmax(1)                  # calibration preserves argmax

    # ---- Point metrics ----
    oa = accuracy_score(y_eval, preds)
    f1w = f1_score(y_eval, preds, average="weighted", zero_division=0)
    f1m = f1_score(y_eval, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_eval, preds)

    # ---- Calibration metrics ----
    ece_u, rows_u = expected_calibration_error(probs_uncal, y_eval, args.n_bins)
    ece_c, rows_c = expected_calibration_error(probs_cal, y_eval, args.n_bins)
    brier_u = brier_score(probs_uncal, y_eval, num_classes)
    brier_c = brier_score(probs_cal, y_eval, num_classes)

    # ---- Selective prediction (use calibrated probs) ----
    coverage, risk, aurc = risk_coverage_curve(probs_cal, y_eval)
    acc80 = accuracy_at_coverage(probs_cal, y_eval, 0.8)

    metrics = {
        "season": args.season, "model_dir": str(args.model_dir),
        "model_type": predictor.model_type, "temperature": round(T, 4),
        "n_test": int(len(y_test)), "n_eval": int(len(y_eval)),
        "OA": round(float(oa), 4), "Macro_F1": round(float(f1m), 4),
        "Weighted_F1": round(float(f1w), 4), "Kappa": round(float(kappa), 4),
        "ECE_uncal": round(ece_u, 4), "ECE_cal": round(ece_c, 4),
        "Brier_uncal": round(brier_u, 4), "Brier_cal": round(brier_c, 4),
        "AURC": round(aurc, 4), "Acc@80cov": round(float(acc80), 4),
    }
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(rows_c).to_csv(out_dir / "reliability_cal.csv", index=False)
    pd.DataFrame({"coverage": coverage, "risk": risk}).to_csv(
        out_dir / "risk_coverage.csv", index=False)
    np.save(out_dir / "eval_preds.npy", preds)
    np.save(out_dir / "eval_labels.npy", y_eval)
    np.save(out_dir / "eval_conf.npy", probs_cal.max(1))

    _plot_reliability(rows_u, rows_c, out_dir / "reliability_diagram.png")
    _plot_risk_coverage(coverage, risk, aurc, out_dir / "risk_coverage.png")

    print(f"\n=== Trustworthy evaluation [{args.season} | {predictor.model_type}] ===")
    for k, v in metrics.items():
        print(f"  {k:<14s}: {v}")
    print(f"\n  Artefacts -> {out_dir}/  (eval_metrics.json, reliability_diagram.png, "
          f"risk_coverage.png)")


if __name__ == "__main__":
    main()
