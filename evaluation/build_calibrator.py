"""
build_calibrator.py — fit a *deployable* trust layer per season and save it next to
the model, so live per-field inference returns a calibrated confidence + a conformal
prediction set + an auto/verify decision (matching the paper).

It reads the out-of-fold artefacts of a --dump-oof run (results/<season>/oof_*.npy)
and the temperature from evaluate_model (results/<season>/eval_metrics.json), then
writes models/<season>/calibration.json with:
  temperature           (for calibrated confidence)
  qhat_robust           (region-robust APS threshold -> the deployment default)
  qhat_marginal         (plain split-conformal threshold, for reference)
  qhat_mondrian         (per-class thresholds)

Usage:
    python run.py build-calibrator --season both
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

from evaluation.conformal import aps_score_all, fit_thresholds


def build_season(season, results_dir, model_dir, alpha, robust_q):
    res = Path(results_dir) / season
    for f in ("oof_probs.npy", "oof_labels.npy", "config.json"):
        if not (res / f).exists():
            print(f"  [skip] {res / f} missing (train with --dump-oof)."); return
    probs = np.load(res / "oof_probs.npy")
    labels = np.load(res / "oof_labels.npy")
    C = probs.shape[1]
    class_names = json.load(open(res / "config.json"))["class_names"]

    meta_path = res / "oof_meta.csv"
    districts = None
    if meta_path.exists():
        m = pd.read_csv(meta_path)
        if "District" in m.columns:
            districts = m["District"].astype(str).to_numpy()

    score_all = aps_score_all(probs)
    q_marg = float(fit_thresholds(score_all, labels, C, alpha, "marginal")[0])
    if districts is not None and len(np.unique(districts)) >= 4:
        q_robust = float(fit_thresholds(score_all, labels, C, alpha, "region-robust",
                                        groups_cal=districts, robust_q=robust_q)[0])
    else:
        q_robust = q_marg
    q_mond = [float(x) for x in fit_thresholds(score_all, labels, C, alpha, "mondrian")]

    T = 1.0
    em = res / "eval_metrics.json"
    if em.exists():
        T = float(json.load(open(em)).get("temperature", 1.0))

    out = {"season": season, "score": "aps", "alpha": alpha, "temperature": round(T, 4),
           "qhat_marginal": round(q_marg, 6), "qhat_robust": round(q_robust, 6),
           "qhat_mondrian": [round(x, 6) for x in q_mond], "class_names": class_names}
    dest = Path(model_dir) / season
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  {season}: T={T:.3f}  qhat_robust={q_robust:.4f}  -> {dest/'calibration.json'}")


def main():
    ap = argparse.ArgumentParser(description="Build a deployable trust layer (calibration.json).")
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="both")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--robust-quantile", type=float, default=0.9)
    args = ap.parse_args()
    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    for s in seasons:
        build_season(s, args.results_dir, args.model_dir, args.alpha, args.robust_quantile)


if __name__ == "__main__":
    main()
