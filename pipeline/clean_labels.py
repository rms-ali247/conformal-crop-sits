"""
clean_labels.py — Uncertainty-guided data-centric label cleaning.

Detects likely-mislabelled training fields from a model's *out-of-fold* (OOF)
predictions and produces a cleaned training set, then reports per-class noise
rates.  This is the data-centric contribution: agricultural ground-truth labels
are noisy, and removing/relabelling confident errors typically lifts macro-F1.

Method (extends Confident Learning, Northcutt et al. JAIR 2021):
  1. Confidence rule — flag field i if the OOF model predicts a class != its
     given label with self-confidence above that class's average
     (the class-conditional threshold t_c).
  2. Uncertainty gate (the novelty) — keep only flagged fields whose predictive
     uncertainty (MC-dropout entropy) is LOW, i.e. the model is *confident* the
     label is wrong.  This raises precision of detected errors and ties the
     data-centric and uncertainty-aware contributions together.
  3. Per-class safety cap so we never relabel more than --max-noise-frac of a class.

Prerequisite — a diagnosis run with OOF dumped over the WHOLE season set
(no test split), e.g.:

    python run.py train --model ms-s4 --season kharif --dump-oof --test-size 0 \
        --data-dir data/processed_monthly \
        --results-dir results_diag

Then:

    python run.py clean-labels --season kharif \
        --results-dir results_diag --data-dir data/processed_monthly \
        --out-dir data/processed_clean --action relabel

Outputs:
  <results-dir>/<season>/label_issues.csv     per-field issues (id, given, suggested, conf, unc)
  <results-dir>/<season>/noise_report.csv      per-class estimated noise rate
  <out-dir>/X_<season>.npy, y_<season>.npy, meta_<season>.csv, label_map.csv  (cleaned)
"""

import sys
import json
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd

from config import RESULTS_DIR, PROCESSED_MONTHLY_DIR, DATA_DIR


def detect_issues(probs: np.ndarray, labels: np.ndarray, unc: np.ndarray,
                  uncertainty_quantile: float, max_noise_frac: float):
    """Return a boolean flag array and the suggested (argmax) label per sample."""
    N, C = probs.shape
    rows = np.arange(N)
    self_conf = probs[rows, labels]
    pred = probs.argmax(1)
    pred_conf = probs[rows, pred]

    # Class-conditional confidence thresholds t_c (Confident Learning).
    t = np.ones(C, dtype=np.float64)
    for c in range(C):
        m = labels == c
        if m.any():
            t[c] = self_conf[m].mean()

    # Rule 1: confident disagreement with the given label.
    candidate = (pred != labels) & (pred_conf >= t[pred])

    # Rule 2: uncertainty gate — keep only low-uncertainty (confident) errors.
    flag = candidate.copy()
    if candidate.any():
        thr = np.quantile(unc[candidate], uncertainty_quantile)
        flag &= unc <= thr

    # Rule 3: per-class safety cap on the GIVEN label.
    for c in range(C):
        idx_c = np.where(flag & (labels == c))[0]
        n_c = int((labels == c).sum())
        cap = int(np.floor(max_noise_frac * n_c))
        if len(idx_c) > cap:
            # keep the most confident errors (highest pred_conf, then lowest unc)
            order = sorted(idx_c, key=lambda i: (-pred_conf[i], unc[i]))
            drop = order[cap:]
            flag[drop] = False

    return flag, pred, pred_conf


def main() -> None:
    ap = argparse.ArgumentParser(description="Uncertainty-guided label cleaning.")
    ap.add_argument("--season", required=True, choices=["rabi", "kharif"])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="Dir holding <season>/oof_*.npy from a --dump-oof run.")
    ap.add_argument("--data-dir", default=str(PROCESSED_MONTHLY_DIR),
                    help="Original per-season tensors to clean.")
    ap.add_argument("--out-dir", default=str(DATA_DIR / "processed_clean"),
                    help="Where to write the cleaned tensors.")
    ap.add_argument("--action", choices=["relabel", "drop"], default="relabel",
                    help="Fix issues by relabelling to the suggested class or dropping them.")
    ap.add_argument("--uncertainty-quantile", type=float, default=0.5,
                    help="Keep flagged errors with uncertainty <= this quantile (0..1).")
    ap.add_argument("--max-noise-frac", type=float, default=0.25,
                    help="Never flag more than this fraction of any class.")
    args = ap.parse_args()

    season = args.season
    res_dir = Path(args.results_dir) / season
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # ---- Load OOF artefacts ----
    for f in ("oof_probs.npy", "oof_labels.npy", "oof_uncertainty.npy", "oof_meta.csv"):
        if not (res_dir / f).exists():
            sys.exit(f"ERROR: {res_dir / f} missing. Run a --dump-oof diagnosis first.")
    probs = np.load(res_dir / "oof_probs.npy")
    labels = np.load(res_dir / "oof_labels.npy")
    unc = np.load(res_dir / "oof_uncertainty.npy")
    oof_meta = pd.read_csv(res_dir / "oof_meta.csv")

    with open(res_dir / "config.json") as fp:
        class_names = json.load(fp)["class_names"]          # season-local order

    if "ID" not in oof_meta.columns:
        sys.exit("ERROR: oof_meta.csv has no 'ID' column — cannot map issues back to fields.")

    # ---- Detect issues ----
    flag, pred, pred_conf = detect_issues(
        probs, labels, unc, args.uncertainty_quantile, args.max_noise_frac
    )
    n_flag = int(flag.sum())
    print(f"[{season}] OOF samples={len(labels)}  flagged label issues={n_flag} "
          f"({100*n_flag/len(labels):.2f}%)")

    # ---- Per-class noise report ----
    rep = []
    for c, name in enumerate(class_names):
        n_c = int((labels == c).sum())
        n_f = int((flag & (labels == c)).sum())
        rep.append({"class": name, "n": n_c, "n_flagged": n_f,
                    "noise_rate": round(n_f / n_c, 4) if n_c else 0.0})
    noise_df = pd.DataFrame(rep).sort_values("noise_rate", ascending=False)
    noise_df.to_csv(res_dir / "noise_report.csv", index=False)
    print("\nEstimated per-class noise rate:")
    print(noise_df.to_string(index=False))

    # ---- Per-field issue list ----
    issues = oof_meta.copy()
    issues["given_label"] = [class_names[l] for l in labels]
    issues["suggested_label"] = [class_names[p] for p in pred]
    issues["pred_confidence"] = np.round(pred_conf, 4)
    issues["uncertainty"] = np.round(unc, 4)
    issues["is_issue"] = flag
    issues[issues["is_issue"]].to_csv(res_dir / "label_issues.csv", index=False)
    print(f"\nSaved per-field issues -> {res_dir / 'label_issues.csv'}")

    # ---- Build cleaned dataset (map back to original tensors by field ID) ----
    X = np.load(data_dir / f"X_{season}.npy")
    y = np.load(data_dir / f"y_{season}.npy")                # GLOBAL labels
    meta = pd.read_csv(data_dir / f"meta_{season}.csv")
    label_map = pd.read_csv(data_dir / "label_map.csv")
    name2global = dict(zip(label_map["crop_type"], label_map["label"]))

    flagged_ids = oof_meta.loc[flag, "ID"].astype(str).tolist()
    suggested_global = {
        str(i): name2global[class_names[p]]
        for i, p in zip(oof_meta.loc[flag, "ID"].astype(str), pred[flag])
    }
    meta_id = meta["ID"].astype(str)
    flagged_mask = meta_id.isin(set(flagged_ids)).to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "relabel":
        y_clean = y.copy()
        for pos, fid in enumerate(meta_id):
            if flagged_mask[pos]:
                y_clean[pos] = suggested_global[fid]
        X_clean, meta_clean = X, meta
        print(f"\nRelabelled {int(flagged_mask.sum())} fields to suggested classes.")
    else:  # drop
        keep = ~flagged_mask
        X_clean, y_clean, meta_clean = X[keep], y[keep], meta[keep].reset_index(drop=True)
        print(f"\nDropped {int(flagged_mask.sum())} fields ({len(X)} -> {len(X_clean)}).")

    np.save(out_dir / f"X_{season}.npy", X_clean)
    np.save(out_dir / f"y_{season}.npy", y_clean)
    meta_clean.to_csv(out_dir / f"meta_{season}.csv", index=False)
    shutil.copy(data_dir / "label_map.csv", out_dir / "label_map.csv")

    print(f"\nCleaned tensors -> {out_dir}/  (action={args.action})")
    print("Retrain on the cleaned data, e.g.:")
    print(f"  python run.py train --model ms-s4 --season {season} --test-size 0.15 "
          f"--spatial-test --data-dir {out_dir}")


if __name__ == "__main__":
    main()
