"""
phenology_shift.py — a stress test standing in for inter-annual variation.

The survey covers a single cropping year, so temporal transfer cannot be
measured directly. The dominant year-to-year effect on a crop time series is a
phenological offset: sowing and harvest slide earlier or later by a few weeks
when the monsoon or a cold spell arrives off schedule, so the same crop lands in
a different month of the composite stack. This script simulates that offset by
re-indexing the monthly window of the held-out test fields by k months (edge
replication, since a shifted season genuinely loses one end of the window), then
asks whether the conformal guarantee survives it.

Calibration is unshifted throughout, which is what a deployer would have from a
previous, normal year; only the evaluation fields are shifted. Both sides are
scored through the same five-fold ensemble on the held-out test districts, which
matters: thresholds fitted on single-fold out-of-fold probabilities do not
transfer to ensemble probabilities, because averaging five folds sharpens the
distribution and an APS threshold near 0.9997 then excludes even the top class.
The test districts are split by district into a calibration and an evaluation
half, repeated over several splits, so the k=0 row reproduces the ordinary
geographic protocol and the change with k isolates the temporal offset.

This is a simulation, not a second year of ground truth, and it is reported as one.

Usage:
    python -m evaluation.phenology_shift --season both --shifts -2 -1 0 1 2
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from evaluation.conformal import score_fn, fit_thresholds, evaluate_sets
from inference.predict import CropPredictor


def shift_window(X, k):
    """Re-index the monthly axis by k months with edge replication.

    k > 0 delays the season (model month t sees real month t-k), k < 0 advances it.
    """
    if k == 0:
        return X
    T = X.shape[1]
    idx = np.clip(np.arange(T) - k, 0, T - 1)
    return X[:, idx, :]


def worst_district_coverage(score_all, labels, districts, thresh, min_n):
    sets = score_all <= thresh[None, :]
    covered = sets[np.arange(len(labels)), labels]
    covs = [float(covered[districts == d].mean())
            for d in np.unique(districts) if (districts == d).sum() >= min_n]
    return float(np.min(covs)) if covs else np.nan


SCHEMES = ["marginal", "region-robust", "class-region-robust"]


def run_season(season, args):
    res_dir = Path(args.results_dir) / season
    num_classes = np.load(res_dir / "oof_probs.npy").shape[1]
    score = score_fn(args.score)

    X = np.load(res_dir / "X_test.npy")
    y = np.load(res_dir / "y_test.npy")
    districts = pd.read_csv(res_dir / "meta_test.csv")["District"].astype(str).to_numpy()
    predictor = CropPredictor(season=season, device=args.device)

    # Score every offset once through the ensemble; k=0 is the unshifted reference.
    probs_by_shift, f1_by_shift = {}, {}
    for k in args.shifts:
        out = predictor.predict(shift_window(X, k), return_attention=False,
                                return_probabilities=True)
        probs_by_shift[k] = out["probabilities"]
        pred = out["probabilities"].argmax(1)
        f1_by_shift[k] = (float((pred == y).mean()),
                          float(f1_score(y, pred, average="macro")))

    # District-grouped calibration/evaluation splits of the held-out test set.
    splits = []
    for s in range(args.seed, args.seed + args.repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=s)
        splits.append(next(gss.split(X, y, groups=districts)))

    rows = []
    for k in args.shifts:
        probs_k = probs_by_shift[k]
        scores_k = score(probs_k)
        acc, mf1 = f1_by_shift[k]
        for scheme in SCHEMES:
            cov, macro, size, wd, single = [], [], [], [], []
            for cal, ev in splits:
                # Thresholds always come from the unshifted calibration fields.
                gc = districts[cal] if scheme != "marginal" else None
                t = fit_thresholds(score(probs_by_shift[0][cal]), y[cal], num_classes,
                                   args.alpha, scheme, groups_cal=gc, robust_q=args.tau)
                m = evaluate_sets(scores_k[ev], y[ev], probs_k[ev], t, num_classes)
                cov.append(m["coverage"]); macro.append(m["macro_coverage"])
                size.append(m["avg_set_size"]); single.append(m["singleton_rate"])
                wd.append(worst_district_coverage(scores_k[ev], y[ev], districts[ev],
                                                  t, args.min_district))
            rows.append({
                "season": season, "shift_months": k, "scheme": scheme,
                "n_test": len(y), "n_splits": len(splits),
                "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                "coverage": round(float(np.mean(cov)), 4),
                "coverage_std": round(float(np.std(cov)), 4),
                "macro_coverage": round(float(np.mean(macro)), 4),
                "worst_district": round(float(np.nanmin(wd)), 4),
                "avg_set_size": round(float(np.mean(size)), 3),
                "singleton_%": round(100 * float(np.mean(single)), 1),
                "violations_%": round(100 * float(np.mean(
                    np.array(cov) < (1 - args.alpha) - args.tol)), 1),
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.results_dir) / f"phenology_shift_{season}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n=== {season.upper()} | phenology-shift stress test "
          f"(target {1 - args.alpha:.2f}, tau={args.tau}, {len(y)} test fields, "
          f"{len(np.unique(districts))} districts, {len(splits)} splits) ===")
    for scheme in SCHEMES:
        sub = df[df.scheme == scheme]
        print(f"\n{scheme}")
        for _, r in sub.iterrows():
            print(f"  shift {int(r['shift_months']):+d} mo   macro-F1={r['macro_f1']:.3f}  "
                  f"cov={r['coverage']:.3f}+-{r['coverage_std']:.3f}  "
                  f"macro-cov={r['macro_coverage']:.3f}  "
                  f"worst-dist={r['worst_district']:.3f}  size={r['avg_set_size']:.2f}  "
                  f"viol={r['violations_%']:.0f}%")
    print(f"\n  -> {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Phenology-shift stress test for the "
                                             "conformal guarantee.")
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="both")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--score", choices=["aps", "lac"], default="aps")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--shifts", type=int, nargs="+", default=[-2, -1, 0, 1, 2],
                    help="Monthly offsets applied to the evaluation window.")
    ap.add_argument("--min-district", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=25,
                    help="District-grouped calibration/evaluation splits of the test set.")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    for s in seasons:
        if not (Path(args.results_dir) / s / "X_test.npy").exists():
            print(f"  [skip] no held-out test set for {s} in {args.results_dir}")
            continue
        run_season(s, args)


if __name__ == "__main__":
    main()
