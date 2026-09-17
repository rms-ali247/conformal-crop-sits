"""
province_transfer.py — leave-one-province-out conformal transfer.

The district splits used in `conformal.py` hold out a random half of the
districts. This script asks a harder question: what happens when the whole
deployment *region* is new? Each of the four provinces (Punjab, Sindh,
Balochistan, Khyber Pakhtunkhwa) is held out in turn, the conformal predictor
is calibrated on every district outside it, and coverage is measured inside it.
Provinces differ in agro-ecology, irrigation regime, and crop mix, so this is a
strictly larger shift than a random district split.

Runs on the out-of-fold artefacts only (`results/<season>/oof_probs.npy`,
`oof_labels.npy`, `oof_meta.csv`) — no model reload, no retraining.

Usage:
    python -m evaluation.province_transfer --season both --alpha 0.1 --tau 0.9
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd

from evaluation.conformal import (score_fn, fit_thresholds, evaluate_sets,
                                  _load_class_names, _MIN_CLASS_CALIB)

SCHEMES = ["marginal", "mondrian", "region-robust", "class-region-robust"]


def _worst_district(score_all, labels, districts, thresh, num_classes, min_n):
    """Coverage of the worst-covered district inside the held-out province."""
    covs = []
    sets = score_all <= thresh[None, :]
    covered = sets[np.arange(len(labels)), labels]
    for d in np.unique(districts):
        m = districts == d
        if m.sum() >= min_n:
            covs.append(float(covered[m].mean()))
    return (float(np.min(covs)), len(covs)) if covs else (np.nan, 0)


def run_season(season, args):
    res_dir = Path(args.results_dir) / season
    probs = np.load(res_dir / "oof_probs.npy")
    labels = np.load(res_dir / "oof_labels.npy")
    meta = pd.read_csv(res_dir / "oof_meta.csv")
    districts = meta["District"].astype(str).to_numpy()
    provinces = meta["Province"].astype(str).to_numpy()
    num_classes = probs.shape[1]
    class_names = _load_class_names(res_dir, num_classes)
    score_all = score_fn(args.score)(probs)
    target = 1 - args.alpha

    rows = []
    for prov in sorted(np.unique(provinces)):
        ev = np.flatnonzero(provinces == prov)
        cal = np.flatnonzero(provinces != prov)
        if len(ev) < args.min_eval:
            print(f"  [skip] {prov}: only {len(ev)} fields")
            continue
        for scheme in SCHEMES:
            gc = districts[cal] if scheme in ("region-robust", "class-region-robust") else None
            thr = fit_thresholds(score_all[cal], labels[cal], num_classes, args.alpha,
                                 scheme, groups_cal=gc, robust_q=args.tau)
            m = evaluate_sets(score_all[ev], labels[ev], probs[ev], thr, num_classes)
            wd, n_d = _worst_district(score_all[ev], labels[ev], districts[ev],
                                      thr, num_classes, args.min_district)
            # Worst-crop and macro coverage are restricted to crops with enough
            # fields in the held-out province; a province can hold a single field
            # of a crop, and its 0/1 coverage is noise, not a finding.
            pc = np.array(m["per_class_coverage"], dtype=float)
            n_per_class = np.array([(labels[ev] == c).sum() for c in range(num_classes)])
            keep = (~np.isnan(pc)) & (n_per_class >= args.min_crop)
            if keep.any():
                worst_i = int(np.flatnonzero(keep)[np.argmin(pc[keep])])
                worst, worst_name = float(pc[worst_i]), class_names[worst_i]
                macro = float(np.mean(pc[keep]))
            else:
                worst, worst_name, macro = np.nan, "-", np.nan
            rows.append({
                "season": season, "province": prov, "scheme": scheme,
                "n_eval": len(ev), "n_cal": len(cal),
                "n_districts_eval": int(len(np.unique(districts[ev]))),
                "n_crops_scored": int(keep.sum()), "n_crops_total": num_classes,
                "coverage": round(m["coverage"], 4),
                "macro_coverage": round(macro, 4),
                "worst_crop": round(worst, 4),
                "worst_crop_name": worst_name,
                "worst_crop_n": int(n_per_class[worst_i]) if keep.any() else 0,
                "worst_district": round(wd, 4) if np.isfinite(wd) else np.nan,
                "n_districts_scored": n_d,
                "avg_set_size": round(m["avg_set_size"], 3),
                "singleton_%": round(100 * m["singleton_rate"], 1),
                "violates": int(m["coverage"] < target - args.tol),
            })

    df = pd.DataFrame(rows)
    out = Path(args.results_dir) / f"province_transfer_{season}.csv"
    df.to_csv(out, index=False)

    print(f"\n=== {season.upper()} | leave-one-province-out "
          f"(target {target:.2f}, tau={args.tau}, {num_classes} crops) ===")
    for scheme in SCHEMES:
        sub = df[df.scheme == scheme]
        if sub.empty:
            continue
        print(f"\n{scheme}")
        for _, r in sub.iterrows():
            flag = "  <-- violates" if r["violates"] else ""
            print(f"  {r['province']:<20} n={r['n_eval']:>6}  cov={r['coverage']:.3f}  "
                  f"macro={r['macro_coverage']:.3f}  worst-crop={r['worst_crop']:.3f} "
                  f"({r['worst_crop_name']}, n={r['worst_crop_n']})  "
                  f"worst-dist={r['worst_district']:.3f}  "
                  f"size={r['avg_set_size']:.2f}{flag}")
        print(f"  {'MEAN':<20} {'':>8}  cov={sub['coverage'].mean():.3f}  "
              f"macro={sub['macro_coverage'].mean():.3f}  "
              f"worst-crop={sub['worst_crop'].min():.3f}"
              f"{'':>14}  size={sub['avg_set_size'].mean():.2f}  "
              f"[{int(sub['violates'].sum())}/{len(sub)} provinces violate]")
    print(f"\n  -> {out}")

    # ---- How large a safety margin does province-scale shift need? ----
    # tau was tuned on district splits; this sweep asks what it would have to be
    # for every held-out province to reach the target, and what that costs.
    sweep = []
    for tau in args.tau_grid:
        covs, sizes = [], []
        for prov in sorted(np.unique(provinces)):
            ev = np.flatnonzero(provinces == prov)
            cal = np.flatnonzero(provinces != prov)
            if len(ev) < args.min_eval:
                continue
            thr = fit_thresholds(score_all[cal], labels[cal], num_classes, args.alpha,
                                 "region-robust", groups_cal=districts[cal], robust_q=tau)
            m = evaluate_sets(score_all[ev], labels[ev], probs[ev], thr, num_classes)
            covs.append(m["coverage"]); sizes.append(m["avg_set_size"])
        covs, sizes = np.array(covs), np.array(sizes)
        sweep.append({"season": season, "tau": tau,
                      "mean_coverage": round(float(covs.mean()), 4),
                      "worst_province": round(float(covs.min()), 4),
                      "avg_set_size": round(float(sizes.mean()), 3),
                      "violations": int((covs < target - args.tol).sum()),
                      "n_provinces": len(covs)})
    sw = pd.DataFrame(sweep)
    out_sw = Path(args.results_dir) / f"province_tau_{season}.csv"
    sw.to_csv(out_sw, index=False)
    print(f"\n  region-robust tau sweep at province scale ({season}):")
    for _, r in sw.iterrows():
        print(f"    tau={r['tau']:.2f}  mean={r['mean_coverage']:.3f}  "
              f"worst-province={r['worst_province']:.3f}  size={r['avg_set_size']:.2f}  "
              f"[{int(r['violations'])}/{int(r['n_provinces'])} violate]")
    print(f"  -> {out_sw}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Leave-one-province-out conformal transfer.")
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="both")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--score", choices=["aps", "lac"], default="aps")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.9,
                    help="Inter-district quantile for the region-robust schemes.")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="Slack below the target before a province counts as a violation.")
    ap.add_argument("--min-eval", type=int, default=200,
                    help="Minimum fields for a province to be evaluated.")
    ap.add_argument("--min-district", type=int, default=_MIN_CLASS_CALIB,
                    help="Minimum fields for a district to enter the worst-district figure.")
    ap.add_argument("--min-crop", type=int, default=30,
                    help="Minimum fields in the held-out province for a crop to enter "
                         "the macro and worst-crop figures.")
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.9, 0.95, 0.99, 1.0],
                    help="Inter-district quantiles for the province-scale safety-margin sweep.")
    args = ap.parse_args()

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    for s in seasons:
        if not (Path(args.results_dir) / s / "oof_probs.npy").exists():
            print(f"  [skip] no OOF for {s} in {args.results_dir}")
            continue
        run_season(s, args)


if __name__ == "__main__":
    main()
