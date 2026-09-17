"""
conformal.py — Geographically-robust conformal prediction for crop mapping.

The novel contribution: distribution-free, *class-conditional* reliability
guarantees for long-tailed crop-type classification that are tested under
**spatial (geographic) distribution shift**.

It runs entirely on the out-of-fold artefacts saved by a `--dump-oof` training
run (`results/<season>/oof_probs.npy`, `oof_labels.npy`, `oof_meta.csv`) — no
model reload, no retraining.

Experiment (per season):
  Calibrate a conformal predictor and measure empirical coverage + set size two ways
    1. i.i.d. split      — random calib/eval split (exchangeable; the optimistic reference)
    2. spatial split     — calibrate on one set of DISTRICTS, evaluate on unseen DISTRICTS
  and with three calibration schemes
    - marginal           — one global threshold (standard split conformal)
    - mondrian           — per-class threshold (class-conditional; guarantees per-crop coverage)
    - region-robust      — district-grouped threshold (robust to the worst region)

The headline result: marginal conformal LOSES its target coverage under the
spatial split, while class-conditional / region-robust calibration restores it
(at the cost of larger sets) — especially for the rare crops that matter.

Score: APS (Adaptive Prediction Sets; Romano et al. 2020) by default, or LAC
(Sadinle et al. 2019) with --score lac.

Usage:
    python -m evaluation.conformal --season kharif --results-dir results --alpha 0.1
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_MIN_CLASS_CALIB = 10   # below this, a class falls back to the marginal threshold


# ------------------------------------------------------------------ #
#  Non-conformity scores
# ------------------------------------------------------------------ #

def aps_score_all(probs: np.ndarray) -> np.ndarray:
    """APS score for every (sample, class): cumulative prob mass of all classes
    at least as probable as c (inclusive). Smaller = more confidently included.
    Returns (N, C)."""
    order = np.argsort(-probs, axis=1)                 # classes, most-probable first
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)                 # mass of top-(k+1) classes
    ranks = np.argsort(order, axis=1)                  # each class's rank
    return np.take_along_axis(csum, ranks, axis=1)


def lac_score_all(probs: np.ndarray) -> np.ndarray:
    """LAC / softmax score: s(c) = 1 - p(c). Returns (N, C)."""
    return 1.0 - probs


def score_fn(name: str):
    return aps_score_all if name == "aps" else lac_score_all


# ------------------------------------------------------------------ #
#  Calibration
# ------------------------------------------------------------------ #

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of true-class scores."""
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_thresholds(score_all_cal, y_cal, num_classes, alpha, scheme,
                   groups_cal=None, robust_q=0.9):
    """Return a threshold vector (C,) for the chosen calibration scheme."""
    s_true = score_all_cal[np.arange(len(y_cal)), y_cal]
    q_marg = conformal_quantile(s_true, alpha)

    if scheme == "marginal":
        return np.full(num_classes, q_marg)

    if scheme == "mondrian":                          # per-class threshold
        q = np.full(num_classes, q_marg)
        for c in range(num_classes):
            m = y_cal == c
            if m.sum() >= _MIN_CLASS_CALIB:
                q[c] = conformal_quantile(s_true[m], alpha)
        return q

    if scheme == "region-robust":                     # district-grouped threshold
        if groups_cal is None:
            return np.full(num_classes, q_marg)
        per_group = []
        for g in np.unique(groups_cal):
            m = groups_cal == g
            if m.sum() >= _MIN_CLASS_CALIB:
                per_group.append(conformal_quantile(s_true[m], alpha))
        if not per_group:
            return np.full(num_classes, q_marg)
        q_robust = float(np.quantile(per_group, robust_q, method="higher"))
        return np.full(num_classes, max(q_robust, q_marg))

    if scheme == "class-region-robust":               # per-class, district-robust
        # Combines the Mondrian objective (per-crop coverage, what the long tail
        # needs) with the region-robust inter-district quantile (validity under
        # geographic shift). For each class the threshold is the tau-quantile of
        # its per-district conformal quantiles, with a fallback hierarchy for the
        # sparse (class, district) cells that the rarest crops produce.
        if groups_cal is None:                        # no districts -> Mondrian
            q = np.full(num_classes, q_marg)
            for c in range(num_classes):
                m = y_cal == c
                if m.sum() >= _MIN_CLASS_CALIB:
                    q[c] = conformal_quantile(s_true[m], alpha)
            return q
        q = np.full(num_classes, q_marg)
        for c in range(num_classes):
            mc = y_cal == c
            if mc.sum() < _MIN_CLASS_CALIB:
                q[c] = q_marg                          # class too sparse -> global marginal
                continue
            q_class = conformal_quantile(s_true[mc], alpha)   # pooled per-class fallback
            per_group = []
            for g in np.unique(groups_cal[mc]):
                mg = mc & (groups_cal == g)
                if mg.sum() >= _MIN_CLASS_CALIB:
                    per_group.append(conformal_quantile(s_true[mg], alpha))
            if len(per_group) >= 2:                    # enough districts to size a spread
                q_crr = float(np.quantile(per_group, robust_q, method="higher"))
                q[c] = max(q_crr, q_class)
            else:                                      # cannot estimate spread -> pooled per-class
                q[c] = q_class
        return q

    raise ValueError(scheme)


# ------------------------------------------------------------------ #
#  Evaluation of prediction sets
# ------------------------------------------------------------------ #

def evaluate_sets(score_all_eval, y_eval, probs_eval, thresh, num_classes):
    """Coverage, set size, and selective-prediction (triage) metrics."""
    sets = score_all_eval <= thresh[None, :]          # (N, C) membership
    N = len(y_eval)
    covered = sets[np.arange(N), y_eval]
    sizes = sets.sum(1)

    # per-class (macro) coverage — the rare-crop reliability that matters
    per_class_cov = []
    for c in range(num_classes):
        m = y_eval == c
        per_class_cov.append(float(covered[m].mean()) if m.any() else np.nan)
    macro_cov = float(np.nanmean(per_class_cov))

    singleton = sizes == 1
    pred = probs_eval.argmax(1)
    singleton_acc = float((pred[singleton] == y_eval[singleton]).mean()) if singleton.any() else np.nan

    return {
        "coverage": float(covered.mean()),
        "macro_coverage": macro_cov,
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_acc": singleton_acc,
        "empty_rate": float((sizes == 0).mean()),
        "per_class_coverage": per_class_cov,
    }


# ------------------------------------------------------------------ #
#  Main experiment
# ------------------------------------------------------------------ #

def run_season(season, args):
    res_dir = Path(args.results_dir) / season
    probs = np.load(res_dir / "oof_probs.npy")
    labels = np.load(res_dir / "oof_labels.npy")
    meta = pd.read_csv(res_dir / "oof_meta.csv")
    with open(res_dir / "config.json") as f:
        class_names = json.load(f)["class_names"]
    num_classes = probs.shape[1]
    districts = meta["District"].astype(str).to_numpy() if "District" in meta.columns else None

    score_all = score_fn(args.score)(probs)
    rng = args.seed
    rows, per_class_records = [], []

    def record(split, scheme, calib_idx, eval_idx, groups_cal=None):
        thr = fit_thresholds(score_all[calib_idx], labels[calib_idx], num_classes,
                             args.alpha, scheme, groups_cal=groups_cal,
                             robust_q=args.robust_quantile)
        m = evaluate_sets(score_all[eval_idx], labels[eval_idx], probs[eval_idx],
                          thr, num_classes)
        rows.append({"split": split, "scheme": scheme,
                     "coverage": round(m["coverage"], 4),
                     "macro_coverage": round(m["macro_coverage"], 4),
                     "avg_set_size": round(m["avg_set_size"], 3),
                     "singleton_rate": round(m["singleton_rate"], 4),
                     "singleton_acc": round(m["singleton_acc"], 4),
                     "empty_rate": round(m["empty_rate"], 4)})
        for c, cov in enumerate(m["per_class_coverage"]):
            per_class_records.append({"split": split, "scheme": scheme,
                                      "class": class_names[c],
                                      "coverage": None if cov is None or np.isnan(cov)
                                      else round(cov, 4)})

    # ---- 1. i.i.d. split (optimistic, exchangeable reference) ----
    tr, ev = train_test_split(np.arange(len(labels)), test_size=0.5,
                              random_state=rng, stratify=labels)
    record("iid", "marginal", tr, ev)
    record("iid", "mondrian", tr, ev)

    # ---- 2. spatial split: calibrate on some districts, evaluate on unseen ----
    if districts is not None and len(np.unique(districts)) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=rng)
        tr_s, ev_s = next(gss.split(probs, labels, groups=districts))
        record("spatial", "marginal", tr_s, ev_s)
        record("spatial", "mondrian", tr_s, ev_s)
        record("spatial", "region-robust", tr_s, ev_s, groups_cal=districts[tr_s])
    else:
        print(f"  [WARN] not enough districts for the spatial experiment ({season}).")

    df = pd.DataFrame(rows)
    pc_df = pd.DataFrame(per_class_records)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "conformal_summary.csv", index=False)
    pc_df.to_csv(res_dir / "conformal_per_class.csv", index=False)

    target = 1 - args.alpha
    print(f"\n=== Conformal [{season} | score={args.score} | target={target:.0%}] ===")
    print(df.to_string(index=False))

    _plot(df, pc_df, target, season, res_dir / "conformal_coverage.png", class_names)
    print(f"  Saved -> {res_dir}/conformal_summary.csv, conformal_per_class.csv, conformal_coverage.png")
    return df


def repeated_study(season, args):
    """Coverage *distribution* over many random geographic (district) splits.

    The robust version of the spatial experiment: instead of one calib/eval
    district split, repeat it N times and report coverage mean/std/min and the
    target-violation rate per scheme. Shows whether each calibration scheme keeps
    coverage reliable under geographic shift.
    """
    res_dir = Path(args.results_dir) / season
    probs = np.load(res_dir / "oof_probs.npy")
    labels = np.load(res_dir / "oof_labels.npy")
    meta = pd.read_csv(res_dir / "oof_meta.csv")
    num_classes = probs.shape[1]
    districts = meta["District"].astype(str).to_numpy()
    score_all = score_fn(args.score)(probs)
    target = 1 - args.alpha
    tol = 0.02   # count a violation if coverage < target - tol

    keys = [("iid", "marginal"), ("spatial", "marginal"),
            ("spatial", "mondrian"), ("spatial", "region-robust"),
            ("spatial", "class-region-robust")]
    acc = {k: {"cov": [], "macro": [], "size": []} for k in keys}

    for s in range(args.seed, args.seed + args.repeats):
        # i.i.d. reference
        tr, ev = train_test_split(np.arange(len(labels)), test_size=0.5,
                                  random_state=s, stratify=labels)
        thr = fit_thresholds(score_all[tr], labels[tr], num_classes, args.alpha, "marginal")
        m = evaluate_sets(score_all[ev], labels[ev], probs[ev], thr, num_classes)
        acc[("iid", "marginal")]["cov"].append(m["coverage"])
        acc[("iid", "marginal")]["macro"].append(m["macro_coverage"])
        acc[("iid", "marginal")]["size"].append(m["avg_set_size"])
        # geographic split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=s)
        trs, evs = next(gss.split(probs, labels, groups=districts))
        for scheme in ("marginal", "mondrian", "region-robust", "class-region-robust"):
            gc = districts[trs] if scheme in ("region-robust", "class-region-robust") else None
            thr = fit_thresholds(score_all[trs], labels[trs], num_classes,
                                 args.alpha, scheme, groups_cal=gc,
                                 robust_q=args.robust_quantile)
            m = evaluate_sets(score_all[evs], labels[evs], probs[evs], thr, num_classes)
            acc[("spatial", scheme)]["cov"].append(m["coverage"])
            acc[("spatial", scheme)]["macro"].append(m["macro_coverage"])
            acc[("spatial", scheme)]["size"].append(m["avg_set_size"])

    rows = []
    for (split, scheme), d in acc.items():
        cov = np.array(d["cov"])
        rows.append({"split": split, "scheme": scheme,
                     "cov_mean": round(float(cov.mean()), 4),
                     "cov_std": round(float(cov.std()), 4),
                     "cov_min": round(float(cov.min()), 4),
                     "viol_rate_%": round(100 * float(np.mean(cov < target - tol)), 1),
                     "macro_mean": round(float(np.mean(d["macro"])), 4),
                     "size_mean": round(float(np.mean(d["size"])), 3)})
    df = pd.DataFrame(rows)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "conformal_robust.csv", index=False)
    print(f"\n=== Conformal robustness over {args.repeats} geographic splits "
          f"[{season} | score={args.score} | target={target:.0%}] ===")
    print(df.to_string(index=False))
    return df


def _plot(df, pc_df, target, season, save_path, class_names):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # (a) marginal vs macro coverage by split/scheme
    labels_x = [f"{r.split}\n{r.scheme}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax[0].bar(x - 0.18, df["coverage"], 0.36, label="marginal coverage", color="#4C72B0")
    ax[0].bar(x + 0.18, df["macro_coverage"], 0.36, label="macro (per-class) coverage", color="#DD8452")
    ax[0].axhline(target, ls="--", c="k", lw=1, label=f"target {target:.0%}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels_x, fontsize=8)
    ax[0].set_ylabel("coverage"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title(f"{season}: coverage by split / scheme"); ax[0].legend(fontsize=8)

    # (b) per-class coverage: spatial marginal vs mondrian (the rare-crop story)
    sub_m = pc_df[(pc_df.split == "spatial") & (pc_df.scheme == "marginal")]
    sub_c = pc_df[(pc_df.split == "spatial") & (pc_df.scheme == "mondrian")]
    if not sub_m.empty:
        xs = np.arange(len(class_names))
        cm = [sub_m[sub_m["class"] == c]["coverage"].values[0] if not sub_m[sub_m["class"] == c].empty else np.nan for c in class_names]
        cc = [sub_c[sub_c["class"] == c]["coverage"].values[0] if not sub_c[sub_c["class"] == c].empty else np.nan for c in class_names]
        cm = [np.nan if v is None else v for v in cm]
        cc = [np.nan if v is None else v for v in cc]
        ax[1].bar(xs - 0.18, cm, 0.36, label="spatial marginal", color="#C44E52")
        ax[1].bar(xs + 0.18, cc, 0.36, label="spatial mondrian", color="#55A868")
        ax[1].axhline(target, ls="--", c="k", lw=1)
        ax[1].set_xticks(xs); ax[1].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax[1].set_ylabel("per-class coverage"); ax[1].set_ylim(0, 1.02)
        ax[1].set_title(f"{season}: per-class coverage under spatial shift"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ------------------------------------------------------------------ #
#  Nested-tau study and cross-backbone reliability efficiency
# ------------------------------------------------------------------ #

QGRID_DEFAULT = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def expected_calibration_error(probs, labels, n_bins=15):
    """Standard top-label ECE, computed directly from probabilities."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / n)
    return float(ece)


def _load_class_names(res_dir, num_classes):
    cfg = Path(res_dir) / "config.json"
    if cfg.exists():
        try:
            with open(cfg) as f:
                names = json.load(f).get("class_names")
            if names and len(names) == num_classes:
                return names
        except Exception:
            pass
    return [f"class_{i}" for i in range(num_classes)]


def _geo_splits(probs, labels, districts, seeds, test_size=0.5):
    out = []
    for s in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        out.append(next(gss.split(probs, labels, groups=districts)))
    return out


def _eval_over_splits(score_all, labels, probs, districts, splits, num_classes,
                      scheme, alpha, tau):
    """Run one scheme over a list of (calib, eval) district splits at quantile tau.
    Returns per-split arrays plus a (n_splits, C) per-class coverage matrix."""
    covs, macros, sizes, singles, empties = [], [], [], [], []
    pc = np.full((len(splits), num_classes), np.nan)
    for i, (tr, ev) in enumerate(splits):
        gc = districts[tr] if scheme in ("region-robust", "class-region-robust") else None
        thr = fit_thresholds(score_all[tr], labels[tr], num_classes, alpha, scheme,
                             groups_cal=gc, robust_q=tau)
        m = evaluate_sets(score_all[ev], labels[ev], probs[ev], thr, num_classes)
        covs.append(m["coverage"]); macros.append(m["macro_coverage"])
        sizes.append(m["avg_set_size"]); singles.append(m["singleton_rate"])
        empties.append(m["empty_rate"])
        pc[i] = [np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                 for v in m["per_class_coverage"]]
    return (np.array(covs), np.array(macros), np.array(sizes),
            np.array(singles), np.array(empties), pc)


def _select_tau(score_all, labels, probs, districts, tune_splits, num_classes,
                scheme, alpha, objective, grid=QGRID_DEFAULT, tol=0.02):
    """Smallest tau on the grid for which no tuning split violates the target on
    the scheme's own objective (overall coverage, or macro/per-class coverage)."""
    target = 1 - alpha
    best = grid[-1]
    for tau in grid:
        covs, macros, *_ = _eval_over_splits(score_all, labels, probs, districts,
                                             tune_splits, num_classes, scheme, alpha, tau)
        if objective == "overall":
            ok = np.mean(covs < target - tol) == 0
        elif objective == "macro":
            ok = np.mean(macros < target - tol) == 0
        else:   # "both": no tuning split violates overall OR per-class coverage
            ok = np.mean((covs < target - tol) | (macros < target - tol)) == 0
        if ok:
            best = tau
            break
    return best


def nested_robust_study(season, args):
    """Tuning/test split of geographic repeats: tau is selected on a disjoint
    tuning pool (region-robust on overall coverage, class-region-robust on
    per-class coverage), then everything is reported on the held-out test pool,
    including per-class and worst-crop coverage."""
    res_dir = Path(args.results_dir) / season
    probs = np.load(res_dir / "oof_probs.npy")
    labels = np.load(res_dir / "oof_labels.npy")
    meta = pd.read_csv(res_dir / "oof_meta.csv")
    districts = meta["District"].astype(str).to_numpy()
    num_classes = probs.shape[1]
    class_names = _load_class_names(res_dir, num_classes)
    score_all = score_fn(args.score)(probs)
    target = 1 - args.alpha
    tol = 0.02
    n = args.repeats

    tune = _geo_splits(probs, labels, districts, range(args.seed, args.seed + n))
    test = _geo_splits(probs, labels, districts, range(args.seed + n, args.seed + 2 * n))

    if getattr(args, "fixed_tau", False):
        chosen = {"region-robust": args.robust_quantile,
                  "class-region-robust": args.robust_quantile}
    else:
        chosen = {
            "region-robust": _select_tau(score_all, labels, probs, districts, tune,
                                         num_classes, "region-robust", args.alpha, "overall"),
            "class-region-robust": _select_tau(score_all, labels, probs, districts, tune,
                                               num_classes, "class-region-robust", args.alpha, "both"),
        }

    rows, perclass_rows = [], []
    schemes = [("marginal", 0.9), ("mondrian", 0.9),
               ("region-robust", chosen["region-robust"]),
               ("class-region-robust", chosen["class-region-robust"])]
    for scheme, tau in schemes:
        covs, macros, sizes, singles, empties, pc = _eval_over_splits(
            score_all, labels, probs, districts, test, num_classes, scheme, args.alpha, tau)
        per_class_mean = np.nanmean(pc, axis=0)
        rows.append({
            "season": season, "scheme": scheme,
            "tau": "" if scheme in ("marginal", "mondrian") else tau,
            "cov_mean": round(float(covs.mean()), 4),
            "cov_std": round(float(covs.std()), 4),
            "cov_min": round(float(covs.min()), 4),
            "viol_%": round(100 * float(np.mean(covs < target - tol)), 1),
            "macro_mean": round(float(np.nanmean(macros)), 4),
            "macro_min": round(float(np.nanmin(macros)), 4),
            "worst_class_cov": round(float(np.nanmin(per_class_mean)), 4),
            "size_mean": round(float(sizes.mean()), 3),
            "singleton_%": round(100 * float(singles.mean()), 1),
            "empty_%": round(100 * float(empties.mean()), 2),
        })
        for c in range(num_classes):
            perclass_rows.append({"season": season, "scheme": scheme,
                                  "class": class_names[c],
                                  "coverage": round(float(per_class_mean[c]), 4)})

    df = pd.DataFrame(rows)
    pc_df = pd.DataFrame(perclass_rows)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "conformal_nested.csv", index=False)
    pc_df.to_csv(res_dir / "conformal_nested_perclass.csv", index=False)
    print(f"\n=== Nested conformal [{season}] target={target:.0%} | "
          f"tau(region-robust)={chosen['region-robust']} "
          f"tau(class-region-robust)={chosen['class-region-robust']} | "
          f"{n} tuning + {n} test splits ===")
    print(df.to_string(index=False))
    print("\nper-class coverage (test pool):")
    print(pc_df.pivot(index="class", columns="scheme", values="coverage")
          .reindex(columns=[s for s, _ in schemes]).to_string())
    return df, pc_df


# Backbone -> results directory. All eight use the cross-entropy benchmark dumps
# under experiments/ so the architecture comparison holds the loss constant.
EFFICIENCY_BACKBONES = [
    ("PhenoSSM", "experiments/ms-s4/results"),
    ("S4D", "experiments/s4d/results"),
    ("Transformer", "experiments/transformer/results"),
    ("MSTACNN", "experiments/mstacnn/results"),
    ("L-TAE", "experiments/ltae/results"),
    ("LSTM", "experiments/lstm/results"),
    ("TempCNN", "experiments/tempcnn/results"),
    ("Random Forest", "experiments/rf/results"),
]


def efficiency_table(season, args):
    """Cross-backbone reliability efficiency: from each backbone's OOF dump,
    report ECE and, under a common tau, the marginal-vs-robust violation rate and
    the set efficiency (avg size, singleton rate, macro coverage) at guaranteed
    coverage. Shows which backbone yields the tightest reliable sets."""
    target = 1 - args.alpha
    tol = 0.02
    n = args.repeats
    rows = []
    for name, rel in EFFICIENCY_BACKBONES:
        rd = _ROOT / rel / season
        if not (rd / "oof_probs.npy").exists():
            print(f"  [skip] {name}: no OOF at {rd}")
            continue
        probs = np.load(rd / "oof_probs.npy")
        labels = np.load(rd / "oof_labels.npy")
        meta = pd.read_csv(rd / "oof_meta.csv")
        districts = meta["District"].astype(str).to_numpy()
        num_classes = probs.shape[1]
        score_all = score_fn(args.score)(probs)
        test = _geo_splits(probs, labels, districts,
                           range(args.seed + n, args.seed + 2 * n))
        out = {"backbone": name, "ECE": round(expected_calibration_error(probs, labels), 4)}
        for scheme, tag in [("marginal", "marg"),
                            ("region-robust", "rr"),
                            ("class-region-robust", "crr")]:
            covs, macros, sizes, singles, _, _ = _eval_over_splits(
                score_all, labels, probs, districts, test, num_classes,
                scheme, args.alpha, args.robust_quantile)
            out[f"viol_{tag}_%"] = round(100 * float(np.mean(covs < target - tol)), 1)
            out[f"size_{tag}"] = round(float(sizes.mean()), 2)
            out[f"single_{tag}_%"] = round(100 * float(singles.mean()), 1)
            out[f"macro_{tag}"] = round(float(np.nanmean(macros)), 3)
        rows.append(out)
    df = pd.DataFrame(rows)
    out_path = _ROOT / args.results_dir / f"efficiency_{season}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n=== Reliability efficiency [{season}] target={target:.0%} "
          f"tau={args.robust_quantile} (test pool, {n} splits) ===")
    print(df.to_string(index=False))
    print(f"  Saved -> {out_path}")
    return df


def perclass_size_table(season, args):
    """Per-true-class coverage, average set size, and singleton rate on the
    held-out test pool, under marginal / region-robust / class-region-robust at a
    common quantile. This is the honest efficiency view: it shows when a scheme
    meets per-crop coverage only by emitting near-vacuous sets for the rare crops
    it targets (large size, near-zero singletons)."""
    rd = Path(args.results_dir) / season
    probs = np.load(rd / "oof_probs.npy")
    labels = np.load(rd / "oof_labels.npy")
    meta = pd.read_csv(rd / "oof_meta.csv")
    districts = meta["District"].astype(str).to_numpy()
    nc = probs.shape[1]
    names = _load_class_names(rd, nc)
    counts = np.bincount(labels, minlength=nc)
    score_all = score_fn(args.score)(probs)
    n = args.repeats
    test = _geo_splits(probs, labels, districts, range(args.seed + n, args.seed + 2 * n))

    rows = []
    for scheme in ["marginal", "region-robust", "class-region-robust"]:
        gc_needed = scheme in ("region-robust", "class-region-robust")
        sz = {c: [] for c in range(nc)}
        sg = {c: [] for c in range(nc)}
        cv = {c: [] for c in range(nc)}
        for tr, ev in test:
            thr = fit_thresholds(score_all[tr], labels[tr], nc, args.alpha, scheme,
                                 groups_cal=districts[tr] if gc_needed else None,
                                 robust_q=args.robust_quantile)
            sets = score_all[ev] <= thr[None, :]
            sizes = sets.sum(1)
            yev = labels[ev]
            cov = sets[np.arange(len(yev)), yev]
            for c in range(nc):
                m = yev == c
                if m.any():
                    sz[c].append(sizes[m].mean())
                    sg[c].append((sizes[m] == 1).mean())
                    cv[c].append(cov[m].mean())
        for c in np.argsort(counts):   # rare crops first
            rows.append({"season": season, "scheme": scheme, "crop": names[c],
                         "n": int(counts[c]),
                         "coverage": round(float(np.mean(cv[c])), 3),
                         "avg_size": round(float(np.mean(sz[c])), 2),
                         "singleton_%": round(100 * float(np.mean(sg[c])), 1)})
    df = pd.DataFrame(rows)
    out = Path(args.results_dir) / f"perclass_size_{season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n=== Per-class set size [{season}] tau={args.robust_quantile} "
          f"(test pool, {n} splits) ===")
    print(df.to_string(index=False))
    print(f"  Saved -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Geographically-robust conformal crop mapping.")
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="both")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--score", choices=["aps", "lac"], default="aps")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (target=1-alpha).")
    ap.add_argument("--robust-quantile", type=float, default=0.9,
                    help="District-quantile for region-robust calibration.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="If >1, run the repeated-geographic-split robustness study. "
                         "For --nested / --efficiency this is the per-pool split count.")
    ap.add_argument("--nested", action="store_true",
                    help="Nested-tau study: select tau on a disjoint tuning pool, report "
                         "on a held-out test pool, with per-class and worst-crop coverage.")
    ap.add_argument("--efficiency", action="store_true",
                    help="Cross-backbone reliability-efficiency table over experiments/*.")
    ap.add_argument("--perclass", action="store_true",
                    help="Per-true-class coverage, set size, and singleton rate "
                         "(shows near-vacuous sets for the rare crops).")
    ap.add_argument("--fixed-tau", action="store_true",
                    help="In --nested, skip tau selection and use --robust-quantile "
                         "for both robust schemes (matched-tau comparison).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    for s in seasons:
        if args.efficiency:
            efficiency_table(s, args)
            continue
        if not (Path(args.results_dir) / s / "oof_probs.npy").exists():
            print(f"  [skip] no OOF for {s} in {args.results_dir} "
                  f"(train with --dump-oof).")
            continue
        if args.perclass:
            perclass_size_table(s, args)
        elif args.nested:
            nested_robust_study(s, args)
        elif args.repeats > 1:
            repeated_study(s, args)
        else:
            run_season(s, args)


if __name__ == "__main__":
    main()
