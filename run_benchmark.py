"""
run_benchmark.py — Train + trustworthy-evaluate every model on one protocol,
then emit a master results table and paired McNemar tests.

For each model it shells out to ``run.py train`` (into experiments/<model>/) and,
for the neural models, ``evaluation.evaluate_model`` (calibration + selective
prediction). Random Forest is reported on point metrics only (it is not a
torch ensemble). McNemar significance is computed for each neural model vs a
reference model on the shared evaluation split.

GPU example (the real run):
    python run_benchmark.py --season kharif --data-dir data/processed_monthly \
        --spatial-test --spatial-cv --loss logit-adj --epochs 150 --folds 5 \
        --reference ms-s4

Quick CPU smoke:
    python run_benchmark.py --season kharif --data-dir data/processed_smoke \
        --models tempcnn ms-s4 --epochs 1 --folds 2 --out experiments_smoke
"""

import sys
import json
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd

from evaluation.metrics import mcnemar_test

ALL_MODELS = ["rf", "tempcnn", "lstm", "ltae", "transformer", "mstacnn", "s4d", "ms-s4"]
PY = sys.executable   # the venv interpreter


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


def _to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free GitHub-flavoured markdown table."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body]) + "\n"


def _train(model, season, args, model_dir, results_dir) -> int:
    cmd = [PY, "run.py", "train", "--model", model, "--season", season,
           "--loss", args.loss, "--epochs", str(args.epochs),
           "--folds", str(args.folds), "--test-size", str(args.test_size),
           "--min-class-size", str(args.min_class_size),
           "--data-dir", args.data_dir, "--model-dir", model_dir,
           "--results-dir", results_dir, "--seed", str(args.seed),
           "--dump-oof", "--mc-samples", str(args.mc_samples)]
    if args.spatial_test:
        cmd.append("--spatial-test")
    if args.spatial_cv:
        cmd.append("--spatial-cv")
    return _run(cmd)


def _conformal_metrics(results_dir, season, args) -> dict:
    """Spatial-shift conformal coverage for one model (reuses its OOF dump)."""
    from types import SimpleNamespace
    from evaluation.conformal import run_season
    ns = SimpleNamespace(results_dir=results_dir, score="aps", alpha=args.alpha,
                         robust_quantile=0.9, seed=args.seed)
    try:
        cdf = run_season(season, ns)
    except Exception as e:
        print(f"  [WARN] conformal failed for {results_dir}: {e}")
        return {}
    out = {}
    for _, r in cdf.iterrows():
        if r["split"] == "spatial" and r["scheme"] == "marginal":
            out["Cov_sp_marginal"] = r["coverage"]
        if r["split"] == "spatial" and r["scheme"] == "mondrian":
            out["MacroCov_sp_mondrian"] = r["macro_coverage"]
        if r["split"] == "spatial" and r["scheme"] == "region-robust":
            out["Cov_sp_robust"] = r["coverage"]
            out["SetSize_sp_robust"] = r["avg_set_size"]
    return out


def _evaluate(model, season, model_dir, results_dir, seed) -> int:
    cmd = [PY, "-m", "evaluation.evaluate_model", "--season", season,
           "--model-dir", model_dir, "--results-dir", results_dir, "--seed", str(seed)]
    return _run(cmd)


def _rf_point_metrics(results_dir, season):
    """RF is not torch-loadable; read its held-out TEST row from fold_metrics.csv."""
    fm = Path(results_dir) / season / "fold_metrics.csv"
    if not fm.exists():
        return None
    df = pd.read_csv(fm)
    row = df[df["fold"] == "TEST"]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"model": "rf", "OA": r.get("OA"), "Macro_F1": r.get("Macro_F1"),
            "Weighted_F1": r.get("Weighted_F1"), "Kappa": r.get("Kappa"),
            "ECE_cal": np.nan, "AURC": np.nan, "Acc@80cov": np.nan}


def benchmark_season(season, args) -> pd.DataFrame:
    out_root = Path(args.out)
    rows, eval_cache = [], {}

    for model in args.models:
        model_dir = str(out_root / model / "models")
        results_dir = str(out_root / model / "results")

        if _train(model, season, args, model_dir, results_dir) != 0:
            print(f"  [WARN] training failed for {model}; skipping.")
            continue

        if model == "rf":
            row = _rf_point_metrics(results_dir, season)
        else:
            if _evaluate(model, season, model_dir, results_dir, args.seed) != 0:
                print(f"  [WARN] evaluation failed for {model}; skipping.")
                continue
            ev_dir = Path(results_dir) / season
            with open(ev_dir / "eval_metrics.json") as f:
                em = json.load(f)
            row = {"model": model, "OA": em["OA"], "Macro_F1": em["Macro_F1"],
                   "Weighted_F1": em["Weighted_F1"], "Kappa": em["Kappa"],
                   "ECE_cal": em["ECE_cal"], "AURC": em["AURC"],
                   "Acc@80cov": em["Acc@80cov"]}
            eval_cache[model] = (np.load(ev_dir / "eval_preds.npy"),
                                 np.load(ev_dir / "eval_labels.npy"))

        if row is None:
            continue
        row.update(_conformal_metrics(results_dir, season, args))   # spatial coverage per model
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  No results for {season}.")
        return df

    # ---- McNemar vs reference (shared eval split) ----
    ref = args.reference
    df["McNemar_p_vs_" + ref] = np.nan
    if ref in eval_cache:
        ref_pred, ref_lab = eval_cache[ref]
        ref_correct = (ref_pred == ref_lab)
        for model, (pred, lab) in eval_cache.items():
            if model == ref:
                continue
            if len(lab) != len(ref_lab) or not np.array_equal(lab, ref_lab):
                print(f"  [WARN] eval split mismatch for {model} vs {ref}; "
                      f"skipping McNemar.")
                continue
            _, _, _, p = mcnemar_test((pred == lab), ref_correct)
            df.loc[df["model"] == model, "McNemar_p_vs_" + ref] = round(p, 5)

    df = df.sort_values("Macro_F1", ascending=False).reset_index(drop=True)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / f"benchmark_{season}.csv", index=False)
    with open(out_root / f"benchmark_{season}.md", "w", encoding="utf-8") as f:
        f.write(f"# Benchmark - {season} "
                f"(loss={args.loss}, spatial_test={args.spatial_test}, "
                f"spatial_cv={args.spatial_cv})\n\n")
        f.write(_to_markdown(df))
    print(f"\n=== Benchmark [{season}] ===")
    print(df.to_string(index=False))
    print(f"\nSaved -> {out_root}/benchmark_{season}.csv / .md")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + trustworthy-evaluate all models.")
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="both")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    help=f"Subset of {ALL_MODELS}.")
    ap.add_argument("--data-dir", default="data/processed_monthly")
    ap.add_argument("--loss", default="ce",
                    help="Same loss for all neural models (fair comparison).")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--min-class-size", type=int, default=5,
                    help="Drop classes with fewer samples than this (e.g. 150 for Rabi).")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Conformal miscoverage rate (target coverage = 1 - alpha).")
    ap.add_argument("--mc-samples", type=int, default=10,
                    help="MC-dropout passes for the OOF uncertainty dump.")
    ap.add_argument("--spatial-test", action="store_true")
    ap.add_argument("--spatial-cv", action="store_true")
    ap.add_argument("--reference", default="ms-s4", help="Model for McNemar comparison.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    for season in seasons:
        benchmark_season(season, args)


if __name__ == "__main__":
    main()
