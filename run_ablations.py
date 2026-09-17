"""
run_ablations.py — Ablations for PhenoSSM (ms-s4) under the spatial protocol.

  loss        : train PhenoSSM with each --loss, report held-out TEST metrics.
  datacentric : regenerate the uncertainty-guided cleaned set from the current
                production OOF, retrain on it, compare to raw (CE, same protocol).
  arch        : component isolation — S4D backbone, +attention, +multi-scale, and
                the full PhenoSSM (matched 2-layer S4D depth, CE).
  features    : raw spectral bands (20) vs bands + indices (28), CE.
  synthnoise  : CE vs SCE under controlled symmetric label noise (one season).

All modes reuse pipeline/train.py via subprocess. Held-out TEST metrics are read
from each run's fold_metrics.csv (TEST row). Spatial protocol + min-class-size 150
are fixed for consistency with the main benchmark.

Usage (GPU):
    python run_ablations.py --do both                 # loss + datacentric
    python run_ablations.py --do arch features synthnoise
    python run_ablations.py --do all                  # every ablation
"""

import sys
import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import pandas as pd

PY = sys.executable
SEASONS = ["rabi", "kharif"]


def _run(cmd):
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


def _test_metrics(results_dir, season):
    fm = Path(results_dir) / season / "fold_metrics.csv"
    if not fm.exists():
        return None
    df = pd.read_csv(fm)
    row = df[df["fold"] == "TEST"]
    if row.empty:
        return None
    d = row.iloc[0]
    return {"OA": float(d["OA"]), "Macro_F1": float(d["Macro_F1"]),
            "Weighted_F1": float(d["Weighted_F1"]), "Kappa": float(d["Kappa"])}


def _train(season, data_dir, model_dir, results_dir, loss, args,
           model="ms-s4", extra=None):
    cmd = [PY, "run.py", "train", "--model", model, "--season", season,
           "--loss", loss, "--epochs", str(args.epochs), "--folds", str(args.folds),
           "--test-size", "0.15", "--min-class-size", "150",
           "--data-dir", data_dir, "--model-dir", model_dir,
           "--results-dir", results_dir, "--seed", str(getattr(args, "seed", 42)),
           "--spatial-test", "--spatial-cv"]
    if extra:
        cmd += list(extra)
    return _run(cmd)


def _to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False, name=None):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def loss_ablation(args):
    out = Path("experiments_loss"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for loss in args.losses:
        for season in SEASONS:
            rd = f"experiments_loss/{loss}/results"
            if _train(season, args.data_dir, f"experiments_loss/{loss}/models", rd, loss, args) != 0:
                print(f"  [WARN] {loss}/{season} failed"); continue
            m = _test_metrics(rd, season)
            if m:
                rows.append({"loss": loss, "season": season,
                             **{k: round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_loss.csv", index=False)
    (out / "ablation_loss.md").write_text(_to_md(df), encoding="utf-8")
    print("\n=== LOSS ABLATION (PhenoSSM, spatial test) ===")
    print(df.to_string(index=False))


def datacentric_ablation(args):
    out = Path("experiments_dc"); out.mkdir(parents=True, exist_ok=True)
    # 1. regenerate cleaned data from the CURRENT production OOF in results/
    for season in SEASONS:
        _run([PY, "run.py", "clean-labels", "--season", season,
              "--results-dir", "results", "--data-dir", args.data_dir,
              "--out-dir", "data/processed_clean", "--action", "relabel"])
    rows = []
    for season in SEASONS:
        # baseline (raw, CE) — reuse the benchmark's ms-s4 if available, else train
        base = _test_metrics("experiments/ms-s4/results", season)
        if base is None:
            _train(season, args.data_dir, "experiments_dc/raw/models",
                   "experiments_dc/raw/results", "ce", args)
            base = _test_metrics("experiments_dc/raw/results", season)
        if base:
            rows.append({"data": "raw", "season": season,
                         **{k: round(v, 4) for k, v in base.items()}})
        # cleaned (CE)
        if _train(season, "data/processed_clean", "experiments_dc/clean/models",
                  "experiments_dc/clean/results", "ce", args) == 0:
            m = _test_metrics("experiments_dc/clean/results", season)
            if m:
                rows.append({"data": "cleaned", "season": season,
                             **{k: round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_datacentric.csv", index=False)
    (out / "ablation_datacentric.md").write_text(_to_md(df), encoding="utf-8")
    print("\n=== DATA-CENTRIC ABLATION (raw vs uncertainty-guided cleaned, CE) ===")
    print(df.to_string(index=False))


# ------------------------------------------------------------------ #
#  Architecture ablation — isolate each PhenoSSM component
# ------------------------------------------------------------------ #

# (model flag, human-readable label). Matched 2-layer S4D depth across all four,
# so only the multi-scale front-end and the attention head vary.
ARCH_VARIANTS = [
    ("ms-s4-bb",     "S4D backbone (mean-pool)"),
    ("ms-s4-noms",   "backbone + attention"),
    ("ms-s4-noattn", "backbone + multi-scale"),
    ("ms-s4",        "PhenoSSM (full)"),
]


def arch_ablation(args):
    out = Path("experiments_arch"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mdl, label in ARCH_VARIANTS:
        for season in SEASONS:
            rd = f"experiments_arch/{mdl}/results"
            if _train(season, args.data_dir, f"experiments_arch/{mdl}/models", rd,
                      "ce", args, model=mdl) != 0:
                print(f"  [WARN] arch {mdl}/{season} failed"); continue
            m = _test_metrics(rd, season)
            if m:
                rows.append({"variant": label, "model": mdl, "season": season,
                             **{k: round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_arch.csv", index=False)
    (out / "ablation_arch.md").write_text(_to_md(df), encoding="utf-8")
    print("\n=== ARCHITECTURE ABLATION (component isolation, CE, spatial test) ===")
    print(df.to_string(index=False))


# ------------------------------------------------------------------ #
#  Feature ablation — raw bands vs bands + indices
# ------------------------------------------------------------------ #

def feature_ablation(args):
    out = Path("experiments_feat"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fs in ["all", "raw"]:
        for season in SEASONS:
            if fs == "all":
                base = _test_metrics("experiments/ms-s4/results", season)
                if base:   # reuse the benchmark's 28-feature PhenoSSM if present
                    rows.append({"features": "bands+indices (28)", "season": season,
                                 **{k: round(v, 4) for k, v in base.items()}})
                    continue
            rd = f"experiments_feat/{fs}/results"
            extra = None if fs == "all" else ["--feature-set", "raw"]
            if _train(season, args.data_dir, f"experiments_feat/{fs}/models", rd,
                      "ce", args, extra=extra) != 0:
                print(f"  [WARN] feat {fs}/{season} failed"); continue
            m = _test_metrics(rd, season)
            if m:
                label = "bands+indices (28)" if fs == "all" else "bands only (20)"
                rows.append({"features": label, "season": season,
                             **{k: round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_features.csv", index=False)
    (out / "ablation_features.md").write_text(_to_md(df), encoding="utf-8")
    print("\n=== FEATURE ABLATION (raw bands vs bands+indices, CE, spatial test) ===")
    print(df.to_string(index=False))


# ------------------------------------------------------------------ #
#  Synthetic label-noise ablation — CE vs SCE under controlled noise
# ------------------------------------------------------------------ #

def synthnoise_ablation(args):
    out = Path("experiments_noise"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    season = args.noise_season
    for rate in args.noise_rates:
        for loss in ["ce", "sce"]:
            tag = f"{loss}_{int(round(rate * 100)):02d}"
            rd = f"experiments_noise/{tag}/results"
            if _train(season, args.data_dir, f"experiments_noise/{tag}/models", rd,
                      loss, args, extra=["--label-noise", str(rate)]) != 0:
                print(f"  [WARN] noise {tag} failed"); continue
            m = _test_metrics(rd, season)
            if m:
                rows.append({"loss": loss, "noise_rate": rate,
                             **{k: round(v, 4) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_noise.csv", index=False)
    (out / "ablation_noise.md").write_text(_to_md(df), encoding="utf-8")
    print(f"\n=== SYNTHETIC LABEL-NOISE ABLATION (CE vs SCE, {season}, spatial test) ===")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="PhenoSSM ablations.")
    ap.add_argument("--do", nargs="+",
                    choices=["loss", "datacentric", "arch", "features",
                             "synthnoise", "both", "all"],
                    default=["both"],
                    help="both = loss+datacentric (default); all = every ablation.")
    ap.add_argument("--data-dir", default="data/processed_monthly")
    ap.add_argument("--losses", nargs="+", default=["ce", "focal", "logit-adj", "gce", "sce"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-season", default="kharif", choices=["rabi", "kharif"])
    ap.add_argument("--noise-rates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    args = ap.parse_args()

    do = set(args.do)
    if "all" in do:
        do = {"loss", "datacentric", "arch", "features", "synthnoise"}
    if "both" in do:
        do |= {"loss", "datacentric"}

    if "loss" in do:
        loss_ablation(args)
    if "datacentric" in do:
        datacentric_ablation(args)
    if "arch" in do:
        arch_ablation(args)
    if "features" in do:
        feature_ablation(args)
    if "synthnoise" in do:
        synthnoise_ablation(args)
    print("\nAblations complete.")


if __name__ == "__main__":
    main()
