"""Generate figs/rare_crop_size.png — what per-crop validity costs on the long
tail. Top: per-crop coverage against the 90% target. Bottom: the average set
size that buys it, against the vacuous ceiling. Crops run rarest first, so the
trade is read left to right: class-conditional region-robust calibration lifts
the rare crops over the target only by returning sets close to the ceiling.

Reads the per-class tables written by `python -m evaluation.conformal --perclass`.
Kharif is shown by default (the longer tail, 7 crops); --season rabi or
--season both for the alternatives.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs" / "rare_crop_size.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

SCHEMES = [("marginal", "#7f7f7f", "o", "marginal"),
           ("region-robust", "#1f77b4", "s", "region-robust"),
           ("class-region-robust", "#d62728", "^", "class-region-robust")]
TARGET = 0.90


def panel(axes, season, show_ylabel=True):
    ax_cov, ax_size = axes
    df = pd.read_csv(RES / f"perclass_size_{season}.csv")
    order = df[df.scheme == "marginal"].sort_values("n")["crop"].tolist()
    counts = df.drop_duplicates("crop").set_index("crop")["n"]
    n_classes = len(order)
    x = list(range(n_classes))

    for scheme, color, marker, label in SCHEMES:
        sub = df[df.scheme == scheme].set_index("crop").reindex(order)
        ax_cov.plot(x, sub["coverage"], marker=marker, color=color, lw=1.8,
                    ms=4.5, label=label)
        ax_size.plot(x, sub["avg_size"], marker=marker, color=color, lw=1.8, ms=4.5)

    ax_cov.axhline(TARGET, ls="--", lw=1.1, color="black", alpha=0.6)
    ax_cov.text(n_classes - 1, TARGET - 0.015, "90% target", fontsize=6.5,
                ha="right", va="top", alpha=0.75)
    ax_cov.set_ylim(0.55, 1.02)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels([])
    ax_cov.grid(alpha=0.3, axis="y")
    ax_cov.set_title(f"{season.capitalize()} ({n_classes} crops)", fontsize=8.5)

    ax_size.axhline(n_classes, ls="--", lw=1.1, color="black", alpha=0.6)
    ax_size.text(0, n_classes - 0.12, "vacuous", fontsize=6.5, va="top", alpha=0.75)
    ax_size.set_ylim(0.8, n_classes + 0.45)
    ax_size.set_xticks(x)
    ax_size.set_xticklabels(
        [f"{c.split()[0]}\n{int(counts[c]):,}" for c in order], fontsize=6.5)
    ax_size.grid(alpha=0.3, axis="y")

    if show_ylabel:
        ax_cov.set_ylabel("Coverage", fontsize=8)
        ax_size.set_ylabel("Set size", fontsize=8)
    for a in (ax_cov, ax_size):
        a.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["rabi", "kharif", "both"], default="kharif")
    args = ap.parse_args()

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]
    width = 3.45 if len(seasons) == 1 else 7.0
    fig, axs = plt.subplots(2, len(seasons), figsize=(width, 2.45), sharex="col")
    axs = axs.reshape(2, len(seasons))

    for j, season in enumerate(seasons):
        panel(axs[:, j], season, show_ylabel=(j == 0))

    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right", ncol=1,
                     handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    fig.supxlabel("Crop, rarest first (fields in the evaluation pool)", fontsize=7.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT, dpi=220)
    print("saved", OUT)


if __name__ == "__main__":
    main()
