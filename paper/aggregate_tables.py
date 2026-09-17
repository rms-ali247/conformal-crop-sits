"""Aggregate CV mean+/-std (over 5 spatial folds) for all paper tables from one
consistent source, plus per-class test metrics. Prevents cross-table contradictions."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MET = ["OA", "Macro_F1", "Weighted_F1", "Kappa"]


def cv(fm):
    fm = Path(fm)
    if not fm.exists():
        return None
    df = pd.read_csv(fm)
    df = df[df["fold"].astype(str) != "TEST"]
    return {m: (df[m].mean(), df[m].std()) for m in MET}


def fmt(stat):
    return f"{stat[0]:.3f}$\\pm${stat[1]:.3f}"


def mcnemar(season):
    f = ROOT / "experiments" / f"benchmark_{season}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    col = [c for c in df.columns if c.startswith("McNemar")]
    if not col:
        return {}
    return dict(zip(df["model"], df[col[0]]))


print("=" * 70, "\nARCHITECTURE BENCHMARK (CV mean+/-std over 5 spatial folds)\n", "=" * 70)
arch = ["ms-s4", "s4d", "mstacnn", "transformer", "rf", "ltae", "tempcnn", "lstm"]
disp = {"ms-s4": "PhenoSSM", "s4d": "S4D", "mstacnn": "MSTACNN", "rf": "Random Forest",
        "ltae": "L-TAE", "tempcnn": "TempCNN", "lstm": "LSTM",
        "transformer": "Transformer"}
for season in ["rabi", "kharif"]:
    mc = mcnemar(season)
    print(f"\n--- {season} ---")
    rows = []
    for m in arch:
        c = cv(ROOT / "experiments" / m / "results" / season / "fold_metrics.csv")
        if c is None:
            continue
        p = mc.get(m, "")
        rows.append((c["Macro_F1"][0], f"{disp[m]} & {fmt(c['OA'])} & {fmt(c['Macro_F1'])} "
                     f"& {fmt(c['Weighted_F1'])} & {fmt(c['Kappa'])} & {p} \\\\"))
    for _, r in sorted(rows, reverse=True):
        print(r)

print("\n", "=" * 70, "\nLOSS ABLATION (PhenoSSM, CV mean+/-std)\n", "=" * 70)
loss_src = {"CE": ROOT / "experiments" / "ms-s4" / "results",
            "Focal": ROOT / "experiments_loss" / "focal" / "results",
            "Logit-adj": ROOT / "experiments_loss" / "logit-adj" / "results",
            "GCE": ROOT / "experiments_loss" / "gce" / "results",
            "SCE": ROOT / "results"}
for loss, base in loss_src.items():
    cells = []
    for season in ["rabi", "kharif"]:
        c = cv(base / season / "fold_metrics.csv")
        cells.append(fmt(c["OA"]) if c else "--")
        cells.append(fmt(c["Macro_F1"]) if c else "--")
    print(f"{loss} & " + " & ".join(cells) + " \\\\")

print("\n", "=" * 70, "\nPER-CLASS (held-out test, production SCE)\n", "=" * 70)
for season in ["rabi", "kharif"]:
    tr = ROOT / "results" / season / "test_report.txt"
    print(f"\n--- {season} ---")
    if tr.exists():
        for ln in tr.read_text().splitlines():
            if re.search(r"\d\.\d{3,4}", ln) and "accuracy" not in ln and "Test OA" not in ln:
                print(ln.strip())
