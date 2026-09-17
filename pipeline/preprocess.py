"""
Stage 3: Merge GEE-exported CSVs, preprocess, and build DL-ready tensors.

NOTE: Normalization is intentionally NOT applied here.  Per-fold
      StandardScaler is applied in pipeline/train.py to prevent
      data leakage between train and validation splits.

Usage:
    python run.py preprocess
    python pipeline/preprocess.py
    python pipeline/preprocess.py --raw-dir data/raw --out-dir data/processed
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    FEATURE_COLS, META_COLS, MONTH_COL, SEASON_MONTH_CODES,
    MAX_MISSING_FRAC, RAW_DIR, PROCESSED_DIR,
)


# ------------------------------------------------------------------ #
#  Load & merge CSVs
# ------------------------------------------------------------------ #

def load_season_csvs(raw_dir: Path, season: str) -> pd.DataFrame:
    """Load all monthly CSVs for one season and concatenate."""
    pattern = f"S2_{season}_*.csv"
    files = sorted(raw_dir.glob(pattern))
    if not files:
        print(f"  WARNING: No files matching {pattern} in {raw_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"    Loaded {f.name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  {season} total rows (all months): {len(merged)}")
    return merged


# ------------------------------------------------------------------ #
#  Pivot to wide: one row per field, columns per (month, feature)
# ------------------------------------------------------------------ #

def pivot_to_timeseries(df: pd.DataFrame, season: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot monthly rows into a wide DataFrame.

    Returns (meta_df, ts_wide):
        meta_df  — one row per field with META_COLS
        ts_wide  — one row per field, (7 months × 28 features) = 196 columns
    """
    expected_months = SEASON_MONTH_CODES[season]

    keep_cols = META_COLS + [MONTH_COL] + FEATURE_COLS
    present_cols = [c for c in keep_cols if c in df.columns]
    df = df[present_cols].copy()

    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    df = df.groupby(["ID", MONTH_COL], as_index=False).agg(
        {**{c: "first" for c in META_COLS if c != "ID" and c in df.columns},
         **{c: "mean" for c in feat_present}}
    )

    field_groups = df.groupby("ID")

    records = []
    meta_records = []

    for field_id, group in field_groups:
        meta_row = group[META_COLS].iloc[0].to_dict()
        meta_records.append(meta_row)

        row_data = {"ID": field_id}
        month_data = group.set_index(MONTH_COL)

        for t_idx, month_label in enumerate(expected_months):
            if month_label in month_data.index:
                for feat in feat_present:
                    val = month_data.loc[month_label, feat]
                    if isinstance(val, pd.Series):
                        val = val.mean()
                    row_data[f"t{t_idx}_{feat}"] = float(val)
            else:
                for feat in feat_present:
                    row_data[f"t{t_idx}_{feat}"] = np.nan

        records.append(row_data)

    ts_wide = pd.DataFrame(records)
    meta_df = pd.DataFrame(meta_records)

    return meta_df, ts_wide


# ------------------------------------------------------------------ #
#  Handle missing data
# ------------------------------------------------------------------ #

def interpolate_and_filter(
    ts_wide: pd.DataFrame,
    meta_df: pd.DataFrame,
    n_months: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop fields with >50% missing months, interpolate the rest."""
    n_features = len(FEATURE_COLS)

    missing_counts = np.zeros(len(ts_wide))
    for t in range(n_months):
        t_cols = [f"t{t}_{f}" for f in FEATURE_COLS if f"t{t}_{f}" in ts_wide.columns]
        if t_cols:
            is_missing = ts_wide[t_cols].isna().all(axis=1)
            missing_counts += is_missing.values

    missing_frac = missing_counts / n_months
    keep_mask = missing_frac <= MAX_MISSING_FRAC

    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"  Dropped {dropped} fields with >{MAX_MISSING_FRAC*100:.0f}% missing months.")

    ts_wide = ts_wide[keep_mask].reset_index(drop=True)
    meta_df = meta_df[keep_mask].reset_index(drop=True)

    N = len(ts_wide)
    arr = np.full((N, n_months, n_features), np.nan, dtype=np.float64)

    for t in range(n_months):
        for c_idx, feat in enumerate(FEATURE_COLS):
            col_name = f"t{t}_{feat}"
            if col_name in ts_wide.columns:
                arr[:, t, c_idx] = ts_wide[col_name].values

    for i in range(N):
        for c in range(n_features):
            series = pd.Series(arr[i, :, c])
            series = series.interpolate(method="linear", limit_direction="both")
            arr[i, :, c] = series.values

    for t in range(n_months):
        for c_idx, feat in enumerate(FEATURE_COLS):
            col_name = f"t{t}_{feat}"
            if col_name in ts_wide.columns:
                ts_wide[col_name] = arr[:, t, c_idx]

    remaining_nans = np.isnan(arr).sum()
    print(f"  After interpolation: {remaining_nans} NaN values remaining.")

    return ts_wide, meta_df


# ------------------------------------------------------------------ #
#  Reshape wide DF → (N, T, C) array
# ------------------------------------------------------------------ #

def wide_to_tensor(ts_wide: pd.DataFrame, n_months: int = 7) -> np.ndarray:
    """Convert wide DataFrame to (N, T=7, C=28) numpy array."""
    N = len(ts_wide)
    C = len(FEATURE_COLS)
    arr = np.full((N, n_months, C), np.nan, dtype=np.float64)

    for t in range(n_months):
        for c_idx, feat in enumerate(FEATURE_COLS):
            col = f"t{t}_{feat}"
            if col in ts_wide.columns:
                arr[:, t, c_idx] = ts_wide[col].values
    return arr


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 — Preprocess GEE CSVs into DL-ready tensors."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR),
                        help="Directory with GEE-exported CSVs.")
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR),
                        help="Output directory for tensors.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        sys.exit(f"ERROR: Raw directory not found: {raw_dir}")

    all_X, all_y, all_meta = [], [], []
    label_encoder = LabelEncoder()

    # First pass: collect all crop types
    all_crop_types = set()
    for season in SEASON_MONTH_CODES:
        df = load_season_csvs(raw_dir, season)
        if not df.empty and "crop_type" in df.columns:
            all_crop_types.update(df["crop_type"].dropna().unique())

    if not all_crop_types:
        sys.exit("ERROR: No crop_type values found. Check data/raw/ files.")

    label_encoder.fit(sorted(all_crop_types))
    print(f"\nLabel encoding ({len(label_encoder.classes_)} classes):")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")

    for season in SEASON_MONTH_CODES:
        print(f"\n{'='*60}")
        print(f"  Processing {season}")
        print(f"{'='*60}")

        df = load_season_csvs(raw_dir, season)
        if df.empty:
            print(f"  Skipping {season} — no data.")
            continue

        print(f"\n  Pivoting to time-series ...")
        meta_df, ts_wide = pivot_to_timeseries(df, season)
        print(f"  Fields: {len(ts_wide)}")

        print(f"\n  Interpolating missing months ...")
        ts_wide, meta_df = interpolate_and_filter(ts_wide, meta_df, n_months=7)

        arr = wide_to_tensor(ts_wide, n_months=7)
        print(f"  Tensor shape: {arr.shape}  (N, T=7, C=28)")

        arr_float = arr.astype(np.float32)
        labels = label_encoder.transform(meta_df["crop_type"].values)

        np.save(out_dir / f"X_{season.lower()}.npy", arr_float)
        np.save(out_dir / f"y_{season.lower()}.npy", labels)
        meta_df.to_csv(out_dir / f"meta_{season.lower()}.csv", index=False)
        print(f"  Saved X_{season.lower()}.npy  shape={arr_float.shape}")
        print(f"  Saved y_{season.lower()}.npy  shape={labels.shape}")
        print(f"  Saved meta_{season.lower()}.csv  ({len(meta_df)} rows)")

        all_X.append(arr_float)
        all_y.append(labels)
        all_meta.append(meta_df)

    # Combine both seasons (reference only)
    if all_X:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        meta_all = pd.concat(all_meta, ignore_index=True)

        np.save(out_dir / "X_all.npy", X_all)
        np.save(out_dir / "y_all.npy", y_all)
        meta_all.to_csv(out_dir / "meta_all.csv", index=False)

        print(f"\n{'='*60}")
        print(f"  Combined: X_all={X_all.shape}, y_all={y_all.shape}")
        print(f"  NOTE: X_all/y_all for reference only — train per-season.")
        print(f"{'='*60}")

    # Save label map
    label_map = pd.DataFrame({
        "class_id": range(len(label_encoder.classes_)),
        "crop_type": label_encoder.classes_,
    })
    label_map.to_csv(out_dir / "label_map.csv", index=False)
    print(f"\n  Saved label_map.csv")

    # Summary
    if all_X:
        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  Total samples   : {X_all.shape[0]}")
        print(f"  Timesteps       : {X_all.shape[1]}")
        print(f"  Features/step   : {X_all.shape[2]}")
        print(f"  Classes         : {len(label_encoder.classes_)}")
        print(f"  NaN remaining   : {np.isnan(X_all).sum()}")
        print(f"\n  Class distribution:")
        for i, cls in enumerate(label_encoder.classes_):
            count = (y_all == i).sum()
            pct = count / len(y_all) * 100
            print(f"    {i}: {cls:<20s} {count:>6d}  ({pct:.1f}%)")
        print(f"{'='*60}")

    print("\nStage 3 complete.")
    print("Next: run  python run.py train  (trains separate Rabi/Kharif models).\n")


if __name__ == "__main__":
    main()
