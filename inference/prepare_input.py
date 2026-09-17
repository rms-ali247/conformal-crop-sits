"""
prepare_input.py — Prepare new data for inference with the crop type model.

Supports multiple input formats:
  1. GEE-exported monthly CSVs (same format as training data)
  2. A single wide CSV (rows = fields, columns = t0_B2_mean, t0_B2_stdDev, ...)
  3. Manual feature entry (interactive mode)

Output: numpy array (N, 7, 28) ready for CropPredictor.predict()

Usage:
    # From GEE CSVs (same directory structure as training)
    python inference/prepare_input.py --mode gee \\
        --csv-dir data/raw_new --season rabi --output inference/inputs/new_rabi.npy

    # From a wide CSV
    python inference/prepare_input.py --mode csv \\
        --csv-file my_fields.csv --output inference/inputs/my_fields.npy

    # Interactive single-sample entry
    python inference/prepare_input.py --mode manual --season rabi
"""

from pathlib import Path
import argparse
import re
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLS, FEATURE_COLS_MEAN, FEATURE_COLS_STD,
    SEASON_MONTH_CODES as SEASONS,
    SEASON_MONTH_LABELS as SEASON_MONTHS,
)


# ------------------------------------------------------------------ #
#  GEE CSV preparation
# ------------------------------------------------------------------ #

def prepare_from_gee_csvs(csv_dir: str | Path, season: str) -> np.ndarray:
    """Load monthly GEE CSVs and build (N, 7, 28) array.

    Expects files named S2_{Season}_{YYYY}_{MM}.csv in csv_dir,
    same format as the training extraction script produces.

    Args:
        csv_dir: Directory containing monthly CSVs
        season:  "rabi" or "kharif"

    Returns:
        X: numpy array of shape (N_fields, 7, 28)
    """
    csv_dir = Path(csv_dir)
    season_cap = season.capitalize()
    expected_months = SEASONS[season_cap]
    pattern = f"S2_{season_cap}_*.csv"

    files = sorted(csv_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern} in {csv_dir}\n"
            f"Expected files like: S2_{season_cap}_2022_10.csv"
        )
    print(f"Found {len(files)} CSV files for {season_cap}")

    # Load all CSVs
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  Loaded {f.name}: {len(df)} rows")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    # Identify ID column
    id_col = "ID" if "ID" in merged.columns else merged.columns[0]

    # Find feature columns present
    feat_present = [c for c in FEATURE_COLS if c in merged.columns]
    if not feat_present:
        raise ValueError(
            f"No expected feature columns found. Expected columns like: {FEATURE_COLS[:4]}\n"
            f"Found columns: {list(merged.columns[:10])}"
        )

    # Check for month column
    month_col = None
    for candidate in ["month", "Month", "MONTH", "date"]:
        if candidate in merged.columns:
            month_col = candidate
            break
    if month_col is None:
        # Try to infer from filename
        file_months = {}
        for f in files:
            match = re.search(r"(\d{4})_(\d{2})", f.stem)
            if match:
                month_label = f"{match.group(1)}_{match.group(2)}"
                df_temp = pd.read_csv(f)
                df_temp["_month"] = month_label
                file_months[f.name] = month_label
        if file_months:
            dfs2 = []
            for f in files:
                df_temp = pd.read_csv(f)
                match = re.search(r"(\d{4})_(\d{2})", f.stem)
                if match:
                    df_temp["month"] = f"{match.group(1)}_{match.group(2)}"
                dfs2.append(df_temp)
            merged = pd.concat(dfs2, ignore_index=True)
            month_col = "month"

    if month_col is None:
        raise ValueError("Cannot determine month for each row. "
                         "Add a 'month' column or use filename convention S2_Rabi_YYYY_MM.csv")

    # Aggregate duplicate (ID, month) rows
    merged = merged.groupby([id_col, month_col], as_index=False).agg(
        {c: "mean" for c in feat_present}
    )

    # Pivot to 3D array
    field_ids = sorted(merged[id_col].unique())
    N = len(field_ids)
    T = len(expected_months)
    C = len(feat_present)

    X = np.full((N, T, C), np.nan, dtype=np.float32)

    for i, fid in enumerate(field_ids):
        field_data = merged[merged[id_col] == fid].set_index(month_col)
        for t, month_label in enumerate(expected_months):
            if month_label in field_data.index:
                X[i, t, :] = field_data.loc[month_label, feat_present].values

    # Interpolate missing timesteps
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Interpolating {nan_count} missing values...")
        for i in range(N):
            for c in range(C):
                series = X[i, :, c]
                if np.any(np.isnan(series)):
                    nans = np.isnan(series)
                    if nans.all():
                        X[i, :, c] = 0.0  # all missing → zero
                    else:
                        x_valid = np.where(~nans)[0]
                        y_valid = series[~nans]
                        X[i, :, c] = np.interp(np.arange(T), x_valid, y_valid)

    print(f"  Prepared array: shape {X.shape}")
    return X, field_ids


# ------------------------------------------------------------------ #
#  Wide CSV preparation
# ------------------------------------------------------------------ #

def prepare_from_wide_csv(csv_file: str | Path, timesteps: int = 7,
                          features: int = 28) -> np.ndarray:
    """Load a wide CSV where each row is a field.

    Columns should be the 28 features × 7 timesteps = 196 values.
    Can be either:
      - Named: t0_B2_mean, t0_B2_stdDev, ..., t6_SAVI_stdDev
      - Unnamed: Just 196 numeric columns in order

    Args:
        csv_file: Path to CSV
        timesteps: Number of timesteps (default 7)
        features: Number of features per timestep (default 28)

    Returns:
        X: numpy array of shape (N, T, C)
    """
    df = pd.read_csv(csv_file)

    # Drop common non-feature columns
    drop_cols = [c for c in df.columns if c.lower() in
                 ("id", "crop_type", "crop_id", "district", "province",
                  "field_id", "label", "season", "unnamed: 0")]
    if drop_cols:
        print(f"  Dropping non-feature columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    n_cols = len(df.columns)
    expected = timesteps * features

    if n_cols != expected:
        print(f"  WARNING: Expected {expected} feature columns, got {n_cols}.")
        if n_cols > expected:
            print(f"  Using first {expected} columns.")
            df = df.iloc[:, :expected]
        else:
            raise ValueError(f"Not enough columns: need {expected}, have {n_cols}")

    X = df.values.astype(np.float32).reshape(len(df), timesteps, features)
    print(f"  Prepared array: shape {X.shape}")
    return X


# ------------------------------------------------------------------ #
#  Synthetic test data generator
# ------------------------------------------------------------------ #

def generate_synthetic_sample(season: str, num_samples: int = 5,
                              seed: int = 42) -> np.ndarray:
    """Generate synthetic test samples for quick validation.

    Creates fake spectral data with realistic value ranges,
    useful for verifying the inference pipeline works end-to-end.

    Args:
        season: "rabi" or "kharif"
        num_samples: Number of synthetic samples
        seed: Random seed

    Returns:
        X: numpy array of shape (num_samples, 7, 28)
    """
    np.random.seed(seed)
    T = 7
    C = 28

    X = np.zeros((num_samples, T, C), dtype=np.float32)

    # Realistic ranges for each feature type
    # First 10 features: spectral band means (0-3000 SR range)
    X[:, :, :10] = np.random.uniform(200, 3000, (num_samples, T, 10))
    # Next 4: index means (NDVI ~0.2-0.8, EVI ~0.1-0.6, NDWI ~-0.3-0.3, SAVI ~0.1-0.7)
    X[:, :, 10] = np.random.uniform(0.1, 0.9, (num_samples, T))   # NDVI
    X[:, :, 11] = np.random.uniform(0.05, 0.7, (num_samples, T))  # EVI
    X[:, :, 12] = np.random.uniform(-0.4, 0.4, (num_samples, T))  # NDWI
    X[:, :, 13] = np.random.uniform(0.05, 0.8, (num_samples, T))  # SAVI
    # Next 10: spectral band stdDevs (50-500)
    X[:, :, 14:24] = np.random.uniform(50, 500, (num_samples, T, 10))
    # Last 4: index stdDevs (0.01-0.2)
    X[:, :, 24:28] = np.random.uniform(0.01, 0.25, (num_samples, T, 4))

    print(f"  Generated {num_samples} synthetic samples: shape {X.shape}")
    return X


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Prepare input data for crop type inference."
    )
    parser.add_argument("--mode", required=True,
                        choices=["gee", "csv", "synthetic"],
                        help="Input mode: gee (monthly CSVs), csv (wide), "
                             "or synthetic (test data).")
    parser.add_argument("--season", default="rabi", choices=["rabi", "kharif"],
                        help="Season (for GEE and synthetic modes).")
    parser.add_argument("--csv-dir", default=None,
                        help="Directory with monthly GEE CSVs (gee mode).")
    parser.add_argument("--csv-file", default=None,
                        help="Path to wide CSV file (csv mode).")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of synthetic samples.")
    parser.add_argument("--output", default=None,
                        help="Output path for .npy file.")
    args = parser.parse_args()

    output_dir = Path("inference/inputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "gee":
        if not args.csv_dir:
            sys.exit("ERROR: --csv-dir required for gee mode.")
        X, field_ids = prepare_from_gee_csvs(args.csv_dir, args.season)
        out_path = args.output or str(output_dir / f"X_{args.season}_new.npy")
        np.save(out_path, X)
        # Also save field IDs for traceability
        pd.DataFrame({"field_id": field_ids}).to_csv(
            str(Path(out_path).with_suffix(".csv")), index=False
        )

    elif args.mode == "csv":
        if not args.csv_file:
            sys.exit("ERROR: --csv-file required for csv mode.")
        X = prepare_from_wide_csv(args.csv_file)
        out_path = args.output or str(output_dir / "X_from_csv.npy")
        np.save(out_path, X)

    elif args.mode == "synthetic":
        X = generate_synthetic_sample(args.season, args.num_samples)
        out_path = args.output or str(output_dir / f"X_synthetic_{args.season}.npy")
        np.save(out_path, X)

    print(f"\nSaved to {out_path}")
    print(f"Shape: {X.shape}  ({X.shape[0]} fields × {X.shape[1]} months × {X.shape[2]} features)")


if __name__ == "__main__":
    main()
