"""
config.py — Single source of truth for project constants and paths.

All band names, season definitions, feature column names, directory
paths, and GEE configuration are defined here.  Import from this module
instead of redefining constants in individual scripts.
"""

from datetime import date, timedelta
from pathlib import Path

# ------------------------------------------------------------------ #
#  Project paths
# ------------------------------------------------------------------ #

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
CLEAN_DIR     = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"
SHAPEFILE_DIR = DATA_DIR / "shapefiles"

# Monthly (7-step) per-season tensors for the PRIMARY MSTACNN model.
# Reconstructed from X_all by pipeline/restore_monthly.py so the monthly
# experiment is not clobbered by later dekadal re-extractions in PROCESSED_DIR.
PROCESSED_MONTHLY_DIR = DATA_DIR / "processed_monthly"
PROCESSED_DEKADAL_DIR = DATA_DIR / "processed_dekadal"

# PRIMARY (MSTACNN, monthly, 7-step) artifacts.
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# SECONDARY (S4D, dekadal, 21-step) artifacts — finer-resolution experiment.
MODELS_DEKADAL_DIR  = PROJECT_ROOT / "models_dekadal"
RESULTS_DEKADAL_DIR = PROJECT_ROOT / "results_dekadal"

# ------------------------------------------------------------------ #
#  Sentinel-2 configuration
# ------------------------------------------------------------------ #

S2_COLLECTION  = "COPERNICUS/S2_SR_HARMONIZED"

SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_NAMES    = ["NDVI", "EVI", "NDWI", "SAVI"]
ALL_BANDS      = SPECTRAL_BANDS + INDEX_NAMES          # 14

FEATURE_COLS_MEAN = [f"{b}_mean"   for b in ALL_BANDS]
FEATURE_COLS_STD  = [f"{b}_stdDev" for b in ALL_BANDS]
FEATURE_COLS      = FEATURE_COLS_MEAN + FEATURE_COLS_STD  # 28

# Column indices of the raw spectral-band features (mean + stdDev), excluding the
# four vegetation indices. Used by the `--feature-set raw` ablation: 10 bands x
# (mean, stdDev) = 20 of the 28 features. Order follows FEATURE_COLS exactly.
RAW_FEATURE_IDX = [i for i, c in enumerate(FEATURE_COLS)
                   if any(c == f"{b}_mean" or c == f"{b}_stdDev"
                          for b in SPECTRAL_BANDS)]

SCL_CLEAR = [4, 5, 6, 7]   # Vegetation, Bare soil, Water, Unclassified-clear

# ------------------------------------------------------------------ #
#  Season definitions
# ------------------------------------------------------------------ #

SEASON_WINDOWS = {
    "Rabi":   {"start": "2022-10-01", "end": "2023-04-30"},
    "Kharif": {"start": "2023-05-01", "end": "2023-11-30"},
}

SEASON_MONTH_CODES = {
    "Rabi":   [f"2022_{m:02d}" for m in [10, 11, 12]]
            + [f"2023_{m:02d}" for m in [1, 2, 3, 4]],
    "Kharif": [f"2023_{m:02d}" for m in [5, 6, 7, 8, 9, 10, 11]],
}

SEASON_MONTH_LABELS = {
    "rabi":   ["Oct 22", "Nov 22", "Dec 22", "Jan 23", "Feb 23", "Mar 23", "Apr 23"],
    "kharif": ["May 23", "Jun 23", "Jul 23", "Aug 23", "Sep 23", "Oct 23", "Nov 23"],
}

# ------------------------------------------------------------------ #
#  Dekadal (10-day) season definitions
# ------------------------------------------------------------------ #

def _build_dekadal_codes(start_str: str, end_str: str) -> list[str]:
    """Generate dekadal period codes: YYYY_MM_D1 / D2 / D3."""
    codes = []
    current = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    while current <= end:
        y, m = current.year, current.month
        if current.day == 1:
            codes.append(f"{y}_{m:02d}_D1")    # 1st–10th
            current = date(y, m, 11)
        elif current.day == 11:
            codes.append(f"{y}_{m:02d}_D2")    # 11th–20th
            current = date(y, m, 21)
        elif current.day == 21:
            codes.append(f"{y}_{m:02d}_D3")    # 21st–end of month
            # advance to 1st of next month
            if m == 12:
                current = date(y + 1, 1, 1)
            else:
                current = date(y, m + 1, 1)
        else:
            # snap to the nearest dekad boundary
            if current.day <= 10:
                current = current.replace(day=1)
            elif current.day <= 20:
                current = current.replace(day=11)
            else:
                current = current.replace(day=21)
    return codes


def _build_dekadal_labels(codes: list[str]) -> list[str]:
    """Human-readable labels: 'Oct-D1', 'Oct-D2', etc."""
    import calendar
    labels = []
    for code in codes:
        parts = code.split("_")  # e.g. "2022_10_D1"
        y, m = int(parts[0]), int(parts[1])
        dekad = parts[2]
        month_abbr = calendar.month_abbr[m]
        yr_short = str(y)[-2:]
        labels.append(f"{month_abbr}-{dekad} {yr_short}")
    return labels


SEASON_DEKADAL_CODES = {
    "Rabi":   _build_dekadal_codes("2022-10-01", "2023-04-30"),
    "Kharif": _build_dekadal_codes("2023-05-01", "2023-11-30"),
}

SEASON_DEKADAL_LABELS = {
    "rabi":   _build_dekadal_labels(SEASON_DEKADAL_CODES["Rabi"]),
    "kharif": _build_dekadal_labels(SEASON_DEKADAL_CODES["Kharif"]),
}

N_DEKADAL_TIMESTEPS = len(SEASON_DEKADAL_CODES["Rabi"])   # 21

# Pilot districts for dekadal extraction test
PILOT_DISTRICTS = ["Khairpur", "Jhal Magsi", "Mardan"]

# ------------------------------------------------------------------ #
#  Metadata columns
# ------------------------------------------------------------------ #

META_COLS = ["ID", "crop_type", "Season", "Province", "District"]
MONTH_COL = "month"

# ------------------------------------------------------------------ #
#  GEE configuration
# ------------------------------------------------------------------ #

GEE_PROJECT      = "ee-ralimsai23seecs"
GEE_ASSET_SUBSET = f"projects/{GEE_PROJECT}/assets/all_districts_major_crops"   # 776 field subset
GEE_ASSET        = f"projects/{GEE_PROJECT}/assets/major_crops_full"            # full 135K fields
GEE_DRIVE_FOLDER = "gee_sentinel2_exports"

# ------------------------------------------------------------------ #
#  Training defaults
# ------------------------------------------------------------------ #

N_TIMESTEPS      = 7
N_FEATURES       = 28
MAX_MISSING_FRAC = 0.5
MIN_CLASS_SIZE   = 5
SEED             = 42

# ------------------------------------------------------------------ #
#  Pixel-level classification settings
# ------------------------------------------------------------------ #

PATCH_SIZE          = 5          # 5×5 spatial patches (50 m context at 10 m)
N_PIXEL_FEATURES    = len(ALL_BANDS)   # 14 raw bands (no mean/std)
MAX_PIXELS_PER_FIELD = 100       # cap random pixel samples per field

PIXEL_DRIVE_FOLDER  = "gee_sentinel2_pixel"
RAW_PIXEL_DIR       = DATA_DIR / "raw_pixel"       # downloaded GeoTIFFs
PROCESSED_PIXEL_DIR = DATA_DIR / "processed_pixel"  # cleaned patches
CHIP_DRIVE_FOLDER   = "gee_sentinel2_chips"
RAW_CHIP_DIR        = DATA_DIR / "raw_chip"        # downloaded chip CSV shards

MAX_FIELDS_PER_CHIP_SHARD = 300   # cap fields per GEE chip export shard

CONVEX_HULL_BUFFER  = 500       # metres — buffer around district convex hull
