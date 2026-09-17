"""
Stage 1: Clean & prepare the field-boundary shapefile for GEE upload.

Steps:
  1. Load shapefile and print summary
  2. Validate & fix geometries
  3. Drop records with missing crop_type or empty/null geometry
  4. Standardise crop_type labels (strip whitespace, title-case)
  5. Add a unique integer field_id column
  6. Ensure CRS is EPSG:4326
  7. Export clean shapefile + GeoPackage + label map

Usage:
    python run.py clean
    python run.py clean --input data/shapefiles/Major_crops.shp
    python pipeline/clean_shapefile.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import geopandas as gpd
import pandas as pd

from config import SHAPEFILE_DIR, CLEAN_DIR


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def print_summary(gdf: gpd.GeoDataFrame, label: str = "") -> None:
    """Print a quick overview of the GeoDataFrame."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Records     : {len(gdf)}")
    print(f"  Columns     : {list(gdf.columns)}")
    print(f"  CRS         : {gdf.crs}")
    if "crop_type" in gdf.columns:
        print(f"  Unique crops: {gdf['crop_type'].nunique()}")
        print(f"  Crop counts :")
        print(gdf["crop_type"].value_counts().to_string())
    print(f"{'='*60}\n")


def validate_and_fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries using vectorized make_valid()."""
    print("  Running make_valid() on all geometries (vectorized) ...")
    gdf["geometry"] = gdf.geometry.make_valid()
    remaining_invalid = (~gdf.is_valid).sum()
    if remaining_invalid > 0:
        print(f"  WARNING: {remaining_invalid} geometries still invalid after make_valid.")
    else:
        print("  All geometries are now valid.")
    return gdf


def drop_bad_records(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove records with null/empty geometry or missing crop_type."""
    before = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if "crop_type" in gdf.columns:
        gdf = gdf[gdf["crop_type"].notna() & (gdf["crop_type"].str.strip() != "")].copy()
    after = len(gdf)
    dropped = before - after
    if dropped:
        print(f"  Dropped {dropped} bad records ({before} -> {after}).")
    else:
        print("  No bad records found.")
    return gdf


def standardise_crop_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Strip whitespace and apply title-case to crop_type."""
    if "crop_type" not in gdf.columns:
        return gdf
    gdf["crop_type"] = gdf["crop_type"].str.strip().str.title()
    print(f"  Standardised crop_type labels: {sorted(gdf['crop_type'].unique())}")
    return gdf


def add_field_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a unique integer field_id starting from 1."""
    gdf = gdf.reset_index(drop=True)
    gdf.insert(0, "field_id", range(1, len(gdf) + 1))
    print(f"  Added field_id column (1 ... {len(gdf)}).")
    return gdf


def ensure_crs(gdf: gpd.GeoDataFrame, target_epsg: int = 4326) -> gpd.GeoDataFrame:
    """Reproject to target CRS if needed."""
    if gdf.crs is None:
        print(f"  WARNING: No CRS set -- assigning EPSG:{target_epsg}.")
        gdf = gdf.set_crs(epsg=target_epsg)
    elif gdf.crs.to_epsg() != target_epsg:
        print(f"  Reprojecting from {gdf.crs} -> EPSG:{target_epsg} ...")
        gdf = gdf.to_crs(epsg=target_epsg)
    else:
        print(f"  CRS already EPSG:{target_epsg}.")
    return gdf


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 — Clean field-boundary shapefile for GEE upload."
    )
    parser.add_argument(
        "--input", default=str(SHAPEFILE_DIR / "Major_crops.shp"),
        help="Input shapefile path.",
    )
    parser.add_argument(
        "--out-dir", default=str(CLEAN_DIR),
        help="Output directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not input_path.exists():
        sys.exit(f"ERROR: Input shapefile not found: {input_path}")

    # --- Load ---
    print(f"\nLoading {input_path} ...")
    gdf = gpd.read_file(input_path)
    print_summary(gdf, label="ORIGINAL DATA")

    # --- Clean ---
    print("Step 1/5  -- Drop bad records (null geometry / missing crop_type)")
    gdf = drop_bad_records(gdf)

    print("Step 2/5  -- Validate & fix geometries")
    gdf = validate_and_fix_geometries(gdf)

    print("Step 3/5  -- Standardise crop labels")
    gdf = standardise_crop_labels(gdf)

    print("Step 4/5  -- Add unique field_id")
    gdf = add_field_id(gdf)

    print("Step 5/5  -- Ensure CRS = EPSG:4326")
    gdf = ensure_crs(gdf)

    print_summary(gdf, label="CLEANED DATA")

    # --- Export ---
    out_dir.mkdir(parents=True, exist_ok=True)

    shp_out = out_dir / "clean_fields.shp"
    gpkg_out = out_dir / "clean_fields.gpkg"

    gdf.to_file(gpkg_out, driver="GPKG")
    print(f"  Saved GeoPackage -> {gpkg_out}")

    gdf.to_file(shp_out)
    print(f"  Saved shapefile  -> {shp_out}")

    if "crop_type" in gdf.columns:
        label_map = (
            gdf[["crop_type"]]
            .drop_duplicates()
            .sort_values("crop_type")
            .reset_index(drop=True)
        )
        label_map.insert(0, "class_id", range(len(label_map)))
        label_csv = out_dir / "label_map.csv"
        label_map.to_csv(label_csv, index=False)
        print(f"  Saved label map  -> {label_csv}")
        print(f"\n  Label mapping:\n{label_map.to_string(index=False)}")

    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    main()
