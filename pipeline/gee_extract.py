"""
Stage 2: Season-aware Sentinel-2 time-series extraction from GEE.

Design rationale
================
A single field may grow different crops in Rabi vs Kharif.  Each record
in the shapefile is a (field, season, crop_type) observation.  We MUST
extract satellite data only from the season the label refers to,
otherwise the spectral signal belongs to a different crop.

Per month we:
    1. Build a cloud-masked Sentinel-2 SR median composite
    2. Compute 4 vegetation indices (NDVI, EVI, NDWI, SAVI)
    3. Run a combined mean+stdDev reduceRegions over each batch
       of fields (GEE handles ~5 000 features per call)
    4. Export each batch result as a CSV to Google Drive

Batching
========
GEE's reduceRegions can reliably process ~5 000 polygons per call.
With 135 000+ fields, we split each season's fields into batches
of --batch-size (default 5 000).  File naming:

    S2_Rabi_2022_10_b001.csv   (batch 1 of Rabi, October 2022)
    S2_Rabi_2022_10_b002.csv   (batch 2, same month)
    ...

The preprocessing stage (pipeline/preprocess.py) uses glob("S2_{season}_*.csv")
so it auto-concatenates all batch files per season.

Usage:
    python run.py extract
    python run.py extract --season Rabi
    python run.py extract --batch-size 3000
    python pipeline/gee_extract.py
"""

import sys
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from datetime import date
from dateutil.relativedelta import relativedelta

import ee

from config import (
    S2_COLLECTION, SPECTRAL_BANDS, INDEX_NAMES, ALL_BANDS,
    SCL_CLEAR, SEASON_WINDOWS, META_COLS, GEE_PROJECT, GEE_ASSET,
    GEE_DRIVE_FOLDER,
)

# Metadata columns to carry through to the CSV
_EXPORT_META = ["ID", "crop_type", "Season", "Province", "District", "Date"]

# Default batch size — max fields per reduceRegions call.
# With district-based batching, most districts fit in one batch.
# Large districts are further split into sub-batches of this size.
DEFAULT_BATCH_SIZE = 2000


# ------------------------------------------------------------------ #
#  Cloud masking
# ------------------------------------------------------------------ #

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds/shadows using the SCL band (server-side)."""
    scl = image.select("SCL")
    clear = ee.Image.constant(0)
    for c in SCL_CLEAR:
        clear = clear.Or(scl.eq(c))
    return image.updateMask(clear)


# ------------------------------------------------------------------ #
#  Vegetation indices
# ------------------------------------------------------------------ #

def add_indices(image: ee.Image) -> ee.Image:
    """Append NDVI, EVI, NDWI, SAVI bands to the image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {"NIR": image.select("B8"),
         "RED": image.select("B4"),
         "BLUE": image.select("B2")},
    ).rename("EVI")
    ndwi = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    savi = image.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
        {"NIR": image.select("B8"),
         "RED": image.select("B4")},
    ).rename("SAVI")
    return image.addBands([ndvi, evi, ndwi, savi])


# ------------------------------------------------------------------ #
#  Monthly composite
# ------------------------------------------------------------------ #

def monthly_composite(start: str, end: str, bounds: ee.Geometry) -> ee.Image:
    """Cloud-free median composite for one calendar month."""
    col = (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start, end)
        .filterBounds(bounds)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        .map(mask_s2_clouds)
        .select(SPECTRAL_BANDS)
        .map(add_indices)
    )
    return col.median()


# ------------------------------------------------------------------ #
#  Per-field reduction
# ------------------------------------------------------------------ #

def extract_field_stats(
    composite: ee.Image,
    fields: ee.FeatureCollection,
    month_label: str,
    scale: int = 10,
) -> ee.FeatureCollection:
    """Single reduceRegions call with a combined mean+stdDev reducer."""
    reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True
    )
    reduced = composite.select(ALL_BANDS).reduceRegions(
        collection=fields, reducer=reducer, scale=scale,
    )

    def tag_month(feat):
        return ee.Feature(feat).set("month", month_label)

    return reduced.map(tag_month)


# ------------------------------------------------------------------ #
#  Batching helper — group by District for geographic locality
# ------------------------------------------------------------------ #

def build_district_batches(
    season_fields: ee.FeatureCollection,
    season_count: int,
    batch_size: int,
) -> list[tuple[str, ee.FeatureCollection]]:
    """Split fields into batches grouped by District.

    Fields in the same district are geographically close, so the batch
    bounding box is small and the S2 composite is much cheaper.
    Large districts are further split into sub-batches of *batch_size*.

    Returns list of (batch_label, FeatureCollection) tuples.
    """
    # Get distinct district names (client-side)
    districts = (
        season_fields
        .aggregate_array("District")
        .distinct()
        .sort()
        .getInfo()
    )

    batches = []
    for district in districts:
        dist_fc = season_fields.filter(ee.Filter.eq("District", district))
        dist_count = dist_fc.size().getInfo()

        if dist_count <= batch_size:
            # Whole district fits in one batch
            safe_name = district.replace(" ", "_")
            batches.append((safe_name, dist_fc))
        else:
            # Split large district into sub-batches
            n_sub = math.ceil(dist_count / batch_size)
            dist_list = dist_fc.toList(dist_count)
            safe_name = district.replace(" ", "_")
            for s in range(n_sub):
                start = s * batch_size
                end = min(start + batch_size, dist_count)
                sub_fc = ee.FeatureCollection(dist_list.slice(start, end))
                batches.append((f"{safe_name}_p{s+1}", sub_fc))

    return batches


# ------------------------------------------------------------------ #
#  Date range helper
# ------------------------------------------------------------------ #

def month_ranges(start_str: str, end_str: str):
    """Return list of (start, end_exclusive, label) per calendar month."""
    current = date.fromisoformat(start_str).replace(day=1)
    end = date.fromisoformat(end_str)
    ranges = []
    while current <= end:
        nxt = current + relativedelta(months=1)
        ranges.append((current.isoformat(), nxt.isoformat(), current.strftime("%Y_%m")))
        current = nxt
    return ranges


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 — Season-aware Sentinel-2 extraction from GEE."
    )
    parser.add_argument("--asset", default=GEE_ASSET,
                        help="GEE asset path for the uploaded shapefile.")
    parser.add_argument("--season", choices=["Rabi", "Kharif", "both"], default="both",
                        help="Which season(s) to process (default: both).")
    parser.add_argument("--scale", type=int, default=10,
                        help="Pixel resolution in metres (default: 10).")
    parser.add_argument("--drive-folder", default=GEE_DRIVE_FOLDER,
                        help=f"Google Drive folder for exports.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Fields per reduceRegions call (default: {DEFAULT_BATCH_SIZE}). "
                             "Lower if GEE times out, raise if you want fewer tasks.")
    args = parser.parse_args()

    # --- Initialise GEE ---
    print("Initialising Earth Engine ...")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        print("Running ee.Authenticate() -- follow the browser prompt.")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)

    # --- Load all fields ---
    print(f"Loading asset: {args.asset}")
    all_fields = ee.FeatureCollection(args.asset)
    total = all_fields.size().getInfo()
    print(f"  Total field records in asset: {total}")

    # --- Determine which seasons to process ---
    seasons_to_run = (
        list(SEASON_WINDOWS.keys()) if args.season == "both"
        else [args.season]
    )

    # Build export property list
    mean_cols   = [f"{b}_mean"   for b in ALL_BANDS]
    stddev_cols = [f"{b}_stdDev" for b in ALL_BANDS]
    export_props = _EXPORT_META + ["month"] + mean_cols + stddev_cols

    tasks = []

    for season in seasons_to_run:
        window = SEASON_WINDOWS[season]
        months = month_ranges(window["start"], window["end"])

        season_fields = all_fields.filter(ee.Filter.eq("Season", season))
        season_count = season_fields.size().getInfo()

        print(f"\n{'='*60}")
        print(f"  {season} season: {season_count:,} fields")
        print(f"  Window: {window['start']} -> {window['end']}  ({len(months)} months)")

        if season_count == 0:
            print(f"  WARNING: No fields found with Season='{season}'. Skipping.")
            print(f"{'='*60}")
            continue

        # --- Build district-based batches ---
        print(f"  Building district-based batches (max {args.batch_size:,} per batch) ...")
        batches = build_district_batches(season_fields, season_count, args.batch_size)
        n_batches = len(batches)
        print(f"  {n_batches} batches across {len(set(b[0].split('_p')[0] for b in batches))} districts")
        print(f"  Export tasks: {n_batches} batches × {len(months)} months = {n_batches * len(months)}")
        print(f"{'='*60}")

        for b_idx, (batch_label, batch_fc) in enumerate(batches, start=1):
            batch_bounds = batch_fc.geometry()

            for m_start, m_end, m_label in months:
                desc = f"S2_{season}_{m_label}_{batch_label}"
                # GEE task description max 100 chars — truncate if needed
                if len(desc) > 100:
                    desc = desc[:100]
                print(f"    Submitting {desc} ...")

                composite = monthly_composite(m_start, m_end, batch_bounds)
                stats_fc = extract_field_stats(
                    composite, batch_fc, m_label, args.scale
                )

                task = ee.batch.Export.table.toDrive(
                    collection=stats_fc.select(
                        propertySelectors=export_props, retainGeometry=False
                    ),
                    description=desc,
                    folder=args.drive_folder,
                    fileNamePrefix=desc,
                    fileFormat="CSV",
                )
                task.start()
                tasks.append((desc, task))

            print(f"    ✓ Batch {b_idx}/{n_batches} [{batch_label}] — {len(months)} tasks submitted")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Submitted {len(tasks)} export tasks to GEE.")
    print(f"  Drive folder : '{args.drive_folder}'")
    print(f"  Monitor at   : https://code.earthengine.google.com/tasks")
    print(f"{'='*60}")

    # Show first few task statuses (avoid flooding console with 196+ lines)
    n_show = min(10, len(tasks))
    print(f"\nFirst {n_show} task statuses:")
    for desc, task in tasks[:n_show]:
        state = task.status().get("state", "UNKNOWN")
        print(f"  {desc}: {state}")
    if len(tasks) > n_show:
        print(f"  ... and {len(tasks) - n_show} more tasks")

    print(f"\nStage 2 complete — {len(tasks)} tasks running on GEE servers.")
    print("Once finished, download ALL CSVs into  data/raw/  and run Stage 3.")
    print("  Rabi CSVs :  S2_Rabi_YYYY_MM_bNNN.csv")
    print("  Kharif CSVs: S2_Kharif_YYYY_MM_bNNN.csv\n")


if __name__ == "__main__":
    main()
