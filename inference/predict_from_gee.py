"""
predict_from_gee.py — Paste GEE console JSON and get an instant crop prediction.

Workflow:
  1. Run  gee_extract_field.js  in GEE Code Editor
  2. Draw a polygon, click Extract, copy the JSON from Console
  3. Run this script and paste when prompted  (or use --json flag)

Usage:
    python inference/predict_from_gee.py                      # interactive paste
    python inference/predict_from_gee.py --season rabi        # skip season prompt
    python inference/predict_from_gee.py --json '{"season":...,"features":[[...]]}'
    python inference/predict_from_gee.py --json-file field1.json
    python inference/predict_from_gee.py --json-file field1.json --plot
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predict import CropPredictor
from config import MODELS_DIR, MODELS_DEKADAL_DIR
from utils import plot_prediction_attention


def parse_gee_json(raw: str) -> tuple[str, np.ndarray]:
    """Parse the JSON output from a GEE feature-extraction script.

    Accepts both resolutions:
      - monthly  (gee/extract_field.js)          -> (7, 28)
      - dekadal  (gee/extract_field_dekadal.js)  -> (21, 28)

    The number of timesteps T is validated later against the loaded model.

    Returns:
        season: "rabi" or "kharif"
        X: numpy array of shape (1, T, 28)
    """
    data = json.loads(raw)

    season = data["season"]
    features = data["features"]  # list of T lists, each 28 values

    X = np.array(features, dtype=np.float32)  # (T, 28)
    if X.ndim != 2 or X.shape[1] != 28:
        raise ValueError(
            f"Expected features of shape (T, 28) — T timesteps × 28 features — "
            f"but got {X.shape}. Check the GEE script exported every feature."
        )

    X = X[np.newaxis, ...]  # (1, T, 28)
    return season, X


def resolve_model_dir(model_dir: str | None, resolution: str | None) -> Path | None:
    """Pick the parent models directory from --model-dir / --resolution.

    --model-dir wins if given. Otherwise --resolution maps:
      monthly -> models/ (MSTACNN)   |   dekadal -> models_dekadal/ (S4D)
    """
    if model_dir:
        return Path(model_dir)
    if resolution == "dekadal":
        return MODELS_DEKADAL_DIR
    if resolution == "monthly":
        return MODELS_DIR
    return None  # CropPredictor default (models/)


def run_prediction(season: str, X: np.ndarray, plot: bool = False,
                   output_dir: Path = None, model_dir: Path | None = None):
    """Load model and run prediction."""
    predictor = CropPredictor(season=season, model_dir=model_dir)
    results = predictor.predict(X)

    # Print formatted results
    print(predictor.format_results(results, top_k=predictor.num_classes))

    # Highlight the main prediction prominently
    pred = results["predictions"][0]
    conf = results["confidence"][0]
    if "decision" in results:
        cc = results["calibrated_confidence"][0]
        pset = results["prediction_set"][0]
        print(f"\n  >>> PREDICTED CROP: {pred}  (calibrated {cc:.1%}) <<<")
        print(f"      Conformal set: {', '.join(pset) if pset else '(empty)'}")
        print(f"      Decision: {results['decision'][0]}\n")
    else:
        print(f"\n  >>> PREDICTED CROP: {pred}  ({conf:.1%} confidence) <<<\n")

    # Attention insight
    if "attention" in results:
        months = results["months"]
        attn = results["attention"][0]
        peak_idx = np.argmax(attn)
        print(f"  Key month: {months[peak_idx]} (attention={attn[peak_idx]:.3f})")
        print(f"  The model focused most on {months[peak_idx]} for this prediction.\n")

    # Save plot
    if plot and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"gee_prediction_{season}_{pred.replace(' ','_')}.png"
        plot_prediction_attention(results, 0, save_path)

    return results


def interactive_mode(season_override: str = None, plot: bool = False,
                     model_dir: Path | None = None):
    """Interactive: prompt user to paste JSON from GEE console."""
    print("=" * 60)
    print("  CROP TYPE PREDICTION FROM GEE FIELD EXTRACTION")
    print("=" * 60)
    print()
    print("Paste the JSON from the GEE Console below.")
    print("(It starts with {\"season\":... and ends with })")
    print("Then press Enter twice (empty line) to submit.")
    print()

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    raw = "\n".join(lines).strip()
    if not raw:
        print("No input received. Exiting.")
        return

    try:
        season, X = parse_gee_json(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON — {e}")
        print("Make sure you copied the full JSON from the GEE Console.")
        return
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    if season_override:
        season = season_override

    print(f"\n  Parsed {season.upper()} field features: shape {X.shape}")
    print(f"  Running inference...\n")

    output_dir = Path("inference/outputs/gee_predictions")
    run_prediction(season, X, plot=plot, output_dir=output_dir, model_dir=model_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Predict crop type from GEE-extracted field features."
    )
    parser.add_argument("--json", default=None,
                        help="GEE JSON string directly (skip interactive mode).")
    parser.add_argument("--json-file", default=None,
                        help="Path to a .json file with GEE output.")
    parser.add_argument("--season", default=None, choices=["rabi", "kharif"],
                        help="Override season from JSON.")
    parser.add_argument("--resolution", default=None, choices=["monthly", "dekadal"],
                        help="monthly -> MSTACNN (models/); dekadal -> S4D (models_dekadal/). "
                             "Default: models/ (primary MSTACNN).")
    parser.add_argument("--model-dir", default=None,
                        help="Parent models directory (overrides --resolution).")
    parser.add_argument("--plot", action="store_true",
                        help="Save temporal attention plot.")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir, args.resolution)
    output_dir = Path("inference/outputs/gee_predictions")

    if args.json:
        season, X = parse_gee_json(args.json)
        if args.season:
            season = args.season
        run_prediction(season, X, plot=args.plot, output_dir=output_dir,
                       model_dir=model_dir)

    elif args.json_file:
        with open(args.json_file) as f:
            raw = f.read()
        season, X = parse_gee_json(raw)
        if args.season:
            season = args.season
        run_prediction(season, X, plot=args.plot, output_dir=output_dir,
                       model_dir=model_dir)

    else:
        interactive_mode(season_override=args.season, plot=args.plot,
                         model_dir=model_dir)


if __name__ == "__main__":
    main()
