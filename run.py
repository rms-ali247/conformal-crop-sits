"""
run.py — Master entry point for the crop type mapping pipeline.

Usage:
    python run.py clean              Stage 1: Clean shapefile
    python run.py extract            Stage 2: GEE Sentinel-2 extraction
    python run.py preprocess         Stage 3: Preprocess time series
    python run.py train              Stage 4: Train models
    python run.py conformal          Stage 5: Conformal calibration and coverage
    python run.py province           Leave-one-province-out transfer
    python run.py phenology          Phenology-offset stress test
    python run.py predict            Run prediction from GEE JSON
    python run.py demo               Run inference demo/tests

Pass extra arguments through to any stage:
    python run.py train --model mstacnn --season rabi
    python run.py predict --json-file inference/inputs/gee_rabi_field.json --plot
"""

import sys
import importlib

STAGES = {
    "clean":              "pipeline.clean_shapefile",
    "extract":            "pipeline.gee_extract",
    "preprocess":         "pipeline.preprocess",
    "clean-labels":       "pipeline.clean_labels",
    "train":              "pipeline.train",
    "evaluate":           "evaluation.evaluate_model",
    "conformal":          "evaluation.conformal",
    "province":           "evaluation.province_transfer",
    "phenology":          "evaluation.phenology_shift",
    "build-calibrator":   "evaluation.build_calibrator",
    "predict":            "inference.predict_from_gee",
    "demo":               "inference.demo",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Available stages:")
        for name, module in STAGES.items():
            print(f"  {name:<15s} -> {module}")
        return

    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"Unknown stage: '{stage}'")
        print(f"Available: {', '.join(STAGES.keys())}")
        sys.exit(1)

    # Remove 'run.py' and stage name from argv, keep remaining args
    sys.argv = [STAGES[stage]] + sys.argv[2:]

    module = importlib.import_module(STAGES[stage])
    module.main()


if __name__ == "__main__":
    main()
