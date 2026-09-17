"""
demo.py — Quick demo / end-to-end test of the inference pipeline.

Run this after training to verify everything works:
    python run.py demo
    python inference/demo.py

Tests:
  1. Synthetic data prediction (no real data needed)
  2. Training data prediction (validates model loading)
  3. Single-sample prediction with attention visualization
"""

from pathlib import Path
import sys

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from inference.predict import CropPredictor
from inference.prepare_input import generate_synthetic_sample
from utils import plot_prediction_attention, plot_batch_summary
# Monthly tensors match the primary MSTACNN models in models/.
from config import PROCESSED_MONTHLY_DIR as PROCESSED_DIR, MODELS_DIR


def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_synthetic(season: str):
    """Test with synthetic data — verifies model loads and runs."""
    separator(f"TEST 1: Synthetic Data — {season.upper()}")

    predictor = CropPredictor(season=season)
    X = generate_synthetic_sample(season, num_samples=5)

    results = predictor.predict(X)
    print(predictor.format_results(results, top_k=3))

    out_dir = Path("inference/outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor.save_results(results, out_dir / f"synthetic_{season}.csv")
    plot_batch_summary(results, out_dir / f"synthetic_{season}_summary.png")


def demo_training_data(season: str):
    """Test on actual training data — should reproduce high accuracy."""
    separator(f"TEST 2: Training Data — {season.upper()}")

    data_path = PROCESSED_DIR / f"X_{season}.npy"
    label_path = PROCESSED_DIR / f"y_{season}.npy"

    if not data_path.exists():
        print(f"  SKIP: {data_path} not found (run preprocessing first)")
        return

    predictor = CropPredictor(season=season)

    X = np.load(data_path)
    y_true = np.load(label_path)

    X_sample = X[:10]
    y_sample = y_true[:10]

    results = predictor.predict(X_sample)
    print(predictor.format_results(results, top_k=3))

    # Compare with true labels
    import pandas as pd
    lm = pd.read_csv(PROCESSED_DIR / "label_map.csv")
    id_col = "label" if "label" in lm.columns else "class_id"
    label_map = {row[id_col]: row["crop_type"] for _, row in lm.iterrows()}

    print("  Comparison with true labels:")
    correct = 0
    for i in range(len(X_sample)):
        true_name = label_map.get(y_sample[i], f"class_{y_sample[i]}")
        pred_name = results["predictions"][i]
        conf = results["confidence"][i]
        match = "✓" if true_name == pred_name else "✗"
        if true_name == pred_name:
            correct += 1
        print(f"    Sample {i+1}: True={true_name:<20s} Pred={pred_name:<20s} "
              f"Conf={conf:.1%} {match}")
    print(f"  Accuracy: {correct}/{len(X_sample)}")


def demo_single_sample(season: str):
    """Test single-sample prediction with attention plot."""
    separator(f"TEST 3: Single Sample with Attention — {season.upper()}")

    data_path = PROCESSED_DIR / f"X_{season}.npy"
    if not data_path.exists():
        print(f"  SKIP: {data_path} not found")
        return

    predictor = CropPredictor(season=season)
    X = np.load(data_path)
    x_single = X[0]

    results = predictor.predict(x_single)
    print(predictor.format_results(results, top_k=predictor.num_classes))

    out_dir = Path("inference/outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_prediction_attention(
        results, 0,
        out_dir / f"single_sample_{season}_attention.png"
    )


def main():
    separator("INFERENCE PIPELINE DEMO")
    print("This demo validates the inference pipeline works correctly.\n")

    available = []
    for season in ["rabi", "kharif"]:
        meta_path = MODELS_DIR / season / "inference_meta.json"
        if meta_path.exists():
            available.append(season)

    if not available:
        print("ERROR: No trained models found!")
        print("Run training first: python run.py train")
        sys.exit(1)

    print(f"Available seasons: {available}\n")

    for season in available:
        try:
            demo_synthetic(season)
        except Exception as e:
            print(f"  ERROR in synthetic demo ({season}): {e}")

        try:
            demo_training_data(season)
        except Exception as e:
            print(f"  ERROR in training data demo ({season}): {e}")

        try:
            demo_single_sample(season)
        except Exception as e:
            print(f"  ERROR in single sample demo ({season}): {e}")

    separator("DEMO COMPLETE")
    print("Check inference/outputs/demo/ for saved files.\n")


if __name__ == "__main__":
    main()
