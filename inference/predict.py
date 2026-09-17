"""
predict.py — Inference engine for trained MSTACNN crop type models.

Loads all 5 CV-fold models for a given season, ensembles their predictions
(soft voting on logits), and optionally returns temporal attention weights
for interpretability.

Supports:
  - Single-sample or batch prediction from numpy arrays
  - Prediction from a CSV file (rows = fields, columns = timestep features)
  - Confidence scores per class
  - Temporal attention heatmaps
  - JSON-serializable output for integration

Usage (standalone):
    python inference/predict.py --season rabi  --input data/processed/X_rabi.npy
    python inference/predict.py --season kharif --input my_new_fields.csv
    python inference/predict.py --season rabi  --input sample.npy --top-k 3

Usage (as module):
    from inference.predict import CropPredictor
    pred = CropPredictor(season="rabi")
    results = pred.predict(X)  # X: (N, 7, 28) numpy array
"""

from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import build_model, ATTENTION_MODELS
from config import SEASON_MONTH_LABELS, SEASON_DEKADAL_LABELS
from utils import plot_prediction_attention, plot_batch_summary


def _aps_score_all(probs: np.ndarray) -> np.ndarray:
    """APS conformal score for every (sample, class): cumulative probability mass
    of all classes at least as probable as c (inclusive). Returns (N, C)."""
    order = np.argsort(-probs, axis=1)
    csum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    ranks = np.argsort(order, axis=1)
    return np.take_along_axis(csum, ranks, axis=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ #
#  CropPredictor — main inference class
# ------------------------------------------------------------------ #

class CropPredictor:
    """Ensemble predictor for a single season's crop type classification.

    Loads all fold models, scaler, and metadata from the models/{season}/ dir.
    Predictions are ensemble-averaged across folds (soft voting on logits).
    """

    def __init__(
        self,
        season: str,
        model_dir: str | Path = None,
        device: str = "auto",
        folds: list[int] | None = None,
    ):
        """
        Args:
            season:    "rabi" or "kharif"
            model_dir: Path to models/ directory (default: auto-detect)
            device:    "cpu", "cuda", or "auto"
            folds:     Subset of folds to use (e.g. [1,3,5]). None = all.
        """
        self.season = season

        # Resolve model directory
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
        self.model_dir = Path(model_dir) / season

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Run training first: python run.py train --season {season}"
            )

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load inference metadata
        meta_path = self.model_dir / "inference_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"inference_meta.json not found in {self.model_dir}\n"
                f"Re-run training to generate it: python run.py train --season {season}"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.num_classes = self.meta["num_classes"]
        self.in_channels = self.meta["in_channels"]
        self.timesteps = self.meta["timesteps"]
        self.model_type = self.meta.get("model_type", "mstacnn")

        # Resolution-aware month labels: 21 timesteps = dekadal, else monthly.
        # Falls back to generic t0..tN if the season isn't recognised.
        label_table = (
            SEASON_DEKADAL_LABELS if self.timesteps == 21 else SEASON_MONTH_LABELS
        )
        self.month_labels = label_table.get(
            season, [f"t{i}" for i in range(self.timesteps)]
        )

        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scaler.pkl not found in {self.model_dir}\n"
                f"Re-run training to generate it."
            )
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load deployable trust layer (temperature + conformal threshold), if present.
        self.calib = None
        calib_path = self.model_dir / "calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                self.calib = json.load(f)

        # Load fold models
        self.models = []
        available_folds = sorted(self.model_dir.glob("best_model_fold*.pt"))
        if not available_folds:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")

        skipped = []
        for pt_path in available_folds:
            fold_num = int(pt_path.stem.replace("best_model_fold", ""))
            if folds is not None and fold_num not in folds:
                continue
            model = self._build_model()
            try:
                model.load_state_dict(
                    torch.load(pt_path, map_location=self.device, weights_only=True)
                )
            except RuntimeError as e:
                # A fold saved by a different run (different #classes/timesteps)
                # won't match the architecture defined by inference_meta.json.
                # Skip it rather than crash the whole ensemble.
                skipped.append((fold_num, str(e).splitlines()[0]))
                continue
            model.eval()
            self.models.append((fold_num, model))

        if skipped:
            print(f"[CropPredictor] WARNING: skipped {len(skipped)} fold(s) whose "
                  f"weights don't match inference_meta.json "
                  f"({self.model_type}, {self.num_classes} classes, "
                  f"T={self.timesteps}):")
            for fold_num, msg in skipped:
                print(f"    fold {fold_num}: {msg}")
            print("    -> Re-train this season to regenerate a consistent set of folds.")

        if not self.models:
            raise ValueError(
                f"No usable models loaded for season='{season}' from {self.model_dir}.\n"
                f"All {len(skipped)} fold(s) were incompatible with inference_meta.json. "
                f"Re-train this season: python run.py train --season {season}"
            )

        self.num_folds = len(self.models)
        print(f"[CropPredictor] Season={season}, {self.num_folds} folds loaded, "
              f"{self.num_classes} classes, device={self.device}")

    def _build_model(self):
        """Instantiate the correct architecture via the shared factory so it
        matches pipeline/train.py exactly (state_dicts load cleanly)."""
        return build_model(
            self.model_type, self.in_channels, self.num_classes
        ).to(self.device)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply the saved full-data scaler to input."""
        N, T, C = X.shape
        X_flat = X.reshape(-1, C)
        X_scaled = self.scaler.transform(X_flat).reshape(N, T, C)
        return X_scaled.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        return_attention: bool = True,
        return_probabilities: bool = True,
    ) -> dict:
        """Run ensemble inference on input samples.

        Args:
            X: Input array, shape (N, T, C) or (T, C) for single sample.
               T must match the loaded model (7 monthly / 21 dekadal),
               C=28 features. UN-NORMALISED (raw GEE values).
            return_attention: Include temporal attention weights (MSTACNN only).
            return_probabilities: Include per-class probability distribution.

        Returns:
            dict with keys:
                "predictions":   list of predicted class names
                "class_ids":     np.array of predicted class indices
                "confidence":    np.array of prediction confidence (max probability)
                "probabilities": np.array (N, num_classes) — per-class probabilities
                "attention":     np.array (N, T) — temporal attention weights
                "class_names":   list of all class names
                "months":        list of month labels
                "num_samples":   int
        """
        # Handle single sample
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        # Validate shape
        N, T, C = X.shape
        if T != self.timesteps:
            raise ValueError(f"Expected {self.timesteps} timesteps, got {T}")
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} features, got {C}")

        # Normalize
        X_norm = self._normalize(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        # Ensemble prediction (soft voting on logits)
        all_logits = []
        all_attn = []

        for fold_num, model in self.models:
            if self.model_type in ATTENTION_MODELS and return_attention:
                logits, attn_w = model(X_tensor, return_attention=True)
                all_attn.append(attn_w.cpu().numpy())
            else:
                logits = model(X_tensor)
            all_logits.append(logits.cpu().numpy())

        # Average logits across folds → softmax → predictions
        mean_logits = np.mean(all_logits, axis=0)  # (N, num_classes)
        # Softmax
        exp_logits = np.exp(mean_logits - mean_logits.max(axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        class_ids = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        predictions = [self.class_names[c] for c in class_ids]

        # Average attention across folds
        attention = None
        if all_attn:
            attention = np.mean(all_attn, axis=0)  # (N, T)

        result = {
            "predictions": predictions,
            "class_ids": class_ids,
            "confidence": confidence,
            "class_names": self.class_names,
            "months": self.month_labels,
            "num_samples": N,
            "logits": mean_logits,          # ensemble mean logits (for calibration)
        }
        if return_probabilities:
            result["probabilities"] = probabilities
        if return_attention and attention is not None:
            result["attention"] = attention

        # Trust layer: calibrated confidence + conformal prediction set + decision.
        if self.calib is not None:
            T = float(self.calib.get("temperature", 1.0))
            qhat = float(self.calib.get("qhat_robust",
                                        self.calib.get("qhat_marginal", 1.0)))
            cal_probs = _softmax(mean_logits / T)
            member = _aps_score_all(probabilities) <= qhat   # (N, C) set membership
            sizes = member.sum(axis=1)
            result["calibrated_confidence"] = cal_probs.max(axis=1)
            result["prediction_set"] = [
                [self.class_names[c] for c in np.where(member[i])[0]] for i in range(N)
            ]
            result["set_size"] = sizes
            result["decision"] = [
                "auto-classify" if s == 1 else
                ("verify (ambiguous)" if s > 1 else "verify (low-confidence)")
                for s in sizes
            ]

        return result

    def predict_csv(self, csv_path: str | Path, **kwargs) -> dict:
        """Predict from a formatted CSV file.

        The CSV should have shape (N_samples, T*C) or be a pivoted GEE CSV.
        See inference/prepare_input.py for converting GEE CSVs.
        """
        csv_path = Path(csv_path)
        if csv_path.suffix == ".npy":
            X = np.load(csv_path)
        else:
            df = pd.read_csv(csv_path)
            # Assume columns are flat features: T*C values per row
            X = df.values.astype(np.float32)
            X = X.reshape(len(X), self.timesteps, self.in_channels)
        return self.predict(X, **kwargs)

    def format_results(self, results: dict, top_k: int = 3) -> str:
        """Format prediction results as a human-readable string."""
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  CROP TYPE PREDICTIONS — {self.season.upper()} SEASON")
        lines.append(f"  Ensemble of {self.num_folds} models | {results['num_samples']} samples")
        lines.append(f"{'=' * 70}\n")

        for i in range(results["num_samples"]):
            lines.append(f"  Sample {i + 1}:")
            lines.append(f"    Predicted: {results['predictions'][i]}  "
                        f"(confidence: {results['confidence'][i]:.1%})")

            # Trust layer: calibrated confidence + conformal set + decision
            if "decision" in results:
                pset = results["prediction_set"][i]
                lines.append(f"    Calibrated confidence: "
                             f"{results['calibrated_confidence'][i]:.1%}")
                lines.append(f"    Conformal set ({int(results['set_size'][i])}): "
                             f"{', '.join(pset) if pset else '(empty)'}")
                lines.append(f"    Decision: {results['decision'][i]}")

            # Top-k probabilities
            if "probabilities" in results:
                probs = results["probabilities"][i]
                sorted_idx = np.argsort(probs)[::-1][:top_k]
                lines.append(f"    Top-{top_k} classes:")
                for rank, idx in enumerate(sorted_idx, 1):
                    lines.append(f"      {rank}. {self.class_names[idx]:<20s} {probs[idx]:.1%}")

            # Temporal attention
            if "attention" in results:
                attn = results["attention"][i]
                months = results["months"]
                peak_month = months[np.argmax(attn)]
                lines.append(f"    Peak attention: {peak_month} ({attn.max():.2f})")
                lines.append(f"    Attention: {' | '.join(f'{m}:{a:.2f}' for m, a in zip(months, attn))}")

            lines.append("")

        return "\n".join(lines)

    def save_results(self, results: dict, output_path: str | Path):
        """Save prediction results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(results["num_samples"]):
            row = {
                "sample_idx": i,
                "predicted_class": results["predictions"][i],
                "class_id": int(results["class_ids"][i]),
                "confidence": round(float(results["confidence"][i]), 4),
            }
            # Add per-class probabilities
            if "probabilities" in results:
                for j, name in enumerate(results["class_names"]):
                    row[f"prob_{name}"] = round(float(results["probabilities"][i, j]), 4)
            # Add attention per month
            if "attention" in results:
                for j, month in enumerate(results["months"]):
                    row[f"attn_{month}"] = round(float(results["attention"][i, j]), 4)
            records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# ------------------------------------------------------------------ #
#  CLI entrypoint
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Inference — Predict crop types using trained MSTACNN ensemble."
    )
    parser.add_argument("--season", required=True, choices=["rabi", "kharif"],
                        help="Season model to use.")
    parser.add_argument("--input", required=True,
                        help="Path to input data (.npy or .csv).")
    parser.add_argument("--model-dir", default=None,
                        help="Path to models/ directory (auto-detected).")
    parser.add_argument("--output", default=None,
                        help="Save results CSV to this path.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Show top-k predictions per sample.")
    parser.add_argument("--plot-attention", action="store_true",
                        help="Save attention plots for each sample.")
    parser.add_argument("--plot-summary", action="store_true",
                        help="Save batch summary plot.")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, or auto.")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subset of folds to use (e.g., 1 3 5).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to first N samples (for quick testing).")
    args = parser.parse_args()

    # Load predictor
    predictor = CropPredictor(
        season=args.season,
        model_dir=args.model_dir,
        device=args.device,
        folds=args.folds,
    )

    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        X = np.load(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
        # Drop non-feature columns if present
        drop_cols = [c for c in df.columns if c.lower() in
                     ("crop_type", "crop_id", "district", "field_id", "label")]
        if drop_cols:
            print(f"  Dropping non-feature columns: {drop_cols}")
            df = df.drop(columns=drop_cols)
        X = df.values.astype(np.float32)
        X = X.reshape(len(X), predictor.timesteps, predictor.in_channels)
    else:
        sys.exit(f"Unsupported input format: {input_path.suffix} (use .npy or .csv)")

    if args.max_samples:
        X = X[:args.max_samples]

    # Predict
    results = predictor.predict(X)

    # Display
    print(predictor.format_results(results, top_k=args.top_k))

    # Save results
    if args.output:
        predictor.save_results(results, args.output)

    # Plots
    output_dir = Path("inference/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_summary:
        plot_batch_summary(results, output_dir / f"batch_summary_{args.season}.png")

    if args.plot_attention:
        attn_dir = output_dir / "attention"
        attn_dir.mkdir(exist_ok=True)
        for i in range(min(results["num_samples"], 20)):  # limit to 20 plots
            plot_prediction_attention(
                results, i,
                attn_dir / f"sample_{i+1}_{results['predictions'][i]}.png"
            )


if __name__ == "__main__":
    main()
