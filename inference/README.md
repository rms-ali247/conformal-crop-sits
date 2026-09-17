# Inference — Crop Type Prediction

Self-contained inference package for the trained crop type models. Loads the
5-fold ensemble, normalizes input, and returns predictions with confidence
scores and temporal attention weights. The engine is **model- and
timestep-agnostic** — it reads `inference_meta.json` and adapts to:

- **MSTACNN** (monthly, 7 timesteps) — in `models/` — *the default*
- **S4D** (dekadal, 21 timesteps) — in `models_dekadal/` — pass `--resolution dekadal`

> The MSTACNN checkpoints must be (re)generated first — see the root
> [README](../README.md#reproducing-the-models). Monthly tensors live in
> `data/processed_monthly/`.

---

## Quick Start

```bash
# 1. Run the demo to verify everything works
python run.py demo

# 2. Predict on your existing training data (monthly tensors → MSTACNN)
python inference/predict.py --season rabi  --input data/processed_monthly/X_rabi.npy --top-k 3
python inference/predict.py --season kharif --input data/processed_monthly/X_kharif.npy

# 3. Predict with attention plots
python inference/predict.py --season rabi --input data/processed_monthly/X_rabi.npy \
    --plot-attention --plot-summary --output inference/outputs/rabi_predictions.csv

# 4. Predict from GEE-extracted JSON
python run.py predict --json-file inference/inputs/gee_rabi_field.json --plot            # monthly / MSTACNN
python run.py predict --json-file my_dekadal_field.json --resolution dekadal --plot      # dekadal / S4D
```

---

## Folder Structure

```
inference/
├── __init__.py           # Package marker
├── predict.py            # Main inference engine (CropPredictor class + CLI)
├── predict_from_gee.py   # CLI: predict from GEE-exported JSON
├── prepare_input.py      # Input data preparation from various formats
├── demo.py               # End-to-end demo / validation script
├── README.md             # This file
├── inputs/               # Place input data files here (JSON, .npy, .csv)
└── outputs/              # Prediction results saved here
    ├── demo/             # Demo outputs
    └── attention/        # Per-sample attention plots
```

> **Note**: Model architectures are defined in the root `models.py`. Constants
> (bands, seasons, feature columns) are in `config.py`. Plotting helpers are
> in `utils.py`. The inference package imports from these shared modules.

---

## Usage Options

### Option A: Python API (recommended for integration)

```python
from inference.predict import CropPredictor

# Load the Rabi season model ensemble
predictor = CropPredictor(season="rabi")

# Predict on your data — shape must be (N, 7, 28) — UN-NORMALISED
import numpy as np
X = np.load("my_new_fields.npy")         # (N, 7, 28)
results = predictor.predict(X)

# Access results
print(results["predictions"])     # ["Wheat", "Maize", ...]
print(results["confidence"])      # [0.94, 0.87, ...]
print(results["probabilities"])   # (N, num_classes) array
print(results["attention"])       # (N, 7) temporal attention weights

# Pretty print
print(predictor.format_results(results, top_k=3))

# Save to CSV
predictor.save_results(results, "my_predictions.csv")
```

### Option B: Command Line

```bash
# Basic prediction (prints to console)
python inference/predict.py --season rabi --input my_data.npy

# Save results + plots
python inference/predict.py --season kharif --input data.npy \
    --output results.csv --plot-attention --plot-summary

# Use specific folds only (e.g., best-performing ones)
python inference/predict.py --season rabi --input data.npy --folds 1 3 5

# Limit samples for quick testing
python inference/predict.py --season rabi --input data.npy --max-samples 10

# Use GPU
python inference/predict.py --season rabi --input data.npy --device cuda
```

### Option C: Single sample prediction

```python
from inference.predict import CropPredictor

predictor = CropPredictor(season="kharif")

# Single field — shape (7, 28)
x = np.load("one_field.npy")   # or construct manually
results = predictor.predict(x)  # automatically adds batch dim

print(f"Predicted: {results['predictions'][0]}")
print(f"Confidence: {results['confidence'][0]:.1%}")
```

---

## Preparing Input Data

The model expects input shape **(N, 7, 28)**: N fields × 7 monthly timesteps × 28 features.

### Feature order (28 features per timestep):
```
B2_mean, B3_mean, B4_mean, B5_mean, B6_mean, B7_mean, B8_mean, B8A_mean,
B11_mean, B12_mean, NDVI_mean, EVI_mean, NDWI_mean, SAVI_mean,
B2_stdDev, B3_stdDev, B4_stdDev, B5_stdDev, B6_stdDev, B7_stdDev,
B8_stdDev, B8A_stdDev, B11_stdDev, B12_stdDev, NDVI_stdDev, EVI_stdDev,
NDWI_stdDev, SAVI_stdDev
```

### Timestep order:
- **Rabi**: Oct 2022, Nov 2022, Dec 2022, Jan 2023, Feb 2023, Mar 2023, Apr 2023
- **Kharif**: May 2023, Jun 2023, Jul 2023, Aug 2023, Sep 2023, Oct 2023, Nov 2023

### From new GEE exports:
```bash
# Same monthly CSV format as training data
python inference/prepare_input.py --mode gee --csv-dir path/to/new_csvs --season rabi \
    --output inference/inputs/new_rabi.npy
```

### From a wide CSV:
```bash
# 196 columns (7 × 28), one row per field
python inference/prepare_input.py --mode csv --csv-file my_fields.csv \
    --output inference/inputs/my_fields.npy
```

### Generate test data:
```bash
# Synthetic data for pipeline validation
python inference/prepare_input.py --mode synthetic --season rabi --num-samples 10 \
    --output inference/inputs/test_data.npy
```

---

## Understanding the Output

### Predictions CSV columns:
| Column | Description |
|--------|-------------|
| `sample_idx` | Sample index (0-based) |
| `predicted_class` | Predicted crop type name |
| `class_id` | Predicted class index |
| `confidence` | Max probability (0-1) |
| `prob_{CropName}` | Probability for each class |
| `attn_{Month}` | Temporal attention weight per month |

### Confidence interpretation:
- **> 90%**: High confidence — reliable prediction
- **70-90%**: Moderate confidence — likely correct
- **50-70%**: Low confidence — review manually
- **< 50%**: Very low confidence — prediction unreliable

### Temporal attention:
Higher attention weight = the model considers that month more discriminative.
For example, Feb-Mar peak attention for Wheat indicates the model focuses on
the heading/grain-fill growth stage for discrimination.

---

## Rabi Classes
| Class | Description |
|-------|-------------|
| Barseem | Forage crop, winter season |
| Maize | Corn, grown in Rabi in some regions |
| Rapeseed Mustard | Oilseed crop |
| Sugarcane | Perennial crop (persists across seasons) |
| Wheat | Dominant Rabi cereal |

## Kharif Classes
| Class | Description |
|-------|-------------|
| Cotton | Cash crop |
| Jantar | Local grain crop |
| Maize | Summer corn |
| Rice | Paddy rice |
| Sesame (Til) | Oilseed |
| Sorghum | Grain/fodder |
| Sugarcane | Perennial |

---

## Troubleshooting

**"inference_meta.json not found"**  
Re-run training: `python run.py train` — the training script saves inference artifacts.

**"scaler.pkl not found"**  
Same fix — re-run training with `python run.py train`.

**Shape mismatch errors**  
Ensure input is (N, 7, 28). Check feature order matches the 28-column spec above.

**Low confidence on all predictions**  
The model was trained on specific crop types in Pakistan's Punjab region.
Data from different regions/years may have different spectral signatures.

**Using with new GEE data**  
Make sure the GEE export uses the same: band selection (B2-B12), cloud masking (SCL),
composite method (monthly median), and reducer (mean+stdDev per polygon).
