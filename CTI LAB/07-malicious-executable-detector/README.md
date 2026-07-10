# 07 · Malicious Executable Detection

Extract features from executable files and train ML models to distinguish between benign and malicious software.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate PE feature dataset
python run.py train      # Train malware detection model
python run.py report     # Generate HTML report
```

## Files

| File | Purpose |
|---|---|
| `generate_pe_features.py` | Generate synthetic PE feature vectors (2381-dim, simulating EMBER v2) |
| `train_malware_model.py` | Train 5+ classifiers with PCA dimensionality reduction |
| `report.py` | Generate HTML report |
| `data/pe_features.npz` | Generated feature matrix (compressed) |
| `results/` | Metrics, models, predictions, report |

## Models

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble (200 trees) |
| Gradient Boosting | Boosted trees |
| MLP | Neural network (128-64) |
| XGBoost (if installed) | Extreme gradient boosting |
| Voting Ensemble | Soft-vote of top-3 |

## Feature Space

Simulates EMBER v2 format (2381 features):
- Byte histogram (256), Byte entropy histogram (256)
- String info (10), General info (10)
- Section info (30), Imports (1280)
- Exports (128), Misc features (411)

PCA reduces to 100 components before training.

## Setup

```bash
cd 07-malicious-executable-detector
pip install -r requirements.txt
```

## Run

```bash
python generate_pe_features.py
python train_malware_model.py
python report.py
```
