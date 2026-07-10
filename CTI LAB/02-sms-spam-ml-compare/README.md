# 02 · SMS Spam — Multi-Model Comparison

Train and compare **6 base classifiers + 2 ensembles** for SMS spam detection on the UCI SMS Spam Collection. Generates a self-contained HTML report.

## Quick Start

```bash
# Run the full pipeline (train + report)
python run.py all

# Or run individual steps
python run.py train      # Train and compare 8 models
python run.py report     # Generate HTML report
python run.py web        # Launch web UI (port 5004)
```

## Models

| Family | Model | Notes |
| --- | --- | --- |
| base | Multinomial Naive Bayes | classic baseline for bag-of-words text |
| base | Complement Naive Bayes | designed for imbalanced text |
| base | Logistic Regression | strong linear baseline with calibrated probs |
| base | Linear SVC | high-margin linear separator |
| base | Random Forest | 300 trees, parallel |
| base | Gradient Boosting | sequential boosting |
| ensemble | Soft Voting | NB + CNB + LR + calibrated LinearSVC |
| ensemble | Stacking | NB + CNB + LR + GBDT → LR meta-learner (5-fold CV) |

All pipelines share the same TF-IDF (1–2 grams, English stop words, max 10k features).

## Files

| File | Purpose |
| --- | --- |
| `sms_preprocessing.py` | UCI download, text cleaner, stratified train/test split |
| `model_comparison.py` | Trains every model, logs metrics, saves predictions to `results/predictions/` |
| `report.py` | Renders `results/report.html` — sortable table + confusion matrices |
| `requirements.txt` | Pinned runtime deps |

## Setup

```bash
cd 02-sms-spam-ml-compare
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Train and score all 8 models
python model_comparison.py

# 2. Render the HTML report
python report.py
```

Then open `results/report.html` in any browser. Click any column header to sort.

## Outputs

- `results/metrics.json` — full per-model metrics (raw, machine-readable)
- `results/metrics.csv` — flat table for diffing in Excel
- `results/predictions/<model>.npz` — `y_true`, `y_pred`, `y_prob` per model
- `results/report.html` — self-contained report (no CDN, no JS deps)
