# 03 · Phishing URL Detection

Train and compare machine learning classifiers to identify phishing websites using URL and domain-based features. Features are extracted from raw URLs — no page content needed.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Build the phishing URL dataset
python run.py train      # Train and compare ML models
python run.py report     # Generate HTML report
python run.py classify   # Classify a URL (interactive CLI)
python run.py web        # Launch Flask web UI (port 5000)
```

## Files

| File | Purpose |
|---|---|
| `url_feature_extractor.py` | Extract 28 hand-crafted features from a raw URL |
| `dataset_builder.py` | Download phishing URLs + generate legitimate URLs, extract features, save dataset |
| `model_comparison.py` | Train 6+ classifiers, evaluate, save metrics and models |
| `phishing_classifier.py` | CLI tool: classify any URL as phishing or legitimate |
| `report.py` | Generate a self-contained HTML report with charts |
| `3.phishing_url_detection.ipynb` | Jupyter notebook walking through the full pipeline |
| `requirements.txt` | Python dependencies |
| `data/phishing_features.csv` | Extracted feature matrix (generated) |
| `results/` | Metrics, models, predictions, report (generated) |

## Setup

```bash
cd 03-phishing-url-detector
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Run

```bash
# 1. Build dataset (download phishing URLs, generate legitimate URLs, extract features)
python dataset_builder.py

# 2. Train and compare models
python model_comparison.py

# 3. Generate HTML report
python report.py

# 4. Classify a single URL
python phishing_classifier.py "https://example.com"

# Interactive mode
python phishing_classifier.py --interactive

# JSON output
python phishing_classifier.py --json "https://example.com"

# Use a specific model
python phishing_classifier.py --model RandomForest "https://..."
```

## Features Extracted (28 total)

- **Length-based:** URL length, path length, domain length, digit ratio, special char ratio
- **Structural:** dots, hyphens, slashes, @ symbols, subdomain count, path depth
- **Presence:** IP address, HTTPS, port, www, double-slash redirect
- **Content:** suspicious keywords, URL shortener, entropy, query params

## Models Trained

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble |
| Gradient Boosting | Boosted trees |
| SVM (RBF) | Kernel method |
| k-Nearest Neighbors | Distance-based |
| Gaussian Naive Bayes | Probabilistic |
| XGBoost (if installed) | Extreme gradient boosting |
| Voting Ensemble | Soft-vote of top-3 |

## Outputs

- `results/metrics.json` — Raw metrics for all models
- `results/metrics.csv` — Flat CSV comparison table
- `results/report.html` — Self-contained HTML report with charts
- `results/models/*.joblib` — Trained model pipelines
- `results/predictions/*.npz` — y_true, y_pred, y_prob per model
