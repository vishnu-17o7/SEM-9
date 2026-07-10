# 04 · Email Phishing Detection with NLP + ML

Analyze email content and headers to detect phishing attempts. Applies NLP and machine learning for classification.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Download phishing email dataset
python run.py train      # Train NLP phishing model
python run.py report     # Generate HTML report
python run.py classify   # Classify an email (interactive CLI)
python run.py web        # Launch web UI (port 5005)
```

## Files

| File | Purpose |
|---|---|
| `download_data.py` | Downloads SpamAssassin corpus + OpenML Spambase |
| `extract_features.py` | Header + body feature extraction (31 features) |
| `train_phishing_model.py` | Train 6+ classifiers, compare, save models |
| `report.py` | Generate self-contained HTML report |
| `classify.py` | CLI to classify any email text |
| `data/` | Datasets (generated) |
| `results/` | Metrics, models, predictions, report (generated) |

## Features (31 total)

- **Header:** recipients, Reply-To mismatch, Return-Path mismatch, missing headers, routing hops, priority flags, subject indicators
- **Body:** HTML ratio, URL analysis, suspicious TLDs, IP-based URLs, shorteners, suspicious keywords, entropy, forms, JavaScript

## Models

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble |
| Gradient Boosting | Boosted trees |
| SVM (RBF) | Kernel method |
| Gaussian Naive Bayes | Probabilistic |
| XGBoost (if installed) | Extreme gradient boosting |
| Voting Ensemble | Soft-vote of top-3 |

## Setup

```bash
cd 04-email-phishing-nlp
pip install -r requirements.txt
```

## Run

```bash
# 1. Download datasets (optional — spambase loads from OpenML)
python download_data.py

# 2. Train models
python train_phishing_model.py

# 3. Generate report
python report.py

# 4. Classify a single email
python classify.py "Your email text here"

# Interactive mode
python classify.py --interactive

# From file
python classify.py --file email.txt
```

## Outputs

- `results/metrics.json` — Full per-model metrics
- `results/metrics.csv` — CSV comparison table
- `results/report.html` — Self-contained HTML report
- `results/models/*.joblib` — Trained model pipelines
- `results/predictions/*.npz` — Predictions per model
