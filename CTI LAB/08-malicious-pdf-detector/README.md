# 08 · Malicious PDF Detection

Analyze PDF files for malicious characteristics such as embedded scripts and suspicious objects. Builds a classifier to identify malicious PDF documents.

## Files

| File | Purpose |
|---|---|
| `generate_pdf_data.py` | Generate synthetic PDF metadata (15K samples, 31 features) |
| `train_pdf_model.py` | Train 5+ classifiers with comparison |
| `report.py` | Generate HTML report |
| `data/pdf_features.csv` | Generated dataset |
| `results/` | Metrics, models, predictions, report |

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate PDF feature dataset
python run.py train      # Train malicious PDF classifier
python run.py report     # Generate HTML report
```

## Features (31 total)

Structure: num_objects, num_pages, file_size, compression_ratio, num_fonts, num_images, num_streams
Security: has_js, num_js_actions, has_openaction, has_launch_action, has_uri_action, has_embedded_file
Suspicious: num_suspicious_strings, num_encoded_streams, entropy metrics
Malformed: obfuscated_names, trailer_after_eof, overlapping_objects, malformed_crossref

## Models

LogisticRegression, RandomForest, GradientBoosting, SVM (RBF), XGBoost, VotingEnsemble

## Setup

```bash
cd 08-malicious-pdf-detector
pip install -r requirements.txt
python generate_pdf_data.py
python train_pdf_model.py
python report.py
```
