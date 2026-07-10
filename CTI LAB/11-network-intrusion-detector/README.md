# 11 · Network Intrusion Detection

Capture and analyze network traffic data to identify unusual communication patterns. Apply machine learning algorithms to detect potential network intrusions using the NSL-KDD benchmark dataset.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Download NSL-KDD dataset
python run.py train      # Train NIDS model
python run.py report     # Generate HTML report
```

## Files

| File | Purpose |
|---|---|
| `download_nsl_kdd.py` | Download NSL-KDD dataset from GitHub mirror (synthetic fallback if unavailable) |
| `train_nids.py` | Train 5 classifiers with voting ensemble |
| `report.py` | Generate HTML report |
| `data/` | NSL-KDD processed CSV files |
| `results/` | Metrics, models, predictions, report |

## Models

LogisticRegression, RandomForest, GradientBoosting, KNN, XGBoost, VotingEnsemble

## Features (41 total)

Basic: duration, protocol_type, service, flag, src_bytes, dst_bytes
Content: hot, num_failed_logins, logged_in, num_compromised, root_shell
Traffic: count, srv_count, serror_rate, same_srv_rate, diff_srv_rate
Host-based: dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, etc.

## Setup

```bash
cd 11-network-intrusion-detector
pip install -r requirements.txt
python download_nsl_kdd.py
python train_nids.py
python report.py
```
