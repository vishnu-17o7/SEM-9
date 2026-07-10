# 06 · Behavioral Profile — UEBA

Create behavioral profiles based on user login and access patterns. Detects unusual activities that may indicate account compromise or insider threats.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate synthetic behavior data
python run.py train      # Build profiles and train detectors
python run.py report     # Generate HTML report
python run.py web        # Launch Flask dashboard (port 5001)
```

## Files

| File | Purpose |
|---|---|
| `generate_behavior_data.py` | Generate enterprise activity logs with 50 users across 8 roles |
| `build_profiles.py` | Feature engineering + 4 anomaly detectors (GMM, PCA+Mahalanobis, IsolationForest, Autoencoder) |
| `report.py` | Generate HTML report |
| `data/user_activity.csv` | Generated activity logs |
| `results/` | Metrics, models, predictions, report (generated) |

## Detectors

| Detector | Method |
|---|---|
| GMM | Gaussian Mixture Model (3 components) |
| PCA + Mahalanobis | Dimensionality reduction + robust covariance distance |
| IsolationForest | Tree-based isolation |
| Autoencoder | MLP reconstruction error |

## Features (17 total)

Per-window: action_count, unique_resources, file_downloads, db_queries, privilege_changes, failure_rate
Per-user: total_actions, avg_hour, resource_diversity, off_hours_ratio
Deviation: action_deviation, resource_deviation, off_hours_deviation

## Anomaly Types Injected

- Off-hours access (3 AM logins)
- Mass file downloads (50-200 in 10 min)
- Role violations (accessing forbidden resources)
- Privilege escalation attempts
- Geo-velocity (logins from multiple countries)

## Setup

```bash
cd 06-behavioral-profile-ueba
pip install -r requirements.txt
```

## Run

```bash
python generate_behavior_data.py
python build_profiles.py
python report.py
```
