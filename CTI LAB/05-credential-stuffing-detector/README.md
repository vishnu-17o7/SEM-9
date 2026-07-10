# 05 · Credential Stuffing Detection

Detect credential stuffing attacks from login activity logs using anomaly detection techniques and threshold-based rules.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate synthetic login logs
python run.py train      # Train detection models
python run.py report     # Generate HTML report
python run.py classify   # Detect credential stuffing (CLI)
python run.py web        # Launch web UI (port 5006)
```

## Files

| File | Purpose |
|---|---|
| `generate_login_logs.py` | Generate synthetic login logs with normal + attack patterns |
| `detect_stuffing.py` | Feature engineering + 4 detectors (IsolationForest, OneClassSVM, LOF, rules) |
| `report.py` | Generate HTML report with metrics and confusion matrices |
| `clidect.py` | CLI for real-time single-log or batch detection |
| `data/login_logs.csv` | Generated login logs |
| `results/` | Metrics, models, predictions, report (generated) |

## Detectors

| Detector | Type |
|---|---|
| Rule-Based | Threshold rules (failure_rate > 0.7, unique_ips > 3, attempts/s > 2) |
| IsolationForest | Tree-based anomaly detection |
| OneClassSVM | One-class classification (RBF kernel) |
| LocalOutlierFactor | Density-based outlier detection |

## Features (13 total)

Time-windowed (5-min): login_count, failure_rate, unique_ips, ip_entropy, unique_uas, unique_users, unique_countries
Per-user: user_failure_rate, user_unique_ips, user_unique_countries
Velocity: time_since_last, attempts_per_second

## Setup

```bash
cd 05-credential-stuffing-detector
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate synthetic login logs
python generate_login_logs.py

# 2. Train detectors
python detect_stuffing.py

# 3. Generate report
python report.py

# 4. CLI detection
python clidect.py --log "username=alice,timestamp=2026-06-22T10:00:00,source_ip=1.2.3.4,success=0"
python clidect.py --file data/login_logs.csv
```

## Outputs

- `results/metrics.json` — Per-detector metrics
- `results/report.html` — Self-contained HTML report
- `results/models/*.joblib` — Trained detectors
