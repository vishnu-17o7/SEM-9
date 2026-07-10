# 13 · User Activity Anomaly Detection

Analyze user and system activity logs to establish normal behavioral baselines. Detects deviations that may indicate malicious or unauthorized activities.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate user activity data
python run.py train      # Run anomaly detection
python run.py report     # Generate HTML report
```

## Files

| File | Purpose |
|---|---|
| `generate_activity_data.py` | Generate synthetic command-line session logs (100 users, 45 days) |
| `detect_anomalies.py` | Feature engineering + 3 anomaly detectors (IsolationForest, PCA+Mahalanobis, Autoencoder) |
| `report.py` | Generate HTML report |
| `data/user_activity_logs.csv` | Generated session logs |
| `results/` | Metrics, models, predictions, report |

## Features (10 per-session)

cmd_count, unique_cmds, avg_duration, max_duration, std_duration, risky_cmd_ratio, exit_code_errors, session_duration_h, user_avg_cmds_per_hour, user_std_cmds_per_hour

## Anomaly Types Injected

- Off-hours access
- Data exfiltration (scp, curl, wget)
- Privilege abuse (sudo, su, chmod)
- Mass deletion (rm, find)
- Sensitive resource access (/etc/passwd, /etc/shadow)

## Setup

```bash
cd 13-user-activity-anomaly-detector
pip install -r requirements.txt
python generate_activity_data.py
python detect_anomalies.py
python report.py
```
