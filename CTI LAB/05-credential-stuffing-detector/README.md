# 05 · Credential Stuffing Detection

Detect web brute-force traffic as a credential-stuffing proxy using a documented public-source data pipeline, anomaly detection, and threshold-based rules.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Download and prepare CICIDS2017 web brute-force events
python run.py train      # Train detection models
python run.py report     # Generate HTML report
python run.py classify   # Detect credential stuffing (CLI)
python run.py web        # Launch web UI (port 5006)
```

## Files

| File | Purpose |
|---|---|
| `prepare_real_data.py` | Downloads CICIDS2017 and normalizes a registered LANL auth-log archive |
| `detect_stuffing.py` | Feature engineering + 4 detectors (IsolationForest, RandomForest, LOF, rules) |
| `report.py` | Generate HTML report with metrics and confusion matrices |
| `clidect.py` | CLI for real-time single-log or batch detection |
| `data/real/cicids2017_web_bruteforce_flows.csv` | Prepared labelled CICIDS2017 native flow records |
| `results/` | Metrics, models, predictions, report (generated) |

## Detectors

| Detector | Type |
|---|---|
| Rule-Based | Flow thresholds using SYN count, packet rate, and forward packet count |
| IsolationForest | Tree-based anomaly detection |
| OneClassSVM | One-class classification (RBF kernel) |
| LocalOutlierFactor | Density-based outlier detection |

## Features (12 total)

destination_port, flow_duration, forward/backward packet and byte counts, flow bytes/s, flow packets/s, SYN/ACK counts, and forward/backward inter-arrival-time means.

## Setup

```bash
cd 05-credential-stuffing-detector
pip install -r requirements.txt
```

## Run

```bash
# 1. Download and prepare public-source data
python prepare_real_data.py

# Optional: import a LANL daily host-event archive after registering on the
# official source site and downloading it yourself.
python prepare_real_data.py --lanl-file path/to/wls_day-01.bz2

# 2. Train detectors
python detect_stuffing.py

# 3. Generate report
python report.py

# 4. CLI detection (all 12 native flow fields are required)
python clidect.py --file data/real/cicids2017_web_bruteforce_flows.csv
```

## Outputs

- `results/metrics.json` — Per-detector metrics
- `results/report.html` — Self-contained HTML report
- `results/models/*.joblib` — Trained detectors

## Dataset scope

- **CICIDS2017** supplies labelled network flows for the Web Attack - Brute Force scenario. The trained web UI accepts the same native flow features as the benchmark model.
- **LANL Unified Host and Network Dataset (2017)** can provide a de-identified enterprise authentication baseline after its official registration step. It contains host authentication events but does not label credential stuffing.
- The generated report records exact local counts, acquisition status, source URLs, feature mapping, and the evaluation limitation: benchmark scores are not claims of production credential-stuffing accuracy.
