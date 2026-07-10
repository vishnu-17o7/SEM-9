# 10 · Crypto-Ransomware Detection

Analyze file encryption patterns and system resource usage to identify crypto-ransomware attacks. Implements anomaly detection techniques for early threat detection.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Simulate crypto-ransomware data
python run.py train      # Train crypto-ransomware detector
python run.py report     # Generate HTML report
```

## Files

| File | Purpose |
|---|---|
| `simulate_crypto_ransomware.py` | Generate synthetic encryption + resource usage data |
| `train_crypto_model.py` | Train DecisionTree, RandomForest, GradientBoosting, LogisticRegression, XGBoost |
| `report.py` | Generate HTML report |
| `data/crypto_ransomware.csv` | Generated dataset (15K samples, 32 features) |
| `results/` | Metrics, models, predictions, report |

## Features (32 total)

File I/O: files_read_per_sec, files_written_per_sec, write_read_ratio, bytes_written
Entropy: avg_entropy_before/after, entropy_delta, max_entropy_delta
CPU/Memory: cpu_usage_pct, memory_usage_mb, cpu_std, memory_std, cpu_spike_freq
Encryption: encryption_calls, decryption_calls, key_operations, file_rename_rate
Suspicious: c2_connections, data_exfil_bytes, num_crypto_procs, powershell_spawns
Headers: file_header_changes, known_headers_ratio, unexpected_headers

## Setup

```bash
cd 10-crypto-ransomware-detector
pip install -r requirements.txt
python simulate_crypto_ransomware.py
python train_crypto_model.py
python report.py
```
