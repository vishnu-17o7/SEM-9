# 09 · Ransomware Behavior Detection

Monitor system activities including file operations and registry modifications. Develop a machine learning model to detect ransomware behavior in Windows environments.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Simulate ransomware behavior data
python run.py train      # Train detection model
python run.py report     # Generate HTML report
```

## Files

| File | Purpose |
|---|---|
| `simulate_ransomware.py` | Generate synthetic system activity data (20K sessions, 29 features) |
| `train_ransomware_model.py` | Train 5+ classifiers with voting ensemble |
| `report.py` | Generate HTML report |
| `data/ransomware_activity.csv` | Generated dataset |
| `results/` | Metrics, models, predictions, report |

## Features (29 total)

File ops: num_files_accessed, modified, renamed, deleted, created
File types: doc_ratio, exe_ratio, compressed_ratio, encrypted_ratio
Registry: reads, writes, deletes, startup writes, run writes
Process: processes_created, terminated, network_connections, dns_queries
Timing: ops_per_second, file_ops_per_second, registry_ops_per_second
Entropy: avg_file_entropy, max_file_entropy, entropy_change
Other: file_ext_diversity, shadow_copy_deletions, backup_file_ops

## Models

LogisticRegression, RandomForest, GradientBoosting, SVM (RBF), XGBoost, VotingEnsemble

## Setup

```bash
cd 09-ransomware-behavior-detector
pip install -r requirements.txt
python simulate_ransomware.py
python train_ransomware_model.py
python report.py
```
