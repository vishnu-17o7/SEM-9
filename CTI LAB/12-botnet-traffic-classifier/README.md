# 12 · Botnet Traffic Classification

Examine network flow characteristics and communication behaviors of infected hosts. Trains a model to classify normal and botnet-generated traffic.

## Quick Start

```bash
# Run the full pipeline (data + train + report)
python run.py all

# Or run individual steps
python run.py data       # Generate botnet traffic data
python run.py train      # Train botnet classifier
python run.py report     # Generate HTML report
python run.py classify   # Classify a single network flow (CLI)
python run.py web        # Launch web UI (port 5007)
```

## Files

| File | Purpose |
|---|---|
| `generate_botnet_traffic.py` | Generate synthetic network flows (20K samples, 32 features) |
| `train_botnet_model.py` | Train 6+ classifiers (LR, RF, GBDT, SVM, MLP, XGBoost, Ensemble) |
| `report.py` | Generate HTML report |
| `classify_flow.py` | CLI to classify a single network flow |
| `data/botnet_traffic.csv` | Generated dataset |
| `results/` | Metrics, models, predictions, report |

## Features (32 total)

Flow: duration, protocol, src/dst ports/bytes, packet counts, packet rate, byte rate
Timing: inter_arrival_mean/std/min/max, flow_duration_ms
TCP flags: tcp_flag_count, syn_count, rst_count, fin_count, ack_count
Payload: small/large_packets_ratio, payload_bytes_mean/std
Behavior: dns_query_count, http_request_count, irc_message_count
Anomaly: c2_communication_score, connection_entropy, num_distinct_ports

## Setup

```bash
cd 12-botnet-traffic-classifier
pip install -r requirements.txt
python generate_botnet_traffic.py
python train_botnet_model.py
python report.py
python classify_flow.py --flow "src_bytes=500,dst_bytes=200,packets_sent=15,..."
```
