# CTI Lab — 18 Cyber Threat Intelligence Projects

A collection of 18 cyber threat intelligence projects covering spam detection, phishing, malware analysis, ransomware detection, network intrusion, threat intelligence repositories, and biometric authentication.

## Quick Start

Each project has a unified `run.py` entry point. From the SEM 9 root directory:

```bash
# Run the full pipeline for any project
.venv\Scripts\python.exe "CTI LAB\<project>\run.py" all

# Or from inside a project directory
python run.py all
```

## Available Commands

| Command | Description |
|---------|-------------|
| `data` | Generate or download the dataset |
| `train` | Train ML models |
| `report` | Generate the HTML report |
| `classify` | Classify input (interactive CLI) |
| `web` | Launch the web UI (if available) |
| `all` | Run the full pipeline (data + train + report) |
| `steps` | List available commands for this project |

```bash
# Example: run individual steps
python run.py data
python run.py train
python run.py report

# Example: launch a web UI
python run.py web
```

## Projects

| # | Title | Web UI | Port | Pipeline |
|---|-------|:------:|------|----------|
| 01 | Spam/Ham Watcher | Yes | 8000 | train |
| 02 | SMS Spam ML Compare | Yes | 5004 | train + report |
| 03 | Phishing URL Detector | Yes | 5000 | data + train + report |
| 04 | Email Phishing NLP | Yes | 5005 | data + train + report |
| 05 | Credential Stuffing Detector | Yes | 5006 | data + train + report |
| 06 | Behavioral Profile / UEBA | Yes | 5001 | data + train + report |
| 07 | Malicious Executable Detector | No | — | data + train + report |
| 08 | Malicious PDF Detector | No | — | data + train + report |
| 09 | Ransomware Behavior Detector | No | — | data + train + report |
| 10 | Crypto-Ransomware Detector | No | — | data + train + report |
| 11 | Network Intrusion Detector | No | — | data + train + report |
| 12 | Botnet Traffic Classifier | Yes | 5007 | data + train + report |
| 13 | User Activity Anomaly Detector | No | — | data + train + report |
| 14 | Threat Intel Repository | Yes | 5008 | data |
| 15 | Social Media Mining | No | — | mine |
| 16 | Threat Intel Dashboard | Yes | 5002 | web only |
| 17 | Facial Recognition Auth | No | — | data + train |
| 18 | Multi-Factor Auth (Face + OTP) | Yes | 5003 | data |

## Web UIs

10 projects have web UIs. Launch them with:

```bash
python run.py web
```

Then open `http://127.0.0.1:<port>` in your browser.

| Project | Port | Description |
|---------|------|-------------|
| 01 Spam/Ham Watcher | 8000 | FastAPI dashboard with live classification stats |
| 02 SMS Spam Classifier | 5004 | Classify SMS text as spam or ham |
| 03 Phishing URL Detector | 5000 | Classify URLs as phishing or legitimate |
| 04 Email Phishing Detector | 5005 | Classify email text as phishing or legitimate |
| 05 Credential Stuffing Detector | 5006 | Detect credential stuffing from login events |
| 06 UEBA Dashboard | 5001 | User behavior analytics with charts and alerts |
| 12 Botnet Traffic Classifier | 5007 | Classify network flows as botnet or normal |
| 14 Threat Intel Repository | 5008 | Search and browse IoC database |
| 16 Threat Intel Dashboard | 5002 | Real-time threat intelligence visualization |
| 18 Multi-Factor Auth | 5003 | Face + OTP authentication with audit log |

## Running via the Hub

The FastAPI hub dashboard provides a web interface for all 18 projects:

```bash
.venv\Scripts\python.exe run.py
# Open http://127.0.0.1:9000
```

Each project page has a "Run All" button that executes `run.py all` with live SSE output streaming. Web UI projects also show an "Open Web UI" button.

## Dependencies

All projects share the root virtual environment at `.venv\`. Install dependencies:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Individual projects may have additional `requirements.txt` files for project-specific dependencies.
