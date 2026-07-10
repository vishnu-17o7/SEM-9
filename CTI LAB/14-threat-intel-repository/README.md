# 14 · Threat Intelligence Repository

Collect threat information from multiple cybersecurity sources and store it in a centralized SQLite repository. Process and organize threat indicators for analysis and reporting.

## Quick Start

```bash
# Run the full pipeline (build repository)
python run.py all

# Or run individual steps
python run.py data       # Build and populate the repository
python run.py query      # Query the repository (CLI)
python run.py web        # Launch web UI (port 5008)
```

## Files

| File | Purpose |
|---|---|
| `build_repository.py` | Initialize DB, insert sample IoCs, fetch from URLhaus, export to CSV/JSON |
| `cli_query.py` | CLI query tool (by type, severity, keyword, add IoCs, stats) |
| `data/threat_intel.db` | SQLite repository |
| `results/all_indicators.csv` | CSV export of all indicators |
| `results/all_indicators.json` | JSON export of all indicators |

## IoC Types Stored

- IP addresses (malicious C2, scanners)
- Domains (malware distribution, phishing)
- URLs (payload delivery, phishing pages)
- File hashes (MD5 of known malware)
- CVE IDs (known vulnerabilities)

## Sources

- URLhaus API (abuse.ch) — live fetch
- Sample bundled threat data (fallback)
- Manual entry via CLI

## Setup

```bash
cd 14-threat-intel-repository
pip install -r requirements.txt
python build_repository.py
python cli_query.py --stats
python cli_query.py --type ip
python cli_query.py --severity critical
python cli_query.py --add "ip" "1.2.3.4" --source "my_feed" --severity high
```
