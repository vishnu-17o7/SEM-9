# 16 · Threat Intelligence Dashboard

Design a real-time threat intelligence dashboard by integrating multiple security feeds. Visualize threats, trends, and alerts to support proactive security monitoring.

## Quick Start

```bash
# Launch the dashboard
python run.py all
# Open http://127.0.0.1:5002
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask web UI with Chart.js visualizations |
| `templates/` | (Generated inline) |

## Features

- Integration with Project 14 (Repository DB) and Project 15 (Social Mining)
- Summary cards: total indicators, critical/high severity counts, active alerts
- Doughnut chart: indicators by type (IP, domain, URL, hash, CVE)
- Bar chart: severity distribution
- Polar area chart: top sources
- Real-time alert feed with severity badges
- Recent indicators table with scroll
- Dark theme UI

## Setup

```bash
cd 16-threat-intel-dashboard
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5002
```

Requires Projects 14 and 15 to be run first. Falls back to sample data if those databases are unavailable.
