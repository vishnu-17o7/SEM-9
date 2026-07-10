"""
Real-time Threat Intelligence Dashboard — Flask Web UI.

Integrates with Project 14 (Repository) and 15 (Social Mining).
Displays threats, trends, and alerts for proactive security monitoring.
"""

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
REPO_DB = BASE_DIR.parent / "14-threat-intel-repository" / "data" / "threat_intel.db"
SOCIAL_DATA = BASE_DIR.parent / "15-threat-intel-social-mining" / "results" / "mined_iocs.json"

# In-memory cache
_cache = {"indicators": [], "alerts": [], "last_update": None}


def load_indicators():
    """Load indicators from repository DB and social mining results."""
    indicators = []

    # From repository
    if REPO_DB.exists():
        conn = None
        try:
            conn = sqlite3.connect(str(REPO_DB))
            df = pd.read_sql("SELECT ioc_type, value, source, confidence, severity, tags, first_seen FROM indicators ORDER BY first_seen DESC", conn)
        except Exception:
            pass
        else:
            for _, row in df.iterrows():
                indicators.append({
                    "type": row["ioc_type"], "value": row["value"],
                    "source": row["source"], "confidence": row["confidence"],
                    "severity": row["severity"], "tags": row["tags"],
                    "first_seen": row["first_seen"],
                })
        finally:
            if conn is not None:
                conn.close()

    # From social mining
    if SOCIAL_DATA.exists():
        try:
            with open(SOCIAL_DATA) as f:
                data = json.load(f)
            for ioc in data.get("iocs", []):
                indicators.append({
                    "type": ioc.get("ioc_type", "unknown"),
                    "value": ioc.get("value", ""),
                    "source": ioc.get("source", "social"),
                    "confidence": 0.7,
                    "severity": "medium",
                    "tags": "social-mined",
                    "first_seen": ioc.get("extracted_at", ""),
                })
        except Exception:
            pass

    # If empty, use built-in sample data
    if not indicators:
        indicators = _sample_indicators()

    # Deduplicate
    seen = set()
    unique = []
    for i in indicators:
        key = (i["type"], i["value"])
        if key not in seen:
            seen.add(key)
            unique.append(i)

    return unique


def _sample_indicators():
    """Fallback sample data if no sources available."""
    types = ["ip", "domain", "url", "hash", "cve"]
    severities = ["low", "medium", "high", "critical"]
    samples = []
    for i in range(100):
        t = random.choice(types)
        sev = random.choices(severities, weights=[0.2, 0.4, 0.3, 0.1])[0]
        dt = (datetime.utcnow() - timedelta(hours=random.randint(0, 168))).strftime("%Y-%m-%dT%H:%M:%SZ")
        if t == "ip":
            v = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        elif t == "domain":
            v = random.choice(["evil", "phish", "malware", "botnet", "ransom"]) + f"{i}.xyz"
        elif t == "url":
            v = f"http://malware{i}.xyz/payload.exe"
        elif t == "hash":
            v = "".join(random.choices("0123456789abcdef", k=32))
        else:
            v = f"CVE-2024-{random.randint(1000,9999)}"
        samples.append({
            "type": t, "value": v, "source": random.choice(["urlhaus", "social", "nvd", "sample"]),
            "confidence": round(random.uniform(0.5, 1.0), 2), "severity": sev,
            "tags": random.choice(["malware", "phishing", "ransomware", "c2", "exploit"]),
            "first_seen": dt,
        })
    return samples


def generate_alerts(indicators):
    """Generate simulated real-time alerts from indicator data."""
    alerts = []
    for ioc in indicators[:20]:
        if random.random() < 0.3:
            continue
        sev = ioc["severity"]
        ts = (datetime.utcnow() - timedelta(minutes=random.randint(0, 1440))).strftime("%H:%M:%S")
        msg = f"New {ioc['type']} indicator detected: {ioc['value']}"
        if ioc.get("tags"):
            msg += f" ({ioc['tags']})"
        alerts.append({
            "time": ts, "severity": sev,
            "type": ioc["type"], "message": msg,
            "source": ioc.get("source", "unknown"),
        })
    return sorted(alerts, key=lambda x: x["time"], reverse=True)[:50]


def refresh_cache():
    """Refresh the in-memory cache."""
    _cache["indicators"] = load_indicators()
    _cache["alerts"] = generate_alerts(_cache["indicators"])
    _cache["last_update"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@app.route("/")
def index():
    if not _cache["last_update"]:
        refresh_cache()

    inds = _cache["indicators"]
    alerts = _cache["alerts"]

    # Compute stats
    type_counts = pd.Series([i["type"] for i in inds]).value_counts().to_dict()
    sev_counts = pd.Series([i["severity"] for i in inds]).value_counts().to_dict()
    src_counts = pd.Series([i.get("source", "unknown") for i in inds]).value_counts().head(8).to_dict()

    recent = inds[:20]

    return render_template_string(HTML_TEMPLATE,
        total_indicators=len(inds),
        type_counts=type_counts,
        sev_counts=sev_counts,
        src_counts=src_counts,
        type_counts_json=json.dumps(type_counts),
        sev_counts_json=json.dumps(sev_counts),
        src_counts_json=json.dumps(src_counts),
        alerts=alerts[:15],
        recent=recent,
        last_update=_cache["last_update"],
    )


@app.route("/api/data")
def api_data():
    if not _cache["last_update"]:
        refresh_cache()
    return jsonify({
        "indicators": _cache["indicators"][:100],
        "alerts": _cache["alerts"][:30],
        "total": len(_cache["indicators"]),
        "last_update": _cache["last_update"],
    })


@app.route("/api/refresh")
def api_refresh():
    refresh_cache()
    return jsonify({"status": "ok", "last_update": _cache["last_update"]})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Threat Intelligence Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:1.5rem}
.dashboard{max-width:1400px;margin:0 auto}
h1{font-size:1.5rem;font-weight:700;margin-bottom:.25rem;color:#f8fafc}
.subtitle{color:#94a3b8;font-size:.85rem;margin-bottom:1.5rem}
.grid{display:grid;gap:1rem;margin-bottom:1.5rem}
.grid-4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.grid-2{grid-template-columns:repeat(auto-fit,minmax(350px,1fr))}
.card{background:#1e293b;border-radius:12px;padding:1.25rem;border:1px solid #334155}
.card .label{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.card .value{font-size:1.4rem;font-weight:700;color:#f8fafc;margin-top:.25rem}
.card .value.critical{color:#ef4444}.card .value.high{color:#f97316}.card .value.medium{color:#eab308}.card .value.low{color:#22c55e}
.chart-container{position:relative;height:200px;width:100%}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{text-align:left;padding:.5rem .75rem;color:#94a3b8;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #334155}
td{padding:.5rem .75rem;border-bottom:1px solid #1e293b;color:#cbd5e1;font-family:monospace;font-size:.75rem}
tr:hover td{background:#334155}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.65rem;font-weight:600;text-transform:uppercase}
.badge-critical{background:#7f1d1d;color:#fecaca}.badge-high{background:#7c2d12;color:#fed7aa}
.badge-medium{background:#713f12;color:#fef08a}.badge-low{background:#14532d;color:#bbf7d0}
.alert-list{max-height:400px;overflow-y:auto}
.alert-item{padding:.5rem 0;border-bottom:1px solid #1e293b;font-size:.75rem}
.alert-item:last-child{border-bottom:none}
.alert-time{color:#64748b;font-family:monospace;font-size:.7rem}
.alert-msg{color:#e2e8f0;margin:2px 0}
.refresh-btn{background:#2563eb;color:white;border:none;padding:.5rem 1rem;border-radius:6px;cursor:pointer;font-size:.8rem;float:right}
.refresh-btn:hover{background:#1d4ed8}
.footer{text-align:center;color:#475569;font-size:.75rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b}
</style>
</head>
<body>
<div class="dashboard">
<div style="display:flex;justify-content:space-between;align-items:center">
<div><h1>Threat Intelligence Dashboard</h1><p class="subtitle">Real-time threat monitoring — Last updated: {{ last_update }}</p></div>
<button class="refresh-btn" onclick="location.reload()">Refresh</button>
</div>

<div class="grid grid-4">
<div class="card"><div class="label">Total Indicators</div><div class="value">{{ total_indicators }}</div></div>
<div class="card"><div class="label">Critical Severity</div><div class="value critical">{{ sev_counts.get('critical', 0) }}</div></div>
<div class="card"><div class="label">High Severity</div><div class="value high">{{ sev_counts.get('high', 0) }}</div></div>
<div class="card"><div class="label">Active Alerts</div><div class="value medium">{{ alerts|length }}</div></div>
</div>

<div class="grid grid-2">
<div class="card">
<div class="label" style="margin-bottom:.75rem">Indicators by Type</div>
<div class="chart-container"><canvas id="typeChart"></canvas></div>
</div>
<div class="card">
<div class="label" style="margin-bottom:.75rem">Severity Distribution</div>
<div class="chart-container"><canvas id="sevChart"></canvas></div>
</div>
</div>

<div class="grid grid-2">
<div class="card">
<div class="label" style="margin-bottom:.75rem">Top Sources</div>
<div class="chart-container"><canvas id="srcChart"></canvas></div>
</div>
<div class="card">
<div class="label" style="margin-bottom:.75rem">Recent Alerts</div>
<div class="alert-list">
{% for alert in alerts %}
<div class="alert-item">
<span class="alert-time">{{ alert.time }}</span>
<span class="badge badge-{{ alert.severity }}">{{ alert.severity }}</span>
<span class="badge badge-{{ 'critical' if alert.severity == 'critical' else 'medium' }}" style="background:#334155;color:#94a3b8;margin-left:4px">{{ alert.type }}</span>
<div class="alert-msg">{{ alert.message }}</div>
</div>
{% endfor %}
</div>
</div>
</div>

<div class="card" style="margin-top:1rem">
<div class="label" style="margin-bottom:.5rem">Recent Indicators</div>
<div style="max-height:300px;overflow-y:auto">
<table>
<thead><tr><th>Type</th><th>Value</th><th>Severity</th><th>Source</th><th>Confidence</th><th>First Seen</th></tr></thead>
<tbody>
{% for ioc in recent %}
<tr>
<td><span class="badge badge-{{ 'high' if ioc.severity == 'high' or ioc.severity == 'critical' else 'medium' }}" style="background:#334155">{{ ioc.type }}</span></td>
<td>{{ ioc.value[:50] }}{% if ioc.value|length > 50 %}...{% endif %}</td>
<td><span class="badge badge-{{ ioc.severity }}">{{ ioc.severity }}</span></td>
<td>{{ ioc.source }}</td>
<td>{{ "%.0f"|format(ioc.confidence * 100) }}%</td>
<td>{{ ioc.first_seen[:10] if ioc.first_seen else '-' }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
</div>

<div class="footer">Threat Intelligence Dashboard — Cybersecurity Monitoring System</div>
</div>

<script>
const typeData = {{ type_counts_json }};
const sevData = {{ sev_counts_json }};
const srcData = {{ src_counts_json }};

const colors = {ip:'#3b82f6',domain:'#8b5cf6',url:'#ec4899',hash:'#f59e0b',cve:'#ef4444',email:'#14b8a6'};
const sevColors = {critical:'#ef4444',high:'#f97316',medium:'#eab308',low:'#22c55e'};

new Chart(document.getElementById('typeChart'), {
type:'doughnut',
data:{labels:Object.keys(typeData),datasets:[{data:Object.values(typeData),backgroundColor:Object.keys(typeData).map(k=>colors[k]||'#6366f1')}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',padding:10}}}}
});

new Chart(document.getElementById('sevChart'), {
type:'bar',
data:{labels:Object.keys(sevData),datasets:[{data:Object.values(sevData),backgroundColor:Object.keys(sevData).map(k=>sevColors[k]||'#6366f1')}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,grid:{color:'#334155'},ticks:{color:'#94a3b8'}},x:{grid:{display:false},ticks:{color:'#94a3b8'}}}}
});

new Chart(document.getElementById('srcChart'), {
type:'polarArea',
data:{labels:Object.keys(srcData),datasets:[{data:Object.values(srcData),backgroundColor:['#3b82f6','#8b5cf6','#ec4899','#f59e0b','#14b8a6','#6366f1','#d946ef','#f97316']}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',padding:8}}}}
});
</script>
</body>
</html>
"""


def main():
    print("  Threat Intelligence Dashboard")
    print(f"  Open http://127.0.0.1:5002")
    print(f"  Press Ctrl+C to stop")
    refresh_cache()
    app.run(debug=False, host="127.0.0.1", port=5002)


if __name__ == "__main__":
    main()
