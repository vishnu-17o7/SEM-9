"""
dashboard.py
Flask web dashboard (port 5001) for the UEBA Behavioral Profile system.
Features:
  - Real-time anomaly feed
  - User profile viewer
  - Role-group comparison charts
  - Alert log
"""

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
DATA_PATH = "data/user_activity.csv"
PRED_CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# ── Flask setup ─────────────────────────────────────────────────────────────────
try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("ERROR: Flask is not installed. Run: pip install flask")
    exit(1)

import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Data loading with caching ──────────────────────────────────────────────────
_data_cache = {}
_cache_timestamps = {}


def load_data(force_reload: bool = False):
    """Load all required datasets with caching."""
    global _data_cache, _cache_timestamps
    cache_key = "all_data"

    if cache_key in _data_cache and not force_reload:
        return _data_cache[cache_key]

    result = {
        "df": None,
        "pred_df": None,
        "metrics": {},
        "predictions": {},
    }

    # Load activity data
    if os.path.exists(DATA_PATH):
        result["df"] = pd.read_csv(DATA_PATH)

    # Load predictions
    if os.path.exists(PRED_CSV_PATH):
        result["pred_df"] = pd.read_csv(PRED_CSV_PATH)

    # Load metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            result["metrics"] = json.load(f)

    # Load .npz predictions
    if os.path.isdir(PREDICTIONS_DIR):
        for fname in os.listdir(PREDICTIONS_DIR):
            if fname.endswith(".npz"):
                name = fname.replace(".npz", "")
                data = np.load(os.path.join(PREDICTIONS_DIR, fname))
                result["predictions"][name] = {
                    "scores": data["scores"].tolist()[:200],  # first 200 for display
                    "prediction": data["prediction"].tolist()[:200],
                }

    _data_cache[cache_key] = result
    _cache_timestamps[cache_key] = datetime.now()
    return result


# ── HTML Templates ──────────────────────────────────────────────────────────────

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UEBA Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1923; color: #e0e0e0; }
    .sidebar { position: fixed; top: 0; left: 0; width: 220px; height: 100vh; background: #1a2332;
               padding: 20px 0; border-right: 1px solid #2a3a4a; z-index: 100; }
    .sidebar h2 { color: #64b5f6; padding: 0 20px 20px; font-size: 18px; border-bottom: 1px solid #2a3a4a; }
    .sidebar a { display: block; padding: 12px 20px; color: #b0bec5; text-decoration: none; font-size: 14px;
                 transition: all 0.2s; }
    .sidebar a:hover, .sidebar a.active { background: #1e3a5f; color: #64b5f6; border-left: 3px solid #64b5f6; }
    .main { margin-left: 220px; padding: 25px; }
    h1 { color: #64b5f6; font-size: 24px; margin-bottom: 5px; }
    .subtitle { color: #78909c; font-size: 13px; margin-bottom: 25px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .card { background: #1a2332; border-radius: 10px; padding: 18px; border: 1px solid #2a3a4a; }
    .card .val { font-size: 26px; font-weight: bold; color: #64b5f6; }
    .card .lbl { font-size: 12px; color: #78909c; margin-top: 5px; }
    .card.green .val { color: #66bb6a; }
    .card.red .val { color: #ef5350; }
    .card.orange .val { color: #ffa726; }
    .section { background: #1a2332; border-radius: 10px; padding: 20px; margin-bottom: 25px; border: 1px solid #2a3a4a; }
    .section h3 { color: #90caf9; font-size: 16px; margin-bottom: 15px; border-bottom: 1px solid #2a3a4a; padding-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { color: #90caf9; text-align: left; padding: 10px 8px; border-bottom: 1px solid #2a3a4a; font-weight: 500; }
    td { padding: 8px; border-bottom: 1px solid #1e2a38; color: #cfd8dc; }
    tr:hover { background: #1e2a3a; }
    .anomaly { color: #ef5350; }
    .chart-container { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px; }
    .chart-box { background: #1a2332; border-radius: 10px; padding: 15px; border: 1px solid #2a3a4a; }
    .chart-box canvas { max-height: 280px; }
    select, input { background: #0f1923; color: #e0e0e0; border: 1px solid #2a3a4a; padding: 6px 10px;
                    border-radius: 4px; font-size: 13px; margin: 0 5px; }
    label { color: #90caf9; font-size: 13px; }
    .filters { margin-bottom: 15px; }
    .refresh-btn { background: #1565c0; color: white; border: none; padding: 6px 14px; border-radius: 4px;
                   cursor: pointer; font-size: 13px; margin-left: 10px; }
    .refresh-btn:hover { background: #1976d2; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
    .badge-anomaly { background: #ef5350; color: white; }
    .badge-normal { background: #2e7d32; color: white; }
    .footer { margin-top: 30px; padding: 15px; text-align: center; color: #546e7a; font-size: 12px; border-top: 1px solid #1e2a38; }
    @media (max-width: 900px) { .sidebar { width: 100%; height: auto; position: relative; }
        .sidebar a { display: inline-block; }
        .main { margin-left: 0; }
        .chart-container { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="sidebar">
    <h2>UEBA Dashboard</h2>
    <a href="#" class="active" onclick="switchTab('overview')">Overview</a>
    <a href="#" onclick="switchTab('users')">User Profiles</a>
    <a href="#" onclick="switchTab('roles')">Role Comparison</a>
    <a href="#" onclick="switchTab('alerts')">Alert Log</a>
    <a href="#" onclick="switchTab('detectors')">Detectors</a>
</div>

<div class="main">
    <h1>User & Entity Behavior Analytics</h1>
    <p class="subtitle">Real-time behavioral monitoring & anomaly detection | Port 5001</p>

    <!-- Overview Tab -->
    <div id="tab-overview" class="tab-content">
        <div class="cards" id="summary-cards"></div>
        <div class="chart-container">
            <div class="chart-box"><h3>Per-Role Anomaly Rate</h3><canvas id="chart-role-anomaly"></canvas></div>
            <div class="chart-box"><h3>Detector AUC Comparison</h3><canvas id="chart-detector-auc"></canvas></div>
        </div>
        <div class="section">
            <h3>Recent Anomaly Feed</h3>
            <table id="anomaly-feed"><thead><tr>
                <th>Timestamp</th><th>User</th><th>Role</th><th>Action</th><th>Resource</th><th>Score</th>
            </tr></thead><tbody></tbody></table>
        </div>
    </div>

    <!-- User Profiles Tab -->
    <div id="tab-users" class="tab-content" style="display:none;">
        <div class="filters">
            <label>Select User:</label>
            <select id="user-select" onchange="loadUserProfile()"></select>
        </div>
        <div class="cards" id="user-cards"></div>
        <div class="section"><h3>User Activity Timeline</h3>
            <div style="overflow-x:auto;"><table id="user-timeline"><thead><tr>
                <th>Window</th><th>Role</th><th>Action Count</th><th>Off-Hours Ratio</th><th>Scope Dev.</th><th>Status</th>
            </tr></thead><tbody></tbody></table></div>
        </div>
    </div>

    <!-- Role Comparison Tab -->
    <div id="tab-roles" class="tab-content" style="display:none;">
        <div class="chart-container" style="grid-template-columns:1fr 1fr;">
            <div class="chart-box"><h3>Feature Means by Role</h3><canvas id="chart-role-features"></canvas></div>
            <div class="chart-box"><h3>User Count by Role</h3><canvas id="chart-role-users"></canvas></div>
        </div>
        <div class="section"><h3>Role Statistics</h3>
            <table id="role-stats"><thead><tr>
                <th>Role</th><th>Users</th><th>Total Events</th><th>Anomaly Rate</th><th>Avg Off-Hours</th><th>Avg Scope Dev.</th>
            </tr></thead><tbody></tbody></table>
        </div>
    </div>

    <!-- Alert Log Tab -->
    <div id="tab-alerts" class="tab-content" style="display:none;">
        <div class="filters">
            <label>Severity:</label>
            <select id="alert-severity" onchange="loadAlerts()">
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
            </select>
            <button class="refresh-btn" onclick="loadAlerts()">Refresh</button>
            <span id="alert-count" style="margin-left:15px;color:#78909c;font-size:13px;"></span>
        </div>
        <div class="section"><table id="alert-table"><thead><tr>
            <th>Timestamp</th><th>User</th><th>Role</th><th>Anomaly Type</th><th>Severity</th><th>Score</th><th>Action</th>
        </tr></thead><tbody></tbody></table></div>
    </div>

    <!-- Detectors Tab -->
    <div id="tab-detectors" class="tab-content" style="display:none;">
        <div class="section"><h3>Detector Performance</h3>
            <table id="detector-table"><thead><tr>
                <th>Detector</th><th>ROC AUC</th><th>F1 Score</th><th>Precision</th><th>Recall</th><th>Accuracy</th>
            </tr></thead><tbody></tbody></table>
        </div>
    </div>

    <div class="footer">UEBA Behavioral Profile System | Data is loaded from generated artifacts</div>
</div>

<script>
let roleAnomalyChart = null, detectorAucChart = null, roleFeaturesChart = null, roleUsersChart = null;

function switchTab(name) {
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.getElementById('tab-' + name).style.display = 'block';
    document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
    document.querySelector('.sidebar a[onclick*="' + name + '"]').classList.add('active');
}

async function fetchData(endpoint) {
    try {
        const resp = await fetch('/api/' + endpoint);
        return await resp.json();
    } catch(e) { console.error(e); return {}; }
}

// Overview
async function loadOverview() {
    const data = await fetchData('overview');
    const cards = document.getElementById('summary-cards');
    cards.innerHTML = Object.entries(data.summary || {}).map(([k,v]) =>
        `<div class="card"><div class="val">${v}</div><div class="lbl">${k.replace(/_/g,' ')}</div></div>`
    ).join('');

    // Role anomaly chart
    const rlabels = Object.keys(data.role_anomaly_rates || {});
    const rvals = Object.values(data.role_anomaly_rates || {});
    if (roleAnomalyChart) roleAnomalyChart.destroy();
    roleAnomalyChart = new Chart(document.getElementById('chart-role-anomaly'), {
        type: 'bar', data: { labels: rlabels, datasets: [{
            label: 'Anomaly Rate (%)', data: rvals, backgroundColor: '#ef5350'
        }]}, options: { responsive: true, plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, grid: { color: '#2a3a4a' } },
                     x: { grid: { display: false } } } }
    });

    // Detector AUC chart
    const dlabels = Object.keys(data.detector_auc || {});
    const dvals = Object.values(data.detector_auc || {});
    if (detectorAucChart) detectorAucChart.destroy();
    detectorAucChart = new Chart(document.getElementById('chart-detector-auc'), {
        type: 'bar', data: { labels: dlabels, datasets: [{
            label: 'ROC AUC', data: dvals, backgroundColor: '#42a5f5'
        }]}, options: { responsive: true, plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 1, grid: { color: '#2a3a4a' } },
                     x: { grid: { display: false } } } }
    });

    // Anomaly feed
    const feed = data.recent_anomalies || [];
    document.getElementById('anomaly-feed').querySelector('tbody').innerHTML =
        feed.map(a => `<tr><td class="anomaly">${a.timestamp}</td><td>${a.user_id}</td>
        <td>${a.role}</td><td>${a.action}</td><td>${a.resource}</td>
        <td><span class="badge badge-anomaly">ANOMALY</span></td></tr>`
    ).join('') || '<tr><td colspan="6" style="text-align:center;color:#546e7a;">No anomalies found</td></tr>';
}

// User profiles
async function loadUserProfile() {
    const uid = document.getElementById('user-select').value;
    if (!uid) return;
    const data = await fetchData('user?user_id=' + encodeURIComponent(uid));
    const cards = document.getElementById('user-cards');
    cards.innerHTML = Object.entries(data.profile || {}).map(([k,v]) =>
        `<div class="card"><div class="val">${typeof v === 'number' ? v.toFixed(4) : v}</div>
        <div class="lbl">${k.replace(/_/g,' ')}</div></div>`
    ).join('');

    const tbody = document.getElementById('user-timeline').querySelector('tbody');
    tbody.innerHTML = (data.timeline || []).map(r =>
        `<tr><td>${r.window}</td><td>${r.role}</td><td>${r.action_count}</td>
        <td>${r.off_hours_ratio.toFixed(4)}</td><td>${r.role_scope_deviation.toFixed(4)}</td>
        <td>${r.is_anomaly ? '<span class="badge badge-anomaly">ANOMALY</span>' : '<span class="badge badge-normal">NORMAL</span>'}</td></tr>`
    ).join('') || '<tr><td colspan="6" style="text-align:center;color:#546e7a;">No data</td></tr>';
}

async function loadUserSelect() {
    const data = await fetchData('users');
    const sel = document.getElementById('user-select');
    sel.innerHTML = (data.users || []).map(u => `<option value="${u}">${u}</option>`).join('');
    if (data.users && data.users.length) loadUserProfile();
}

// Role comparison
async function loadRoleComparison() {
    const data = await fetchData('roles');
    const rnames = Object.keys(data.role_stats || {});
    const fnames = ['login_hour_mean','off_hours_ratio','role_scope_deviation'];

    if (roleFeaturesChart) roleFeaturesChart.destroy();
    roleFeaturesChart = new Chart(document.getElementById('chart-role-features'), {
        type: 'bar', data: {
            labels: rnames,
            datasets: fnames.map((f,i) => ({
                label: f.replace(/_/g,' '),
                data: rnames.map(r => (data.role_stats[r]?.[f] || 0)),
                backgroundColor: ['#42a5f5','#ef5350','#66bb6a'][i]
            }))
        }, options: { responsive: true, plugins: { legend: { labels: { color: '#b0bec5' } } },
            scales: { y: { beginAtZero: true, grid: { color: '#2a3a4a' } },
                     x: { grid: { display: false } } } }
    });

    if (roleUsersChart) roleUsersChart.destroy();
    roleUsersChart = new Chart(document.getElementById('chart-role-users'), {
        type: 'doughnut', data: {
            labels: rnames,
            datasets: [{ data: rnames.map(r => data.role_stats[r]?.user_count || 0),
                         backgroundColor: ['#42a5f5','#ef5350','#66bb6a','#ffa726','#ab47bc'] }]
        }, options: { responsive: true, plugins: { legend: { labels: { color: '#b0bec5' } } } }
    });

    const tbody = document.getElementById('role-stats').querySelector('tbody');
    tbody.innerHTML = rnames.map(r => {
        const s = data.role_stats[r] || {};
        return `<tr><td><strong>${r}</strong></td><td>${s.user_count||0}</td>
        <td>${s.total_events||0}</td><td>${(s.anomaly_rate||0).toFixed(2)}%</td>
        <td>${(s.avg_off_hours||0).toFixed(4)}</td><td>${(s.avg_scope_dev||0).toFixed(4)}</td></tr>`;
    }).join('');
}

// Alerts
async function loadAlerts() {
    const sev = document.getElementById('alert-severity').value;
    const data = await fetchData('alerts?severity=' + sev);
    document.getElementById('alert-count').textContent = data.count + ' alerts';
    const tbody = document.getElementById('alert-table').querySelector('tbody');
    tbody.innerHTML = (data.alerts || []).map(a =>
        `<tr><td>${a.timestamp}</td><td>${a.user_id}</td><td>${a.role}</td>
        <td>${a.anomaly_type}</td><td><span class="badge badge-${a.severity === 'high' ? 'anomaly' : 'normal'}">${a.severity}</span></td>
        <td>${a.score.toFixed(4)}</td><td>${a.action}</td></tr>`
    ).join('') || '<tr><td colspan="7" style="text-align:center;color:#546e7a;">No alerts</td></tr>';
}

// Detectors
async function loadDetectors() {
    const data = await fetchData('detectors');
    const tbody = document.getElementById('detector-table').querySelector('tbody');
    tbody.innerHTML = Object.entries(data.detectors || {}).map(([name, m]) =>
        `<tr><td><strong>${name}</strong></td>
        <td>${(m.roc_auc||0).toFixed(4)}</td>
        <td>${(m.f1_score||0).toFixed(4)}</td>
        <td>${(m.precision||0).toFixed(4)}</td>
        <td>${(m.recall||0).toFixed(4)}</td>
        <td>${(m.accuracy||0).toFixed(4)}</td></tr>`
    ).join('');
}

// Init
async function init() {
    await loadOverview();
    await loadUserSelect();
    await loadRoleComparison();
    await loadAlerts();
    await loadDetectors();
}
init();
</script>
</body>
</html>
"""

# ── API Endpoints ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template_string(MAIN_TEMPLATE)


@app.route("/api/overview")
def api_overview():
    """Return overview summary data."""
    data = load_data()
    df = data.get("df")
    pred_df = data.get("pred_df")
    metrics = data.get("metrics", {})

    response = {"summary": {}, "role_anomaly_rates": {}, "detector_auc": {}, "recent_anomalies": []}

    if df is not None:
        response["summary"] = {
            "Total Entries": f"{len(df):,}",
            "Unique Users": df["user_id"].nunique(),
            "Roles": df["role"].nunique(),
            "Anomalies": int(df["is_anomaly"].sum()),
            "Anomaly Rate": f"{df['is_anomaly'].mean()*100:.2f}%",
        }
        # Recent anomalies
        anoms = df[df["is_anomaly"] == 1].tail(20)
        response["recent_anomalies"] = anoms.to_dict(orient="records")

    if pred_df is not None:
        rates = pred_df.groupby("role")["is_anomaly"].mean() * 100
        response["role_anomaly_rates"] = rates.to_dict()

    for name, m in metrics.items():
        if "roc_auc" in m:
            response["detector_auc"][name] = m["roc_auc"]

    return jsonify(response)


@app.route("/api/users")
def api_users():
    """Return list of all user IDs."""
    data = load_data()
    df = data.get("df")
    if df is None:
        return jsonify({"users": []})
    users = sorted(df["user_id"].unique().tolist())
    return jsonify({"users": users})


@app.route("/api/user")
def api_user():
    """Return profile and timeline for a specific user."""
    user_id = request.args.get("user_id", "")
    data = load_data()
    df = data.get("df")
    pred_df = data.get("pred_df")

    if df is None or pred_df is None:
        return jsonify({"profile": {}, "timeline": []})

    user_df = df[df["user_id"] == user_id]
    user_pred = pred_df[pred_df["user_id"] == user_id]

    if user_df.empty:
        return jsonify({"profile": {}, "timeline": []})

    profile = {
        "user_id": user_id,
        "role": user_df["role"].iloc[0],
        "total_events": len(user_df),
        "login_events": int((user_df["action"] == "login").sum()),
        "file_accesses": int((user_df["action"] == "file_access").sum()),
        "db_queries": int((user_df["action"] == "db_query").sum()),
        "privilege_changes": int((user_df["action"] == "privilege_change").sum()),
        "anomaly_events": int(user_df["is_anomaly"].sum()),
        "off_hours_events": int(((user_df["hour_of_day"] < 7) | (user_df["hour_of_day"] > 19)).sum()),
        "success_rate": float(user_df["success"].mean()),
    }

    timeline = user_pred.sort_values("window").tail(50).to_dict(orient="records")
    # Convert numpy types
    for t in timeline:
        for k, v in t.items():
            if isinstance(v, (np.integer,)):
                t[k] = int(v)
            elif isinstance(v, (np.floating,)):
                t[k] = float(v)
            elif isinstance(v, pd.Timestamp):
                t[k] = str(v)

    return jsonify({"profile": profile, "timeline": timeline})


@app.route("/api/roles")
def api_roles():
    """Return per-role statistics."""
    data = load_data()
    df = data.get("df")
    pred_df = data.get("pred_df")

    if df is None or pred_df is None:
        return jsonify({"role_stats": {}})

    role_stats = {}
    for role in sorted(df["role"].unique()):
        role_df = df[df["role"] == role]
        role_pred = pred_df[pred_df["role"] == role]
        stats = {
            "user_count": int(role_df["user_id"].nunique()),
            "total_events": len(role_df),
            "anomaly_rate": float(role_df["is_anomaly"].mean() * 100),
            "avg_login_hour": float(role_df["hour_of_day"].mean()),
            "avg_off_hours": float(((role_df["hour_of_day"] < 7) | (role_df["hour_of_day"] > 19)).mean()),
            "avg_scope_dev": float(0),
            "login_hour_mean": float(role_df["hour_of_day"].mean()),
            "off_hours_ratio": float(((role_df["hour_of_day"] < 7) | (role_df["hour_of_day"] > 19)).mean()),
            "role_scope_deviation": float(0),
            "success_rate": float(role_df["success"].mean()),
        }
        role_stats[role] = stats

    return jsonify({"role_stats": role_stats})


@app.route("/api/alerts")
def api_alerts():
    """Return alert log entries."""
    severity = request.args.get("severity", "all")
    data = load_data()
    df = data.get("df")

    if df is None:
        return jsonify({"alerts": [], "count": 0})

    anomaly_df = df[df["is_anomaly"] == 1].copy()

    alerts = []
    for _, row in anomaly_df.iterrows():
        hour = row["hour_of_day"]
        # Determine severity based on context
        if hour < 5 or hour > 22:
            sev = "high"
        elif row["action"] == "privilege_change":
            sev = "high"
        elif row["action"] == "db_query":
            sev = "medium"
        else:
            sev = "low"

        if severity != "all" and sev != severity:
            continue

        alert = {
            "timestamp": row["timestamp"],
            "user_id": row["user_id"],
            "role": row["role"],
            "anomaly_type": "Off-Hours" if (hour < 7 or hour > 19) else
                           "Privilege Escalation" if row["action"] == "privilege_change" else
                           "Out-of-Scope Access" if row["action"] in ("file_access", "db_query") else
                           "Anomalous Login",
            "severity": sev,
            "score": 1.0 if row["is_anomaly"] else 0.0,
            "action": row["action"],
        }
        alerts.append(alert)

    return jsonify({"alerts": alerts[-100:], "count": len(alerts)})


@app.route("/api/detectors")
def api_detectors():
    """Return detector performance metrics."""
    data = load_data()
    metrics = data.get("metrics", {})
    return jsonify({"detectors": metrics})


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("UEBA Dashboard starting on http://0.0.0.0:5001")
    print("Data directory:", os.path.abspath(RESULTS_DIR))
    app.run(host="0.0.0.0", port=5001, debug=False)
