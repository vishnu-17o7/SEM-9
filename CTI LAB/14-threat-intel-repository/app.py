"""Flask web UI for Threat Intel Repository (port 5008).

Searches and browses the SQLite threat intelligence database.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "threat_intel.db"

app = Flask(__name__)


def get_conn():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH))


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Threat Intel Repository</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2e3a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --crit: #ff5252; --high: #ff9800; --med: #ffc107; --low: #4caf50;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
       min-height: 100vh; padding: 2rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.container { max-width: 960px; margin: 0 auto; }
.toolbar { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.toolbar input, .toolbar select { padding: 0.5rem 0.75rem; font-size: 0.9rem; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); color: var(--text); outline: none; }
.toolbar input { flex: 1; min-width: 200px; }
.toolbar input:focus, .toolbar select:focus { border-color: var(--accent); }
button { padding: 0.5rem 1rem; font-size: 0.9rem; border: none; border-radius: 6px;
  background: var(--accent); color: white; cursor: pointer; }
button:hover { opacity: 0.85; }
.stats { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.stat { padding: 0.75rem 1.25rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.stat-num { font-size: 1.4rem; font-weight: 700; }
.stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; padding: 0.5rem; color: var(--muted); border-bottom: 1px solid var(--border);
     font-size: 0.75rem; text-transform: uppercase; }
td { padding: 0.5rem; border-bottom: 1px solid var(--surface); }
tr:hover td { background: var(--surface); }
.sev { padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.sev.critical { background: rgba(255,82,82,0.15); color: var(--crit); }
.sev.high { background: rgba(255,152,0,0.15); color: var(--high); }
.sev.medium { background: rgba(255,193,7,0.15); color: var(--med); }
.sev.low { background: rgba(76,175,80,0.15); color: var(--low); }
.error { color: var(--crit); margin: 1rem 0; }
</style>
</head>
<body>
<div class="container">
  <h1>Threat Intel <span>Repository</span></h1>
  <p class="subtitle">Search and browse indicators of compromise (IoCs).</p>
  <div class="stats" id="stats"></div>
  <div class="toolbar">
    <input id="search" type="text" placeholder="Search IoC value or description..." />
    <select id="type"><option value="">All Types</option><option value="ip">IP</option><option value="domain">Domain</option><option value="url">URL</option><option value="hash">Hash</option><option value="cve">CVE</option></select>
    <select id="severity"><option value="">All Severities</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
    <button onclick="search()">Search</button>
  </div>
  <div id="error" class="error"></div>
  <table>
    <thead><tr><th>Type</th><th>Value</th><th>Source</th><th>Severity</th><th>Confidence</th><th>First Seen</th></tr></thead>
    <tbody id="results"></tbody>
  </table>
</div>
<script>
async function loadStats() {
  try {
    const res = await fetch('/api/stats'); const data = await res.json();
    const div = document.getElementById('stats');
    div.innerHTML = '';
    data.forEach(s => {
      const stat = document.createElement('div');
      stat.className = 'stat';
      stat.innerHTML = '<div class="stat-num">' + s.count + '</div><div class="stat-label">' + s.type + '</div>';
      div.appendChild(stat);
    });
  } catch(e) {}
}
async function search() {
  const params = new URLSearchParams();
  const q = document.getElementById('search').value;
  const t = document.getElementById('type').value;
  const s = document.getElementById('severity').value;
  if (q) params.set('q', q);
  if (t) params.set('type', t);
  if (s) params.set('severity', s);
  try {
    const res = await fetch('/api/search?' + params);
    const data = await res.json();
    if (!res.ok) { document.getElementById('error').textContent = data.error || 'Failed'; return; }
    document.getElementById('error').textContent = '';
    const tbody = document.getElementById('results');
    tbody.innerHTML = '';
    data.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + r.ioc_type + '</td><td style="font-family:monospace;font-size:0.8rem;">' + r.value.substring(0,60) +
        '</td><td>' + r.source + '</td><td><span class="sev ' + r.severity + '">' + r.severity + '</span></td><td>' +
        (r.confidence||0).toFixed(2) + '</td><td>' + (r.first_seen||'').substring(0,10) + '</td>';
      tbody.appendChild(tr);
    });
  } catch(e) { document.getElementById('error').textContent = 'Network error'; }
}
loadStats();
search();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/stats")
def stats():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "Database not found. Run 'python run.py data' first."}), 500

    cursor = conn.execute("""
        SELECT ioc_type, COUNT(*) FROM indicators GROUP BY ioc_type ORDER BY COUNT(*) DESC
    """)
    result = [{"type": row[0], "count": row[1]} for row in cursor]
    conn.close()
    return jsonify(result)


@app.route("/api/search")
def search():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "Database not found. Run 'python run.py data' first."}), 500

    q = request.args.get("q", "").strip()
    ioc_type = request.args.get("type", "").strip()
    severity = request.args.get("severity", "").strip()

    query = "SELECT ioc_type, value, source, confidence, severity, first_seen FROM indicators WHERE 1=1"
    params: list[str] = []
    if ioc_type:
        query += " AND ioc_type = ?"
        params.append(ioc_type)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if q:
        query += " AND (value LIKE ? OR description LIKE ?)"
        params.append(f"%{q}%")
        params.append(f"%{q}%")
    query += " ORDER BY first_seen DESC LIMIT 100"

    cursor = conn.execute(query, params)
    result = [
        {
            "ioc_type": row[0],
            "value": row[1],
            "source": row[2],
            "confidence": row[3],
            "severity": row[4],
            "first_seen": row[5],
        }
        for row in cursor
    ]
    conn.close()
    return jsonify(result)


@app.route("/api/add", methods=["POST"])
def add():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "Database not found. Run 'python run.py data' first."}), 500

    data = request.get_json()
    if not data or "type" not in data or "value" not in data:
        return jsonify({"error": "Missing type or value"}), 400

    ioc_type = data["type"]
    value = data["value"]
    source = data.get("source", "manual")
    confidence = float(data.get("confidence", 0.5))
    severity = data.get("severity", "medium")
    tags = data.get("tags", "")

    ioc_id = hashlib.md5(f"{ioc_type}:{value}".encode()).hexdigest()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        conn.execute("""
            INSERT OR IGNORE INTO indicators (id, ioc_type, value, source, confidence, severity, tags, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ioc_id, ioc_type, value, source, confidence, severity, tags, now, now))
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

    conn.close()
    return jsonify({"status": "added", "type": ioc_type, "value": value})


if __name__ == "__main__":
    print("  Threat Intel Repository -- Web UI")
    print("  Open http://127.0.0.1:5008")
    app.run(debug=True, host="127.0.0.1", port=5008)
