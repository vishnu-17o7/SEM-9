"""Flask web UI for Credential Stuffing Detector (port 5006).

Loads the trained RandomForest model and classifies login log entries.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
DEFAULT_MODEL = "RandomForest"

app = Flask(__name__)
_artifact = None


def load_artifact():
    global _artifact
    path = MODEL_DIR / f"{DEFAULT_MODEL}.joblib"
    if path.exists():
        _artifact = joblib.load(path)
        print(f"  [OK] Loaded model: {DEFAULT_MODEL}")
    else:
        print(f"  [!] Model not found: {path}")
        print("      Run 'python run.py train' first.")


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Credential Stuffing Detector</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2e3a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --attack: #ff5252; --normal: #4caf50;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
       min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.container { width: 100%; max-width: 580px; }
.form-group { margin-bottom: 0.75rem; }
.form-group label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.2rem; }
.form-group input, .form-group select {
  width: 100%; padding: 0.6rem; font-size: 0.95rem; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); color: var(--text); outline: none;
}
.form-group input:focus, .form-group select:focus { border-color: var(--accent); }
.form-row { display: flex; gap: 0.75rem; }
.form-row .form-group { flex: 1; }
button { margin-top: 0.5rem; padding: 0.75rem 1.5rem; font-size: 1rem; border: none; border-radius: 8px;
         background: var(--accent); color: white; cursor: pointer; }
button:hover { opacity: 0.85; }
button:disabled { opacity: 0.4; }
.result { margin-top: 1.5rem; display: none; }
.result.show { display: block; }
.badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
.badge.ATTACK { background: rgba(255,82,82,0.15); color: var(--attack); }
.badge.NORMAL { background: rgba(76,175,80,0.15); color: var(--normal); }
.risk { margin-top: 0.5rem; font-size: 0.85rem; color: var(--muted); }
.error { color: var(--attack); margin-top: 0.75rem; font-size: 0.9rem; display: none; }
.error.show { display: block; }
</style>
</head>
<body>
<div class="container">
  <h1>Credential Stuffing <span>Detector</span></h1>
  <p class="subtitle">Enter a login event to check for credential stuffing.</p>
  <div class="form-group"><label>Username</label><input id="username" type="text" value="alice" /></div>
  <div class="form-row">
    <div class="form-group"><label>Source IP</label><input id="source_ip" type="text" value="192.168.1.50" /></div>
    <div class="form-group"><label>Geo Country</label><input id="geo_country" type="text" value="US" /></div>
  </div>
  <div class="form-row">
    <div class="form-group"><label>Success</label><select id="success"><option value="0">Failed (0)</option><option value="1">Success (1)</option></select></div>
    <div class="form-group"><label>Failures (last 5 min)</label><input id="failure_count" type="number" value="3" /></div>
  </div>
  <div class="form-row">
    <div class="form-group"><label>Unique IPs (5 min)</label><input id="unique_ips" type="number" value="1" /></div>
    <div class="form-group"><label>Attempts/sec</label><input id="attempts_per_second" type="number" step="0.1" value="0.5" /></div>
  </div>
  <button id="btn" onclick="detect()">Detect</button>
  <div id="error" class="error"></div>
  <div id="result" class="result">
    <span id="badge" class="badge"></span>
    <div id="risk" class="risk"></div>
  </div>
</div>
<script>
async function detect() {
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'Detecting...';
  document.getElementById('error').classList.remove('show');
  document.getElementById('result').classList.remove('show');
  const payload = {
    username: document.getElementById('username').value,
    source_ip: document.getElementById('source_ip').value,
    geo_country: document.getElementById('geo_country').value,
    success: document.getElementById('success').value,
    failure_count: parseInt(document.getElementById('failure_count').value),
    unique_ips: parseInt(document.getElementById('unique_ips').value),
    attempts_per_second: parseFloat(document.getElementById('attempts_per_second').value),
  };
  try {
    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) { document.getElementById('error').textContent = data.error || 'Failed'; document.getElementById('error').classList.add('show'); return; }
    const badge = document.getElementById('badge');
    badge.className = 'badge ' + data.verdict;
    badge.textContent = data.verdict;
    document.getElementById('risk').textContent = 'Model: ' + data.model + ' | ' +
      (data.verdict === 'ATTACK' ? 'Credential stuffing pattern detected' : 'No attack pattern');
    document.getElementById('result').classList.add('show');
  } catch(e) { document.getElementById('error').textContent = 'Network error'; document.getElementById('error').classList.add('show'); }
  finally { btn.disabled = false; btn.textContent = 'Detect'; }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if _artifact is None:
        load_artifact()
    if _artifact is None:
        return jsonify({"error": "No trained model found. Run 'python run.py train' first."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    model = _artifact["model"]
    scaler = _artifact.get("scaler")
    features = _artifact.get("features", [])

    success = int(data.get("success", 0))
    failure_count = int(data.get("failure_count", 1))
    unique_ips = int(data.get("unique_ips", 1))

    feats = {
        "login_count": max(failure_count, 1),
        "failure_count": failure_count,
        "failure_rate": failure_count / max(failure_count, 1) if success == 0 else 0.0,
        "unique_ips": unique_ips,
        "unique_uas": 1,
        "unique_users": 1,
        "unique_countries": 1,
        "ip_entropy": 0.0,
        "user_failure_rate": 0.5,
        "user_unique_ips": unique_ips,
        "user_unique_countries": 1,
        "time_since_last": 0.1,
        "attempts_per_second": float(data.get("attempts_per_second", 0.5)),
    }

    X = np.array([[feats.get(f, 0) for f in features]], dtype=np.float64)
    if scaler:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    is_anomaly = bool(pred == -1 if hasattr(model, "predict") else pred == 1)

    return jsonify({
        "anomaly": is_anomaly,
        "verdict": "ATTACK" if is_anomaly else "NORMAL",
        "model": DEFAULT_MODEL,
    })


if __name__ == "__main__":
    print("  Credential Stuffing Detector -- Web UI")
    load_artifact()
    print("  Open http://127.0.0.1:5006")
    app.run(debug=True, host="127.0.0.1", port=5006)
