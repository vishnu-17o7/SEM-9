"""Flask web UI for Botnet Traffic Classifier (port 5007).

Loads the trained VotingEnsemble and classifies network flows.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from generate_botnet_traffic import FEATURE_NAMES

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
DEFAULT_MODEL = "VotingEnsemble"

app = Flask(__name__)
_model = None


def load_model():
    global _model
    path = MODEL_DIR / f"{DEFAULT_MODEL}.joblib"
    if path.exists():
        _model = joblib.load(path)
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
<title>Botnet Traffic Classifier</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2e3a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --bot: #ff5252; --norm: #4caf50;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
       min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.container { width: 100%; max-width: 640px; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
.form-group { margin-bottom: 0.5rem; }
.form-group label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.15rem; }
.form-group input { width: 100%; padding: 0.5rem; font-size: 0.9rem; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); color: var(--text); outline: none; }
.form-group input:focus { border-color: var(--accent); }
button { margin-top: 0.75rem; padding: 0.75rem 1.5rem; font-size: 1rem; border: none; border-radius: 8px;
  background: var(--accent); color: white; cursor: pointer; }
button:hover { opacity: 0.85; }
button:disabled { opacity: 0.4; }
.result { margin-top: 1.5rem; display: none; }
.result.show { display: block; }
.badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
.badge.BOTNET { background: rgba(255,82,82,0.15); color: var(--bot); }
.badge.NORMAL { background: rgba(76,175,80,0.15); color: var(--norm); }
.prob { margin-top: 0.5rem; font-size: 0.85rem; color: var(--muted); }
.error { color: var(--bot); margin-top: 0.75rem; font-size: 0.9rem; display: none; }
.error.show { display: block; }
</style>
</head>
<body>
<div class="container">
  <h1>Botnet Traffic <span>Classifier</span></h1>
  <p class="subtitle">Enter network flow features to classify as botnet or normal.</p>
  <div class="form-grid" id="formGrid"></div>
  <button id="btn" onclick="classify()">Classify</button>
  <div id="error" class="error"></div>
  <div id="result" class="result">
    <span id="badge" class="badge"></span>
    <div id="prob" class="prob"></div>
  </div>
</div>
<script>
const features = {{ features_json|safe }};
const grid = document.getElementById('formGrid');
features.forEach(f => {
  const div = document.createElement('div');
  div.className = 'form-group';
  div.innerHTML = '<label>' + f + '</label><input id="f_' + f + '" type="number" step="any" value="0">';
  grid.appendChild(div);
});
async function classify() {
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'Classifying...';
  document.getElementById('error').classList.remove('show');
  document.getElementById('result').classList.remove('show');
  const payload = {};
  features.forEach(f => { payload[f] = parseFloat(document.getElementById('f_' + f).value) || 0; });
  try {
    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) { document.getElementById('error').textContent = data.error || 'Failed'; document.getElementById('error').classList.add('show'); return; }
    const badge = document.getElementById('badge');
    badge.className = 'badge ' + data.label;
    badge.textContent = data.label;
    document.getElementById('prob').textContent = 'Botnet probability: ' + (data.botnet_probability * 100).toFixed(1) + '%';
    document.getElementById('result').classList.add('show');
  } catch(e) { document.getElementById('error').textContent = 'Network error'; document.getElementById('error').classList.add('show'); }
  finally { btn.disabled = false; btn.textContent = 'Classify'; }
}
</script>
</body>
</html>
"""

import json as _json


@app.route("/")
def index():
    return render_template_string(HTML, features_json=_json.dumps(FEATURE_NAMES))


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        load_model()
    if _model is None:
        return jsonify({"error": "No trained model found. Run 'python run.py train' first."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        X = np.array([[float(data.get(name, 0)) for name in FEATURE_NAMES]], dtype=np.float64)
        pred = int(_model.predict(X)[0])
        prob = float(_model.predict_proba(X)[0, 1])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": pred,
        "label": "BOTNET" if pred == 1 else "NORMAL",
        "botnet_probability": round(prob, 4),
    })


if __name__ == "__main__":
    print("  Botnet Traffic Classifier -- Web UI")
    load_model()
    print("  Open http://127.0.0.1:5007")
    app.run(debug=True, host="127.0.0.1", port=5007)
