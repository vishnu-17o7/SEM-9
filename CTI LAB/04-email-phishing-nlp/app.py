"""Flask web UI for Email Phishing Detector (port 5005).

Loads the trained VotingEnsemble and classifies email text as phishing or legitimate.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from classify import classify as classify_email, load_model

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Phishing Detector</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2e3a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --phish: #ff5252; --safe: #4caf50;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
       min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.container { width: 100%; max-width: 680px; }
textarea { width: 100%; height: 160px; padding: 1rem; font-size: 0.95rem; border: 1px solid var(--border);
           border-radius: 8px; background: var(--surface); color: var(--text); resize: vertical; outline: none; font-family: monospace; }
textarea:focus { border-color: var(--accent); }
button { margin-top: 0.75rem; padding: 0.75rem 1.5rem; font-size: 1rem; border: none; border-radius: 8px;
         background: var(--accent); color: white; cursor: pointer; }
button:hover { opacity: 0.85; }
button:disabled { opacity: 0.4; }
.result { margin-top: 1.5rem; display: none; }
.result.show { display: block; }
.badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
.badge.PHISHING { background: rgba(255,82,82,0.15); color: var(--phish); }
.badge.LEGITIMATE { background: rgba(76,175,80,0.15); color: var(--safe); }
.conf { margin-top: 0.5rem; font-size: 0.85rem; color: var(--muted); }
.signals { margin-top: 1rem; }
.signals h3 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.4rem; }
.signal { display: inline-block; padding: 0.2rem 0.6rem; margin: 0.2rem; border-radius: 4px; font-size: 0.8rem;
          font-family: monospace; background: var(--surface); border: 1px solid var(--border); }
.error { color: var(--phish); margin-top: 0.75rem; font-size: 0.9rem; display: none; }
.error.show { display: block; }
</style>
</head>
<body>
<div class="container">
  <h1>Email Phishing <span>Detector</span></h1>
  <p class="subtitle">Paste email text (body or raw email with headers) to classify.</p>
  <textarea id="text" placeholder="Enter email text here..."></textarea>
  <button id="btn" onclick="classify()">Classify</button>
  <div id="error" class="error"></div>
  <div id="result" class="result">
    <span id="badge" class="badge"></span>
    <div id="conf" class="conf"></div>
    <div class="signals"><h3>Top Signals</h3><div id="signals"></div></div>
  </div>
</div>
<script>
async function classify() {
  const text = document.getElementById('text').value.trim();
  if (!text) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'Classifying...';
  document.getElementById('error').classList.remove('show');
  document.getElementById('result').classList.remove('show');
  try {
    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text}) });
    const data = await res.json();
    if (!res.ok) { document.getElementById('error').textContent = data.error || 'Failed'; document.getElementById('error').classList.add('show'); return; }
    const badge = document.getElementById('badge');
    badge.className = 'badge ' + data.label;
    badge.textContent = data.label;
    document.getElementById('conf').textContent = 'Phishing probability: ' + (data.phishing_probability * 100).toFixed(1) + '% | Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
    const sigDiv = document.getElementById('signals');
    sigDiv.innerHTML = '';
    data.top_signals.forEach(s => {
      const span = document.createElement('span');
      span.className = 'signal';
      span.textContent = s[0] + ' = ' + s[1];
      sigDiv.appendChild(span);
    });
    document.getElementById('result').classList.add('show');
  } catch(e) { document.getElementById('error').textContent = 'Network error'; document.getElementById('error').classList.add('show'); }
  finally { btn.disabled = false; btn.textContent = 'Classify'; }
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
    model = get_model()
    if model is None:
        return jsonify({"error": "No trained model found. Run 'python run.py train' first."}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        result = classify_email(text, model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("  Email Phishing Detector -- Web UI")
    print("  Open http://127.0.0.1:5005")
    app.run(debug=True, host="127.0.0.1", port=5005)
