"""Flask web UI for SMS Spam Classifier (port 5004).

Loads the best trained model and classifies SMS text as spam or ham.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template_string, request

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_PATH = RESULTS_DIR / "models" / "best_model.joblib"

app = Flask(__name__)
model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"  [OK] Loaded model: {MODEL_PATH.name}")
    else:
        print(f"  [!] Model not found: {MODEL_PATH}")
        print("      Run 'python run.py train' first.")


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMS Spam Classifier</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2e3a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --spam: #ff5252; --ham: #4caf50;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
       min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.container { width: 100%; max-width: 640px; }
textarea { width: 100%; height: 120px; padding: 1rem; font-size: 1rem; border: 1px solid var(--border);
           border-radius: 8px; background: var(--surface); color: var(--text); resize: vertical; outline: none; }
textarea:focus { border-color: var(--accent); }
button { margin-top: 0.75rem; padding: 0.75rem 1.5rem; font-size: 1rem; border: none; border-radius: 8px;
         background: var(--accent); color: white; cursor: pointer; transition: opacity 0.2s; }
button:hover { opacity: 0.85; }
button:disabled { opacity: 0.4; cursor: default; }
.result { margin-top: 1.5rem; display: none; }
.result.show { display: block; }
.badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
.badge.spam { background: rgba(255,82,82,0.15); color: var(--spam); }
.badge.ham { background: rgba(76,175,80,0.15); color: var(--ham); }
.prob-bar { margin-top: 0.75rem; height: 8px; background: var(--surface); border-radius: 4px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; transition: width 0.4s; width: 0%; }
.prob-fill.spam { background: var(--spam); }
.prob-fill.ham { background: var(--ham); }
.prob-text { margin-top: 0.3rem; font-size: 0.85rem; color: var(--muted); text-align: right; }
.error { color: var(--spam); margin-top: 0.75rem; font-size: 0.9rem; display: none; }
.error.show { display: block; }
</style>
</head>
<body>
<div class="container">
  <h1>SMS Spam <span>Classifier</span></h1>
  <p class="subtitle">Paste an SMS message to classify it as spam or ham.</p>
  <textarea id="text" placeholder="Enter SMS text here..."></textarea>
  <button id="btn" onclick="classify()">Classify</button>
  <div id="error" class="error"></div>
  <div id="result" class="result">
    <span id="badge" class="badge"></span>
    <div class="prob-bar"><div id="probFill" class="prob-fill"></div></div>
    <div id="probText" class="prob-text"></div>
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
    const isSpam = data.prediction === 1;
    const badge = document.getElementById('badge');
    badge.className = 'badge ' + (isSpam ? 'spam' : 'ham');
    badge.textContent = isSpam ? 'SPAM' : 'HAM';
    const fill = document.getElementById('probFill');
    fill.className = 'prob-fill ' + (isSpam ? 'spam' : 'ham');
    const pct = (data.probability * 100).toFixed(1);
    fill.style.width = pct + '%';
    document.getElementById('probText').textContent = (isSpam ? 'Spam' : 'Ham') + ' probability: ' + pct + '%';
    document.getElementById('result').classList.add('show');
  } catch(e) { document.getElementById('error').textContent = 'Network error'; document.getElementById('error').classList.add('show'); }
  finally { btn.disabled = false; btn.textContent = 'Classify'; }
}
document.getElementById('text').addEventListener('keydown', e => { if(e.key==='Enter' && e.ctrlKey) classify(); });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        load_model()
    if model is None:
        return jsonify({"error": "No trained model found. Run 'python run.py train' first."}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        prediction = int(model.predict([text])[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba([text])[0, 1])
        else:
            probability = 0.9 if prediction == 1 else 0.1
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": prediction,
        "label": "spam" if prediction == 1 else "ham",
        "probability": probability,
    })


if __name__ == "__main__":
    print("  SMS Spam Classifier -- Web UI")
    load_model()
    print("  Open http://127.0.0.1:5004")
    app.run(debug=True, host="127.0.0.1", port=5004)
