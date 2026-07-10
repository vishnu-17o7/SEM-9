"""
Flask web UI for the Phishing URL Detector.
Loads trained model, dataset lookup, and serves prediction via a web interface.

Classification flow:
  1. Check the URL against the training dataset (fast, exact lookup).
  2. If not found, fall back to the ML ensemble model.

Usage:
    .\venv\Scripts\activate
    pip install flask
    python app.py
    # Open http://127.0.0.1:5000
"""

import csv
import sys
import json
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tldextract

from url_feature_extractor import extract_features, FEATURE_NAMES, BRAND_DOMAINS

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
DATASET_FILE = Path(__file__).parent / "data" / "phishing_url_dataset.csv"
DEFAULT_MODEL = "VotingEnsemble"

app = Flask(__name__)

PHISHING_INDICATORS = [
    "has_ip_address", "has_suspicious_words", "has_shortener",
    "num_at_symbols", "has_double_slash_redirect",
    "has_typosquatting", "has_brand_in_domain",
]
LEGITIMATE_INDICATORS = ["has_https", "has_www"]


def _normalize_url_for_lookup(url: str) -> str:
    """Normalize a URL for dataset comparison: lowercase hostname, strip trailing slash, consistent scheme."""
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        p = urlparse(url)
        hostname = (p.hostname or "").lower()
        path = p.path or ""
        query = ("?" + p.query) if p.query else ""
        return f"{p.scheme}://{hostname}{path}{query}"
    except Exception:
        return url.lower()


def load_dataset_lookup() -> dict:
    """Build an O(1) lookup dict: normalized URL -> label (0=legitimate, 1=phishing)."""
    lookup = {}
    if not DATASET_FILE.exists():
        print(f"  [!] Dataset not found at {DATASET_FILE} — dataset lookup disabled.")
        return lookup

    print(f"  Loading dataset lookup from {DATASET_FILE.name} ...")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            label = int(row.get("label", "0"))
            if url:
                lookup[_normalize_url_for_lookup(url)] = label

    print(f"  [OK] {len(lookup):,} URLs indexed for fast lookup.")
    return lookup


def load_model(model_name=DEFAULT_MODEL):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


model = load_model()
dataset_lookup = load_dataset_lookup()


@app.route("/")
def index():
    models = sorted(m.stem for m in MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else []
    return render_template("index.html", models=models, default_model=DEFAULT_MODEL)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"].strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        features = extract_features(url)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    # ── Tier 1: Dataset lookup ──
    global dataset_lookup
    normalized = _normalize_url_for_lookup(url)
    dataset_match = dataset_lookup.get(normalized) if dataset_lookup else None

    if dataset_match is not None:
        prediction = dataset_match
        probability = 0.99 if prediction == 1 else 0.01
        source = "dataset"
        confidence = 0.99
    else:
        # ── Tier 2: ML model ──
        global model
        if model is None:
            model = load_model()
        if model is None:
            return jsonify({"error": "No trained model found. Run model_comparison.py first."}), 500

        X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        source = "model"

    # ── Whitelist override (applied regardless of source) ──
    try:
        extracted = tldextract.extract(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        if registered_domain in BRAND_DOMAINS:
            prediction = 0
            probability = 0.01
        elif extracted.domain.lower() in {d.split(".")[0] for d in BRAND_DOMAINS}:
            prediction = 0
            probability = 0.01
    except Exception:
        pass

    confidence = probability if prediction == 1 else 1 - probability

    # Build significant features list
    sig_features = []
    for name, val in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(val) > 0:
            sig_features.append({
                "name": name.replace("_", " ").title(),
                "value": val,
                "is_suspicious": name in PHISHING_INDICATORS and val > 0,
                "is_legitimate": name in LEGITIMATE_INDICATORS and val > 0,
            })

    sus_signals = [f.replace("_", " ").title() for f in PHISHING_INDICATORS
                   if f in features and features[f] > 0]
    leg_signals = [f.replace("_", " ").title() for f in LEGITIMATE_INDICATORS
                   if f in features and features[f] > 0]

    return jsonify({
        "url": url,
        "prediction": prediction,
        "label": "phishing" if prediction == 1 else "legitimate",
        "probability": probability,
        "confidence": confidence,
        "source": source,
        "sus_signals": sus_signals,
        "leg_signals": leg_signals,
        "features": sig_features,
    })


@app.route("/models", methods=["GET"])
def list_models():
    models_list = sorted(m.stem for m in MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else []
    return jsonify({"models": models_list, "default": DEFAULT_MODEL})


if __name__ == "__main__":
    print(f"  Phishing URL Detector — Web UI")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Open http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
