"""
CLI tool to classify a single email text as phishing or legitimate.

Usage:
    python classify.py "Your email text here"
    python classify.py --file email.txt
    python classify.py --interactive
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from extract_features import extract_all_features, features_to_vector, FEATURE_NAMES

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
DEFAULT_MODEL = "VotingEnsemble"


def load_model(model_name=DEFAULT_MODEL):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        print(f"Model not found: {path}")
        print("Run train_phishing_model.py first.")
        sys.exit(1)
    return joblib.load(path)


def classify(email_text, model=None, model_name=DEFAULT_MODEL):
    if model is None:
        model = load_model(model_name)

    features = extract_all_features(email_text)
    X = np.array([features_to_vector(features)], dtype=np.float64)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    spam_prob = float(proba[1])
    confidence = spam_prob if pred == 1 else 1 - spam_prob

    # Top signals
    signals = []
    for name in FEATURE_NAMES:
        val = features.get(name, 0)
        if abs(val) > 0:
            signals.append((name.replace("_", " ").title(), val))

    return {
        "prediction": pred,
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_probability": round(spam_prob, 4),
        "confidence": round(confidence, 4),
        "top_signals": sorted(signals, key=lambda x: abs(x[1]), reverse=True)[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Classify email as phishing or legitimate")
    parser.add_argument("text", nargs="?", help="Email text to classify")
    parser.add_argument("--file", "-f", help="Read email from file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.interactive:
        print(f"Email Phishing Classifier (model: {args.model})")
        print("Enter email text (Ctrl+Z then Enter to finish):")
        lines = []
        try:
            for line in sys.stdin:
                lines.append(line)
        except KeyboardInterrupt:
            pass
        text = "".join(lines).strip()
        if not text:
            print("No input provided.")
            return
        result = classify(text, model)
    elif args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        result = classify(text, model)
    elif args.text:
        result = classify(args.text, model)
    else:
        parser.print_help()
        return

    if args.json or args.interactive:
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"\n  {'=' * 50}")
        print(f"  Classification: {result['label']}")
        print(f"  {'=' * 50}")
        print(f"  Phishing probability: {result['phishing_probability']:.4f}")
        print(f"  Confidence:          {result['confidence']:.4f}")
        print(f"\n  Top signals:")
        for name, val in result["top_signals"]:
            print(f"    {name:35s} {val}")


if __name__ == "__main__":
    main()
