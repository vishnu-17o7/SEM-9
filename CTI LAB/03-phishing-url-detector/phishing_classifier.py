"""
CLI tool to classify a single URL — phishing or legitimate.

Classification flow:
  1. Check the URL against the training dataset (fast, exact lookup).
  2. If not found, fall back to the ML ensemble model.

Usage:
    python phishing_classifier.py "https://example.com"
    python phishing_classifier.py --interactive
    python phishing_classifier.py --model RandomForest "https://..."
    echo "https://..." | python phishing_classifier.py
"""

import csv
import sys
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import joblib
import tldextract
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from url_feature_extractor import extract_features, FEATURE_NAMES, BRAND_DOMAINS

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
DATASET_FILE = Path(__file__).parent / "data" / "phishing_url_dataset.csv"

# Best model (highest F1 from training)
DEFAULT_MODEL = "VotingEnsemble"  # or "RandomForest"

PHISHING_INDICATORS: list[str] = [
    "has_ip_address",
    "has_suspicious_words",
    "has_shortener",
    "num_at_symbols",
    "has_double_slash_redirect",
    "has_typosquatting",
    "has_brand_in_domain",
]

LEGITIMATE_INDICATORS: list[str] = [
    "has_https",
    "has_www",
]


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

    print(f"  Loading dataset lookup ({DATASET_FILE.stat().st_size / 1_000_000:.1f} MB) ...")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            label = int(row.get("label", "0"))
            if url:
                lookup[_normalize_url_for_lookup(url)] = label

    print(f"  [OK] {len(lookup):,} URLs indexed for fast lookup.")
    return lookup


_dataset_lookup_cache: dict | None = None


def _get_dataset_lookup() -> dict:
    global _dataset_lookup_cache
    if _dataset_lookup_cache is None:
        _dataset_lookup_cache = load_dataset_lookup()
    return _dataset_lookup_cache


def load_model(model_name: str = DEFAULT_MODEL):
    """Load a trained model pipeline."""
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        print(f"  ✗ Model not found: {path}")
        print(f"    Run `python model_comparison.py` first to train models.")
        sys.exit(1)
    return joblib.load(path)


def _fmt(val, width=14):
    """Format a value for table display."""
    if isinstance(val, float):
        return f"{val:<{width}.4f}"
    return f"{str(val):<{width}}"


def explain_prediction(features: dict, prediction: int, probability: float) -> str:
    """Generate a human-readable explanation of the prediction."""
    lines = []
    # Suspicious signals
    sus_signals = []
    for indicator in PHISHING_INDICATORS:
        if indicator in features and features[indicator] > 0:
            sus_signals.append(indicator.replace("_", " ").title())

    # Legitimate signals
    leg_signals = []
    for indicator in LEGITIMATE_INDICATORS:
        if indicator in features and features[indicator] > 0:
            leg_signals.append(indicator.replace("_", " ").title())

    confidence = probability if prediction == 1 else 1 - probability

    if prediction == 1:
        lines.append("  🔴 Phishing detected")
        if sus_signals:
            lines.append(f"     Suspicious signals: {', '.join(sus_signals)}")
        if leg_signals:
            lines.append(f"     (still has: {', '.join(leg_signals)})")
    else:
        lines.append("  🟢 Legitimate website")
        if leg_signals:
            lines.append(f"     Positive signals: {', '.join(leg_signals)}")
        if sus_signals:
            lines.append(f"     Warning: contains {', '.join(sus_signals)} "
                         f"— verify manually")

    lines.append(f"     Confidence: {confidence:.1%}")
    return "\n".join(lines)


def classify_url(url: str, model_name: str = DEFAULT_MODEL, verbose: bool = True) -> dict:
    """Classify a single URL, return result dict. Tier 1: dataset lookup → Tier 2: ML model."""
    features = extract_features(url)

    # ── Tier 1: Dataset lookup ──
    lookup = _get_dataset_lookup()
    normalized = _normalize_url_for_lookup(url)
    dataset_match = lookup.get(normalized)

    if dataset_match is not None:
        prediction = dataset_match
        probability = 0.99 if prediction == 1 else 0.01
        source = "dataset"
    else:
        # ── Tier 2: ML model ──
        model = load_model(model_name)
        X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        source = "model"

    # ── Whitelist override ──
    try:
        extracted = tldextract.extract(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        if registered_domain in BRAND_DOMAINS:
            prediction = 0
            probability = min(probability, 0.01)
        elif extracted.domain.lower() in {d.split(".")[0] for d in BRAND_DOMAINS}:
            prediction = 0
            probability = min(probability, 0.01)
    except Exception:
        pass

    label = "phishing" if prediction == 1 else "legitimate"

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  URL: {url}")
        print(f"{'=' * 60}")
        print(f"  Source: {'📚 DATASET (exact match)' if source == 'dataset' else '🤖 ML MODEL'}")
        print(f"  Prediction: {'🔴 PHISHING' if prediction == 1 else '🟢 LEGITIMATE'}")
        print(f"  Probability: {probability:.2%}")
        print(f"  Model: {model_name}")
        print(f"\n  Feature breakdown:")
        print(f"  {'─' * 50}")

        # Show only non-zero/significant features
        significant = {k: v for k, v in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
                       if abs(v) > 0}
        for name, val in significant.items():
            print(f"    {name.replace('_', ' ').title():30s} = {_fmt(val)}")

        print(f"\n  Explanation:")
        print(explain_prediction(features, prediction, probability))
        print()

    return {"url": url, "prediction": prediction, "label": label,
            "probability": probability, "features": features, "source": source}


def interactive_mode(model_name: str = DEFAULT_MODEL):
    """Interactive loop for classifying URLs."""
    print(f"\n  Phishing URL Classifier (model: {model_name})")
    print(f"  Tier 1: Dataset lookup → Tier 2: ML model fallback")
    print(f"  Type a URL to classify, or 'quit' to exit.\n")
    while True:
        try:
            url = input("  URL> ").strip()
            if not url:
                continue
            if url.lower() in ("quit", "exit", "q"):
                break
            # Add https:// if no scheme
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            classify_url(url, model_name=model_name)
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a URL as phishing or legitimate.",
    )
    parser.add_argument("url", nargs="?", help="URL to classify")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON (quiet)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")

    args = parser.parse_args()

    # List models
    if args.list_models:
        models = list(MODEL_DIR.glob("*.joblib"))
        if not models:
            print("  No trained models found. Run `python model_comparison.py` first.")
            return
        print("  Available models:")
        for m in sorted(models):
            print(f"    {m.stem}")
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(model_name=args.model)
        return

    # URL from pipe/stdin
    if args.url is None and not sys.stdin.isatty():
        url = sys.stdin.read().strip()
        if url:
            args.url = url

    if not args.url:
        parser.print_help()
        return

    # Add https:// if no scheme
    url = args.url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    result = classify_url(url, model_name=args.model, verbose=not args.json)

    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
