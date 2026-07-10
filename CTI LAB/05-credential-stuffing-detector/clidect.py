"""
CLI for real-time credential stuffing detection.

Usage:
    python clidect.py --log "username=alice,timestamp=2026-06-22T10:00:00,source_ip=1.2.3.4,success=0"
    python clidect.py --file logs.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"


def parse_log_line(line):
    """Parse a comma-separated key=value log line into a dict."""
    parts = {}
    for kv in line.strip().split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            parts[k.strip()] = v.strip()
    # Add defaults for missing fields
    parts.setdefault("user_agent", "Mozilla/5.0")
    parts.setdefault("geo_country", "US")
    return parts


def main():
    parser = argparse.ArgumentParser(description="Credential stuffing real-time detection")
    parser.add_argument("--log", help="Single log entry as comma-separated key=value pairs")
    parser.add_argument("--file", help="Batch detect from CSV file")
    parser.add_argument("--model", default="RandomForest", help="Model to use (RandomForest, RuleBased, IsolationForest, LOF)")
    parser.add_argument("--list-models", action="store_true", help="List available trained models")
    args = parser.parse_args()

    if args.list_models:
        if MODEL_DIR.exists():
            print("Available models:")
            for p in MODEL_DIR.glob("*.joblib"):
                print(f"  {p.stem}")
        else:
            print("No trained models found. Run detect_stuffing.py first.")
        return

    if not args.log and not args.file:
        parser.print_help()
        return

    # Load model
    model_path = MODEL_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        print(f"Model {args.model} not found. Available models:")
        for p in MODEL_DIR.glob("*.joblib"):
            print(f"  {p.stem}")
        sys.exit(1)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact.get("scaler")
    features = artifact.get("features", [])

    if args.log:
        entry = parse_log_line(args.log)
        # Create quick feature vector with defaults
        feats = {
            "login_count": 1, "failure_count": 1 if entry.get("success") == "0" else 0,
            "failure_rate": 1.0 if entry.get("success") == "0" else 0.0,
            "unique_ips": 1, "unique_uas": 1, "unique_users": 1, "unique_countries": 1,
            "ip_entropy": 0.0, "user_failure_rate": 0.5, "user_unique_ips": 1,
            "user_unique_countries": 1, "time_since_last": 0.1, "attempts_per_second": 10.0,
        }
        X = np.array([[feats.get(f, 0) for f in features]], dtype=np.float64)
        if scaler:
            X = scaler.transform(X)
        pred = model.predict(X)[0]
        is_anomaly = bool(pred == -1 if hasattr(model, "predict") else pred == 1)

        result = {
            "log": entry,
            "anomaly": is_anomaly,
            "verdict": "ATTACK" if is_anomaly else "NORMAL",
            "model": args.model,
        }
        print(json.dumps(result, indent=2))

    elif args.file:
        df = pd.read_csv(args.file)
        print(f"Processing {len(df)} entries...")
        # Use predefined features - average across file
        feature_cols = features or [c for c in df.columns if c not in ("timestamp", "is_attack")]
        if not all(c in df.columns for c in feature_cols):
            print("CSV missing required columns. Ensure it has the same structure as login_logs.csv")
            return
        X = df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X)
        if scaler:
            X = scaler.transform(X)
        preds = model.predict(X)
        df["prediction"] = (preds == -1).astype(int) if hasattr(model, "predict") else preds
        attack_count = df["prediction"].sum()
        print(f"  Detected {attack_count} attacks out of {len(df)} entries ({attack_count / len(df) * 100:.1f}%)")
        out_path = Path(args.file).parent / f"detected_{Path(args.file).name}"
        df.to_csv(out_path, index=False)
        print(f"  Saved results to {out_path}")


if __name__ == "__main__":
    main()
