"""Classify CICIDS2017-compatible network-flow records with a trained CTI 05 model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).parent / "results" / "models"


def parse_flow(text: str, features: list[str]) -> dict[str, float]:
    """Parse comma-separated feature=value flow fields and validate completeness."""
    values: dict[str, float] = {}
    for part in text.split(","):
        name, separator, value = part.partition("=")
        if not separator:
            raise ValueError("Flow fields must use feature=value syntax.")
        values[name.strip()] = float(value)
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing required flow features: {', '.join(missing)}")
    return values


def predict(model_artifact: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run model prediction and return labels plus attack probabilities."""
    features = model_artifact["features"]
    values = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Flow values must be finite numbers.")
    scaler = model_artifact.get("scaler")
    if scaler is not None:
        values = scaler.transform(values)
    model = model_artifact["model"]
    labels = model.predict(values).astype(int)
    probabilities = model.predict_proba(values)[:, 1] if hasattr(model, "predict_proba") else labels.astype(float)
    return labels, probabilities


def main() -> None:
    """Run single-flow or CSV classification using a saved model artifact."""
    parser = argparse.ArgumentParser(description="CTI 05 CICIDS2017 flow classifier")
    parser.add_argument("--flow", help="Comma-separated native CIC flow fields as feature=value")
    parser.add_argument("--file", type=Path, help="CSV containing every trained flow feature")
    parser.add_argument("--model", default="RandomForest", help="Saved model name")
    parser.add_argument("--list-models", action="store_true", help="List saved model artifacts")
    args = parser.parse_args()
    if args.list_models:
        for path in sorted(MODEL_DIR.glob("*.joblib")):
            print(f"  {path.stem}")
        return
    if not args.flow and not args.file:
        parser.error("Provide --flow or --file.")

    path = MODEL_DIR / f"{args.model}.joblib"
    if not path.exists():
        print(f"Model not found: {path}")
        sys.exit(1)
    artifact = joblib.load(path)
    features = artifact["features"]

    try:
        if args.flow:
            row = parse_flow(args.flow, features)
            labels, probabilities = predict(artifact, pd.DataFrame([row]))
            print(json.dumps({
                "verdict": "ATTACK" if labels[0] else "NORMAL",
                "attack_probability": round(float(probabilities[0]), 6),
                "model": args.model,
            }, indent=2))
            return

        frame = pd.read_csv(args.file)
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing flow features: {', '.join(missing)}")
        labels, probabilities = predict(artifact, frame)
        frame["prediction"] = labels
        frame["attack_probability"] = probabilities
        output = args.file.with_name(f"detected_{args.file.name}")
        frame.to_csv(output, index=False)
        print(f"Detected {int(labels.sum()):,} attacks out of {len(frame):,} flows.")
        print(f"Saved {output}")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
