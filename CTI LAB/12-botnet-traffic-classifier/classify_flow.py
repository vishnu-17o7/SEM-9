"""
CLI to classify a network flow as botnet or normal.
Usage: python classify_flow.py --flow "src_bytes=1000,dst_bytes=2000,packets_sent=10,..."
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np
RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"

def main():
    parser = argparse.ArgumentParser(description="Classify network flow as botnet or normal")
    parser.add_argument("--flow", help="Comma-separated key=value flow features")
    parser.add_argument("--model", default="VotingEnsemble")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        if MODEL_DIR.exists():
            for p in MODEL_DIR.glob("*.joblib"): print(f"  {p.stem}")
        return
    if not args.flow: parser.print_help(); return
    model_path = MODEL_DIR / f"{args.model}.joblib"
    if not model_path.exists(): print(f"Model not found"); return
    pipe = joblib.load(model_path)
    parts = {}
    for kv in args.flow.split(","):
        if "=" in kv: k, v = kv.split("=", 1); parts[k.strip()] = float(v.strip())
    from generate_botnet_traffic import FEATURE_NAMES
    X = np.array([[parts.get(n, 0) for n in FEATURE_NAMES]], dtype=np.float64)
    pred = int(pipe.predict(X)[0])
    prob = float(pipe.predict_proba(X)[0, 1])
    print(json.dumps({"prediction": pred, "label": "BOTNET" if pred else "NORMAL", "botnet_probability": round(prob, 4)}, indent=2))

if __name__ == "__main__":
    main()
