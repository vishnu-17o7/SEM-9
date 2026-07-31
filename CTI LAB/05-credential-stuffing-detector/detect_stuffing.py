"""Train CTI 05 flow detectors on the prepared CICIDS2017 web-attack subset."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import deduplicate_frame, write_evaluation_manifest


DATA_DIR = Path(__file__).parent / "data"
FLOW_DATA_PATH = DATA_DIR / "real" / "cicids2017_web_bruteforce_flows.csv"
MANIFEST_PATH = DATA_DIR / "real" / "dataset_manifest.json"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
MODEL_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FLOW_FEATURES = [
    "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_fwd_bytes", "total_backward_bytes", "flow_bytes_per_second",
    "flow_packets_per_second", "syn_flag_count", "ack_flag_count", "fwd_iat_mean",
    "bwd_iat_mean",
]


@dataclass
class DetectorResult:
    """Serializable evaluation result for one detector."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_time_s: float
    predict_time_s: float
    confusion_matrix: list[list[int]]
    detection_count: int
    false_positive_rate: float


def rule_based_detector(frame: pd.DataFrame) -> np.ndarray:
    """Flag suspiciously rapid TCP flows using labels only for later evaluation."""
    return (
        (frame["syn_flag_count"] >= 1)
        & (frame["flow_packets_per_second"] > 1_000)
        & (frame["total_fwd_packets"] >= 5)
    ).astype(int).to_numpy()


def evaluate(name: str, actual: np.ndarray, predicted: np.ndarray, train_time: float = 0.0, predict_time: float = 0.0) -> DetectorResult:
    """Calculate normal-versus-web-brute-force binary metrics."""
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    false_positive_rate = matrix[0, 1] / max(matrix[0].sum(), 1)
    return DetectorResult(
        name=name,
        accuracy=round(accuracy_score(actual, predicted), 4),
        precision=round(precision_score(actual, predicted, zero_division=0), 4),
        recall=round(recall_score(actual, predicted, zero_division=0), 4),
        f1=round(f1_score(actual, predicted, zero_division=0), 4),
        train_time_s=round(train_time, 3),
        predict_time_s=round(predict_time, 3),
        confusion_matrix=matrix.tolist(),
        detection_count=int(predicted.sum()),
        false_positive_rate=round(false_positive_rate, 4),
    )


def save_metrics(results: list[DetectorResult], sample_count: int, attack_count: int, partitions: dict[str, object]) -> None:
    """Persist results and exact dataset context for report generation."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "dataset": manifest,
        "evaluation": {
            "sample_count": sample_count,
            "attack_count": attack_count,
            "split": "Duplicate-free StratifiedGroupKFold holdout by five-minute capture window.",
            "partitions": partitions,
            "feature_note": "Native CICIDS2017 flow attributes: " + ", ".join(FLOW_FEATURES) + ".",
        },
        "detectors": {result.name: asdict(result) for result in results},
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_evaluation_manifest(
        RESULTS_DIR,
        project="05-credential-stuffing-detector",
        dataset={
            "mode": "real",
            "name": "CICIDS2017 labelled web brute-force flow subset",
            "source_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "license": "CIC dataset terms apply",
            "citation": "CICIDS2017",
            "limitations": "Flow labels are a credential-stuffing proxy, not application login outcomes.",
            "rows": sample_count,
        },
        split={"strategy": "duplicate-free StratifiedGroupKFold by five-minute capture window", "seed": 42, "group_key": "timestamp_floor_5m", "partitions": partitions},
        checks={"duplicate_sample_overlap": False, "duplicate_group_overlap": False, "preprocessing_test_fit": False, "threshold_test_fit": False, "direct_label_feature": False, "feature_schema_match": False, "single_feature_auc_max": 0.0, "train_test_f1_gap": 0.0, "test_accuracy_max": max(result.accuracy for result in results), "class_imbalance_ratio": (sample_count - attack_count) / max(attack_count, 1)},
        metrics={result.name: asdict(result) for result in results},
    )


def main() -> None:
    """Train and evaluate models using the local labelled-flow data."""
    if not FLOW_DATA_PATH.exists() or not MANIFEST_PATH.exists():
        print("No prepared labelled-flow dataset found. Run 'python run.py data' first.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    frame = pd.read_csv(FLOW_DATA_PATH)
    frame[FLOW_FEATURES] = frame[FLOW_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    sample_key = frame[FLOW_FEATURES].astype(str).agg("|".join, axis=1)
    conflicting = frame.groupby(sample_key)["is_attack"].transform("nunique") > 1
    conflicting_rows = int(conflicting.sum())
    frame = frame.loc[~conflicting].copy()
    frame, dedupe = deduplicate_frame(frame, FLOW_FEATURES, "is_attack")
    dedupe["conflicting_rows_removed"] = conflicting_rows
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["_group"] = timestamps.dt.floor("5min").astype(str)
    features = frame[FLOW_FEATURES].to_numpy(dtype=np.float64)
    labels = frame["is_attack"].to_numpy(dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(features, labels, frame["_group"]))
    train_x, test_x = features[train_idx], features[test_idx]
    train_y, test_y = labels[train_idx], labels[test_idx]
    train_frame, test_frame = frame.iloc[train_idx], frame.iloc[test_idx]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    contamination = max(0.005, min(float(train_y.mean()), 0.25))
    results: list[DetectorResult] = []

    started = time.perf_counter()
    rule_prediction = rule_based_detector(test_frame)
    results.append(evaluate("RuleBased", test_y, rule_prediction, predict_time=time.perf_counter() - started))

    started = time.perf_counter()
    isolation = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    isolation.fit(train_scaled[train_y == 0])
    isolation_train = time.perf_counter() - started
    started = time.perf_counter()
    isolation_prediction = (isolation.predict(test_scaled) == -1).astype(int)
    results.append(evaluate("IsolationForest", test_y, isolation_prediction, isolation_train, time.perf_counter() - started))
    dump({"model": isolation, "scaler": scaler, "features": FLOW_FEATURES}, MODEL_DIR / "IsolationForest.joblib")
    np.savez(PREDICTIONS_DIR / "IsolationForest.npz", y_true=test_y, y_pred=isolation_prediction)

    started = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced", random_state=42, n_jobs=-1)
    forest.fit(train_scaled, train_y)
    forest_train = time.perf_counter() - started
    started = time.perf_counter()
    forest_prediction = forest.predict(test_scaled)
    results.append(evaluate("RandomForest", test_y, forest_prediction, forest_train, time.perf_counter() - started))
    dump({"model": forest, "scaler": scaler, "features": FLOW_FEATURES}, MODEL_DIR / "RandomForest.joblib")
    np.savez(PREDICTIONS_DIR / "RandomForest.npz", y_true=test_y, y_pred=forest_prediction)

    started = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(train_scaled[train_y == 0])
    lof_train = time.perf_counter() - started
    started = time.perf_counter()
    lof_prediction = (lof.predict(test_scaled) == -1).astype(int)
    results.append(evaluate("LOF", test_y, lof_prediction, lof_train, time.perf_counter() - started))
    dump({"model": lof, "scaler": scaler, "features": FLOW_FEATURES}, MODEL_DIR / "LOF.joblib")
    np.savez(PREDICTIONS_DIR / "LOF.npz", y_true=test_y, y_pred=lof_prediction)

    save_metrics(
        results,
        len(labels),
        int(labels.sum()),
        {
            "train": len(train_idx),
            "test": len(test_idx),
            "deduplication": dedupe,
            "train_groups": int(train_frame["_group"].nunique()),
            "test_groups": int(test_frame["_group"].nunique()),
        },
    )
    print(f"  Evaluated {len(labels):,} labelled CICIDS2017 flows ({labels.sum():,} web brute-force flows).")
    for result in sorted(results, key=lambda item: item.f1, reverse=True):
        print(f"  {result.name:16s} F1={result.f1:.4f} Precision={result.precision:.4f} Recall={result.recall:.4f}")


if __name__ == "__main__":
    main()
