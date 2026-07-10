"""
Credential stuffing detection — anomaly detection + supervised comparison.

Models: Rule-based, IsolationForest, RandomForest, LocalOutlierFactor
Dataset: Generated from generate_login_logs.py (690k+ entries)

Reference: Wiefling et al., ACM TOPS 2022 (RBA on large-scale SSO)
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from joblib import dump

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"
TIME_WINDOW_MINUTES = 5


@dataclass
class DetectorResult:
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


def engineer_features(df):
    """Extract time-windowed and per-user features from login logs."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Per-5-minute window features ──────────────────────────────────────
    df["window"] = df["timestamp"].dt.floor(f"{TIME_WINDOW_MINUTES}min")

    window_agg = df.groupby("window").agg(
        login_count=("success", "count"),
        failure_count=("success", lambda x: (x == 0).sum()),
        unique_ips=("source_ip", "nunique"),
        unique_uas=("user_agent", "nunique"),
        unique_users=("username", "nunique"),
        unique_countries=("geo_country", "nunique"),
    ).reset_index()
    window_agg["failure_rate"] = window_agg["failure_count"] / window_agg["login_count"].clip(lower=1)

    # IP entropy per window
    ip_entropy = df.groupby("window")["source_ip"].apply(
        lambda x: -sum((v / len(x)) * np.log2(v / len(x)) for v in x.value_counts().values)
    ).reset_index(name="ip_entropy")

    window_agg = window_agg.merge(ip_entropy, on="window")

    # ── Per-user features ─────────────────────────────────────────────────
    user_agg = df.groupby("username").agg(
        user_total_logins=("success", "count"),
        user_failures=("success", lambda x: (x == 0).sum()),
        user_unique_ips=("source_ip", "nunique"),
        user_unique_countries=("geo_country", "nunique"),
    ).reset_index()
    user_agg["user_failure_rate"] = user_agg["user_failures"] / user_agg["user_total_logins"].clip(lower=1)

    # Merge window features into original df, then add user features
    df = df.merge(window_agg, on="window", how="left")
    df = df.merge(user_agg, on="username", how="left")

    # ── Velocity features ─────────────────────────────────────────────────
    df_sorted = df.sort_values(["username", "timestamp"])
    df_sorted["prev_timestamp"] = df_sorted.groupby("username")["timestamp"].shift(1)
    df_sorted["time_since_last"] = (
        df_sorted["timestamp"] - df_sorted["prev_timestamp"]
    ).dt.total_seconds().fillna(300)

    # Attempts per second in current window (per user)
    df_sorted["attempts_per_second"] = 1.0 / df_sorted["time_since_last"].clip(lower=0.1)

    df = df_sorted

    # ── Feature columns ───────────────────────────────────────────────────
    feature_cols = [
        "login_count", "failure_count", "failure_rate",
        "unique_ips", "unique_uas", "unique_users", "unique_countries",
        "ip_entropy", "user_failure_rate", "user_unique_ips",
        "user_unique_countries", "time_since_last", "attempts_per_second",
    ]

    # Fill NaN values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df, feature_cols


def rule_based_detector(df):
    """Simple threshold-based detection rules."""
    conditions = (
        (df["failure_rate"] > 0.7) &
        (df["unique_ips"] >= 3) &
        (df["attempts_per_second"] > 2)
    )
    return conditions.astype(int).values


def evaluate_detector(name, y_true, y_pred, train_time=0, predict_time=0):
    """Compute detection metrics."""
    # Handle potential issues
    cm = confusion_matrix(y_true, y_pred).tolist() if len(np.unique(y_true)) > 1 else \
        [[len(y_true[y_true == 0]), 0], [0, len(y_true[y_true == 1])]]

    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fpr = fp / max(fp + tn, 1)

    return DetectorResult(
        name=name,
        accuracy=round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
        train_time_s=round(train_time, 3),
        predict_time_s=round(predict_time, 3),
        confusion_matrix=cm,
        detection_count=int(y_pred.sum()),
        false_positive_rate=round(fpr, 4),
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Credential Stuffing Detection — Model Comparison")
    print("=" * 60)

    # Load data
    csv_path = DATA_DIR / "login_logs.csv"
    if not csv_path.exists():
        print("  No login_logs.csv found. Run generate_login_logs.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df)} login entries")

    # Engineer features
    df, feature_cols = engineer_features(df)
    print(f"  Engineered {len(feature_cols)} features")

    X = df[feature_cols].values.astype(np.float64)
    y = df["is_attack"].values.astype(np.int64)

    # Remove any rows with NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"  After cleaning: {len(X)} samples, {X.shape[1]} features")
    print(f"  Attack rate: {y.mean():.1%}")

    # Train/test split (temporal — use first 80% for training)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ── 1. Rule-based ─────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  Rule-Based Detector...")
    t0 = time.perf_counter()
    # Rules apply to the data beyond split_idx
    rule_pred = rule_based_detector(df.iloc[split_idx:])
    pred_time = time.perf_counter() - t0
    result = evaluate_detector("RuleBased", y_test, rule_pred, predict_time=pred_time)
    results.append(result)
    print(f"  F1: {result.f1:.4f}  Prec: {result.precision:.4f}  Rec: {result.recall:.4f}")

    # ── 2. IsolationForest ────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  IsolationForest...")
    t0 = time.perf_counter()
    # Use actual attack rate as contamination
    contam = max(0.01, y_train.mean())
    iso = IsolationForest(contamination=contam, random_state=42, n_jobs=-1)
    iso.fit(X_train_scaled)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_pred = iso.predict(X_test_scaled)
    pred_time = time.perf_counter() - t0
    iso_pred = (iso_pred == -1).astype(int)

    result = evaluate_detector("IsolationForest", y_test, iso_pred, train_time, pred_time)
    results.append(result)
    dump({"model": iso, "scaler": scaler, "features": feature_cols},
         MODEL_DIR / "IsolationForest.joblib")
    print(f"  F1: {result.f1:.4f}  Prec: {result.precision:.4f}  Rec: {result.recall:.4f}")
    np.savez(PREDICTIONS_DIR / "IsolationForest.npz", y_true=y_test, y_pred=iso_pred)

    # ── 3. RandomForest (supervised) ─────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  RandomForest (supervised)...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rf_pred = rf.predict(X_test_scaled)
    pred_time = time.perf_counter() - t0

    result = evaluate_detector("RandomForest", y_test, rf_pred, train_time, pred_time)
    results.append(result)
    dump({"model": rf, "scaler": scaler, "features": feature_cols},
         MODEL_DIR / "RandomForest.joblib")
    print(f"  F1: {result.f1:.4f}  Prec: {result.precision:.4f}  Rec: {result.recall:.4f}")
    np.savez(PREDICTIONS_DIR / "RandomForest.npz", y_true=y_test, y_pred=rf_pred)

    # ── 4. LocalOutlierFactor ─────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  LocalOutlierFactor...")
    # Subsample for LOF since it's O(n²)
    sample_n = min(20000, len(X_test_scaled))
    idx = np.random.default_rng(42).choice(len(X_test_scaled), sample_n, replace=False)
    X_sub = X_test_scaled[idx]
    y_sub = y_test[idx]
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contam, novelty=False)
    lof_pred = lof.fit_predict(X_sub)
    pred_time = time.perf_counter() - t0
    lof_pred = (lof_pred == -1).astype(int)

    result = evaluate_detector("LOF", y_sub, lof_pred, predict_time=pred_time)
    results.append(result)
    print(f"  F1: {result.f1:.4f}  Prec: {result.precision:.4f}  Rec: {result.recall:.4f}")
    np.savez(PREDICTIONS_DIR / "LOF.npz", y_true=y_sub, y_pred=lof_pred)

    # Save metrics
    print(f"\n  {'─' * 50}")
    print("  Saving results...")

    metrics_dict = {}
    for r in results:
        d = asdict(r)
        d["confusion_matrix"] = r.confusion_matrix
        metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  {METRICS_JSON}")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  {'Detector':20s} {'F1':8s} {'Precision':10s} {'Recall':8s} {'FPR':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:20s} {r.f1:.4f}  {r.precision:.4f}     {r.recall:.4f}     {r.false_positive_rate:.4f}")

    # Per-class report for the best supervised model
    print(f"\n  Classification Report (RandomForest):")
    print(classification_report(y_test, rf_pred, target_names=["Normal", "Attack"], zero_division=0))
    print()


if __name__ == "__main__":
    main()
