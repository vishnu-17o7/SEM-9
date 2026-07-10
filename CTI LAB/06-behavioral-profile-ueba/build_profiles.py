"""
Build behavioral baselines and detect anomalies using multiple techniques.

Methods:
- Gaussian Mixture Model (GMM)
- PCA + Mahalanobis distance
- IsolationForest
- Autoencoder (MLPRegressor reconstruction error)
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.covariance import MinCovDet
from joblib import dump

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"


@dataclass
class DetectorResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_time_s: float
    predict_time_s: float
    confusion_matrix: list[list[int]]
    anomaly_count: int


def engineer_features(df):
    """Extract per-user features from activity logs."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Per-user, per-hour window features ────────────────────────────────
    df["hour_window"] = df["timestamp"].dt.floor("1h")

    window_agg = df.groupby(["user_id", "hour_window"]).agg(
        action_count=("action", "count"),
        unique_resources=("resource", "nunique"),
        file_downloads=("action", lambda x: (x == "file_download").sum()),
        db_queries=("action", lambda x: (x == "db_query").sum()),
        privilege_changes=("action", lambda x: (x == "privilege_change").sum()),
        failure_count=("success", lambda x: (x == 0).sum()),
        unique_actions=("action", "nunique"),
    ).reset_index()
    window_agg["failure_rate"] = window_agg["failure_count"] / window_agg["action_count"].clip(lower=1)

    # Add hour and day features
    window_agg["hour_of_day"] = window_agg["hour_window"].dt.hour
    window_agg["day_of_week"] = window_agg["hour_window"].dt.dayofweek

    # ── Per-user long-term statistics (for deviation scoring) ─────────────
    user_stats = df.groupby("user_id").agg(
        user_total_actions=("action", "count"),
        user_avg_hour=("hour_of_day", "mean"),
        user_resource_diversity=("resource", "nunique"),
        user_failure_rate=("success", lambda x: 1 - x.mean()),
        user_off_hours_ratio=("hour_of_day", lambda x: (x < 7).mean()),
    ).reset_index()

    # Merge user stats
    window_agg = window_agg.merge(user_stats, on="user_id", how="left")

    # ── Role-based peer group features ────────────────────────────────────
    role_map = df[["user_id", "role"]].drop_duplicates().set_index("user_id")["role"].to_dict()
    window_agg["role"] = window_agg["user_id"].map(role_map)

    role_stats = window_agg.groupby("role").agg(
        role_avg_actions=("action_count", "mean"),
        role_avg_resources=("unique_resources", "mean"),
        role_avg_failure_rate=("failure_rate", "mean"),
        role_avg_off_hours=("hour_of_day", lambda x: (x < 7).mean()),
    ).reset_index().rename(columns={
        "role_avg_actions": "role_avg_actions",
        "role_avg_resources": "role_avg_resources",
        "role_avg_failure_rate": "role_avg_failure_rate",
        "role_avg_off_hours": "role_avg_off_hours",
    })

    window_agg = window_agg.merge(role_stats, on="role", how="left")

    # Deviation scores
    window_agg["action_deviation"] = (window_agg["action_count"] - window_agg["role_avg_actions"]) / window_agg["role_avg_actions"].clip(lower=1)
    window_agg["resource_deviation"] = (window_agg["unique_resources"] - window_agg["role_avg_resources"]) / window_agg["role_avg_resources"].clip(lower=1)
    window_agg["off_hours_deviation"] = window_agg["hour_of_day"].apply(lambda x: 1 if x < 7 or x > 20 else 0)

    # Carry the per-window anomaly label from the original event-level data
    anomaly_per_window = df.groupby(["user_id", "hour_window"])["is_anomaly"].max().reset_index()
    window_agg = window_agg.merge(anomaly_per_window, on=["user_id", "hour_window"], how="left")

    feature_cols = [
        "action_count", "unique_resources", "file_downloads", "db_queries",
        "privilege_changes", "failure_count", "failure_rate",
        "unique_actions", "hour_of_day", "user_total_actions", "user_avg_hour",
        "user_resource_diversity", "user_failure_rate", "user_off_hours_ratio",
        "action_deviation", "resource_deviation", "off_hours_deviation",
    ]

    df_features = window_agg.fillna(0).copy()
    # Ensure is_anomaly is integer for downstream evaluation
    if "is_anomaly" in df_features.columns:
        df_features["is_anomaly"] = df_features["is_anomaly"].astype(int)
    return df_features, feature_cols


def evaluate_detector(name, y_true, y_pred, y_score=None, train_time=0, predict_time=0):
    cm = confusion_matrix(y_true, y_pred).tolist() if len(np.unique(y_true)) > 1 else \
        [[len(y_true[y_true == 0]), 0], [0, len(y_true[y_true == 1])]]
    roc = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) > 1 else 0.0

    return DetectorResult(
        name=name, accuracy=round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
        roc_auc=round(roc, 4),
        train_time_s=round(train_time, 3), predict_time_s=round(predict_time, 3),
        confusion_matrix=cm, anomaly_count=int(y_pred.sum()),
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Behavioral Profile / UEBA — Anomaly Detection")
    print("=" * 60)

    csv_path = DATA_DIR / "user_activity.csv"
    if not csv_path.exists():
        print("  Run generate_behavior_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df)} activity entries")

    raw_df, feature_cols = engineer_features(df)
    print(f"  Engineered {len(feature_cols)} features across {len(raw_df)} windows")

    X = raw_df[feature_cols].values.astype(np.float64)
    y = raw_df["is_anomaly"].values.astype(np.int64) if "is_anomaly" in raw_df.columns else None

    if y is None:
        print("  No ground truth (is_anomaly) column — using unsupervised mode only")
        y = np.zeros(len(X))

    # Clean
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"  After cleaning: {len(X)} samples, anomaly rate: {y.mean():.1%}")

    # Temporal split
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # ── 1. GMM ────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  Gaussian Mixture Model...")
    t0 = time.perf_counter()
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(X_train_s)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    gmm_scores = -gmm.score_samples(X_test_s)
    pred_time = time.perf_counter() - t0
    threshold = np.percentile(gmm_scores, 95)
    gmm_pred = (gmm_scores > threshold).astype(int)
    results.append(evaluate_detector("GMM", y_test, gmm_pred, gmm_scores, train_time, pred_time))
    dump({"model": gmm, "scaler": scaler, "features": feature_cols}, MODEL_DIR / "GMM.joblib")
    np.savez(PREDICTIONS_DIR / "GMM.npz", y_true=y_test, y_pred=gmm_pred, y_score=gmm_scores)
    print(f"  F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    # ── 2. PCA + Mahalanobis ──────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  PCA + Mahalanobis...")
    t0 = time.perf_counter()
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_train_s)
    # Robust covariance estimation
    mcd = MinCovDet(random_state=42).fit(X_pca)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_test_pca = pca.transform(X_test_s)
    pca_scores = mcd.mahalanobis(X_test_pca)
    pred_time = time.perf_counter() - t0
    threshold = np.percentile(pca_scores, 95)
    pca_pred = (pca_scores > threshold).astype(int)
    results.append(evaluate_detector("PCA_Mahalanobis", y_test, pca_pred, pca_scores, train_time, pred_time))
    print(f"  F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")
    np.savez(PREDICTIONS_DIR / "PCA_Mahalanobis.npz", y_true=y_test, y_pred=pca_pred, y_score=pca_scores)

    # ── 3. IsolationForest ────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  IsolationForest...")
    t0 = time.perf_counter()
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    iso_scores = -iso.score_samples(X_test_s)
    pred_time = time.perf_counter() - t0
    iso_pred = iso.predict(X_test_s)
    iso_pred = (iso_pred == -1).astype(int)
    results.append(evaluate_detector("IsolationForest", y_test, iso_pred, iso_scores, train_time, pred_time))
    dump({"model": iso, "scaler": scaler, "features": feature_cols}, MODEL_DIR / "IsolationForest.joblib")
    np.savez(PREDICTIONS_DIR / "IsolationForest.npz", y_true=y_test, y_pred=iso_pred, y_score=iso_scores)
    print(f"  F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    # ── 4. Autoencoder (MLPRegressor) ─────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  Autoencoder (MLP reconstruction)...")
    t0 = time.perf_counter()
    ae = MLPRegressor(
        hidden_layer_sizes=(X_train_s.shape[1] * 2, X_train_s.shape[1] // 2, X_train_s.shape[1] * 2),
        activation="relu", max_iter=200, random_state=42, early_stopping=True,
    )
    ae.fit(X_train_s, X_train_s)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_recon = ae.predict(X_test_s)
    pred_time = time.perf_counter() - t0
    ae_scores = np.mean((X_test_s - X_recon) ** 2, axis=1)
    threshold = np.percentile(ae_scores, 95)
    ae_pred = (ae_scores > threshold).astype(int)
    results.append(evaluate_detector("Autoencoder", y_test, ae_pred, ae_scores, train_time, pred_time))
    dump({"model": ae, "scaler": scaler, "features": feature_cols}, MODEL_DIR / "Autoencoder.joblib")
    np.savez(PREDICTIONS_DIR / "Autoencoder.npz", y_true=y_test, y_pred=ae_pred, y_score=ae_scores)
    print(f"  F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    # Save metrics
    metrics_dict = {}
    for r in results:
        d = asdict(r)
        d["confusion_matrix"] = r.confusion_matrix
        metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n  {METRICS_JSON}")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  {'Detector':20s} {'F1':8s} {'ROC-AUC':8s} {'Precision':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:20s} {r.f1:.4f}  {r.roc_auc:.4f}  {r.precision:.4f}")
    print()


if __name__ == "__main__":
    main()
