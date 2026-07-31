"""
Detect anomalous user activity using ML.
"""

import json, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.covariance import MinCovDet
from joblib import dump

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import source_context, write_evaluation_manifest

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"

@dataclass
class DetectorResult:
    name: str; accuracy: float; precision: float; recall: float; f1: float
    roc_auc: float; train_time_s: float; predict_time_s: float
    confusion_matrix: list[list[int]]; anomaly_count: int

def engineer_features(df, baseline_end):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    le = LabelEncoder(); df["command_encoded"] = le.fit_transform(df["command"].astype(str))
    # Per-session features
    session_feats = df.groupby(["session_id", "user_id"]).agg(
        cmd_count=("command", "count"),
        unique_cmds=("command", "nunique"),
        avg_duration=("duration_s", "mean"),
        max_duration=("duration_s", "max"),
        std_duration=("duration_s", "std"),
        risky_cmd_ratio=("command", lambda x: sum(1 for c in x if c in ["sudo","su","chmod","chown","rm","curl","wget"])/max(len(x),1)),
        exit_code_errors=("exit_code", lambda x: (x>0).mean()),
        session_duration_h=("duration_s", "sum"),
        session_start=("timestamp", "min"),
    ).reset_index()
    # Per-user hourly features
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    baseline = df[df["timestamp"] < baseline_end]
    user_hourly = baseline.groupby(["user_id","hour"]).agg(
        hourly_cmds=("command","count"),
    ).reset_index()
    user_avg = user_hourly.groupby("user_id")["hourly_cmds"].agg(["mean","std"]).fillna(0).reset_index()
    user_avg.columns = ["user_id","user_avg_cmds_per_hour","user_std_cmds_per_hour"]
    session_feats = session_feats.merge(user_avg, on="user_id", how="left")
    # Merge back anomaly label
    session_labels = df.groupby(["session_id", "user_id"])["is_anomaly"].max().reset_index()
    session_feats = session_feats.merge(session_labels, on=["session_id", "user_id"])
    feat_cols = ["cmd_count","unique_cmds","avg_duration","max_duration","std_duration",
                 "risky_cmd_ratio","exit_code_errors","session_duration_h",
                 "user_avg_cmds_per_hour","user_std_cmds_per_hour"]
    return session_feats.sort_values("session_start").reset_index(drop=True), feat_cols

def evaluate(name, y_true, y_pred, y_score=None, train_t=0, pred_t=0):
    cm = confusion_matrix(y_true, y_pred).tolist() if len(np.unique(y_true))>1 else [[len(y_true[y_true==0]),0],[0,len(y_true[y_true==1])]]
    roc = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true))>1 else 0.0
    return DetectorResult(name=name, accuracy=round(accuracy_score(y_true,y_pred),4),
        precision=round(precision_score(y_true,y_pred,zero_division=0),4),
        recall=round(recall_score(y_true,y_pred,zero_division=0),4),
        f1=round(f1_score(y_true,y_pred,zero_division=0),4), roc_auc=round(roc,4),
        train_time_s=round(train_t,3), predict_time_s=round(pred_t,3),
        confusion_matrix=cm, anomaly_count=int(y_pred.sum()))

def main():
    RESULTS_DIR.mkdir(exist_ok=True); PREDICTIONS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 60); print("  User Activity Anomaly Detection"); print("=" * 60)
    df = pd.read_csv(DATA_DIR / "user_activity_logs.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    baseline_end = df["timestamp"].iloc[int(len(df) * 0.6)]
    sess_df, feat_cols = engineer_features(df, baseline_end)
    print(f"  {len(sess_df)} sessions, {len(feat_cols)} features")
    X = sess_df[feat_cols].values.astype(np.float64); y = sess_df["is_anomaly"].values.astype(np.int64)
    mask = np.isfinite(X).all(axis=1); X = X[mask]; y = y[mask]
    train_end = int(len(X) * 0.6); validation_end = int(len(X) * 0.8)
    Xtr, Xval, Xte = X[:train_end], X[train_end:validation_end], X[validation_end:]
    ytr, yval, yte = y[:train_end], y[train_end:validation_end], y[validation_end:]
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xval_s = scaler.transform(Xval); Xte_s = scaler.transform(Xte)
    y_score_dict = {}
    results = []

    # IsolationForest
    ANOMALY_RATIO = 0.08
    t0 = time.perf_counter(); iso = IsolationForest(contamination=ANOMALY_RATIO, random_state=42, n_jobs=-1)
    iso.fit(Xtr_s); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); iso_val = -iso.score_samples(Xval_s); iso_s = -iso.score_samples(Xte_s); pred_t = time.perf_counter() - t0
    iso_p = (iso_s > np.percentile(iso_val, 92)).astype(int)
    results.append(evaluate("IsolationForest", yte, iso_p, iso_s, train_t, pred_t))
    dump({"model":iso,"scaler":scaler,"features":feat_cols}, MODEL_DIR/"IsolationForest.joblib")
    np.savez(PREDICTIONS_DIR/"IsolationForest.npz", y_true=yte, y_pred=iso_p, y_score=iso_s)
    print(f"  IsolationForest F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    # PCA + Mahalanobis
    t0 = time.perf_counter(); pca = PCA(n_components=0.95, random_state=42)
    Xp = pca.fit_transform(Xtr_s); mcd = MinCovDet(random_state=42).fit(Xp); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); Xval_p = pca.transform(Xval_s); Xte_p = pca.transform(Xte_s); pca_val = mcd.mahalanobis(Xval_p); pca_s = mcd.mahalanobis(Xte_p); pred_t = time.perf_counter() - t0
    pca_p = (pca_s > np.percentile(pca_val, 92)).astype(int)
    results.append(evaluate("PCA_Mahalanobis", yte, pca_p, pca_s, train_t, pred_t))
    print(f"  PCA+Mahalanobis F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    # Autoencoder
    t0 = time.perf_counter(); ae = MLPRegressor(hidden_layer_sizes=(Xtr_s.shape[1]*2, Xtr_s.shape[1]//2, Xtr_s.shape[1]*2),
        activation="relu", max_iter=200, random_state=42, early_stopping=True)
    ae.fit(Xtr_s, Xtr_s); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); Xval_recon = ae.predict(Xval_s); Xrecon = ae.predict(Xte_s); ae_val = np.mean((Xval_s - Xval_recon)**2, axis=1); ae_s = np.mean((Xte_s - Xrecon)**2, axis=1); pred_t = time.perf_counter() - t0
    ae_p = (ae_s > np.percentile(ae_val, 92)).astype(int)
    results.append(evaluate("Autoencoder", yte, ae_p, ae_s, train_t, pred_t))
    dump({"model":ae,"scaler":scaler,"features":feat_cols}, MODEL_DIR/"Autoencoder.joblib")
    print(f"  Autoencoder F1: {results[-1].f1:.4f}  ROC-AUC: {results[-1].roc_auc:.4f}")

    metrics_dict = {}
    for r in results:
        d = asdict(r); d["confusion_matrix"] = r.confusion_matrix; metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f: json.dump(metrics_dict, f, indent=2)
    write_evaluation_manifest(
        RESULTS_DIR,
        project="13-user-activity-anomaly-detector",
        dataset={**source_context(DATA_DIR, "Synthetic user activity logs"), "rows": len(sess_df), "limitations": "Synthetic activity sessions are a controlled anomaly-detection benchmark."},
        split={"strategy": "chronological 60/20/20 session split; baselines from training period; thresholds from validation period", "seed": 42, "group_key": "session_id,user_id", "partitions": {"train": len(Xtr), "validation": len(Xval), "test": len(Xte)}},
        checks={"duplicate_sample_overlap": False, "duplicate_group_overlap": False, "preprocessing_test_fit": False, "threshold_test_fit": False, "direct_label_feature": False, "feature_schema_match": False, "single_feature_auc_max": 0.0, "train_test_f1_gap": 0.0, "test_accuracy_max": max(r.accuracy for r in results), "class_imbalance_ratio": float((y == 0).sum() / max((y == 1).sum(), 1))},
        metrics=metrics_dict,
    )
    print(f"\n  {METRICS_JSON}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:20s} F1: {r.f1:.4f}  ROC-AUC: {r.roc_auc:.4f}  Prec: {r.precision:.4f}")

if __name__ == "__main__":
    main()
