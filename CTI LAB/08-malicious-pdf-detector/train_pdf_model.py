"""
Train classifiers for malicious PDF detection.

Models: LogisticRegression, RandomForest, XGBoost, SVM, Voting Ensemble
"""

import json, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump, load

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"
TEST_SIZE = 0.25; RANDOM_STATE = 42

@dataclass
class ModelResult:
    name: str; family: str; accuracy: float; precision: float; recall: float
    f1: float; roc_auc: float; train_time_s: float; predict_time_s: float
    confusion_matrix: list[list[int]]; notes: str = ""

def load_data():
    csv_path = DATA_DIR / "pdf_features.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "label"]
    else:
        npz_path = DATA_DIR / "pdf_features.npz"
        if not npz_path.exists():
            raise RuntimeError("Run generate_pdf_data.py first.")
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        from generate_pdf_data import FEATURE_NAMES
        feature_cols = FEATURE_NAMES
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        return X_train, X_test, y_train, y_test, feature_cols

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    t0 = time.perf_counter(); pipeline.fit(X_train, y_train); train_time = time.perf_counter() - t0
    t0 = time.perf_counter(); y_pred = pipeline.predict(X_test); predict_time = time.perf_counter() - t0
    try: y_prob = pipeline.predict_proba(X_test)[:, 1]
    except: y_prob = y_pred.astype(float)
    acc = accuracy_score(y_test, y_pred); prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0); f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob); cm = confusion_matrix(y_test, y_pred).tolist()
    np.savez(PREDICTIONS_DIR / f"{name}.npz", y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    dump(pipeline, MODEL_DIR / f"{name}.joblib")
    return ModelResult(name=name, family=pipeline.named_steps.get("clf", pipeline).__class__.__name__,
                       accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4),
                       f1=round(f1,4), roc_auc=round(roc,4), train_time_s=round(train_time,3),
                       predict_time_s=round(predict_time,3), confusion_matrix=cm)

def build_ensemble(X_train, y_train, X_test, y_test, results):
    top3 = sorted(results, key=lambda r: r.f1, reverse=True)[:3]
    estimators = []
    for r in top3:
        try:
            pipe = load(MODEL_DIR / f"{r.name}.joblib"); estimators.append((r.name, pipe))
        except: pass
    if len(estimators) < 2:
        return ModelResult(name="VotingEnsemble", family="ensemble", accuracy=0, precision=0,
                           recall=0, f1=0, roc_auc=0, train_time_s=0, predict_time_s=0,
                           confusion_matrix=[[0,0],[0,0]], notes="Not enough models")
    t0 = time.perf_counter(); ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train); train_time = time.perf_counter() - t0
    t0 = time.perf_counter(); y_pred = ensemble.predict(X_test); predict_time = time.perf_counter() - t0
    try: y_prob = ensemble.predict_proba(X_test)[:, 1]
    except: y_prob = y_pred.astype(float)
    acc = accuracy_score(y_test, y_pred); prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0); f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob); cm = confusion_matrix(y_test, y_pred).tolist()
    dump(ensemble, MODEL_DIR / "VotingEnsemble.joblib")
    np.savez(PREDICTIONS_DIR / "VotingEnsemble.npz", y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    return ModelResult(name="VotingEnsemble", family="ensemble",
                       accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4),
                       f1=round(f1,4), roc_auc=round(roc,4), train_time_s=round(train_time,3),
                       predict_time_s=round(predict_time,3), confusion_matrix=cm,
                       notes=f"Soft-vote of {', '.join(r.name for r in top3)}")

def main():
    RESULTS_DIR.mkdir(exist_ok=True); PREDICTIONS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 60); print("  Malicious PDF Detection — Model Comparison"); print("=" * 60)
    X_train, X_test, y_train, y_test, _ = load_data()
    print(f"\n  Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    print(f"  Malicious rate: {y_train.mean():.1%} train, {y_test.mean():.1%} test")

    models = [
        ("LogisticRegression", "linear", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE))])),
        ("RandomForest", "ensemble", Pipeline([("clf", RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE))])),
        ("GradientBoosting", "ensemble", Pipeline([("clf", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE))])),
        ("SVM_RBF", "svm", Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE))])),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost", "ensemble", Pipeline([("clf", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss"))])))
    except: pass

    results = []
    for name, family, pipeline in models:
        print(f"\n  Training {name}...")
        r = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test); results.append(r)
        print(f"  F1: {r.f1:.4f}  ROC-AUC: {r.roc_auc:.4f}  Acc: {r.accuracy:.4f}")

    print("\n  Building Voting Ensemble...")
    ensemble_r = build_ensemble(X_train, y_train, X_test, y_test, results); results.append(ensemble_r)
    print(f"  F1: {ensemble_r.f1:.4f}  ROC-AUC: {ensemble_r.roc_auc:.4f}")

    metrics_dict = {}
    for r in results:
        d = asdict(r); d["confusion_matrix"] = r.confusion_matrix; metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f: json.dump(metrics_dict, f, indent=2)
    print(f"\n  {METRICS_JSON}")

    print(f"\n  {'Model':22s} {'F1':8s} {'ROC-AUC':8s} {'Acc':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:22s} {r.f1:.4f}  {r.roc_auc:.4f}  {r.accuracy:.4f}")

if __name__ == "__main__":
    main()
