"""
Train multiple ML classifiers for email phishing detection.

Primary dataset: OpenML spambase.
Alternative: SpamAssassin corpus features.

Models: LogisticRegression, RandomForest, GradientBoosting, SVM, XGBoost, Voting Ensemble
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from joblib import dump, load

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"

TEST_SIZE = 0.25
RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    family: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_time_s: float
    predict_time_s: float
    confusion_matrix: list[list[int]]
    notes: str = ""


def _make_pipeline(clf, scaler=True):
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


MODELS: list[tuple[str, str, Pipeline]] = [
    ("LogisticRegression", "linear",
     _make_pipeline(LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE))),
    ("RandomForest", "ensemble",
     _make_pipeline(RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                           random_state=RANDOM_STATE), scaler=False)),
    ("GradientBoosting", "ensemble",
     _make_pipeline(GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                               random_state=RANDOM_STATE), scaler=False)),
    ("SVM_RBF", "svm",
     _make_pipeline(SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=RANDOM_STATE))),
    ("GaussianNB", "probabilistic",
     _make_pipeline(GaussianNB())),
]


def _try_xgboost():
    try:
        from xgboost import XGBClassifier
        return [("XGBoost", "ensemble",
                 _make_pipeline(XGBClassifier(n_estimators=200, max_depth=6,
                                              learning_rate=0.1, random_state=RANDOM_STATE,
                                              verbosity=0, use_label_encoder=False,
                                              eval_metric="logloss"), scaler=False))]
    except ImportError:
        return []


def load_data():
    """Load Spambase from CSV (fallback) or OpenML."""
    csv_path = DATA_DIR / "spambase.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values.astype(np.float64)
        y = df["label"].values.astype(np.int64)
        print(f"  Loaded spambase.csv ({len(df)} samples, {len(feature_cols)} features)")
    else:
        try:
            from sklearn.datasets import fetch_openml
            spambase = fetch_openml(name="spambase", version=1, as_frame=True)
            df = spambase.frame
            feature_cols = [c for c in df.columns if c != "class"]
            X = df[feature_cols].values.astype(np.float64)
            y = df["class"].astype(int).values
            print(f"  Loaded spambase from OpenML ({len(df)} samples, {len(feature_cols)} features)")
        except Exception as e:
            raise RuntimeError(f"Cannot load dataset: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    np.savez(PREDICTIONS_DIR / f"{name}.npz", y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    dump(pipeline, MODEL_DIR / f"{name}.joblib")

    return ModelResult(
        name=name,
        family=pipeline.named_steps.get("clf", pipeline).__class__.__name__,
        accuracy=round(acc, 4), precision=round(prec, 4),
        recall=round(rec, 4), f1=round(f1, 4), roc_auc=round(roc, 4),
        train_time_s=round(train_time, 3), predict_time_s=round(predict_time, 3),
        confusion_matrix=cm,
    )


def build_ensemble(X_train, y_train, X_test, y_test, results):
    """Build Voting ensemble from top-3 models by F1."""
    sorted_results = sorted(results, key=lambda r: r.f1, reverse=True)
    top3 = sorted_results[:3]

    estimators = []
    for r in top3:
        try:
            pipe = load(MODEL_DIR / f"{r.name}.joblib")
            estimators.append((r.name, pipe))
        except Exception:
            pass

    if len(estimators) < 2:
        return ModelResult(name="VotingEnsemble", family="ensemble",
                           accuracy=0, precision=0, recall=0, f1=0, roc_auc=0,
                           train_time_s=0, predict_time_s=0,
                           confusion_matrix=[[0, 0], [0, 0]],
                           notes="Not enough models for ensemble")

    t0 = time.perf_counter()
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = ensemble.predict(X_test)
    predict_time = time.perf_counter() - t0

    try:
        y_prob = ensemble.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    dump(ensemble, MODEL_DIR / "VotingEnsemble.joblib")
    np.savez(PREDICTIONS_DIR / "VotingEnsemble.npz",
             y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    return ModelResult(
        name="VotingEnsemble", family="ensemble",
        accuracy=round(acc, 4), precision=round(prec, 4),
        recall=round(rec, 4), f1=round(f1, 4), roc_auc=round(roc, 4),
        train_time_s=round(train_time, 3), predict_time_s=round(predict_time, 3),
        confusion_matrix=cm,
        notes=f"Soft-vote of {', '.join(r.name for r in top3)}",
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Email Phishing Detection — Model Comparison")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"\n  Data: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{len(feature_cols)} features")
    print(f"  Phishing rate: {y_train.mean():.1%} train, {y_test.mean():.1%} test")

    models = list(MODELS)
    models.extend(_try_xgboost())

    results = []
    for name, family, pipeline in models:
        print(f"\n  {'─' * 50}")
        print(f"  Training {name}...")
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)
        print(f"  Accuracy:  {result.accuracy:.4f}")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1 Score:  {result.f1:.4f}")
        print(f"  ROC-AUC:   {result.roc_auc:.4f}")
        print(f"  T: {result.train_time_s:.3f}s  P: {result.predict_time_s:.3f}s")

    # Ensemble
    print(f"\n  {'─' * 50}")
    print("  Building Voting Ensemble...")
    ensemble_result = build_ensemble(X_train, y_train, X_test, y_test, results)
    results.append(ensemble_result)
    print(f"  F1 Score:  {ensemble_result.f1:.4f}  ROC-AUC: {ensemble_result.roc_auc:.4f}")

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

    rows = []
    for r in results:
        rows.append({
            "Model": r.name, "Family": r.family,
            "Accuracy": r.accuracy, "Precision": r.precision,
            "Recall": r.recall, "F1": r.f1, "ROC-AUC": r.roc_auc,
            "Train_Time_s": r.train_time_s, "Predict_Time_s": r.predict_time_s,
        })
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"  {METRICS_CSV}")

    # Feature importance from RF
    try:
        rf_pipe = load(MODEL_DIR / "RandomForest.joblib")
        rf = rf_pipe.named_steps["clf"]
        importances = sorted(zip(feature_cols, rf.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        pd.DataFrame(importances, columns=["Feature", "Importance"]).to_csv(
            RESULTS_DIR / "feature_importance.csv", index=False)
        print("\n  Top-10 features:")
        for name, imp in importances[:10]:
            print(f"    {name:30s} {imp:.4f}")
    except Exception:
        pass

    # Final summary
    print(f"\n  {'=' * 50}")
    print(f"  {'Model':25s} {'F1':8s} {'ROC-AUC':8s} {'Accuracy':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:25s} {r.f1:.4f}   {r.roc_auc:.4f}   {r.accuracy:.4f}")
    print()


if __name__ == "__main__":
    main()
