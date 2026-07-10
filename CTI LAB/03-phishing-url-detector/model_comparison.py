"""
Train and compare multiple ML classifiers for phishing URL detection.

Models:
  - Logistic Regression        (baseline linear)
  - Random Forest              (tree ensemble)
  - Gradient Boosting          (boosted trees)
  - Support Vector Machine     (RBF kernel)
  - k-Nearest Neighbors        (distance-based)
  - Gaussian Naive Bayes       (probabilistic)
  - XGBoost (if available)     (extreme gradient boosting)
  - Voting Ensemble            (soft-vote of top-3)
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
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from joblib import dump, load

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
FEATURES_FILE = DATA_DIR / "phishing_features.csv"
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


# ── Model definitions ──────────────────────────────────────────────────────

def _make_pipeline(clf, scaler=True):
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


MODELS: list[tuple[str, str, Pipeline]] = [
    (
        "LogisticRegression",
        "linear",
        _make_pipeline(LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ),
    (
        "RandomForest",
        "ensemble",
        _make_pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE),
            scaler=False,
        ),
    ),
    (
        "GradientBoosting",
        "ensemble",
        _make_pipeline(
            GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
            scaler=False,
        ),
    ),
    (
        "SVM_RBF",
        "svm",
        _make_pipeline(SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
    ),
    (
        "KNN",
        "distance",
        _make_pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ),
    (
        "GaussianNB",
        "probabilistic",
        _make_pipeline(GaussianNB()),
    ),
]


def _try_xgboost() -> list[tuple[str, str, Pipeline]]:
    """Add XGBoost if the package is available."""
    try:
        from xgboost import XGBClassifier
        return [
            (
                "XGBoost",
                "ensemble",
                _make_pipeline(
                    XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric="logloss",
                                  random_state=RANDOM_STATE, verbosity=0),
                    scaler=False,
                ),
            ),
        ]
    except ImportError:
        return []


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load features, split into train/test, return feature names."""
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> ModelResult:
    """Train a model, evaluate, return ModelResult."""
    # Train
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Predict
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Probabilities for ROC-AUC
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

    # Save predictions
    np.savez(
        PREDICTIONS_DIR / f"{name}.npz",
        y_true=y_test, y_pred=y_pred, y_prob=y_prob,
    )

    # Save model
    dump(pipeline, MODEL_DIR / f"{name}.joblib")

    return ModelResult(
        name=name,
        family=pipeline.named_steps.get("clf", pipeline).__class__.__name__,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(roc, 4),
        train_time_s=round(train_time, 3),
        predict_time_s=round(predict_time, 3),
        confusion_matrix=cm,
        notes="",
    )


def build_ensemble(X_train, y_train, X_test, y_test, results: list[ModelResult]) -> ModelResult:
    """Build a Voting ensemble from the top-3 models by F1."""
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
        return ModelResult(
            name="VotingEnsemble", family="ensemble",
            accuracy=0, precision=0, recall=0, f1=0, roc_auc=0,
            train_time_s=0, predict_time_s=0, confusion_matrix=[[0, 0], [0, 0]],
            notes="Not enough models for ensemble",
        )

    # Re-train ensemble on training data
    # Use raw features (no scaler) and let each internal pipeline handle it
    X_train_raw = X_train
    X_test_raw = X_test

    t0 = time.perf_counter()
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train_raw, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = ensemble.predict(X_test_raw)
    predict_time = time.perf_counter() - t0

    try:
        y_prob = ensemble.predict_proba(X_test_raw)[:, 1]
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
        recall=round(rec, 4), f1=round(f1, 4),
        roc_auc=round(roc, 4), train_time_s=round(train_time, 3),
        predict_time_s=round(predict_time, 3), confusion_matrix=cm,
        notes=f"Soft-vote of {', '.join(r.name for r in top3)}",
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Phishing URL Detection — Model Comparison")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"\nData: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{len(feature_cols)} features")
    print(f"Phishing rate: {y_train.mean():.1%} train, {y_test.mean():.1%} test")

    # Build model list
    models = list(MODELS)
    models.extend(_try_xgboost())

    # Train & evaluate each model
    results: list[ModelResult] = []
    for name, family, pipeline in models:
        print(f"\n{'─' * 50}")
        print(f"  Training {name}...")
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

        print(f"  Accuracy:  {result.accuracy:.4f}")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1 Score:  {result.f1:.4f}")
        print(f"  ROC-AUC:   {result.roc_auc:.4f}")
        print(f"  ⏱  Train: {result.train_time_s:.3f}s  Predict: {result.predict_time_s:.3f}s")

    # Build ensemble
    print(f"\n{'─' * 50}")
    print("  Building Voting Ensemble from top-3 models...")
    ensemble_result = build_ensemble(X_train, y_train, X_test, y_test, results)
    results.append(ensemble_result)
    print(f"  F1 Score:  {ensemble_result.f1:.4f}")
    print(f"  ROC-AUC:   {ensemble_result.roc_auc:.4f}")

    # Save metrics
    print(f"\n{'─' * 50}")
    print("  Saving results...")

    # JSON
    metrics_dict = {}
    for r in results:
        d = asdict(r)
        d["confusion_matrix"] = r.confusion_matrix
        metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  ✓ {METRICS_JSON}")

    # CSV
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "Family": r.family,
            "Accuracy": r.accuracy,
            "Precision": r.precision,
            "Recall": r.recall,
            "F1": r.f1,
            "ROC-AUC": r.roc_auc,
            "Train_Time_s": r.train_time_s,
            "Predict_Time_s": r.predict_time_s,
            "Notes": r.notes,
        })
    pdf = pd.DataFrame(rows)
    pdf.to_csv(METRICS_CSV, index=False)
    print(f"  ✓ {METRICS_CSV}")

    # Save feature importance (from Random Forest)
    try:
        rf_pipe = dump(MODEL_DIR / "RandomForest.joblib", None)
        rf = rf_pipe.named_steps["clf"]
        importances = sorted(zip(feature_cols, rf.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        imp_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
        imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
        print(f"  ✓ feature_importance.csv")
        print("\n  Top-10 features:")
        for name, imp in importances[:10]:
            print(f"    {name:30s} {imp:.4f}")
    except Exception:
        pass

    # Summary
    print(f"\n{'=' * 60}")
    print("  Model Comparison Complete — Summary")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':25s} {'F1':8s} {'ROC-AUC':8s} {'Accuracy':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:25s} {r.f1:.4f}   {r.roc_auc:.4f}   {r.accuracy:.4f}")
    print()


if __name__ == "__main__":
    main()
