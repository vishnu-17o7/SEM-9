"""Train, evaluate, and persist comparisons of multiple SMS spam classifiers.

Compares 6 base models + 2 ensembles on the same train/test split:

Base models
-----------
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC
- Random Forest
- Gradient Boosting
- Complement Naive Bayes (good for imbalanced text)

Ensembles
---------
- Soft Voting (NB + LR + LinearSVC via calibrated probabilities)
- Stacking (NB + LR + GBDT → Logistic Regression meta-learner)

All models share the same TF-IDF (1-2 grams, English stop words, max 10k features).

Outputs
-------
- results/metrics.json  — raw metrics for every model
- results/metrics.csv   — flat table for easy diffing
- results/predictions/*.npz  — y_true + y_pred + y_prob per model
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sms_preprocessing import load_split

RESULTS_DIR = Path(__file__).parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
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
    confusion_matrix: list[list[int]] = field(default_factory=list)
    notes: str = ""
    estimator: str = ""


def _build_models() -> dict[str, tuple[str, Any]]:
    """Return {name: (family, sklearn-compatible estimator)}."""
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", MultinomialNB()),
    ])
    cnb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", ComplementNB()),
    ])
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    svm = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    gb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])

    # LinearSVC wrapped with calibration so soft-voting can use its probabilities
    svm_cal = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3, n_jobs=-1)),
    ])

    soft_vote = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", VotingClassifier(
            estimators=[
                ("nb", MultinomialNB()),
                ("cnb", ComplementNB()),
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ("svm", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3, n_jobs=-1)),
            ],
            voting="soft",
            n_jobs=-1,
        )),
    ])

    stack = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
        ("clf", StackingClassifier(
            estimators=[
                ("nb", MultinomialNB()),
                ("cnb", ComplementNB()),
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            cv=5,
            n_jobs=-1,
            passthrough=False,
        )),
    ])

    return {
        "MultinomialNB":      ("base",     nb),
        "ComplementNB":       ("base",     cnb),
        "LogisticRegression": ("base",     lr),
        "LinearSVC":          ("base",     svm),
        "RandomForest":       ("base",     rf),
        "GradientBoosting":   ("base",     gb),
        "SoftVoting":         ("ensemble", soft_vote),
        "Stacking":           ("ensemble", stack),
    }


def _score(name: str, family: str, model, X_train, X_test, y_train, y_test) -> ModelResult:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # LinearSVC: use decision_function, min-max scaled to [0, 1]
        from scipy.special import expit
        raw = model.decision_function(X_test)
        y_prob = expit(raw)

    cm = confusion_matrix(y_test, y_pred).tolist()

    notes = ""
    if family == "ensemble":
        notes = "ensemble of NB + CNB + LR + SVM/GBDT"

    return ModelResult(
        name=name,
        family=family,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_prob),
        train_time_s=round(train_time, 3),
        predict_time_s=round(predict_time, 3),
        confusion_matrix=cm,
        notes=notes,
    )


def main() -> list[ModelResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SMS Spam Collection...")
    X_train, X_test, y_train, y_test = load_split()
    print(f"  train: {len(X_train):,}  test: {len(X_test):,}  "
          f"spam ratio train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")
    print("-" * 72)

    models = _build_models()
    results: list[ModelResult] = []
    for name, (family, model) in models.items():
        print(f"\n[{family:8s}] {name}")
        result = _score(name, family, model, X_train, X_test, y_train, y_test)
        results.append(result)
        print(f"  acc={result.accuracy:.4f}  prec={result.precision:.4f}  "
              f"rec={result.recall:.4f}  f1={result.f1:.4f}  "
              f"auc={result.roc_auc:.4f}  "
              f"train={result.train_time_s:.2f}s  predict={result.predict_time_s:.3f}s")
        print(f"  confusion: {result.confusion_matrix}")

        # save predictions for the report
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            from scipy.special import expit
            y_prob = expit(model.decision_function(X_test))
        np.savez(
            PRED_DIR / f"{name}.npz",
            y_true=np.asarray(y_test),
            y_pred=np.asarray(y_pred),
            y_prob=np.asarray(y_prob),
        )

    # Save best model for web UI / inference
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(results, key=lambda r: r.f1).name
    joblib.dump(models[best_name][1], model_dir / "best_model.joblib")
    print(f"\n  Saved best model: {best_name} (F1={max(r.f1 for r in results):.4f})"
          f" -> {model_dir / 'best_model.joblib'}")

    # persist
    metrics_json = RESULTS_DIR / "metrics.json"
    metrics_json.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    csv_lines = ["name,family,accuracy,precision,recall,f1,roc_auc,train_time_s,predict_time_s,notes"]
    for r in results:
        csv_lines.append(
            f"{r.name},{r.family},{r.accuracy:.6f},{r.precision:.6f},"
            f"{r.recall:.6f},{r.f1:.6f},{r.roc_auc:.6f},"
            f"{r.train_time_s},{r.predict_time_s},\"{r.notes}\""
        )
    (RESULTS_DIR / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("Saved:")
    print(f"  {metrics_json}")
    print(f"  {RESULTS_DIR / 'metrics.csv'}")
    print(f"  {PRED_DIR}/*.npz")
    return results


if __name__ == "__main__":
    main()
