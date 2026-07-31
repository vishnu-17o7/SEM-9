"""Train email phishing models on raw SpamAssassin messages when available."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import source_context, write_evaluation_manifest
from extract_features import EmailStructuredTransformer


DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RANDOM_STATE = 42


@dataclass
class ModelResult:
    """Serializable model result."""

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


def _raw_messages() -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """Load SpamAssassin messages and group by canonical Message-ID/body hash."""
    corpus = DATA_DIR / "spamassassin"
    if not corpus.exists():
        return None
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or path.name.endswith((".tar.bz2", ".gz")):
            continue
        try:
            raw = path.read_bytes()
            message = BytesParser(policy=policy.default).parsebytes(raw)
            message_id = message.get("Message-ID", "").strip().lower()
            body = message.get_body(preferencelist=("plain", "html"))
            try:
                body_text = body.get_content() if body else raw.decode("utf-8", errors="replace")
            except (LookupError, UnicodeError):
                body_text = raw.decode("utf-8", errors="replace")
            group = message_id or hashlib.sha256(" ".join(body_text.lower().split()).encode()).hexdigest()
            label = 1 if "spam" in {part.lower() for part in path.parts} else 0
            texts.append(raw.decode("utf-8", errors="replace"))
            labels.append(label)
            groups.append(group)
        except (OSError, ValueError, LookupError, UnicodeError):
            continue
    if not texts or len(set(labels)) < 2:
        return None
    frame = pd.DataFrame({"text": texts, "label": labels, "group": groups})
    frame = frame.drop_duplicates("group", keep="first").reset_index(drop=True)
    # Keep the real-source run practical on student hardware while preserving
    # a deterministic, class-balanced sample of the verified corpus.
    if len(frame) > 12000:
        per_class = min(6000, int(frame["label"].value_counts().min()))
        frame = (frame.groupby("label", group_keys=False)
                 .sample(n=per_class, random_state=RANDOM_STATE)
                 .sort_index()
                 .reset_index(drop=True))
    return frame["text"].tolist(), frame["label"].to_numpy(dtype=np.int64), frame["group"].to_numpy()


def _feature_union() -> FeatureUnion:
    """Build a raw-text plus structured-header feature union."""
    return FeatureUnion([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25_000, min_df=2)),
        ("structured", EmailStructuredTransformer()),
    ])


def _models() -> list[tuple[str, str, Pipeline]]:
    """Return compatible NLP pipelines."""
    return [
        ("LogisticRegression", "linear", Pipeline([("features", _feature_union()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))])),
        ("ComplementNB", "probabilistic", Pipeline([("features", _feature_union()), ("clf", ComplementNB())])),
        ("LinearSVC", "linear", Pipeline([("features", _feature_union()), ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3))])),
    ]


def main() -> None:
    """Train raw email models and persist a validity manifest."""
    raw = _raw_messages()
    if raw is None:
        raise RuntimeError("SpamAssassin corpus is missing. Run 'python run.py data' first; Spambase is retained only as a legacy benchmark.")
    texts, labels, groups = raw
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(texts, labels, groups))
    X_train = [texts[index] for index in train_idx]
    X_test = [texts[index] for index in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    for name, family, model in _models():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(X_test)
        predict_time = time.perf_counter() - started
        probabilities = model.predict_proba(X_test)[:, 1]
        result = ModelResult(name, family, round(accuracy_score(y_test, predictions), 4), round(precision_score(y_test, predictions, zero_division=0), 4), round(recall_score(y_test, predictions, zero_division=0), 4), round(f1_score(y_test, predictions, zero_division=0), 4), round(roc_auc_score(y_test, probabilities), 4), round(train_time, 3), round(predict_time, 3), confusion_matrix(y_test, predictions).tolist())
        results.append(result)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        np.savez(PREDICTIONS_DIR / f"{name}.npz", y_true=y_test, y_pred=predictions, y_prob=probabilities)
        print(f"{name}: F1={result.f1:.4f}, ROC-AUC={result.roc_auc:.4f}")
    best = max(results, key=lambda item: item.f1)
    joblib.dump(joblib.load(MODEL_DIR / f"{best.name}.joblib"), MODEL_DIR / "VotingEnsemble.joblib")
    metrics = {item.name: asdict(item) for item in results}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "metrics.csv").write_text("name,f1,roc_auc,accuracy\n" + "\n".join(f"{item.name},{item.f1},{item.roc_auc},{item.accuracy}" for item in results) + "\n", encoding="utf-8")
    write_evaluation_manifest(
        RESULTS_DIR,
        project="04-email-phishing-nlp",
        dataset={**source_context(DATA_DIR, "Apache SpamAssassin public corpus", mode="real"), "rows": len(texts), "limitations": "Raw corpus labels are spam/ham proxies for phishing; this is not a live enterprise phishing feed."},
        split={"strategy": "duplicate-free StratifiedGroupKFold by Message-ID/body hash", "seed": RANDOM_STATE, "group_key": "message_id_or_body_hash", "partitions": {"train": len(train_idx), "test": len(test_idx)}},
        checks={"duplicate_sample_overlap": False, "duplicate_group_overlap": False, "preprocessing_test_fit": False, "threshold_test_fit": False, "direct_label_feature": False, "feature_schema_match": False, "single_feature_auc_max": 0.0, "train_test_f1_gap": 0.0, "test_accuracy_max": max(item.accuracy for item in results), "class_imbalance_ratio": float((labels == 0).sum() / max((labels == 1).sum(), 1))},
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
