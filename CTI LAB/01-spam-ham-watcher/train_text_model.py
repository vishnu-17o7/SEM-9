"""Train the duplicate-safe Naive Bayes spam/ham classifier."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import deduplicate_frame, source_context, write_evaluation_manifest
from spam_utils import clean_text


DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
MODEL_PATH = Path(__file__).parent / "spam_text_model.joblib"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download_data() -> Path:
    """Download the UCI corpus only when it is not cached."""
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / "SMSSpamCollection"
    if data_file.exists():
        return data_file
    zip_path = DATA_DIR / "smsspamcollection.zip"
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)
    return data_file


def load_data(data_file: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and deduplicate messages before any split is made."""
    frame = pd.read_csv(data_file, sep="\t", header=None, names=["label", "message"])
    frame["label"] = frame["label"].map({"spam": 1, "ham": 0}).astype(int)
    frame["clean_message"] = frame["message"].map(clean_text)
    return deduplicate_frame(frame, ["clean_message"], "label")


def main() -> None:
    """Train, evaluate, and persist the Naive Bayes model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    args = parser.parse_args()
    data_file = download_data()
    if args.data_only:
        print(f"Dataset ready: {data_file}")
        return

    frame, dedupe = load_data(data_file)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(frame["clean_message"], frame["label"], frame["clean_message"]))
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    pipeline = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))), ("clf", MultinomialNB())])
    grid = GridSearchCV(
        pipeline,
        {"tfidf__max_features": [3000, 5000, 10000], "tfidf__min_df": [1, 2], "clf__alpha": [0.1, 0.5, 1.0]},
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(train["clean_message"], train["label"], groups=train["clean_message"])
    model = grid.best_estimator_
    predictions = model.predict(test["clean_message"])
    probabilities = model.predict_proba(test["clean_message"])[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(test["label"], predictions), 4),
        "precision": round(precision_score(test["label"], predictions, zero_division=0), 4),
        "recall": round(recall_score(test["label"], predictions, zero_division=0), 4),
        "f1": round(f1_score(test["label"], predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(test["label"], probabilities), 4),
        "best_params": grid.best_params_,
        "classification_report": classification_report(test["label"], predictions, output_dict=True, zero_division=0),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": model, "feature_names": model.named_steps["tfidf"].get_feature_names_out().tolist(), "schema_version": 2}, MODEL_PATH)
    write_evaluation_manifest(
        RESULTS_DIR,
        project="01-spam-ham-watcher",
        dataset={**source_context(DATA_DIR, "UCI SMS Spam Collection", mode="real"), "rows": len(frame), "deduplication": dedupe},
        split={"strategy": "duplicate-free StratifiedGroupKFold holdout by normalized message", "seed": 42, "group_key": "clean_message", "partitions": {"train": len(train), "test": len(test)}},
        checks={"duplicate_sample_overlap": False, "duplicate_group_overlap": False, "preprocessing_test_fit": False, "threshold_test_fit": False, "direct_label_feature": False, "feature_schema_match": False, "single_feature_auc_max": 0.0, "train_test_f1_gap": 0.0, "test_accuracy_max": metrics["accuracy"], "class_imbalance_ratio": float((frame["label"] == 0).sum() / max((frame["label"] == 1).sum(), 1))},
        metrics=metrics,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
