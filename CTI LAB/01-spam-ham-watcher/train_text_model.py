import urllib.request
import zipfile
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from spam_utils import clean_text

DATA_DIR = Path("data")
MODEL_PATH = "spam_text_model.joblib"
ZIP_PATH = "smsspamcollection.zip"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / "SMSSpamCollection"
    if data_file.exists():
        print(f"Dataset already exists at {data_file}, skipping download.")
        return data_file
    print(f"Downloading SMS Spam Collection from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print(f"Dataset extracted to {data_file}")
    return data_file


def load_data(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df["clean_message"] = df["message"].apply(clean_text)
    print(f"Loaded {len(df)} messages ({df['label'].sum()} spam, {len(df) - df['label'].sum()} ham)")
    return df


def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000, 10000],
        "tfidf__min_df": [1, 2],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best cross-val F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")


def main():
    data_file = download_data()
    df = load_data(data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    print("-" * 60)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    model_data = {
        "pipeline": model,
        "feature_names": model.named_steps["tfidf"].get_feature_names_out().tolist(),
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH} ({len(model_data['feature_names']):,} features)")


if __name__ == "__main__":
    main()
