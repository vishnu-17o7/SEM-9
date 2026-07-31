"""Preprocessing + dataset loading for the SMS Spam Collection (UCI).

Same data format as project 1, but re-packaged as a clean module that
project 2 can import independently. No external state, no IMAP, no DB.
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import deduplicate_frame

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs / punctuation / digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_data(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not force:
        return DATA_FILE
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    ZIP_PATH.unlink(missing_ok=True)
    return DATA_FILE


def load_dataframe(force_download: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns: label (0/1), message (str), clean_message (str)."""
    path = download_data(force=force_download)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"spam": 1, "ham": 0}).astype(int)
    df["clean_message"] = df["message"].apply(clean_text)
    cleaned, _ = deduplicate_frame(df, ["clean_message"], "label")
    return cleaned


def load_split(test_size: float = 0.2, random_state: int = 42):
    """Return a duplicate-free, deterministic stratified split."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = load_dataframe()
    folds = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df["clean_message"], df["label"], df["clean_message"]))
    X_train = df.iloc[train_idx]["clean_message"]
    X_test = df.iloc[test_idx]["clean_message"]
    y_train = df.iloc[train_idx]["label"]
    y_test = df.iloc[test_idx]["label"]
    return X_train, X_test, y_train, y_test
