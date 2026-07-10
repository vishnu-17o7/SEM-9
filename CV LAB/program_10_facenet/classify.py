"""Classifier on top of FaceNet embeddings."""
from __future__ import annotations

import numpy as np


def make_classifier(name: str, n_neighbors: int = 5):
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=max(1, n_neighbors), metric="cosine")
    raise ValueError(f"Unknown classifier: {name}")


def train_classifier(name: str, X_train: np.ndarray, y_train: np.ndarray):
    n_neighbors = min(5, len(X_train)) if name == "knn" else 5
    return make_classifier(name, n_neighbors=n_neighbors).fit(X_train, y_train)
