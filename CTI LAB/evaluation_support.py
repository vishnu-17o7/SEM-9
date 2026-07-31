"""Shared helpers for reproducible CTI dataset and evaluation metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_context(data_dir: Path, default_name: str, mode: str = "synthetic_fallback") -> dict[str, Any]:
    """Load the data context written by a data preparation step."""
    path = data_dir / "evaluation_context.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return {
        "mode": mode,
        "name": default_name,
        "source_url": "",
        "license": "Synthetic fallback generated locally",
        "citation": "",
        "limitations": "Synthetic fallback; metrics do not establish real-world generalization.",
    }


def write_source_context(data_dir: Path, payload: dict[str, Any]) -> None:
    """Persist dataset provenance beside generated data."""
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **payload}
    (data_dir / "evaluation_context.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deduplicate_frame(frame: pd.DataFrame, keys: Iterable[str], label_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicate samples and reject duplicate keys with conflicting labels."""
    key_list = list(keys)
    duplicate_keys = frame.duplicated(key_list, keep=False)
    conflicting = int(frame.loc[duplicate_keys].groupby(key_list, dropna=False)[label_column].nunique().gt(1).sum())
    if conflicting:
        raise ValueError(f"Found {conflicting} duplicate sample keys with conflicting labels")
    before = len(frame)
    cleaned = frame.drop_duplicates(key_list, keep="first").reset_index(drop=True)
    return cleaned, {"rows_before": before, "rows_after": len(cleaned), "duplicates_removed": before - len(cleaned), "conflicting_keys": conflicting}


def stratified_group_indices(y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic stratified group holdout."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_index, test_index = next(splitter.split(np.zeros(len(y)), y, groups))
    return train_index, test_index


def max_univariate_auc(X: np.ndarray, y: np.ndarray) -> float:
    """Measure the strongest single numeric feature without changing training."""
    best = 0.5
    for column in range(X.shape[1]):
        values = X[:, column]
        if np.unique(values).size < 2:
            continue
        try:
            score = roc_auc_score(y, values)
            best = max(best, score, 1.0 - score)
        except ValueError:
            continue
    return float(best)


def write_evaluation_manifest(results_dir: Path, *, project: str, dataset: dict[str, Any], split: dict[str, Any], checks: dict[str, Any], metrics: Any) -> Path:
    """Write the shared evaluation manifest consumed by evaluation_guard.py."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "dataset": dataset,
        "split": split,
        "checks": checks,
        "metrics": metrics,
    }
    path = results_dir / "evaluation_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
