"""Validate CTI model evaluation manifests before reports are generated."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_CHECKS = (
    "duplicate_sample_overlap",
    "duplicate_group_overlap",
    "preprocessing_test_fit",
    "threshold_test_fit",
    "direct_label_feature",
    "feature_schema_match",
)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a JSON evaluation manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation manifest must contain a JSON object")
    return payload


def validate_manifest(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return hard errors and non-blocking warnings for a manifest."""
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for key in ("project", "dataset", "split", "checks", "metrics"):
        if key not in payload:
            errors.append(f"missing required section: {key}")

    dataset = payload.get("dataset", {})
    if not isinstance(dataset, dict):
        errors.append("dataset must be an object")
    else:
        if dataset.get("mode") not in {"real", "synthetic_fallback", "legacy"}:
            errors.append("dataset.mode must be real, synthetic_fallback, or legacy")
        if not dataset.get("name"):
            errors.append("dataset.name is required")
        if not dataset.get("limitations"):
            warnings.append("dataset limitations are missing")

    split = payload.get("split", {})
    if not isinstance(split, dict):
        errors.append("split must be an object")
    else:
        if not split.get("strategy"):
            errors.append("split.strategy is required")
        if not split.get("partitions"):
            errors.append("split.partitions is required")

    checks = payload.get("checks", {})
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
    else:
        for name in REQUIRED_CHECKS:
            value = checks.get(name)
            if value is not False:
                errors.append(f"hard evaluation check failed: {name}={value!r}")
        for name in (
            "single_feature_auc_max",
            "train_test_f1_gap",
            "test_accuracy_max",
            "class_imbalance_ratio",
        ):
            value = checks.get(name)
            if isinstance(value, (int, float)):
                if name == "single_feature_auc_max" and value > 0.98:
                    warnings.append(f"single-feature AUC is high: {value:.4f}")
                elif name == "train_test_f1_gap" and value > 0.05:
                    warnings.append(f"train/test F1 gap is high: {value:.4f}")
                elif name == "test_accuracy_max" and value > 0.995:
                    warnings.append(f"test accuracy is suspiciously high: {value:.4f}")
                elif name == "class_imbalance_ratio" and value > 20:
                    warnings.append(f"class imbalance ratio is high: {value:.2f}")

    if not payload.get("metrics"):
        warnings.append("no model metrics were recorded")
    return errors, warnings


def main() -> int:
    """Validate one project manifest and print an actionable summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    manifest = args.manifest or Path(args.project) / "results" / "evaluation_manifest.json"
    try:
        payload = load_manifest(manifest)
        errors, warnings = validate_manifest(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[AUDIT FAIL] {exc}", file=sys.stderr)
        return 1

    print(f"Evaluation audit: {payload.get('project', args.project)}")
    if warnings:
        for warning in warnings:
            print(f"  [WARN] {warning}")
    if errors:
        for error in errors:
            print(f"  [FAIL] {error}", file=sys.stderr)
        return 1
    print("  [OK] hard validity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
