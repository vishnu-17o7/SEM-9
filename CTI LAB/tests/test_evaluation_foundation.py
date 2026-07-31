"""Small adversarial tests for the shared CTI evaluation contract."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_guard import validate_manifest
from evaluation_support import deduplicate_frame, stratified_group_indices


class EvaluationFoundationTests(unittest.TestCase):
    """Verify the checks that prevent the most damaging leakage modes."""

    def test_conflicting_duplicate_labels_are_rejected(self) -> None:
        """A repeated sample key cannot silently receive two labels."""
        frame = pd.DataFrame({"message": ["same", "same"], "label": [0, 1]})
        with self.assertRaises(ValueError):
            deduplicate_frame(frame, ["message"], "label")

    def test_group_holdout_has_no_overlap_and_is_deterministic(self) -> None:
        """Repeated grouped splitting produces identical, disjoint partitions."""
        labels = np.array([0, 0, 1, 1] * 8)
        groups = np.array([f"g{index // 2}" for index in range(len(labels))])
        train_a, test_a = stratified_group_indices(labels, groups, n_splits=4)
        train_b, test_b = stratified_group_indices(labels, groups, n_splits=4)
        self.assertEqual(train_a.tolist(), train_b.tolist())
        self.assertEqual(test_a.tolist(), test_b.tolist())
        self.assertTrue(set(groups[train_a]).isdisjoint(set(groups[test_a])))

    def test_direct_label_feature_is_a_hard_failure(self) -> None:
        """A manifest declaring label leakage must fail the audit."""
        manifest = {
            "schema_version": 1,
            "project": "fixture",
            "dataset": {"mode": "synthetic_fallback", "name": "fixture", "limitations": "test"},
            "split": {"strategy": "grouped", "partitions": {"train": 1, "test": 1}},
            "checks": {
                "duplicate_sample_overlap": False,
                "duplicate_group_overlap": False,
                "preprocessing_test_fit": False,
                "threshold_test_fit": False,
                "direct_label_feature": True,
                "feature_schema_match": False,
            },
            "metrics": {"fixture": {"f1": 0.5}},
        }
        errors, _ = validate_manifest(manifest)
        self.assertTrue(any("direct_label_feature" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
