from __future__ import annotations

import unittest
import hashlib

from facade_training_worker.challenger_pipeline import _cycle_rows
from facade_training_worker.data import parse_classes, parse_image_records
from facade_training_worker.failure_pipeline import consensus_slices
from facade_training_worker.runtime import load_worker_manifest


class CompletedWorkerTests(unittest.TestCase):
    def test_all_four_production_pipelines_are_enabled(self) -> None:
        manifest = load_worker_manifest()
        self.assertEqual(manifest["phase"], "5_COMPLETE")
        self.assertTrue(all(manifest["pipelines"].values()))
        self.assertEqual(manifest["minimum_visible_gpus"], 1)
        self.assertEqual(manifest["minimum_compute_capability"], [7, 0])

    def test_challenger_pool_cycle_is_exact_and_deterministic(self) -> None:
        rows = [{"image_id": "a"}, {"image_id": "b"}]
        self.assertEqual([row["image_id"] for row in _cycle_rows(rows, 5)], ["a", "b", "a", "b", "a"])
        with self.assertRaises(ValueError):
            _cycle_rows([], 1)

    def test_failure_consensus_produces_stable_minimum_size_slices(self) -> None:
        failures = []
        features = {}
        for index in range(12):
            image_id = f"i{index:02d}"
            failures.append({"image_id": image_id, "class_id": "crack", "error_type": "FP", "severity": 0.2 + index / 100})
            features[image_id] = ([1.0, 0.01 * index, 0.0] if index < 6 else [0.0, 0.01 * index, 1.0])
        profile = {
            "projection": {"pca_components_max": 3},
            "clustering": {
                "runs": 3, "kmeans_n_init_per_run": 20, "cluster_seed": 7,
                "min_failures_per_class_error_pool": 12, "min_cluster_size": 6,
                "max_clusters": 2, "min_silhouette": 0.08,
                "min_mean_pairwise_ari": 0.6, "stable_member_coassociation": 2 / 3,
            },
        }
        slices, members = consensus_slices(failures, features, profile)
        self.assertEqual(len(slices), 2)
        self.assertEqual(sum(item["support"] for item in slices), 12)
        self.assertEqual(len(members), 12)

    def test_evaluation_parser_accepts_two_logical_splits_with_cohort_provenance(self) -> None:
        classes = [{"class_id": "crack"}]
        payload = b"image"
        digest = hashlib.sha256(payload).hexdigest()
        rows = [
            {
                "image_id": "current-1",
                "file": "holdout/images/current-1.jpg",
                "sha256": digest,
                "class_ids": ["crack"],
                "no_defect": False,
                "split": "current_gate",
                "cohort_id": "cohort-current",
                "source_round": "Round B",
                "origin_role": "current_gate",
            },
            {
                "image_id": "safety-1",
                "file": "holdout/images/safety-1.jpg",
                "sha256": digest,
                "class_ids": ["crack"],
                "no_defect": False,
                "split": "core_safety",
                "cohort_id": "cohort-safety",
                "source_round": "Round A",
                "origin_role": "core_safety_seed",
            },
        ]
        class_ids = parse_classes(classes)
        records = parse_image_records(
            rows,
            class_ids,
            {"holdout/images/current-1.jpg", "holdout/images/safety-1.jpg"},
        )
        self.assertEqual([record.image_id for record in records], ["current-1", "safety-1"])


if __name__ == "__main__":
    unittest.main()
