from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from facade_agent.application.environment import EnvironmentManager
from facade_agent.storage import Store
from facade_training_worker.data import ImageRecord, deterministic_multilabel_split, split_payload
from facade_training_worker.initial_pipeline import build_result_bundle
from facade_training_worker.metrics import calibrate_thresholds, multilabel_metrics


class Phase5BInitialChampionWorkerTests(unittest.TestCase):
    def test_deterministic_multilabel_split_preserves_positive_support(self) -> None:
        records = [
            ImageRecord("i1", "images/i1.jpg", "1" * 64, ("crack",), False),
            ImageRecord("i2", "images/i2.jpg", "2" * 64, ("crack", "spall"), False),
            ImageRecord("i3", "images/i3.jpg", "3" * 64, ("spall",), False),
            ImageRecord("i4", "images/i4.jpg", "4" * 64, (), True),
            ImageRecord("i5", "images/i5.jpg", "5" * 64, ("crack",), False),
            ImageRecord("i6", "images/i6.jpg", "6" * 64, ("spall",), False),
        ]
        first = deterministic_multilabel_split(records, ["crack", "spall"], development_fraction=0.34, seed=11)
        second = deterministic_multilabel_split(records, ["crack", "spall"], development_fraction=0.34, seed=11)
        self.assertEqual(first, second)
        train, development = first
        by_id = {record.image_id: record for record in records}
        for class_id in ("crack", "spall"):
            self.assertTrue(any(class_id in by_id[image_id].class_ids for image_id in train))
            self.assertTrue(any(class_id in by_id[image_id].class_ids for image_id in development))

    def test_split_rejects_a_class_with_only_one_positive(self) -> None:
        records = [
            ImageRecord("i1", "images/i1.jpg", "1" * 64, ("crack",), False),
            ImageRecord("i2", "images/i2.jpg", "2" * 64, (), True),
            ImageRecord("i3", "images/i3.jpg", "3" * 64, (), True),
        ]
        with self.assertRaisesRegex(ValueError, "at least two positive"):
            deterministic_multilabel_split(records, ["crack"], development_fraction=0.2, seed=11)

    def test_development_only_thresholds_and_metrics_are_exact(self) -> None:
        targets = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]])
        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.7], [0.2, 0.8], [0.1, 0.2]])
        class_ids = ["crack", "spall"]
        thresholds = calibrate_thresholds(targets, probabilities, class_ids)
        metrics = multilabel_metrics(targets, probabilities, class_ids, thresholds)
        self.assertEqual(thresholds, {"crack": 0.8, "spall": 0.7})
        self.assertEqual(metrics["development_macro_map"], 1.0)
        self.assertEqual(metrics["development_exact_match_accuracy"], 1.0)
        self.assertEqual(metrics["development_false_positive_count"], 0.0)
        self.assertEqual(metrics["development_false_negative_count"], 0.0)

    def test_real_result_packager_emits_agent_verifiable_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            weights = output / "weights"
            weights.mkdir(parents=True)
            seed_results = []
            for seed, score in ((1, 0.7), (2, 0.8), (3, 0.75)):
                path = weights / f"seed-{seed}.pt"
                path.write_bytes(f"checkpoint-{seed}".encode())
                seed_results.append({
                    "seed": seed,
                    "weights_file": f"weights/{path.name}",
                    "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "best_epoch": 2,
                    "metrics": {"development_macro_map": score},
                })
            profile = (json.dumps({"profile_id": "test"}, sort_keys=True) + "\n").encode()
            split = split_payload(["i1", "i2"], ["i3"], 7)
            result = build_result_bundle(
                output=output,
                run_id="run_1",
                job_manifest={"job_id": "job_1", "content_fingerprint": "a" * 64},
                profile_bytes=profile,
                seed_results=seed_results,
                selected_seed=2,
                thresholds={"crack": 0.5},
                split_bytes=split,
                weights_dir=weights,
            )
            with zipfile.ZipFile(result) as archive:
                names = archive.namelist()
                checksums = json.loads(archive.read("checksums.json"))
                manifest = json.loads(archive.read("result_manifest.json"))
                self.assertEqual(set(checksums), set(names) - {"checksums.json"})
                for name, digest in checksums.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                self.assertEqual(manifest["selected_seed"], 2)
                self.assertFalse(manifest["test_read"])

    def test_preflight_allows_only_the_requested_enabled_pipeline(self) -> None:
        def detector(_: Path) -> dict:
            return {
                "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
                "python": {"version": "3.12", "executable": "python", "supported": True},
                "cpu": {"logical_cores": 16},
                "memory": {"total_bytes": 64 * 1024**3},
                "disk": {"root": "test", "free_bytes": 1, "total_bytes": 2},
                "nvidia": {"available": True, "gpus": [
                    {"index": 0, "name": "RTX 4090"}
                ], "error": None},
                "torch": {"installed": True, "version": "2.6.0", "cuda_available": True},
                "training_runtime": {
                    "available": True,
                    "ready": True,
                    "pipelines": {"initial_champion": True, "challenger_update": False},
                },
                "capabilities": {"interface_ready": True, "supported_platform": True, "local_gpu_training_ready": True},
                "warnings": [],
            }

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = EnvironmentManager(
                Store(root / "agent.sqlite3"),
                root,
                detector=detector,
                smoke_tester=lambda devices: {"passed": devices == [0]},
            )
            self.assertTrue(manager.training_preflight("initial_champion")["passed"])
            failed = manager.training_preflight("challenger_update")
            self.assertFalse(failed["passed"])
            self.assertIn("worker_pipeline_enabled", [item["name"] for item in failed["checks"] if not item["passed"]])


if __name__ == "__main__":
    unittest.main()
