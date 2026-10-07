from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from facade_training_worker.bundle import BundleValidationError, verify_bundle
from facade_training_worker.entrypoint import CONTRACTS
from facade_training_worker.runtime import collect_runtime_report, load_worker_manifest


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_bundle(path: Path, payloads: dict[str, bytes]) -> None:
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
        archive.writestr("checksums.json", json_bytes(checksums))


class WorkerRuntimeTests(unittest.TestCase):
    def test_manifest_exposes_per_pipeline_production_gates(self) -> None:
        manifest = load_worker_manifest()
        self.assertEqual(manifest["phase"], "5_COMPLETE")
        self.assertEqual(set(manifest["pipelines"]), set(CONTRACTS))
        self.assertTrue(all(manifest["pipelines"].values()))

    def test_runtime_report_is_machine_readable_and_not_a_module_presence_probe(self) -> None:
        report = collect_runtime_report(required_gpus=1, pipeline="not_a_pipeline")
        self.assertTrue(report["available"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["minimum_visible_gpus"], 1)
        self.assertTrue(any("Unknown governed Worker pipeline" in issue for issue in report["issues"]))

    def test_initial_champion_pretrained_weights_are_environment_managed(self) -> None:
        report = collect_runtime_report(required_gpus=1, pipeline="initial_champion")
        readiness = report["pipeline_readiness"]["initial_champion"]
        asset = readiness["assets"]["convnext_tiny_imagenet1k_v1"]
        self.assertTrue(readiness["ready"])
        self.assertEqual(asset["managed_by"], "torchvision")
        self.assertFalse(asset["readiness_gate"])
        self.assertFalse(any("cached" in issue.lower() for issue in readiness["issues"]))

    def test_initial_champion_bundle_is_verified_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "job.zip"
            payloads = {
                "job_manifest.json": json_bytes({"schema_version": 1, "job_id": "train_1", "kind": "initial_champion"}),
                "classes.json": json_bytes([{"class_id": "crack"}]),
                "labels.jsonl": json_bytes({"image_id": "image_1", "file": "images/image_1.jpg"}),
                "training_profile.json": json_bytes({"profile_id": "test"}),
            }
            write_bundle(bundle, payloads)
            verified = verify_bundle(
                bundle,
                required_members=CONTRACTS["initial_champion"].required_members,
                expected_kind="initial_champion",
            )
            self.assertEqual(verified.manifest["job_id"], "train_1")
            self.assertEqual(len(verified.checksums), 4)

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "bad.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("job_manifest.json", json_bytes({"schema_version": 1, "job_id": "x", "kind": "initial_champion"}))
                archive.writestr("classes.json", b"[]\n")
                archive.writestr("labels.jsonl", b"")
                archive.writestr("training_profile.json", b"{}\n")
                archive.writestr("checksums.json", json_bytes({
                    "job_manifest.json": "0" * 64,
                    "classes.json": hashlib.sha256(b"[]\n").hexdigest(),
                    "labels.jsonl": hashlib.sha256(b"").hexdigest(),
                    "training_profile.json": hashlib.sha256(b"{}\n").hexdigest(),
                }))
            with self.assertRaisesRegex(BundleValidationError, "SHA-256"):
                verify_bundle(bundle, required_members=CONTRACTS["initial_champion"].required_members, expected_kind="initial_champion")

    def test_path_traversal_is_rejected_before_content_use(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "unsafe.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("../escape", b"x")
            with self.assertRaisesRegex(BundleValidationError, "Unsafe ZIP member"):
                verify_bundle(bundle, required_members=(), expected_kind=None)

    def test_four_fixed_entry_modules_import(self) -> None:
        import facade_training_worker.challenger_update
        import facade_training_worker.champion_challenger_evaluation
        import facade_training_worker.champion_failure_discovery
        import facade_training_worker.initial_champion

        self.assertEqual(len(CONTRACTS), 4)


if __name__ == "__main__":
    unittest.main()
