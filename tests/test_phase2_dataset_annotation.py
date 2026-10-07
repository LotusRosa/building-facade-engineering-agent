from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_agent.application.image_io import PNG_SIGNATURE, probe_image, safe_filename
from facade_agent.storage import Store


class Phase2DatasetAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "agent.sqlite3")
        created = self.store.create_project(
            project_name="Facade screening", class_names=["crack", "spalling"]
        )["project"]
        self.project_id = created["project_id"]
        self.classes = created["classes"]
        self.store.transition_project(
            project_id=self.project_id,
            action="confirm_taxonomy",
            confirmed=True,
        )
        self.dataset = self.store.create_dataset(
            project_id=self.project_id,
            name="initial-v1",
            role="initial_training",
        )["dataset"]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_image(self, name: str, digest: str) -> dict:
        return self.store.register_image(
            dataset_id=self.dataset["dataset_id"],
            filename=name,
            stored_path=str(Path(self.temp.name) / name),
            sha256=digest,
            size_bytes=24,
            mime_type="image/png",
            width=16,
            height=12,
            health_status="ok",
        )

    def test_no_defect_is_mutually_exclusive(self) -> None:
        image = self.add_image("one.png", "a" * 64)
        with self.assertRaises(ValueError):
            self.store.save_annotation(
                image_id=image["image_id"],
                class_ids=[self.classes[0]["class_id"]],
                no_defect=True,
            )
        saved = self.store.save_annotation(
            image_id=image["image_id"], class_ids=[], no_defect=True
        )
        self.assertEqual(saved["annotation_status"], "complete")
        self.assertEqual(saved["class_ids"], [])

    def test_validation_freezes_label_version_and_is_idempotent(self) -> None:
        first = self.add_image("one.png", "a" * 64)
        second = self.add_image("two.png", "b" * 64)
        self.store.save_annotation(
            image_id=first["image_id"],
            class_ids=[self.classes[0]["class_id"]],
            no_defect=False,
        )
        failed = self.store.validate_dataset(dataset_id=self.dataset["dataset_id"])
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["missing_annotations"], 1)
        self.store.save_annotation(
            image_id=second["image_id"], class_ids=[], no_defect=True
        )
        passed = self.store.validate_dataset(dataset_id=self.dataset["dataset_id"])
        self.assertTrue(passed["valid"])
        self.assertIn("labels_sha256", passed)
        self.assertEqual(self.store.get_project(self.project_id)["state"], "DATA_VALIDATED")
        repeated = self.store.validate_dataset(dataset_id=self.dataset["dataset_id"])
        self.assertEqual(repeated["labels_sha256"], passed["labels_sha256"])
        with self.store.connect() as db:
            versions = db.execute(
                "SELECT count(*) FROM label_versions WHERE dataset_id=?",
                (self.dataset["dataset_id"],),
            ).fetchone()[0]
        self.assertEqual(versions, 1)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_exact_duplicates_block_validation(self) -> None:
        first = self.add_image("one.png", "a" * 64)
        second = self.add_image("two.png", "a" * 64)
        for image in (first, second):
            self.store.save_annotation(
                image_id=image["image_id"], class_ids=[], no_defect=True
            )
        report = self.store.validate_dataset(dataset_id=self.dataset["dataset_id"])
        self.assertFalse(report["valid"])
        self.assertEqual(report["exact_duplicate_groups"], 1)

    def test_image_probe_and_filename_safety(self) -> None:
        png = PNG_SIGNATURE + b"\x00\x00\x00\x0dIHDR" + (16).to_bytes(4, "big") + (12).to_bytes(4, "big")
        result = probe_image(png, "sample.png")
        self.assertEqual((result.health_status, result.width, result.height), ("ok", 16, 12))
        self.assertEqual(safe_filename("../sample.png"), "sample.png")


if __name__ == "__main__":
    unittest.main()
