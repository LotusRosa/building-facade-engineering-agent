from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from facade_agent.application.training_jobs import (
    InitialChampionJobService,
    register_initial_training_tool,
)
from facade_agent.storage import Store
from facade_agent.tools import build_phase1_registry


class Phase3BTrainingJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project_root = self.root / "projects"
        self.project_root.mkdir()
        self.store = Store(self.root / "agent.sqlite3")
        created = self.store.create_project(
            project_name="Facade", class_names=["crack", "spalling"]
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
        for index in range(2):
            content = b"\x89PNG\r\n\x1a\n" + bytes([index]) * 32
            path = self.project_root / f"sample-{index}.png"
            path.write_bytes(content)
            image = self.store.register_image(
                dataset_id=self.dataset["dataset_id"],
                filename=path.name,
                stored_path=str(path),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                mime_type="image/png",
                width=16,
                height=12,
                health_status="ok",
            )
            self.store.save_annotation(
                image_id=image["image_id"],
                class_ids=[self.classes[0]["class_id"]] if index == 0 else [],
                no_defect=index == 1,
            )
        self.store.validate_dataset(dataset_id=self.dataset["dataset_id"])
        profile = self.root / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile_id": "initial_champion_convnext_tiny_768_v2",
                    "task": "image_level_multilabel_classification",
                }
            ),
            encoding="utf-8",
        )
        self.service = InitialChampionJobService(
            self.store,
            self.project_root,
            self.root / "exports",
            profile,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_bundle_is_immutable_auditable_and_idempotent(self) -> None:
        created = self.service.create_job(
            project_id=self.project_id,
            dataset_id=self.dataset["dataset_id"],
        )
        job = created["job"]
        self.assertFalse(created["idempotent"])
        self.assertEqual(self.store.get_project(self.project_id)["state"], "INITIAL_TRAINING_READY")
        self.assertEqual(self.store.get_dataset(self.dataset["dataset_id"])["status"], "frozen")
        bundle = Path(job["bundle_path"])
        self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(), job["bundle_sha256"])
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            self.assertTrue({"job_manifest.json", "classes.json", "labels.jsonl", "training_profile.json", "checksums.json"} <= names)
            self.assertEqual(len([name for name in names if name.startswith("images/")]), 2)
            manifest = json.loads(archive.read("job_manifest.json"))
            self.assertEqual(manifest["image_count"], 2)
            self.assertEqual(manifest["class_count"], 2)
        repeated = self.service.create_job(
            project_id=self.project_id,
            dataset_id=self.dataset["dataset_id"],
        )
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["job"]["job_id"], job["job_id"])
        self.assertTrue(self.store.verify_audit_chain()["valid"])
        history = self.store.get_project_history(self.project_id)
        self.assertEqual(history["counts"]["training_jobs"], 1)

    def test_training_job_requires_explicit_human_confirmation(self) -> None:
        registry = build_phase1_registry(self.store)
        register_initial_training_tool(registry, self.service)
        arguments = {"project_id": self.project_id, "dataset_id": self.dataset["dataset_id"]}
        with self.assertRaises(PermissionError):
            registry.execute(
                "create_initial_training_job",
                arguments,
                actor_type="llm",
                actor_id="model",
                confirmed=False,
            )
        result = registry.execute(
            "create_initial_training_job",
            arguments,
            actor_type="human",
            actor_id="engineer",
            confirmed=True,
        )
        self.assertEqual(result["job"]["status"], "exported")


if __name__ == "__main__":
    unittest.main()
