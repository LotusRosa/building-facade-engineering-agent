from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_agent.core.states import BatchState, ProjectState
from facade_agent.storage import Store, utc_now
from facade_agent.tools import build_phase1_registry
from facade_agent.adapters.llm import LLMManager
from facade_agent.application.agent_runtime import AgentRuntime
from facade_agent.application.workflow import WorkflowSnapshotService


class Phase4AMaintenanceBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "agent.sqlite3")
        project = self.store.create_project(
            project_name="Facade maintenance", class_names=["crack", "spalling"]
        )["project"]
        self.project_id = project["project_id"]
        self.classes = project["classes"]
        self.store.transition_project(
            project_id=self.project_id,
            action="confirm_taxonomy",
            confirmed=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def register_champion_fixture(self) -> None:
        now = utc_now()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    "champion-fixture",
                    self.project_id,
                    "champion",
                    str(Path(self.temp.name) / "champion.pt"),
                    "f" * 64,
                    "[0.5,0.5]",
                    "fixture-v1",
                    now,
                ),
            )
            db.execute(
                "INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)",
                (self.project_id, "champion-fixture", now),
            )
            db.execute(
                "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                (ProjectState.CHAMPION_READY, now, self.project_id),
            )

    def add_image(self, dataset_id: str, name: str, digest: str) -> dict:
        return self.store.register_image(
            dataset_id=dataset_id,
            filename=name,
            stored_path=str(Path(self.temp.name) / name),
            sha256=digest,
            size_bytes=24,
            mime_type="image/png",
            width=16,
            height=12,
            health_status="ok",
        )

    def create_batch_dataset(self) -> tuple[dict, dict]:
        batch = self.store.create_maintenance_batch(
            project_id=self.project_id, batch_name="inspection-2026-09"
        )["maintenance_batch"]
        created = self.store.create_maintenance_dataset(
            batch_id=batch["batch_id"], name="inspection-2026-09-images"
        )
        return created["maintenance_batch"], created["dataset"]

    def test_requires_registered_active_champion(self) -> None:
        with self.assertRaises(PermissionError):
            self.store.create_maintenance_batch(
                project_id=self.project_id, batch_name="not-ready"
            )

    def test_chat_skips_repeat_annotation_but_confirms_validation_and_freeze(self):
        self.register_champion_fixture()
        batch, dataset = self.create_batch_dataset()
        image = self.add_image(dataset['dataset_id'], 'new.png', '1' * 64)
        workflow = WorkflowSnapshotService(self.store)
        runtime = AgentRuntime(self.store, build_phase1_registry(self.store), LLMManager(), workflow)
        first = runtime.chat(text='continue', project_id=self.project_id, language='en')
        self.assertEqual(first['status'], 'requires_human_input')
        self.assertIn('1 unannotated', first['message'])
        self.store.save_annotation(image_id=image['image_id'], class_ids=[self.classes[0]['class_id']], no_defect=False)
        ready = runtime.chat(text='continue', project_id=self.project_id, language='en')
        self.assertEqual(ready['pending']['tool_name'], 'validate_dataset')
        self.assertEqual(self.store.get_dataset(dataset['dataset_id'])['status'], 'open')
        runtime.resolve_confirmation(pending_id=ready['pending']['pending_id'], approved=True, language='en')
        frozen = runtime.chat(text='continue', project_id=self.project_id, language='en')
        self.assertEqual(frozen['pending']['tool_name'], 'freeze_maintenance_batch')
        runtime.resolve_confirmation(pending_id=frozen['pending']['pending_id'], approved=True, language='en')
        self.assertEqual(self.store.get_maintenance_batch(batch['batch_id'])['state'], 'MAINTENANCE_BATCH_FROZEN')
        self.assertEqual(workflow.get(self.project_id)['next_actions'][0]['action_id'], 'start_failure_discovery')

    def test_import_validate_and_human_freeze_do_not_regress_project(self) -> None:
        self.register_champion_fixture()
        batch, dataset = self.create_batch_dataset()
        self.assertEqual(batch["state"], BatchState.MAINTENANCE_DATA_IMPORTED)
        self.assertEqual(dataset["role"], "maintenance")
        self.assertEqual(dataset["batch_id"], batch["batch_id"])
        self.assertEqual(
            self.store.get_project(self.project_id)["state"], ProjectState.CHAMPION_READY
        )

        image = self.add_image(dataset["dataset_id"], "new.png", "1" * 64)
        self.store.save_annotation(
            image_id=image["image_id"],
            class_ids=[self.classes[0]["class_id"]],
            no_defect=False,
        )
        report = self.store.validate_dataset(dataset_id=dataset["dataset_id"])
        self.assertTrue(report["valid"])
        self.assertEqual(
            self.store.get_maintenance_batch(batch["batch_id"])["state"],
            BatchState.MAINTENANCE_LABELS_READY,
        )
        self.assertEqual(
            self.store.get_project(self.project_id)["state"], ProjectState.CHAMPION_READY
        )

        registry = build_phase1_registry(self.store)
        arguments = {"batch_id": batch["batch_id"]}
        with self.assertRaises(PermissionError):
            registry.execute(
                "freeze_maintenance_batch",
                arguments,
                actor_type="llm",
                actor_id="model",
                confirmed=True,
            )
        with self.assertRaises(PermissionError):
            registry.execute(
                "freeze_maintenance_batch",
                arguments,
                actor_type="human",
                actor_id="engineer",
                confirmed=False,
            )
        result = registry.execute(
            "freeze_maintenance_batch",
            arguments,
            actor_type="human",
            actor_id="engineer",
            confirmed=True,
        )
        self.assertEqual(
            result["maintenance_batch"]["state"],
            BatchState.MAINTENANCE_BATCH_FROZEN,
        )
        self.assertEqual(result["project"]["state"], ProjectState.SCREENING_READY)
        self.assertEqual(result["maintenance_batch"]["dataset"]["status"], "frozen")
        with self.assertRaises(PermissionError):
            self.store.save_annotation(
                image_id=image["image_id"], class_ids=[], no_defect=True
            )
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_historical_duplicate_blocks_maintenance_validation(self) -> None:
        initial = self.store.create_dataset(
            project_id=self.project_id,
            name="initial-fixture",
            role="initial_training",
        )["dataset"]
        self.add_image(initial["dataset_id"], "old.png", "a" * 64)
        self.register_champion_fixture()
        batch, maintenance = self.create_batch_dataset()
        duplicate = self.add_image(
            maintenance["dataset_id"], "copied-old.png", "a" * 64
        )
        self.store.save_annotation(
            image_id=duplicate["image_id"], class_ids=[], no_defect=True
        )
        report = self.store.validate_dataset(dataset_id=maintenance["dataset_id"])
        self.assertFalse(report["valid"])
        self.assertEqual(report["historical_duplicate_images"], 1)
        self.assertEqual(
            self.store.get_maintenance_batch(batch["batch_id"])["state"],
            BatchState.MAINTENANCE_DATA_IMPORTED,
        )

    def test_one_dataset_per_active_batch(self) -> None:
        self.register_champion_fixture()
        batch, _ = self.create_batch_dataset()
        with self.assertRaises(PermissionError):
            self.store.create_maintenance_dataset(
                batch_id=batch["batch_id"], name="second-dataset"
            )


if __name__ == "__main__":
    unittest.main()
