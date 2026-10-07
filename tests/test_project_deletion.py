from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_agent.storage import Store
from facade_agent.tools import build_phase1_registry


class ProjectDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "data" / "facade_agent.sqlite3")
        self.registry = build_phase1_registry(self.store)
        self.keep = self.store.create_project(
            project_name="keep-project", class_names=["crack"]
        )["project"]
        self.target = self.store.create_project(
            project_name="delete-project", class_names=["spalling"]
        )["project"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_name_and_human_confirmation_are_required(self) -> None:
        with self.assertRaises(PermissionError):
            self.registry.execute(
                "delete_project",
                {
                    "project_id": self.target["project_id"],
                    "confirmation_name": "wrong-name",
                },
                actor_type="human",
                actor_id="tester",
                confirmed=True,
            )
        with self.assertRaises(PermissionError):
            self.registry.execute(
                "delete_project",
                {
                    "project_id": self.target["project_id"],
                    "confirmation_name": self.target["name"],
                },
                actor_type="llm",
                actor_id="model",
                confirmed=True,
            )

    def test_project_data_is_removed_but_audit_chain_stays_valid(self) -> None:
        project_dir = self.root / "projects" / self.target["project_id"]
        project_dir.mkdir(parents=True)
        (project_dir / "sample.txt").write_text("test", encoding="utf-8")

        result = self.registry.execute(
            "delete_project",
            {
                "project_id": self.target["project_id"],
                "confirmation_name": self.target["name"],
            },
            actor_type="human",
            actor_id="tester",
            confirmed=True,
        )

        self.assertTrue(result["deleted"])
        self.assertTrue(result["audit_tombstone_retained"])
        self.assertFalse(project_dir.exists())
        self.assertEqual(
            [project["name"] for project in self.store.list_projects()],
            [self.keep["name"]],
        )
        with self.assertRaises(KeyError):
            self.store.get_project(self.target["project_id"])
        with self.store.connect() as db:
            tombstone = db.execute(
                "SELECT state,deleted_at FROM projects WHERE project_id=?",
                (self.target["project_id"],),
            ).fetchone()
            class_count = db.execute(
                "SELECT COUNT(*) AS count FROM classes WHERE project_id=?",
                (self.target["project_id"],),
            ).fetchone()["count"]
        self.assertEqual(tombstone["state"], "DELETED")
        self.assertIsNotNone(tombstone["deleted_at"])
        self.assertEqual(class_count, 0)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_external_project_model_root_is_moved_to_recoverable_trash(self) -> None:
        with tempfile.TemporaryDirectory() as external_temporary:
            model_root = Path(external_temporary) / "project-models"
            model_root.mkdir()
            project = self.store.create_project(
                project_name="external-model-project",
                class_names=["crack"],
                model_storage_root=str(model_root.resolve()),
            )["project"]
            model_dir = model_root / "champion_fixture"
            model_dir.mkdir()
            (model_dir / "checkpoint.pt").write_bytes(b"registered-model")
            (model_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

            result = self.registry.execute(
                "delete_project",
                {
                    "project_id": project["project_id"],
                    "confirmation_name": project["name"],
                },
                actor_type="human",
                actor_id="tester",
                confirmed=True,
            )

            self.assertFalse(model_root.exists())
            quarantine = Path(result["quarantine_path"])
            recovered = quarantine / "external_model_storage" / model_root.name
            self.assertEqual(
                (recovered / "champion_fixture" / "checkpoint.pt").read_bytes(),
                b"registered-model",
            )


if __name__ == "__main__":
    unittest.main()
