from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_agent.core.states import PROJECT_TRANSITIONS
from facade_agent.storage import Store, utc_now
from facade_agent.tools import build_phase1_registry


class Phase1FoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "agent.sqlite3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_migration_creates_required_tables(self) -> None:
        required = {
            "projects", "classes", "maintenance_batches", "model_versions",
            "active_models", "failure_slices", "failure_slice_members",
            "slice_reviews", "tool_runs", "evidence_reports",
            "deployment_decisions", "artifact_refs", "audit_events",
            "evaluation_jobs", "evaluation_runs", "evaluation_run_events",
        }
        with self.store.connect() as db:
            actual = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
        self.assertTrue(required <= actual)

    def test_project_classes_have_stable_ids_and_confirmation(self) -> None:
        created = self.store.create_project(
            project_name="Bridge surface screening",
            class_names=["crack", "spalling"],
        )
        project = created["project"]
        self.assertEqual(project["state"], "PROJECT_CREATED")
        self.assertEqual(
            [item["class_id"] for item in project["classes"]],
            [
                f"{project['project_id']}_class_001",
                f"{project['project_id']}_class_002",
            ],
        )
        with self.assertRaises(PermissionError):
            self.store.transition_project(
                project_id=project["project_id"],
                action="confirm_taxonomy",
                confirmed=False,
            )
        confirmed = self.store.transition_project(
            project_id=project["project_id"],
            action="confirm_taxonomy",
            confirmed=True,
        )
        self.assertEqual(confirmed["project"]["state"], "TAXONOMY_CONFIRMED")
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_llm_cannot_confirm_critical_tool(self) -> None:
        project = self.store.create_project(
            project_name="Facade", class_names=["crack"]
        )["project"]
        registry = build_phase1_registry(self.store)
        with self.assertRaises(PermissionError):
            registry.execute(
                "confirm_taxonomy",
                {"project_id": project["project_id"]},
                actor_type="llm",
                actor_id="assistant",
                confirmed=True,
            )

    def test_maintenance_batch_has_one_terminal_decision(self) -> None:
        project = self.store.create_project(
            project_name="Facade", class_names=["crack"]
        )["project"]
        project_id = project["project_id"]
        for action in PROJECT_TRANSITIONS:
            result = self.store.transition_project(
                project_id=project_id,
                action=action,
                confirmed=True,
            )
        self.assertEqual(result["project"]["state"], "SCREENING_READY")

        now = utc_now()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    "champion-fixture",
                    project_id,
                    "champion",
                    str(Path(self.temp.name) / "champion.pt"),
                    "f" * 64,
                    "[0.5]",
                    "fixture-v1",
                    now,
                ),
            )
            db.execute(
                "INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)",
                (project_id, "champion-fixture", now),
            )

        batch = self.store.create_maintenance_batch(
            project_id=project_id, batch_name="round-a"
        )["maintenance_batch"]
        batch_id = batch["batch_id"]
        actions = [
            "confirm_batch_labels",
            "record_batch_inference",
            "record_failure_discovery",
            "freeze_failure_review",
            "record_challenger_training",
            "record_challenger_evaluation",
            "open_human_decision",
        ]
        for action in actions:
            result = self.store.transition_batch(
                batch_id=batch_id,
                action=action,
                confirmed=True,
            )
        self.assertEqual(result["maintenance_batch"]["state"], "DECISION_PENDING")

        with self.store.connect() as db:
            db.execute(
                "INSERT INTO evidence_reports(evidence_id,project_id,batch_id,champion_model_id,challenger_model_id,"
                "metrics_json,artifact_path,artifact_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                ("evidence-fixture", project_id, batch_id, "champion-fixture", "challenger-fixture",
                 "{}", str(Path(self.temp.name) / "evidence.zip"), "e" * 64, now),
            )

        decision = self.store.record_batch_decision(
            batch_id=batch_id,
            decision="hold",
            reason="Historical-safety trade-off was not acceptable.",
            confirmed=True,
        )
        self.assertEqual(decision["maintenance_batch"]["state"], "HELD")
        with self.assertRaises(PermissionError):
            self.store.record_batch_decision(
                batch_id=batch_id,
                decision="deploy",
                reason="Trying to overwrite the final decision.",
                confirmed=True,
            )
        with self.store.connect() as db:
            count = db.execute(
                "SELECT count(*) FROM deployment_decisions WHERE batch_id=?",
                (batch_id,),
            ).fetchone()[0]
        self.assertEqual(count, 1)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_tool_schema_rejects_missing_and_unknown_arguments(self) -> None:
        registry = build_phase1_registry(self.store)
        with self.assertRaises(ValueError):
            registry.execute(
                "create_project",
                {"project_name": "Facade"},
                actor_type="human",
                actor_id="local_engineer",
                confirmed=False,
            )
        with self.assertRaises(ValueError):
            registry.execute(
                "create_project",
                {
                    "project_name": "Facade",
                    "classes": ["crack"],
                    "unexpected": True,
                },
                actor_type="human",
                actor_id="local_engineer",
                confirmed=False,
            )

    def test_duplicate_class_names_are_rejected_case_insensitively(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_project(
                project_name="Facade", class_names=["Crack", "crack"]
            )


if __name__ == "__main__":
    unittest.main()


