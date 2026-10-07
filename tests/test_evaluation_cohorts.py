from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from facade_agent.application.model_evaluation import ModelEvaluationService
from facade_agent.storage import Store


LEGACY_MIGRATION_CEILING = "015_project_model_storage.sql"
NOW = "2026-09-30T00:00:00+00:00"


class EvaluationCohortMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "data" / "facade_agent.sqlite3"
        self.database.parent.mkdir(parents=True)
        self.project_id = "project_legacy"
        self._create_legacy_database()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _job_values(job_id: str, batch_id: str, status: str) -> tuple[object, ...]:
        return (
            job_id,
            "project_legacy",
            batch_id,
            "champion_legacy",
            f"challenger_{batch_id}",
            status,
            "evaluation-profile-v1",
            "{}",
            f"{job_id}.zip",
            hashlib.sha256(f"snapshot:{job_id}".encode()).hexdigest(),
            "{}",
            hashlib.sha256(f"content:{job_id}".encode()).hexdigest(),
            f"/immutable/{job_id}.zip",
            hashlib.sha256(f"bundle:{job_id}".encode()).hexdigest(),
            100,
            NOW,
            NOW,
        )

    def _create_legacy_database(self) -> None:
        migrations = (
            Path(__file__).resolve().parents[1]
            / "facade_agent"
            / "adapters"
            / "storage"
            / "migrations"
        )
        db = sqlite3.connect(self.database)
        try:
            db.execute(
                "CREATE TABLE schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            for migration in sorted(migrations.glob("*.sql")):
                if migration.name > LEGACY_MIGRATION_CEILING:
                    continue
                sql = migration.read_text(encoding="utf-8")
                db.executescript(sql)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,sha256) VALUES(?,?,?)",
                    (migration.name, NOW, hashlib.sha256(sql.encode()).hexdigest()),
                )

            db.execute(
                "INSERT INTO projects(project_id,name,state,created_at,updated_at) VALUES(?,?,?,?,?)",
                (self.project_id, "Legacy project", "SCREENING_READY", NOW, NOW),
            )
            batch_rows = (
                ("batch_seed", "Seed round", "HELD"),
                ("batch_a", "Round A", "HELD"),
                ("batch_b", "Round B", "EVIDENCE_READY"),
                ("batch_failed", "Failed round", "CHALLENGER_TRAINED"),
            )
            db.executemany(
                "INSERT INTO maintenance_batches(batch_id,project_id,name,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                [
                    (batch_id, self.project_id, name, state, NOW, NOW)
                    for batch_id, name, state in batch_rows
                ],
            )
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,checkpoint_sha256,"
                "thresholds_json,training_profile_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    "champion_legacy",
                    self.project_id,
                    "champion",
                    "/immutable/champion.pt",
                    "1" * 64,
                    "{}",
                    "initial-profile",
                    NOW,
                ),
            )
            for batch_id, _, _ in batch_rows:
                db.execute(
                    "INSERT INTO model_versions(model_id,project_id,parent_model_id,source_batch_id,role,"
                    "checkpoint_path,checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        f"challenger_{batch_id}",
                        self.project_id,
                        "champion_legacy",
                        batch_id,
                        "challenger",
                        f"/immutable/challenger-{batch_id}.pt",
                        hashlib.sha256(batch_id.encode()).hexdigest(),
                        "{}",
                        "challenger-profile",
                        NOW,
                    ),
                )

            job_sql = (
                "INSERT INTO evaluation_jobs(job_id,project_id,batch_id,champion_model_id,"
                "challenger_model_id,status,evaluation_profile_id,profile_json,source_snapshot_name,"
                "source_snapshot_sha256,split_counts_json,content_fingerprint,bundle_path,bundle_sha256,"
                "bundle_size_bytes,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            )
            db.executemany(
                job_sql,
                (
                    self._job_values("evaluation_seed", "batch_seed", "registered"),
                    self._job_values("evaluation_a", "batch_a", "registered"),
                    self._job_values("evaluation_b", "batch_b", "result_verified"),
                    self._job_values("evaluation_failed", "batch_failed", "failed"),
                ),
            )
            db.executemany(
                "INSERT INTO evaluation_gates(gate_id,project_id,source_batch_id,gate_key,role,status,"
                "source_evaluation_job_id,artifact_path,source_split,content_sha256,image_count,"
                "created_at,activated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    (
                        "gate_seed",
                        self.project_id,
                        None,
                        "core_safety",
                        "core_safety",
                        "active",
                        "evaluation_seed",
                        "/immutable/evaluation_seed.zip",
                        "core_safety",
                        "a" * 64,
                        25,
                        NOW,
                        NOW,
                    ),
                    (
                        "gate_a",
                        self.project_id,
                        "batch_a",
                        "round_a_gate",
                        "round_gate",
                        "active",
                        "evaluation_a",
                        "/immutable/evaluation_a.zip",
                        "round_a_gate",
                        "b" * 64,
                        25,
                        NOW,
                        NOW,
                    ),
                    (
                        "gate_b",
                        self.project_id,
                        "batch_b",
                        "round_b_gate",
                        "round_gate",
                        "pending",
                        "evaluation_b",
                        "/immutable/evaluation_b.zip",
                        "round_b_gate",
                        "c" * 64,
                        25,
                        NOW,
                        None,
                    ),
                ),
            )
            db.commit()
        finally:
            db.close()

    def _service(self, store: Store) -> ModelEvaluationService:
        profile = (
            Path(__file__).resolve().parents[1]
            / "facade_agent"
            / "protocols"
            / "champion_challenger_evaluation_profile_v1.json"
        )
        return ModelEvaluationService(store, object(), self.root, profile, runner=object())

    def test_legacy_gates_migrate_without_rewriting_artifact_identity(self) -> None:
        store = Store(self.database)
        with store.connect() as db:
            cohorts = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM evaluation_cohorts WHERE project_id=? ORDER BY content_sha256",
                    (self.project_id,),
                )
            ]

        self.assertEqual(len(cohorts), 3)
        self.assertEqual(
            [(row["origin_role"], row["status"], row["source_round_name"]) for row in cohorts],
            [
                ("core_safety_seed", "active", "Seed round"),
                ("current_gate", "active", "Round A"),
                ("current_gate", "pending", "Round B"),
            ],
        )
        self.assertEqual(
            [
                (row["artifact_path"], row["source_split"], row["image_count"], row["activated_at"])
                for row in cohorts
            ],
            [
                ("/immutable/evaluation_seed.zip", "core_safety", 25, NOW),
                ("/immutable/evaluation_a.zip", "round_a_gate", 25, NOW),
                ("/immutable/evaluation_b.zip", "round_b_gate", 25, None),
            ],
        )
        self.assertEqual([row["source_batch_id"] for row in cohorts], ["batch_seed", "batch_a", "batch_b"])
        self.assertFalse(any("final" in row["origin_role"] for row in cohorts))

    def test_migration_is_idempotent_and_list_cohorts_filters_status(self) -> None:
        first = Store(self.database)
        second = Store(self.database)
        service = self._service(second)

        self.assertEqual(len(service.list_cohorts(self.project_id)), 3)
        active = service.list_cohorts(self.project_id, status="active")
        self.assertEqual([row["content_sha256"] for row in active], ["a" * 64, "b" * 64])
        with second.connect() as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM evaluation_cohorts").fetchone()[0],
                3,
            )
            applied = {
                row[0]
                for row in db.execute(
                    "SELECT version FROM schema_migrations WHERE version LIKE '01%_%.sql'"
                )
            }
        self.assertIn("016_evaluation_cohorts.sql", applied)
        self.assertIn("017_evaluation_job_revisions.sql", applied)
        self.assertIsNotNone(first)

    def test_evaluation_jobs_gain_one_current_revision_per_batch(self) -> None:
        store = Store(self.database)
        with store.connect() as db:
            failed = dict(
                db.execute(
                    "SELECT * FROM evaluation_jobs WHERE job_id='evaluation_failed'"
                ).fetchone()
            )
            self.assertEqual(failed["revision_number"], 1)
            self.assertEqual(failed["is_current"], 1)
            self.assertIsNone(failed["supersedes_job_id"])
            self.assertEqual(failed["bundle_path"], "/immutable/evaluation_failed.zip")
            self.assertEqual(
                failed["bundle_sha256"],
                hashlib.sha256(b"bundle:evaluation_failed").hexdigest(),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO evaluation_jobs(job_id,project_id,batch_id,champion_model_id,"
                    "challenger_model_id,status,evaluation_profile_id,profile_json,source_snapshot_name,"
                    "source_snapshot_sha256,split_counts_json,content_fingerprint,bundle_path,bundle_sha256,"
                    "bundle_size_bytes,revision_number,is_current,supersedes_job_id,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        *self._job_values("evaluation_failed_duplicate", "batch_failed", "failed")[:15],
                        2,
                        1,
                        "evaluation_failed",
                        NOW,
                        NOW,
                    ),
                )

    def test_history_and_project_deletion_include_cohorts(self) -> None:
        store = Store(self.database)
        with store.connect() as db:
            db.execute(
                "INSERT INTO final_test_image_inventory(project_id,image_id,content_sha256,"
                "source_artifact_id,created_at) VALUES(?,?,?,?,?)",
                (self.project_id, "final-1", "f" * 64, "sealed-final-v1", NOW),
            )
        history = store.get_project_history(self.project_id)
        self.assertEqual(history["counts"]["evaluation_cohorts"], 3)
        self.assertEqual(
            len([item for item in history["timeline"] if item["kind"] == "evaluation_cohort"]),
            3,
        )

        result = store.delete_project(
            project_id=self.project_id,
            confirmation_name="Legacy project",
            actor_type="human",
            actor_id="engineer",
        )
        self.assertEqual(result["records_deleted"]["evaluation_cohorts"], 3)
        self.assertEqual(result["records_deleted"]["final_test_image_inventory"], 1)
        with store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM evaluation_cohorts").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM final_test_image_inventory").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM evaluation_jobs").fetchone()[0], 0)

    def test_round_completion_summary_schema_is_ready_for_atomic_decisions(self) -> None:
        store = Store(self.database)
        self.assertIsNone(store.get_round_completion("batch_failed"))
        with store.connect() as db:
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(round_completion_summaries)")
            }
        self.assertTrue(
            {
                "batch_id",
                "project_id",
                "decision",
                "reason",
                "current_gate_cohort_id",
                "core_safety_cohort_id",
                "cumulative_core_safety_images",
                "final_test_read",
            }.issubset(columns)
        )


if __name__ == "__main__":
    unittest.main()
