from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from facade_agent.application.model_storage import (
    ModelStorageService,
    backfill_legacy_model_storage,
)
from facade_agent.storage import Store, canonical_json, utc_now
from facade_agent.tools import build_phase1_registry


class ProjectModelStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app_root = self.root / "app"
        self.store = Store(self.app_root / "data" / "agent.sqlite3")
        self.storage = ModelStorageService(self.store, self.app_root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_project_requires_safe_model_root(self) -> None:
        with self.store.connect() as db:
            project_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(projects)")
            }
            model_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(model_versions)")
            }

        self.assertIn("model_storage_root", project_columns)
        self.assertIn("model_storage_locked_at", project_columns)
        self.assertIn("checkpoint_relpath", model_columns)
        self.assertIn("manifest_relpath", model_columns)
        self.assertIn("manifest_sha256", model_columns)

    def test_public_project_creation_requires_human_confirmed_safe_root(self) -> None:
        registry = build_phase1_registry(self.store, self.storage)
        with self.assertRaisesRegex(ValueError, "model_storage_root"):
            registry.execute(
                "create_project",
                {"project_name": "Facade", "classes": ["crack"]},
                actor_type="human",
                actor_id="engineer",
                confirmed=True,
            )

        model_root = self.root / "project-models"
        with patch("facade_agent.application.model_storage.tempfile.gettempdir", return_value=str(self.root / "system-temp")):
            with self.assertRaises(PermissionError):
                registry.execute(
                    "create_project",
                    {
                        "project_name": "Facade",
                        "classes": ["crack"],
                        "model_storage_root": str(model_root),
                    },
                    actor_type="llm",
                    actor_id="assistant",
                    confirmed=True,
                )
            created = registry.execute(
                "create_project",
                {
                    "project_name": "Facade",
                    "classes": ["crack"],
                    "model_storage_root": str(model_root),
                },
                actor_type="human",
                actor_id="engineer",
                confirmed=True,
            )

        self.assertEqual(created["project"]["model_storage_root"], str(model_root.resolve()))
        self.assertTrue(model_root.is_dir())
        with self.store.connect() as db:
            payload = db.execute(
                "SELECT payload_json FROM audit_events WHERE tool_name='create_project'"
            ).fetchone()["payload_json"]
        self.assertNotIn(str(model_root.parent), payload)
        self.assertIn(model_root.name, payload)

    def test_rejects_unsafe_roots(self) -> None:
        relative = "relative/models"
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "existing.pt").write_bytes(b"weights")
        file_path = self.root / "not-a-directory"
        file_path.write_text("x", encoding="utf-8")
        reserved_temp = self.root / "system-temp"
        reserved_temp.mkdir()

        with patch("facade_agent.application.model_storage.tempfile.gettempdir", return_value=str(reserved_temp)):
            for candidate in (
                relative,
                str(file_path),
                str(occupied),
                str(reserved_temp),
                str(self.app_root / "data"),
                str(self.app_root / "audit"),
            ):
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        self.storage.validate_root(candidate)

    def test_rejects_unwritable_root(self) -> None:
        candidate = self.root / "models"
        candidate.mkdir()
        with patch.object(
            self.storage,
            "_probe_writable",
            side_effect=PermissionError("read only"),
        ):
            with self.assertRaises(ValueError):
                self.storage.validate_root(str(candidate))

    def test_rejects_symlink_root(self) -> None:
        target = self.root / "real-models"
        target.mkdir()
        link = self.root / "linked-models"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("This Windows account cannot create directory symlinks.")
        with self.assertRaises(ValueError):
            self.storage.validate_root(str(link))

    def test_rejects_root_reuse_and_nested_roots(self) -> None:
        registry = build_phase1_registry(self.store, self.storage)
        first_root = self.root / "first-models"
        with patch("facade_agent.application.model_storage.tempfile.gettempdir", return_value=str(self.root / "system-temp")):
            registry.execute(
                "create_project",
                {
                    "project_name": "First",
                    "classes": ["crack"],
                    "model_storage_root": str(first_root),
                },
                actor_type="human",
                actor_id="engineer",
                confirmed=True,
            )
            for candidate in (
                first_root,
                first_root / "nested",
                first_root.parent,
            ):
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        self.storage.validate_root(str(candidate))

    def test_resolve_model_file_accepts_only_contained_posix_relative_paths(self) -> None:
        model_root = self.root / "project-models"
        created = self.store.create_project(
            project_name="Facade",
            class_names=["crack"],
            model_storage_root=str(model_root.resolve()),
        )["project"]
        expected = model_root / "model_001" / "checkpoint.pt"
        self.assertEqual(
            self.storage.resolve_model_file(
                created["project_id"], "model_001/checkpoint.pt"
            ),
            expected.resolve(),
        )
        for relpath in (
            "../checkpoint.pt",
            "/absolute/checkpoint.pt",
            r"model_001\checkpoint.pt",
            r"C:\checkpoint.pt",
        ):
            with self.subTest(relpath=relpath):
                with self.assertRaises(ValueError):
                    self.storage.resolve_model_file(created["project_id"], relpath)

    def test_prepared_model_publishes_canonical_files_and_requires_commit(self) -> None:
        model_root = self.root / "project-models"
        model_root.mkdir()
        project = self.store.create_project(
            project_name="Facade",
            class_names=["crack"],
            model_storage_root=str(model_root.resolve()),
        )["project"]
        source = self.root / "selected.pt"
        source.write_bytes(b"verified checkpoint")
        expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "project_id": project["project_id"],
            "model_id": "champion_fixture",
            "role": "champion",
        }

        with self.storage.prepare_registration(
            project_id=project["project_id"],
            model_id="champion_fixture",
            source_checkpoint=source,
            expected_sha256=expected_sha256,
            manifest=manifest,
        ) as prepared:
            published = prepared.publish()
            checkpoint = self.storage.resolve_model_file(
                project["project_id"], published["checkpoint_relpath"]
            )
            manifest_path = self.storage.resolve_model_file(
                project["project_id"], published["manifest_relpath"]
            )
            self.assertEqual(checkpoint.name, "checkpoint.pt")
            self.assertEqual(manifest_path.name, "manifest.json")
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_sha256"], expected_sha256)
        self.assertEqual(list(model_root.iterdir()), [])

        with self.storage.prepare_registration(
            project_id=project["project_id"],
            model_id="champion_fixture",
            source_checkpoint=source,
            expected_sha256=expected_sha256,
            manifest=manifest,
        ) as prepared:
            published = prepared.publish()
            prepared.commit()
        self.assertTrue((model_root / "champion_fixture" / "checkpoint.pt").is_file())
        self.assertEqual(
            hashlib.sha256((model_root / "champion_fixture" / "manifest.json").read_bytes()).hexdigest(),
            published["manifest_sha256"],
        )

    def test_prepared_model_hash_failure_and_orphans_never_become_models(self) -> None:
        model_root = self.root / "project-models"
        model_root.mkdir()
        project = self.store.create_project(
            project_name="Facade",
            class_names=["crack"],
            model_storage_root=str(model_root.resolve()),
        )["project"]
        source = self.root / "selected.pt"
        source.write_bytes(b"checkpoint")
        with self.assertRaises(ValueError):
            with self.storage.prepare_registration(
                project_id=project["project_id"],
                model_id="bad_hash",
                source_checkpoint=source,
                expected_sha256="0" * 64,
                manifest={"schema_version": 1},
            ) as prepared:
                prepared.publish()
        self.assertEqual(list(model_root.iterdir()), [])

        expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(Path, "write_text", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                with self.storage.prepare_registration(
                    project_id=project["project_id"],
                    model_id="manifest_failure",
                    source_checkpoint=source,
                    expected_sha256=expected_sha256,
                    manifest={"schema_version": 1},
                ) as prepared:
                    prepared.publish()
        self.assertEqual(list(model_root.iterdir()), [])

        orphan = model_root / "orphan_model"
        orphan.mkdir()
        (orphan / "checkpoint.pt").write_bytes(b"orphan")
        result = self.storage.reconcile_orphans()
        self.assertEqual(result["quarantined"], 1)
        self.assertFalse(orphan.exists())
        with self.store.connect() as db:
            count = db.execute(
                "SELECT count(*) FROM model_versions WHERE project_id=?",
                (project["project_id"],),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_legacy_backfill_is_verified_and_idempotent_without_moving_weights(self) -> None:
        project = self.store.create_project(
            project_name="Legacy",
            class_names=["crack"],
        )["project"]
        model_id = "champion_legacy"
        checkpoint = (
            self.app_root / "models" / project["project_id"] / model_id / "legacy.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"legacy-verified-checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        now = utc_now()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    model_id,
                    project["project_id"],
                    "champion",
                    str(checkpoint),
                    digest,
                    canonical_json({project["classes"][0]["class_id"]: 0.5}),
                    "legacy-profile",
                    now,
                ),
            )

        with self.store.connect() as db:
            first = backfill_legacy_model_storage(db, self.app_root)
        self.assertEqual(first["projects_migrated"], 1)
        self.assertTrue(checkpoint.is_file())
        migrated = self.store.get_project(project["project_id"])
        self.assertEqual(
            migrated["model_storage_root"],
            str((self.app_root / "models" / project["project_id"]).resolve()),
        )
        with self.store.connect() as db:
            model = db.execute(
                "SELECT * FROM model_versions WHERE model_id=?", (model_id,)
            ).fetchone()
        self.assertEqual(model["checkpoint_relpath"], f"{model_id}/legacy.pt")
        self.assertEqual(model["manifest_relpath"], f"{model_id}/manifest.json")
        self.assertTrue((checkpoint.parent / "manifest.json").is_file())
        snapshot = {
            item.relative_to(checkpoint.parents[1]).as_posix(): item.read_bytes()
            for item in checkpoint.parents[1].rglob("*")
            if item.is_file()
        }
        with self.store.connect() as db:
            second = backfill_legacy_model_storage(db, self.app_root)
        self.assertEqual(second["projects_migrated"], 0)
        self.assertEqual(
            snapshot,
            {
                item.relative_to(checkpoint.parents[1]).as_posix(): item.read_bytes()
                for item in checkpoint.parents[1].rglob("*")
                if item.is_file()
            },
        )

    def test_legacy_backfill_rejects_missing_or_tampered_weight_atomically(self) -> None:
        for case in ("missing", "tampered"):
            with self.subTest(case=case):
                case_root = self.root / case
                store = Store(case_root / "data" / "agent.sqlite3")
                project = store.create_project(
                    project_name=case,
                    class_names=["crack"],
                )["project"]
                model_id = f"champion_{case}"
                checkpoint = case_root / "models" / project["project_id"] / model_id / "checkpoint.pt"
                if case == "tampered":
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"tampered")
                with store.connect() as db:
                    db.execute(
                        "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                        "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            model_id,
                            project["project_id"],
                            "champion",
                            str(checkpoint),
                            "f" * 64,
                            "{}",
                            "legacy",
                            utc_now(),
                        ),
                    )
                with store.connect() as db:
                    with self.assertRaises(ValueError):
                        backfill_legacy_model_storage(db, case_root)
                self.assertIsNone(store.get_project(project["project_id"])["model_storage_root"])
                self.assertFalse((checkpoint.parent / "manifest.json").exists())

    def test_server_backfills_before_orphan_reconciliation(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "facade_agent" / "server.py"
        ).read_text(encoding="utf-8-sig")
        self.assertLess(
            source.index("backfill_legacy_model_storage(startup_db, ROOT)"),
            source.index("MODEL_STORAGE.reconcile_orphans()"),
        )


if __name__ == "__main__":
    unittest.main()
