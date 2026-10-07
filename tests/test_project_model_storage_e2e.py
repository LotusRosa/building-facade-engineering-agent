from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_agent.application.model_storage import (
    ModelStorageService,
    backfill_legacy_model_storage,
)
from facade_agent.storage import Store, canonical_json, utc_now
from facade_agent.tools import build_phase1_registry


class ProjectModelStorageE2ETests(unittest.TestCase):
    def test_initial_and_challenger_are_the_only_project_model_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "data" / "agent.sqlite3")
            model_storage = ModelStorageService(store, root)
            registry = build_phase1_registry(store, model_storage)
            model_root = root / "external-project-models"
            with patch(
                "facade_agent.application.model_storage.tempfile.gettempdir",
                return_value=str(root / "system-temp"),
            ):
                project = registry.execute(
                    "create_project",
                    {
                        "project_name": "Facade lifecycle",
                        "classes": ["crack", "spalling"],
                        "model_storage_root": str(model_root),
                    },
                    actor_type="human",
                    actor_id="engineer",
                    confirmed=True,
                )["project"]
            taxonomy_sha256 = model_storage.taxonomy_sha256(project["project_id"])

            champion = self._register_fixture_model(
                store,
                model_storage,
                root,
                project["project_id"],
                model_id="champion_e2e",
                role="champion",
                parent_model_id=None,
                taxonomy_sha256=taxonomy_sha256,
            )
            challenger = self._register_fixture_model(
                store,
                model_storage,
                root,
                project["project_id"],
                model_id="challenger_e2e",
                role="challenger",
                parent_model_id=champion["model_id"],
                taxonomy_sha256=taxonomy_sha256,
            )

            (root / "artifact_store" / "training_results").mkdir(parents=True)
            (root / "projects" / project["project_id"]).mkdir(parents=True)
            self.assertEqual(
                {item.name for item in model_root.iterdir()},
                {champion["model_id"], challenger["model_id"]},
            )
            for model in (champion, challenger):
                checkpoint = model_storage.resolve_model_file(
                    project["project_id"], model["checkpoint_relpath"]
                )
                manifest_path = model_storage.resolve_model_file(
                    project["project_id"], model["manifest_relpath"]
                )
                self.assertEqual(
                    {item.name for item in checkpoint.parent.iterdir()},
                    {checkpoint.name, "manifest.json"},
                )
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    model["checkpoint_sha256"],
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["taxonomy_sha256"], taxonomy_sha256)
                self.assertEqual(manifest["checkpoint_sha256"], model["checkpoint_sha256"])
            with store.connect() as db:
                restart_check = backfill_legacy_model_storage(db, root)
            self.assertEqual(restart_check, {"projects_migrated": 0, "models_migrated": 0})

    @staticmethod
    def _register_fixture_model(
        store: Store,
        model_storage: ModelStorageService,
        root: Path,
        project_id: str,
        *,
        model_id: str,
        role: str,
        parent_model_id: str | None,
        taxonomy_sha256: str,
    ) -> dict:
        source = root / "artifact_store" / "verified" / f"{model_id}.pt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"verified-{model_id}".encode())
        checkpoint_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        now = utc_now()
        manifest = {
            "schema_version": 1,
            "project_id": project_id,
            "model_id": model_id,
            "parent_model_id": parent_model_id,
            "source_batch_id": "round-a" if role == "challenger" else None,
            "role": role,
            "taxonomy_sha256": taxonomy_sha256,
            "thresholds": {"fixture": 0.5},
            "training_profile_id": "e2e-profile",
            "registered_at": now,
        }
        with model_storage.prepare_registration(
            project_id=project_id,
            model_id=model_id,
            source_checkpoint=source,
            expected_sha256=checkpoint_sha256,
            manifest=manifest,
        ) as prepared:
            published = prepared.publish()
            checkpoint = model_storage.resolve_model_file(
                project_id, published["checkpoint_relpath"]
            )
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO model_versions(model_id,project_id,parent_model_id,role,"
                    "checkpoint_path,checkpoint_relpath,checkpoint_sha256,manifest_relpath,"
                    "manifest_sha256,thresholds_json,training_profile_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        model_id,
                        project_id,
                        parent_model_id,
                        role,
                        str(checkpoint),
                        published["checkpoint_relpath"],
                        checkpoint_sha256,
                        published["manifest_relpath"],
                        published["manifest_sha256"],
                        canonical_json({"fixture": 0.5}),
                        "e2e-profile",
                        now,
                    ),
                )
                db.execute(
                    "UPDATE projects SET model_storage_locked_at=COALESCE(model_storage_locked_at,?) "
                    "WHERE project_id=?",
                    (now, project_id),
                )
            prepared.commit()
        with store.connect() as db:
            return dict(
                db.execute(
                    "SELECT * FROM model_versions WHERE model_id=?", (model_id,)
                ).fetchone()
            )


if __name__ == "__main__":
    unittest.main()
