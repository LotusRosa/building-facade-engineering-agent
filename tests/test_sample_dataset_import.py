from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from facade_agent.application.sample_dataset import (
    SampleDatasetImportService,
    import_sample_bundle,
    receive_bundle,
    validate_sample_bundle,
)
from facade_agent.core.states import BatchState, ProjectState
from facade_agent.storage import Store, utc_now


def png_bytes(width: int, height: int, marker: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + bytes([marker])


def jpeg_with_late_sof(width: int, height: int) -> bytes:
    app_payload = b"x" * 65533
    return (
        b"\xff\xd8\xff\xe1"
        + struct.pack(">H", 65535)
        + app_payload
        + b"\xff\xc0\x00\x07\x08"
        + struct.pack(">HH", height, width)
        + b"\xff\xd9"
    )


def make_bundle(
    path: Path,
    *,
    agent_role: str = "initial_training",
    source_split: str | None = None,
    tamper: bool = False,
    late_jpeg: bool = False,
    prefix: str = "DFWI",
    count: int = 4,
    marker_offset: int = 0,
    duplicate_content: bool = False,
) -> None:
    if source_split is None:
        source_split = (
            "core_train" if agent_role == "initial_training" else "adaptation_a_discovery"
        )
    classes = ["crack", "spalling", "hollow"]
    images = {
        f"images/{prefix}_{index:05d}.png": png_bytes(8, 8, marker_offset + index)
        for index in range(1, count + 1)
    }
    if duplicate_content and count >= 2:
        images[f"images/{prefix}_00002.png"] = images[f"images/{prefix}_00001.png"]
    if late_jpeg:
        images[f"images/{prefix}_00001.jpg"] = jpeg_with_late_sof(16, 12)
        del images[f"images/{prefix}_00001.png"]
    label_sets = (["crack"], ["spalling"], ["hollow"], [])
    labels = []
    for index, (member, content) in enumerate(images.items()):
        selected = list(label_sets[index % len(label_sets)])
        labels.append(
            {
                "image_id": Path(member).stem,
                "filename": Path(member).name,
                "source_split": source_split,
                "temporal_unit_id": Path(member).stem,
                "capture_group": Path(member).stem,
                "labels": selected,
                "no_defect": not selected,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    files = {
        "classes.json": (json.dumps(classes, sort_keys=True) + "\n").encode(),
        "dataset_manifest.json": (
            json.dumps(
                {
                    "schema_version": 2,
                    "name": f"BFEA {agent_role} fixture",
                    "agent_role": agent_role,
                    "source_split": source_split,
                    "sample_size": count,
                    "selection_seed": 20260930,
                    "classes": classes,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "labels.jsonl": ("\n".join(json.dumps(row, sort_keys=True) for row in labels) + "\n").encode(),
        **images,
    }
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    if tamper:
        first_image = next(name for name in files if name.startswith("images/"))
        checksums[first_image] = "0" * 64
    files["checksums.json"] = (json.dumps(checksums, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)


class SampleDatasetImportTests(unittest.TestCase):
    def test_verified_frozen_labels_import_to_audited_validated_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "sample.zip"
            make_bundle(bundle, late_jpeg=True)
            loaded = validate_sample_bundle(bundle)
            self.assertEqual(len(loaded.labels), 4)

            app_root = root / "agent"
            model_root = root / "model-storage"
            with patch(
                "facade_agent.application.model_storage.tempfile.gettempdir",
                return_value=str(root / "unrelated-system-temp"),
            ):
                result = import_sample_bundle(
                    app_root=app_root,
                    bundle_path=bundle,
                    project_name="Linux smoke 200",
                    model_storage_root=model_root,
                    confirmed=True,
                )

            store = Store(app_root / "data" / "facade_agent.sqlite3")
            project = store.get_project(result["project_id"])
            dataset = store.get_dataset(result["dataset_id"])
            images = store.list_images(result["dataset_id"])
            self.assertEqual(project["state"], "DATA_VALIDATED")
            self.assertEqual(dataset["image_count"], 4)
            self.assertEqual(dataset["labeled_count"], 4)
            self.assertEqual({item["annotation_status"] for item in images}, {"complete"})
            self.assertTrue(store.verify_audit_chain()["valid"])
            self.assertTrue(model_root.is_dir())
            self.assertEqual(list(model_root.iterdir()), [])

    def test_initial_bundle_imports_into_an_existing_confirmed_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "initial.zip"
            make_bundle(bundle, count=3)
            store = Store(root / "agent.sqlite3")
            project = store.create_project(
                project_name="Browser trial",
                class_names=["crack", "spalling", "hollow"],
            )["project"]
            store.transition_project(
                project_id=project["project_id"],
                action="confirm_taxonomy",
                confirmed=True,
            )

            result = SampleDatasetImportService(store, root).import_initial_bundle(
                project_id=project["project_id"],
                bundle_path=bundle,
                actor_id="browser_engineer",
                confirmed=True,
            )

            self.assertEqual(result["image_count"], 3)
            self.assertEqual(
                store.get_project(project["project_id"])["state"],
                ProjectState.DATA_VALIDATED,
            )
            self.assertEqual(store.get_dataset(result["dataset_id"])["status"], "validated")
            history = store.get_project_history(project["project_id"])
            provenance = next(
                item["data"] for item in history["timeline"]
                if item["kind"] == "audit"
                and item["data"]["tool_name"] == "import_verified_labeled_bundle"
            )
            self.assertEqual(provenance["payload"]["agent_role"], "initial_training")
            self.assertTrue(store.verify_audit_chain()["valid"])

    def test_maintenance_bundle_imports_without_freezing_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "round-one.zip"
            make_bundle(
                bundle,
                agent_role="maintenance",
                prefix="ROUND1",
                count=5,
                marker_offset=20,
            )
            store = Store(root / "agent.sqlite3")
            project = store.create_project(
                project_name="Multi-round project",
                class_names=["crack", "spalling", "hollow"],
            )["project"]
            project_id = project["project_id"]
            store.transition_project(
                project_id=project_id,
                action="confirm_taxonomy",
                confirmed=True,
            )
            now = utc_now()
            with store.connect() as db:
                db.execute(
                    "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                    "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        "champion-fixture",
                        project_id,
                        "champion",
                        str(root / "champion.pt"),
                        "f" * 64,
                        "[0.5,0.5,0.5]",
                        "fixture-v1",
                        now,
                    ),
                )
                db.execute(
                    "INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)",
                    (project_id, "champion-fixture", now),
                )
                db.execute(
                    "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                    (ProjectState.CHAMPION_READY, now, project_id),
                )
            batch = store.create_maintenance_batch(
                project_id=project_id,
                batch_name="Round one",
            )["maintenance_batch"]

            result = SampleDatasetImportService(store, root).import_maintenance_bundle(
                batch_id=batch["batch_id"],
                bundle_path=bundle,
                actor_id="browser_engineer",
                confirmed=True,
            )

            self.assertEqual(result["image_count"], 5)
            self.assertEqual(
                store.get_maintenance_batch(batch["batch_id"])["state"],
                BatchState.MAINTENANCE_LABELS_READY,
            )
            self.assertEqual(store.get_project(project_id)["state"], ProjectState.CHAMPION_READY)
            self.assertEqual(store.get_dataset(result["dataset_id"])["status"], "validated")

    def test_tampered_or_wrong_role_bundle_is_rejected_before_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tampered = root / "tampered.zip"
            make_bundle(tampered, tamper=True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                import_sample_bundle(
                    app_root=root / "agent-a",
                    bundle_path=tampered,
                    project_name="bad",
                    model_storage_root=root / "models-a",
                    confirmed=True,
                )
            self.assertFalse((root / "agent-a" / "data" / "facade_agent.sqlite3").exists())

            wrong_role = root / "wrong-role.zip"
            make_bundle(wrong_role, agent_role="maintenance", prefix="ROUND1")
            store = Store(root / "existing.sqlite3")
            project = store.create_project(
                project_name="No mutation",
                class_names=["crack", "spalling", "hollow"],
            )["project"]
            store.transition_project(
                project_id=project["project_id"],
                action="confirm_taxonomy",
                confirmed=True,
            )
            with self.assertRaisesRegex(ValueError, "initial_training"):
                SampleDatasetImportService(store, root).import_initial_bundle(
                    project_id=project["project_id"],
                    bundle_path=wrong_role,
                    actor_id="browser_engineer",
                    confirmed=True,
                )
            self.assertEqual(store.list_datasets(project["project_id"]), [])
            self.assertEqual(
                store.get_project(project["project_id"])["state"],
                ProjectState.TAXONOMY_CONFIRMED,
            )

            protected = root / "protected.zip"
            make_bundle(protected, source_split="final_test")
            with self.assertRaisesRegex(ValueError, "source split"):
                validate_sample_bundle(protected)

    def test_taxonomy_mismatch_is_rejected_before_dataset_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "sample.zip"
            make_bundle(bundle)
            store = Store(root / "agent.sqlite3")
            project = store.create_project(
                project_name="Wrong classes",
                class_names=["crack", "hollow"],
            )["project"]
            store.transition_project(
                project_id=project["project_id"],
                action="confirm_taxonomy",
                confirmed=True,
            )
            with self.assertRaisesRegex(ValueError, "taxonomy"):
                SampleDatasetImportService(store, root).import_initial_bundle(
                    project_id=project["project_id"],
                    bundle_path=bundle,
                    actor_id="browser_engineer",
                    confirmed=True,
                )
            self.assertEqual(store.list_datasets(project["project_id"]), [])

    def test_duplicate_content_is_rejected_before_dataset_or_batch_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_bundle = root / "duplicate.zip"
            make_bundle(duplicate_bundle, duplicate_content=True)
            with self.assertRaisesRegex(ValueError, "duplicate image content"):
                validate_sample_bundle(duplicate_bundle)

            round_bundle = root / "round.zip"
            make_bundle(
                round_bundle,
                agent_role="maintenance",
                prefix="ROUND",
                marker_offset=40,
            )
            store = Store(root / "agent.sqlite3")
            project = store.create_project(
                project_name="Historical duplicate",
                class_names=["crack", "spalling", "hollow"],
            )["project"]
            project_id = project["project_id"]
            store.transition_project(
                project_id=project_id,
                action="confirm_taxonomy",
                confirmed=True,
            )
            prior = store.create_dataset(
                project_id=project_id,
                name="prior",
                role="initial_training",
            )["dataset"]
            duplicate_hash = hashlib.sha256(png_bytes(8, 8, 41)).hexdigest()
            store.register_image(
                dataset_id=prior["dataset_id"],
                filename="prior.png",
                stored_path=str(root / "prior.png"),
                sha256=duplicate_hash,
                size_bytes=24,
                mime_type="image/png",
                width=8,
                height=8,
                health_status="ok",
            )
            now = utc_now()
            with store.connect() as db:
                db.execute(
                    "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,"
                    "checkpoint_sha256,thresholds_json,training_profile_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        "champion-duplicate",
                        project_id,
                        "champion",
                        str(root / "champion.pt"),
                        "f" * 64,
                        "[0.5,0.5,0.5]",
                        "fixture-v1",
                        now,
                    ),
                )
                db.execute(
                    "INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)",
                    (project_id, "champion-duplicate", now),
                )
                db.execute(
                    "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                    (ProjectState.CHAMPION_READY, now, project_id),
                )
            batch = store.create_maintenance_batch(
                project_id=project_id,
                batch_name="Duplicate round",
            )["maintenance_batch"]

            with self.assertRaisesRegex(ValueError, "already exists in project history"):
                SampleDatasetImportService(store, root).import_maintenance_bundle(
                    batch_id=batch["batch_id"],
                    bundle_path=round_bundle,
                    actor_id="browser_engineer",
                    confirmed=True,
                )
            self.assertEqual(
                store.get_maintenance_batch(batch["batch_id"])["state"],
                BatchState.CREATED,
            )
            self.assertIsNone(store.get_maintenance_batch(batch["batch_id"])["dataset"])

    def test_import_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "sample.zip"
            make_bundle(bundle)
            with self.assertRaises(PermissionError):
                SampleDatasetImportService(Store(root / "agent.sqlite3"), root).import_initial_bundle(
                    project_id="not-used",
                    bundle_path=bundle,
                    actor_id="browser_engineer",
                    confirmed=False,
                )

    def test_bundle_receipt_streams_exact_length_and_removes_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "received.zip"
            digest = receive_bundle(
                io.BytesIO(b"abcdef"),
                content_length=6,
                destination=destination,
                max_bytes=10,
            )
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(digest, hashlib.sha256(b"abcdef").hexdigest())

            oversized = root / "oversized.zip"
            with self.assertRaisesRegex(ValueError, "size limit"):
                receive_bundle(
                    io.BytesIO(b"01234567890"),
                    content_length=11,
                    destination=oversized,
                    max_bytes=10,
                )
            self.assertFalse(oversized.exists())

            incomplete = root / "incomplete.zip"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                receive_bundle(
                    io.BytesIO(b"abc"),
                    content_length=6,
                    destination=incomplete,
                    max_bytes=10,
                )
            self.assertFalse(incomplete.exists())


if __name__ == "__main__":
    unittest.main()
