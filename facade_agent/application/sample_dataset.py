from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..storage import Store
from ..tools.registry import build_phase1_registry
from .image_io import probe_image, safe_filename
from .model_storage import ModelStorageService


MAX_SAMPLE_IMAGES = 10_000
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_ARCHIVE_BYTES = 64 * 1024**3
MAX_UPLOAD_BYTES = 2 * 1024**3
PROTECTED_SOURCE_SPLITS = {
    "development_calib",
    "core_safety",
    "adaptation_a_gate",
    "adaptation_b_gate",
    "final_test",
}
SUPPORTED_AGENT_ROLES = {"initial_training", "maintenance"}


@dataclass(frozen=True)
class SampleDatasetBundle:
    path: Path
    manifest: dict[str, Any]
    classes: tuple[str, ...]
    labels: tuple[dict[str, Any], ...]
    checksums: dict[str, str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receive_bundle(
    stream: Any,
    *,
    content_length: int,
    destination: Path,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> str:
    """Receive one request body without retaining the archive in memory."""
    length = int(content_length)
    target = Path(destination)
    if length <= 0:
        raise ValueError("The labeled bundle upload is empty.")
    if length > int(max_bytes):
        raise ValueError("The labeled bundle exceeds the upload size limit.")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = length
    try:
        with target.open("xb") as output:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("The labeled bundle upload is incomplete.")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_bundle(
    bundle_path: Path,
    *,
    expected_agent_role: str | None = None,
) -> SampleDatasetBundle:
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise ValueError("The labeled sample ZIP does not exist.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("The labeled sample is not a valid ZIP archive.") from exc
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or not names or len(names) > MAX_SAMPLE_IMAGES + 10:
            raise ValueError("The labeled sample has duplicate or excessive members.")
        total = 0
        for info in infos:
            if not _safe_member(info.filename):
                raise ValueError("The labeled sample contains an unsafe member path.")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError("The labeled sample cannot contain symbolic links.")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError("A labeled sample member exceeds the safety limit.")
            total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("The labeled sample exceeds the uncompressed safety limit.")

        required = {"dataset_manifest.json", "classes.json", "labels.jsonl", "checksums.json"}
        if not required.issubset(names):
            raise ValueError("The labeled sample is missing required metadata.")
        try:
            manifest = json.loads(archive.read("dataset_manifest.json"))
            classes_value = json.loads(archive.read("classes.json"))
            labels = [
                json.loads(line)
                for line in archive.read("labels.jsonl").decode("utf-8").splitlines()
                if line.strip()
            ]
            checksums = json.loads(archive.read("checksums.json"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("The labeled sample metadata is invalid JSON.") from exc

        schema_version = manifest.get("schema_version")
        if schema_version == 1:
            agent_role = "initial_training"
            source_split = manifest.get("source_split")
            if source_split != "core_train":
                raise ValueError("A legacy schema-v1 bundle must use the Core Train source split.")
        elif schema_version == 2:
            agent_role = str(manifest.get("agent_role", "")).strip()
            source_split = str(manifest.get("source_split", "")).strip()
            if agent_role not in SUPPORTED_AGENT_ROLES:
                raise ValueError("The labeled bundle Agent role is invalid.")
            if not source_split or source_split in PROTECTED_SOURCE_SPLITS or source_split.endswith("_gate"):
                raise ValueError("The labeled bundle source split is protected or invalid for training.")
        else:
            raise ValueError("The labeled bundle schema version is unsupported.")
        if expected_agent_role is not None and agent_role != expected_agent_role:
            raise ValueError(
                f"This import requires agent_role={expected_agent_role}; received {agent_role}."
            )
        if not isinstance(classes_value, list) or not classes_value:
            raise ValueError("The labeled sample classes are invalid.")
        classes = tuple(str(value).strip() for value in classes_value)
        if any(not value for value in classes) or len(set(classes)) != len(classes):
            raise ValueError("The labeled sample classes must be non-empty and unique.")
        if manifest.get("classes") != list(classes):
            raise ValueError("The labeled sample class manifest does not match classes.json.")
        if manifest.get("sample_size") != len(labels) or not 1 <= len(labels) <= MAX_SAMPLE_IMAGES:
            raise ValueError("The labeled sample image count does not match its manifest.")

        expected_checksum_members = set(names) - {"checksums.json"}
        if not isinstance(checksums, dict) or set(checksums) != expected_checksum_members:
            raise ValueError("The labeled sample checksum inventory is incomplete.")
        for name in sorted(expected_checksum_members):
            expected = checksums.get(name)
            if not isinstance(expected, str) or len(expected) != 64 or _member_sha256(archive, name) != expected:
                raise ValueError(f"The labeled sample checksum failed: {name}")

        image_members = {name for name in names if name.startswith("images/")}
        observed_members: set[str] = set()
        observed_ids: set[str] = set()
        observed_filenames: set[str] = set()
        observed_hashes: set[str] = set()
        normalized_labels: list[dict[str, Any]] = []
        for raw in labels:
            if not isinstance(raw, dict):
                raise ValueError("A labeled sample row is not an object.")
            image_id = str(raw.get("image_id", "")).strip()
            filename = safe_filename(str(raw.get("filename", "")))
            member = f"images/{filename}"
            selected = raw.get("labels")
            no_defect = raw.get("no_defect")
            if raw.get("source_split") != source_split:
                raise ValueError("A labeled sample row does not match the manifest source split.")
            if not image_id or image_id in observed_ids or filename in observed_filenames:
                raise ValueError("The labeled sample contains duplicate image identities.")
            if member not in image_members or member in observed_members:
                raise ValueError("The labeled sample image inventory does not match its labels.")
            if not isinstance(selected, list) or any(value not in classes for value in selected):
                raise ValueError("A labeled sample row contains an unknown class.")
            selected = sorted(set(selected), key=classes.index)
            if not isinstance(no_defect, bool) or no_defect == bool(selected):
                raise ValueError("Each row must contain defect labels or No defect, exclusively.")
            declared_sha = raw.get("sha256")
            if declared_sha != checksums[member]:
                raise ValueError("A labeled sample image hash does not match its label row.")
            if declared_sha in observed_hashes:
                raise ValueError("The labeled sample contains duplicate image content.")
            with archive.open(member) as stream:
                image_bytes = stream.read()
            if probe_image(image_bytes, filename).health_status != "ok":
                raise ValueError(f"The labeled sample contains an unreadable image: {filename}")
            observed_ids.add(image_id)
            observed_filenames.add(filename)
            observed_hashes.add(declared_sha)
            observed_members.add(member)
            normalized_labels.append(
                {
                    **raw,
                    "image_id": image_id,
                    "filename": filename,
                    "labels": selected,
                    "no_defect": no_defect,
                    "sha256": declared_sha,
                }
            )
        if observed_members != image_members:
            raise ValueError("The labeled sample contains unreferenced images.")
        return SampleDatasetBundle(
            path=path,
            manifest={**dict(manifest), "agent_role": agent_role, "source_split": source_split},
            classes=classes,
            labels=tuple(normalized_labels),
            checksums=dict(checksums),
        )


class SampleDatasetImportService:
    """Import a fully verified labeled bundle into an existing Agent workflow."""

    def __init__(self, store: Store, app_root: Path):
        self.store = store
        self.app_root = Path(app_root).resolve()

    @staticmethod
    def _require_confirmation(confirmed: bool) -> None:
        if not confirmed:
            raise PermissionError("Importing frozen labels requires explicit engineer confirmation.")

    @staticmethod
    def _require_taxonomy(project: dict[str, Any], bundle: SampleDatasetBundle) -> None:
        project_classes = tuple(item["display_name"] for item in project["classes"])
        if project_classes != bundle.classes:
            raise ValueError(
                "The bundle taxonomy must exactly match the confirmed project taxonomy."
            )

    def _require_new_project_images(
        self,
        project_id: str,
        bundle: SampleDatasetBundle,
    ) -> None:
        historical = self.store.list_project_image_hashes(project_id)
        repeated = sorted({row["sha256"] for row in bundle.labels} & historical)
        if repeated:
            raise ValueError(
                "A labeled bundle image already exists in project history: "
                + ", ".join(repeated[:5])
            )

    def _import_rows(
        self,
        *,
        project_id: str,
        dataset_id: str,
        bundle: SampleDatasetBundle,
    ) -> None:
        project = self.store.get_project(project_id)
        class_ids = {item["display_name"]: item["class_id"] for item in project["classes"]}
        image_folder = self.app_root / "projects" / project_id / "datasets" / dataset_id / "images"
        image_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle.path) as archive:
            for row in bundle.labels:
                member = f"images/{row['filename']}"
                destination = image_folder / f"{row['sha256'][:12]}__{row['filename']}"
                temporary = destination.with_name(destination.name + ".part")
                digest = hashlib.sha256()
                try:
                    with archive.open(member) as source, temporary.open("xb") as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                    if digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"Image changed during import: {row['filename']}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                probe = probe_image(destination.read_bytes(), row["filename"])
                image = self.store.register_image(
                    dataset_id=dataset_id,
                    filename=row["filename"],
                    stored_path=str(destination.resolve()),
                    sha256=row["sha256"],
                    size_bytes=destination.stat().st_size,
                    mime_type=probe.mime_type,
                    width=probe.width,
                    height=probe.height,
                    health_status=probe.health_status,
                )
                self.store.save_annotation(
                    image_id=image["image_id"],
                    class_ids=[class_ids[name] for name in row["labels"]],
                    no_defect=row["no_defect"],
                )

    def _finish_import(
        self,
        *,
        project_id: str,
        batch_id: str | None,
        dataset_id: str,
        bundle: SampleDatasetBundle,
        actor_id: str,
    ) -> dict[str, Any]:
        self._import_rows(
            project_id=project_id,
            dataset_id=dataset_id,
            bundle=bundle,
        )
        report = self.store.validate_dataset(
            dataset_id=dataset_id,
            actor_type="human",
            actor_id=actor_id,
        )
        if not report["valid"]:
            raise ValueError("The imported labeled bundle failed Agent dataset validation.")
        self.store.record_labeled_bundle_import(
            dataset_id=dataset_id,
            bundle_filename=bundle.path.name,
            bundle_sha256=_file_sha256(bundle.path),
            agent_role=bundle.manifest["agent_role"],
            source_split=bundle.manifest["source_split"],
            image_count=len(bundle.labels),
            actor_type="human",
            actor_id=actor_id,
        )
        return {
            "project_id": project_id,
            "batch_id": batch_id,
            "dataset_id": dataset_id,
            "image_count": len(bundle.labels),
            "validation": report,
            "audit_chain": self.store.verify_audit_chain(),
        }

    def import_initial_bundle(
        self,
        *,
        project_id: str,
        bundle_path: Path,
        actor_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_confirmation(confirmed)
        bundle = validate_sample_bundle(
            bundle_path,
            expected_agent_role="initial_training",
        )
        project = self.store.get_project(project_id)
        self._require_taxonomy(project, bundle)
        dataset = self.store.create_dataset(
            project_id=project_id,
            name=str(bundle.manifest.get("name") or "Imported initial training data"),
            role="initial_training",
            actor_type="human",
            actor_id=actor_id,
        )["dataset"]
        return self._finish_import(
            project_id=project_id,
            batch_id=None,
            dataset_id=dataset["dataset_id"],
            bundle=bundle,
            actor_id=actor_id,
        )

    def import_maintenance_bundle(
        self,
        *,
        batch_id: str,
        bundle_path: Path,
        actor_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_confirmation(confirmed)
        bundle = validate_sample_bundle(
            bundle_path,
            expected_agent_role="maintenance",
        )
        batch = self.store.get_maintenance_batch(batch_id)
        project_id = batch["project_id"]
        self._require_taxonomy(self.store.get_project(project_id), bundle)
        self._require_new_project_images(project_id, bundle)
        dataset = self.store.create_maintenance_dataset(
            batch_id=batch_id,
            name=str(bundle.manifest.get("name") or "Imported maintenance data"),
            actor_type="human",
            actor_id=actor_id,
        )["dataset"]
        return self._finish_import(
            project_id=project_id,
            batch_id=batch_id,
            dataset_id=dataset["dataset_id"],
            bundle=bundle,
            actor_id=actor_id,
        )


def import_sample_bundle(
    *,
    app_root: Path,
    bundle_path: Path,
    project_name: str,
    model_storage_root: Path,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("Importing frozen labels requires explicit engineer confirmation.")
    bundle = validate_sample_bundle(bundle_path, expected_agent_role="initial_training")
    root = Path(app_root).resolve()
    store = Store(root / "data" / "facade_agent.sqlite3")
    model_storage = ModelStorageService(store, root)
    registry = build_phase1_registry(store, model_storage)
    created = registry.execute(
        "create_project",
        {
            "project_name": str(project_name).strip(),
            "classes": list(bundle.classes),
            "model_storage_root": str(Path(model_storage_root).resolve()),
        },
        actor_type="human",
        actor_id="sample_dataset_importer",
        confirmed=True,
    )
    project = created["project"]
    project_id = project["project_id"]
    store.transition_project(
        project_id=project_id,
        action="confirm_taxonomy",
        confirmed=True,
        actor_type="human",
        actor_id="sample_dataset_importer",
        payload={"source": bundle.path.name, "labels": "frozen_formal_core_train"},
    )
    result = SampleDatasetImportService(store, root).import_initial_bundle(
        project_id=project_id,
        bundle_path=bundle.path,
        actor_id="sample_dataset_importer",
        confirmed=True,
    )
    return {
        **result,
        "project_state": store.get_project(project_id)["state"],
    }
