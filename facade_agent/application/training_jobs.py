from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from ..core.permissions import ToolPolicy
from ..core.states import ProjectState
from ..storage import canonical_json, make_id, utc_now
from ..tools.registry import ToolDefinition, ToolRegistry


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


class InitialChampionJobService:
    """Build immutable, auditable jobs for the governed local GPU worker."""

    def __init__(self, store: Any, project_root: Path, export_root: Path, profile_path: Path) -> None:
        self.store = store
        self.project_root = project_root.resolve()
        self.export_root = export_root.resolve()
        self.profile_path = profile_path.resolve()

    def _profile(self) -> dict[str, Any]:
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if profile.get("profile_id") != "initial_champion_convnext_tiny_768_v2":
            raise ValueError("Unexpected Initial Champion training profile.")
        return profile

    def _row_to_job(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        return item

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Training job not found: {job_id}")
        return self._row_to_job(row)

    def list_jobs(self, project_id: str) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM training_jobs WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def verify_bundle(self, job_id: str) -> dict[str, Any]:
        """Verify the immutable outer bundle and every member before execution."""
        job = self.get_job(job_id)
        bundle = Path(job["bundle_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Training bundle is outside the managed export directory.")
        if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
            raise ValueError("Training bundle checksum verification failed.")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or len(names) > 100_000:
                raise ValueError("Training bundle has duplicate or excessive entries.")
            for name in names:
                pure = Path(name)
                if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                    raise ValueError("Training bundle contains an unsafe path.")
            required = {"job_manifest.json", "classes.json", "labels.jsonl", "training_profile.json", "checksums.json"}
            if not required.issubset(names):
                raise ValueError("Training bundle is missing a required payload.")
            checksums = json.loads(archive.read("checksums.json"))
            payload_names = set(names) - {"checksums.json"}
            if set(checksums) != payload_names:
                raise ValueError("Training bundle checksum inventory is incomplete.")
            for name, expected in checksums.items():
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                if actual != expected:
                    raise ValueError(f"Training bundle member checksum failed: {name}")
            manifest = json.loads(archive.read("job_manifest.json"))
            profile = json.loads(archive.read("training_profile.json"))
            if manifest.get("job_id") != job_id or manifest.get("content_fingerprint") != job["content_fingerprint"]:
                raise ValueError("Training manifest does not match the database job.")
            if profile != job["profile"] or profile != self._profile():
                raise ValueError("Training preset differs from the locked official profile.")
            seeds = profile.get("seeds")
            if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3 or any(not isinstance(seed, int) for seed in seeds):
                raise ValueError("Initial Champion preset must contain exactly three distinct integer seeds.")
        return {"job": job, "manifest": manifest, "profile": profile, "bundle_path": str(bundle)}

    def _load_snapshot(self, project_id: str, dataset_id: str) -> dict[str, Any]:
        project = self.store.get_project(project_id)
        dataset = self.store.get_dataset(dataset_id)
        if dataset["project_id"] != project_id:
            raise PermissionError("Dataset does not belong to the selected project.")
        if dataset["role"] != "initial_training":
            raise ValueError("Initial Champion jobs require an initial_training dataset.")
        if dataset["status"] not in {"validated", "frozen"}:
            raise PermissionError("Validate and freeze the dataset before preparing training.")
        if project["state"] not in {
            ProjectState.DATA_VALIDATED,
            ProjectState.ANNOTATION_READY,
            ProjectState.INITIAL_TRAINING_READY,
        }:
            raise PermissionError("Project is not ready to prepare Initial Champion training.")
        with self.store.connect() as db:
            version = db.execute(
                "SELECT * FROM label_versions WHERE dataset_id=? "
                "ORDER BY version_number DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
            if version is None:
                raise ValueError("No frozen label version exists for this dataset.")
            classes = [
                dict(row)
                for row in db.execute(
                    "SELECT class_id,display_name,display_name_zh,display_name_en,description,sort_order "
                    "FROM classes WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
                    (project_id,),
                )
            ]
            images = [
                dict(row)
                for row in db.execute(
                    "SELECT i.image_id,i.filename,i.stored_path,i.sha256,i.size_bytes,i.mime_type,"
                    "i.width,i.height,a.no_defect FROM images i JOIN image_annotations a "
                    "ON a.image_id=i.image_id WHERE i.dataset_id=? AND a.status='complete' "
                    "ORDER BY i.image_id",
                    (dataset_id,),
                )
            ]
            for image in images:
                image["class_ids"] = [
                    row["class_id"]
                    for row in db.execute(
                        "SELECT class_id FROM image_annotation_labels WHERE image_id=? ORDER BY class_id",
                        (image["image_id"],),
                    )
                ]
        if len(images) != int(version["image_count"]):
            raise ValueError("Frozen label image count does not match the current dataset.")
        return {
            "project": project,
            "dataset": dataset,
            "label_version": dict(version),
            "classes": classes,
            "images": images,
        }

    def _managed_image(self, raw_path: str, expected_sha256: str) -> Path:
        path = Path(raw_path).resolve()
        if path != self.project_root and self.project_root not in path.parents:
            raise PermissionError("An image is outside the managed project directory.")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Image checksum changed after validation: {path.name}")
        return path

    @staticmethod
    def _zip_write(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
        info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100444 << 16
        archive.writestr(info, content)

    def create_job(
        self,
        *,
        project_id: str,
        dataset_id: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        snapshot = self._load_snapshot(project_id, dataset_id)
        profile = self._profile()
        profile_bytes = json_bytes(profile)
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        image_fingerprint = [
            {"image_id": row["image_id"], "sha256": row["sha256"]}
            for row in snapshot["images"]
        ]
        fingerprint_payload = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "label_version_id": snapshot["label_version"]["label_version_id"],
            "labels_sha256": snapshot["label_version"]["labels_sha256"],
            "profile_sha256": profile_sha256,
            "classes": [row["class_id"] for row in snapshot["classes"]],
            "images": image_fingerprint,
        }
        content_fingerprint = hashlib.sha256(
            canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        with self.store.connect() as db:
            existing = db.execute(
                "SELECT * FROM training_jobs WHERE content_fingerprint=?",
                (content_fingerprint,),
            ).fetchone()
        if existing is not None:
            job = self._row_to_job(existing)
            bundle = Path(job["bundle_path"])
            if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
                raise ValueError("Existing immutable training bundle is missing or changed.")
            return {"job": job, "idempotent": True}

        job_id = f"train_{content_fingerprint[:16]}"
        final_dir = self.export_root / job_id
        final_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = final_dir / f"{job_id}.zip"
        temporary_path = final_dir / f".{job_id}.tmp"
        manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "kind": "initial_champion",
            "project_id": project_id,
            "project_name": snapshot["project"]["name"],
            "dataset_id": dataset_id,
            "dataset_name": snapshot["dataset"]["name"],
            "label_version_id": snapshot["label_version"]["label_version_id"],
            "labels_sha256": snapshot["label_version"]["labels_sha256"],
            "training_profile_id": profile["profile_id"],
            "profile_sha256": profile_sha256,
            "class_count": len(snapshot["classes"]),
            "image_count": len(snapshot["images"]),
            "content_fingerprint": content_fingerprint,
            "contract": "The local GPU worker must verify all checksums before training and register a separately checksummed result.",
        }
        labels_lines: list[bytes] = []
        payloads: dict[str, bytes] = {
            "job_manifest.json": json_bytes(manifest),
            "classes.json": json_bytes(snapshot["classes"]),
            "training_profile.json": profile_bytes,
        }
        image_sources: list[tuple[str, Path]] = []
        for image in snapshot["images"]:
            source = self._managed_image(image["stored_path"], image["sha256"])
            extension = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".bin"
            archive_name = f"images/{image['image_id']}{extension}"
            label = {
                "image_id": image["image_id"],
                "original_filename": image["filename"],
                "file": archive_name,
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "class_ids": image["class_ids"],
                "no_defect": bool(image["no_defect"]),
            }
            labels_lines.append(canonical_json(label).encode("utf-8") + b"\n")
            image_sources.append((archive_name, source))
        payloads["labels.jsonl"] = b"".join(labels_lines)
        checksums = {
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())
        }
        checksums.update({name: sha256_file(source) for name, source in image_sources})
        payloads["checksums.json"] = json_bytes(checksums)
        try:
            with zipfile.ZipFile(temporary_path, "w") as archive:
                for name in sorted(payloads):
                    self._zip_write(archive, name, payloads[name])
                for name, source in sorted(image_sources):
                    self._zip_write(archive, name, source.read_bytes())
            os.replace(temporary_path, bundle_path)
            bundle_sha256 = sha256_file(bundle_path)
            bundle_size = bundle_path.stat().st_size
            manifest_sha256 = hashlib.sha256(payloads["job_manifest.json"]).hexdigest()
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute(
                    "SELECT state FROM projects WHERE project_id=?", (project_id,)
                ).fetchone()
                if current is None:
                    raise KeyError(f"Project not found: {project_id}")
                if current["state"] == ProjectState.DATA_VALIDATED:
                    db.execute(
                        "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                        (ProjectState.ANNOTATION_READY, now, project_id),
                    )
                    self.store._append_audit(
                        db,
                        project_id=project_id,
                        batch_id=None,
                        actor_type=actor_type,
                        actor_id=actor_id,
                        tool_name="record_annotation_ready",
                        from_state=ProjectState.DATA_VALIDATED,
                        to_state=ProjectState.ANNOTATION_READY,
                        payload={
                            "dataset_id": dataset_id,
                            "label_version_id": snapshot["label_version"]["label_version_id"],
                            "labels_sha256": snapshot["label_version"]["labels_sha256"],
                        },
                    )
                    current_state = ProjectState.ANNOTATION_READY
                else:
                    current_state = current["state"]
                if current_state != ProjectState.ANNOTATION_READY:
                    raise PermissionError("Project state changed while the training bundle was prepared.")
                db.execute(
                    "INSERT INTO training_jobs(job_id,project_id,dataset_id,label_version_id,kind,status,"
                    "training_profile_id,profile_json,manifest_sha256,content_fingerprint,bundle_path,"
                    "bundle_sha256,bundle_size_bytes,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        project_id,
                        dataset_id,
                        snapshot["label_version"]["label_version_id"],
                        "initial_champion",
                        "exported",
                        profile["profile_id"],
                        canonical_json(profile),
                        manifest_sha256,
                        content_fingerprint,
                        str(bundle_path),
                        bundle_sha256,
                        bundle_size,
                        now,
                        now,
                    ),
                )
                db.execute(
                    "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (make_id("artifact"), project_id, None, "initial_champion_training_bundle", str(bundle_path), bundle_sha256, bundle_size, now),
                )
                db.execute(
                    "UPDATE datasets SET status='frozen',updated_at=? WHERE dataset_id=?",
                    (now, dataset_id),
                )
                db.execute(
                    "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                    (ProjectState.INITIAL_TRAINING_READY, now, project_id),
                )
                audit_sha256 = self.store._append_audit(
                    db,
                    project_id=project_id,
                    batch_id=None,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name="create_initial_training_job",
                    from_state=ProjectState.ANNOTATION_READY,
                    to_state=ProjectState.INITIAL_TRAINING_READY,
                    payload={
                        "job_id": job_id,
                        "dataset_id": dataset_id,
                        "training_profile_id": profile["profile_id"],
                        "bundle_sha256": bundle_sha256,
                        "bundle_size_bytes": bundle_size,
                    },
                )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            bundle_path.unlink(missing_ok=True)
            raise
        return {"job": self.get_job(job_id), "audit_event_sha256": audit_sha256, "idempotent": False}


def register_initial_training_tool(registry: ToolRegistry, service: InitialChampionJobService) -> None:
    registry.register(
        ToolDefinition(
            "create_initial_training_job",
            "After deterministic validation, export the frozen image-level dataset as an immutable Initial Champion GPU training job. The versioned backend preset cannot be changed by the language model.",
            {
                "type": "object",
                "required": ["project_id", "dataset_id"],
                "properties": {
                    "project_id": {"type": "string", "minLength": 1},
                    "dataset_id": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True),
            lambda args, ctx: service.create_job(
                project_id=args["project_id"],
                dataset_id=args["dataset_id"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )

