from __future__ import annotations

import tempfile
import uuid
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ..storage import canonical_json, utc_now

if TYPE_CHECKING:
    from ..storage import Store


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.chmod(0o666)
        path.unlink()
        return
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            item.chmod(0o666)
    shutil.rmtree(path)


def _assert_no_symlink_path(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("Legacy model paths cannot contain symbolic links.")


def backfill_legacy_model_storage(
    db: sqlite3.Connection,
    app_root: Path,
) -> dict[str, int]:
    """Backfill authoritative roots and manifests without moving legacy weights."""

    root = Path(app_root).resolve()
    projects = db.execute(
        "SELECT project_id,model_storage_root,model_storage_locked_at FROM projects "
        "WHERE deleted_at IS NULL ORDER BY project_id"
    ).fetchall()
    plans: list[dict[str, Any]] = []
    projects_migrated = 0
    models_migrated = 0
    manifests_to_write: list[tuple[Path, bytes]] = []

    for project in projects:
        project_id = project["project_id"]
        configured_root = project["model_storage_root"]
        model_root = (
            Path(configured_root).resolve(strict=False)
            if configured_root
            else (root / "models" / project_id).resolve(strict=False)
        )
        _assert_no_symlink_path(model_root)
        models = db.execute(
            "SELECT * FROM model_versions WHERE project_id=? ORDER BY created_at,model_id",
            (project_id,),
        ).fetchall()
        taxonomy_rows = db.execute(
            "SELECT class_id,display_name,sort_order FROM classes "
            "WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
            (project_id,),
        ).fetchall()
        taxonomy_sha256 = hashlib.sha256(
            canonical_json([dict(row) for row in taxonomy_rows]).encode("utf-8")
        ).hexdigest()
        model_updates: list[dict[str, str]] = []

        for model in models:
            fully_migrated = bool(
                model["checkpoint_relpath"]
                and model["manifest_relpath"]
                and model["manifest_sha256"]
            )
            if fully_migrated:
                for field in ("checkpoint_relpath", "manifest_relpath"):
                    value = model[field]
                    if "\\" in value or ":" in value:
                        raise ValueError(
                            f"Stored model path is not portable: {model['model_id']}"
                        )
                    relative_path = PurePosixPath(value)
                    if (
                        relative_path.is_absolute()
                        or any(part in {"", ".", ".."} for part in value.split("/"))
                        or relative_path.parts[0] != model["model_id"]
                    ):
                        raise ValueError(
                            f"Stored model path is unsafe: {model['model_id']}"
                        )
                checkpoint = model_root.joinpath(
                    *PurePosixPath(model["checkpoint_relpath"]).parts
                )
                manifest_path = model_root.joinpath(
                    *PurePosixPath(model["manifest_relpath"]).parts
                )
                _assert_no_symlink_path(checkpoint)
                _assert_no_symlink_path(manifest_path)
                try:
                    checkpoint = checkpoint.resolve(strict=True)
                    manifest_path = manifest_path.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Registered model file is missing: {model['model_id']}"
                    ) from exc
                if (
                    not checkpoint.is_file()
                    or not manifest_path.is_file()
                    or _sha256_file(checkpoint) != model["checkpoint_sha256"]
                    or _sha256_file(manifest_path) != model["manifest_sha256"]
                ):
                    raise ValueError(
                        f"Registered model hash verification failed: {model['model_id']}"
                    )
                manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
                if any(
                    (
                        manifest_document.get("project_id") != project_id,
                        manifest_document.get("model_id") != model["model_id"],
                        manifest_document.get("parent_model_id") != model["parent_model_id"],
                        manifest_document.get("checkpoint_sha256")
                        != model["checkpoint_sha256"],
                        manifest_document.get("taxonomy_sha256") != taxonomy_sha256,
                        manifest_document.get("thresholds")
                        != json.loads(model["thresholds_json"]),
                        manifest_document.get("training_profile_id")
                        != model["training_profile_id"],
                    )
                ):
                    raise ValueError(
                        f"Registered manifest does not match the database: {model['model_id']}"
                    )
                model_updates.append(
                    {
                        "model_id": model["model_id"],
                        "checkpoint_relpath": model["checkpoint_relpath"],
                        "manifest_relpath": model["manifest_relpath"],
                        "manifest_sha256": model["manifest_sha256"],
                    }
                )
                continue

            checkpoint_unresolved = Path(model["checkpoint_path"])
            if not checkpoint_unresolved.is_absolute():
                raise ValueError(
                    f"Legacy checkpoint path is not absolute: {model['model_id']}"
                )
            _assert_no_symlink_path(checkpoint_unresolved)
            try:
                checkpoint = checkpoint_unresolved.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Legacy checkpoint is missing: {model['model_id']}"
                ) from exc
            if not checkpoint.is_file():
                raise ValueError(
                    f"Legacy checkpoint is not a regular file: {model['model_id']}"
                )
            try:
                relative = checkpoint.relative_to(model_root)
            except ValueError as exc:
                raise ValueError(
                    f"Legacy checkpoint is outside its project model root: {model['model_id']}"
                ) from exc
            if len(relative.parts) < 2 or relative.parts[0] != model["model_id"]:
                raise ValueError(
                    f"Legacy checkpoint is not stored under its model ID: {model['model_id']}"
                )
            checkpoint_sha256 = _sha256_file(checkpoint)
            if checkpoint_sha256 != model["checkpoint_sha256"]:
                raise ValueError(
                    f"Legacy checkpoint SHA-256 mismatch: {model['model_id']}"
                )

            manifest_path = checkpoint.parent / "manifest.json"
            manifest_document = {
                "schema_version": 1,
                "project_id": project_id,
                "model_id": model["model_id"],
                "parent_model_id": model["parent_model_id"],
                "source_batch_id": model["source_batch_id"],
                "role": model["role"],
                "taxonomy_sha256": taxonomy_sha256,
                "thresholds": json.loads(model["thresholds_json"]),
                "training_profile_id": model["training_profile_id"],
                "registered_at": model["created_at"],
                "checkpoint_filename": checkpoint.name,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "legacy_backfill": True,
            }
            manifest_bytes = (canonical_json(manifest_document) + "\n").encode("utf-8")
            expected_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_path.exists():
                _assert_no_symlink_path(manifest_path)
                if not manifest_path.is_file():
                    raise ValueError(
                        f"Legacy manifest is not a regular file: {model['model_id']}"
                    )
                actual_manifest_sha256 = _sha256_file(manifest_path)
                recorded_manifest_sha256 = model["manifest_sha256"]
                if (
                    actual_manifest_sha256 != expected_manifest_sha256
                    or (
                        recorded_manifest_sha256
                        and actual_manifest_sha256 != recorded_manifest_sha256
                    )
                ):
                    raise ValueError(
                        f"Legacy manifest does not match the model record: {model['model_id']}"
                    )
            else:
                manifests_to_write.append((manifest_path, manifest_bytes))
            model_updates.append(
                {
                    "model_id": model["model_id"],
                    "checkpoint_relpath": PurePosixPath(*relative.parts).as_posix(),
                    "manifest_relpath": PurePosixPath(
                        *manifest_path.relative_to(model_root).parts
                    ).as_posix(),
                    "manifest_sha256": expected_manifest_sha256,
                }
            )
            if not fully_migrated:
                models_migrated += 1

        project_needs_update = not configured_root or any(
            not (
                model["checkpoint_relpath"]
                and model["manifest_relpath"]
                and model["manifest_sha256"]
            )
            for model in models
        )
        if project_needs_update:
            projects_migrated += 1
        plans.append(
            {
                "project_id": project_id,
                "model_root": str(model_root),
                "locked_at": (
                    project["model_storage_locked_at"]
                    or (models[0]["created_at"] if models else None)
                ),
                "model_updates": model_updates,
                "needs_directory": not model_root.exists(),
            }
        )

    created_manifests: list[Path] = []
    created_roots: list[Path] = []
    temporary_manifests: list[Path] = []
    try:
        for plan in plans:
            model_root = Path(plan["model_root"])
            if plan["needs_directory"]:
                model_root.mkdir(parents=True, exist_ok=False)
                created_roots.append(model_root)
        for manifest_path, manifest_bytes in manifests_to_write:
            temporary = manifest_path.with_name(
                f".manifest-backfill-{uuid.uuid4().hex}.tmp"
            )
            temporary_manifests.append(temporary)
            with temporary.open("xb") as stream:
                stream.write(manifest_bytes)
            os.replace(temporary, manifest_path)
            temporary_manifests.remove(temporary)
            manifest_path.chmod(0o444)
            created_manifests.append(manifest_path)

        db.execute("BEGIN IMMEDIATE")
        for plan in plans:
            db.execute(
                "UPDATE projects SET model_storage_root=?,"
                "model_storage_locked_at=COALESCE(model_storage_locked_at,?) "
                "WHERE project_id=?",
                (plan["model_root"], plan["locked_at"], plan["project_id"]),
            )
            for model in plan["model_updates"]:
                db.execute(
                    "UPDATE model_versions SET checkpoint_relpath=?,manifest_relpath=?,"
                    "manifest_sha256=? WHERE model_id=?",
                    (
                        model["checkpoint_relpath"],
                        model["manifest_relpath"],
                        model["manifest_sha256"],
                        model["model_id"],
                    ),
                )
        db.commit()
    except Exception:
        db.rollback()
        for temporary in temporary_manifests:
            temporary.unlink(missing_ok=True)
        for manifest_path in created_manifests:
            manifest_path.chmod(0o666)
            manifest_path.unlink(missing_ok=True)
        for model_root in reversed(created_roots):
            try:
                model_root.rmdir()
            except OSError:
                pass
        raise

    return {
        "projects_migrated": projects_migrated,
        "models_migrated": models_migrated,
    }


class PreparedModel:
    """A staged model that is removed unless its database transaction commits."""

    def __init__(
        self,
        service: "ModelStorageService",
        *,
        project_id: str,
        model_id: str,
        source_checkpoint: Path,
        expected_sha256: str,
        manifest: dict[str, Any],
    ) -> None:
        self.service = service
        self.project_id = project_id
        self.model_id = model_id
        self.source_checkpoint = Path(source_checkpoint)
        self.expected_sha256 = expected_sha256.lower()
        self.manifest = json.loads(canonical_json(manifest))
        self._staging_dir: Path | None = None
        self._target_dir: Path | None = None
        self._published: dict[str, Any] | None = None
        self._committed = False

    def __enter__(self) -> "PreparedModel":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._staging_dir is not None:
            _remove_tree(self._staging_dir)
        if self._target_dir is not None and not self._committed:
            _remove_tree(self._target_dir)

    def publish(self) -> dict[str, Any]:
        if self._published is not None:
            return dict(self._published)
        if (
            len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("Expected checkpoint SHA-256 is invalid.")
        if not self.source_checkpoint.is_file() or self.source_checkpoint.is_symlink():
            raise ValueError("Checkpoint source must be a regular file.")
        if _sha256_file(self.source_checkpoint) != self.expected_sha256:
            raise ValueError("Checkpoint source SHA-256 does not match the verified result.")

        root = self.service._project_root(self.project_id)
        if not root.is_dir():
            raise ValueError("Project model storage folder is missing.")
        if not self.model_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in self.model_id
        ) or self.model_id in {".", ".."}:
            raise ValueError("Model identifier is not safe for storage.")
        for identity_key, expected in (
            ("project_id", self.project_id),
            ("model_id", self.model_id),
        ):
            if self.manifest.get(identity_key, expected) != expected:
                raise ValueError(f"Manifest {identity_key} does not match registration.")

        target = root / self.model_id
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Model storage target already exists: {self.model_id}")
        staging = root / f".staging-{self.model_id}-{uuid.uuid4().hex}"
        staging.mkdir()
        self._staging_dir = staging
        self._target_dir = target
        suffix = self.source_checkpoint.suffix.lower()
        if not suffix or len(suffix) > 16 or any(
            character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in suffix
        ):
            suffix = ".bin"
        checkpoint_name = f"checkpoint{suffix}"
        checkpoint = staging / checkpoint_name
        temporary_checkpoint = staging / f".{checkpoint_name}.tmp"
        manifest_path = staging / "manifest.json"
        temporary_manifest = staging / ".manifest.json.tmp"
        try:
            shutil.copyfile(self.source_checkpoint, temporary_checkpoint)
            if _sha256_file(temporary_checkpoint) != self.expected_sha256:
                raise ValueError("Checkpoint changed while entering project storage.")
            os.replace(temporary_checkpoint, checkpoint)
            size_bytes = checkpoint.stat().st_size
            document = {
                **self.manifest,
                "project_id": self.project_id,
                "model_id": self.model_id,
                "checkpoint_filename": checkpoint_name,
                "checkpoint_sha256": self.expected_sha256,
                "checkpoint_size_bytes": size_bytes,
            }
            temporary_manifest.write_text(
                canonical_json(document) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temporary_manifest, manifest_path)
            manifest_sha256 = _sha256_file(manifest_path)
            checkpoint.chmod(0o444)
            manifest_path.chmod(0o444)
            os.replace(staging, target)
            self._staging_dir = None
        except Exception:
            _remove_tree(staging)
            raise

        self._published = {
            "checkpoint_relpath": PurePosixPath(self.model_id, checkpoint_name).as_posix(),
            "manifest_relpath": PurePosixPath(self.model_id, "manifest.json").as_posix(),
            "checkpoint_sha256": self.expected_sha256,
            "manifest_sha256": manifest_sha256,
            "checkpoint_size_bytes": size_bytes,
        }
        return dict(self._published)

    def commit(self) -> None:
        if self._published is None:
            raise RuntimeError("Model must be published before it can be committed.")
        self._committed = True


class ModelStorageService:
    """Owns validation and resolution for project-scoped model roots."""

    def __init__(self, store: "Store", app_root: Path) -> None:
        self.store = store
        self.app_root = Path(app_root).resolve()

    @staticmethod
    def _is_within(candidate: Path, parent: Path) -> bool:
        try:
            candidate.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def _assert_no_symlinks(path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ValueError("Model storage paths cannot contain symbolic links.")

    @staticmethod
    def _probe_writable(directory: Path) -> None:
        probe = directory / f".model-storage-probe-{uuid.uuid4().hex}"
        try:
            with probe.open("xb") as stream:
                stream.write(b"ok")
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

    def _reserved_roots(self) -> tuple[Path, ...]:
        return tuple(
            (self.app_root / name).resolve()
            for name in (
                "data",
                "audit",
                "artifact_store",
                "exports",
                "inbox",
                "projects",
                "trash",
            )
        )

    def validate_root(
        self,
        raw_path: str,
        *,
        exclude_project_id: str | None = None,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("An absolute model storage folder is required.")
        untrusted = Path(raw_path.strip())
        if not untrusted.is_absolute():
            raise ValueError("Model storage folder must be an absolute path.")

        self._assert_no_symlinks(untrusted)
        candidate = untrusted.resolve(strict=False)
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if self._is_within(candidate, temporary_root):
            raise ValueError("Model storage folder cannot be inside the temporary directory.")
        for reserved in self._reserved_roots():
            if self._is_within(candidate, reserved):
                raise ValueError("Model storage folder overlaps an Agent-managed directory.")

        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError("Model storage path must be a directory.")
            if any(candidate.iterdir()):
                raise ValueError("Model storage folder must be empty for a new project.")
            probe_parent = candidate
        else:
            probe_parent = candidate.parent
            while not probe_parent.exists() and probe_parent != probe_parent.parent:
                probe_parent = probe_parent.parent
            if not probe_parent.is_dir():
                raise ValueError("Model storage parent is not a directory.")

        with self.store.connect() as db:
            rows = db.execute(
                "SELECT project_id,model_storage_root FROM projects "
                "WHERE deleted_at IS NULL AND model_storage_root IS NOT NULL"
            ).fetchall()
        for row in rows:
            if exclude_project_id and row["project_id"] == exclude_project_id:
                continue
            existing = Path(row["model_storage_root"]).resolve(strict=False)
            if (
                candidate == existing
                or self._is_within(candidate, existing)
                or self._is_within(existing, candidate)
            ):
                raise ValueError(
                    "Model storage folder conflicts with another project folder."
                )

        try:
            self._probe_writable(probe_parent)
        except (OSError, PermissionError) as exc:
            raise ValueError("Model storage folder is not writable.") from exc
        return candidate

    def resolve_model_file(self, project_id: str, relpath: str) -> Path:
        if not isinstance(relpath, str) or not relpath.strip():
            raise ValueError("A model-relative path is required.")
        if "\\" in relpath or ":" in relpath:
            raise ValueError("Model paths must use portable POSIX relative syntax.")
        raw_parts = relpath.split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise ValueError("Model path contains an unsafe segment.")
        relative = PurePosixPath(relpath)
        if relative.is_absolute():
            raise ValueError("Absolute model paths are not allowed.")

        with self.store.connect() as db:
            project = db.execute(
                "SELECT model_storage_root FROM projects "
                "WHERE project_id=? AND deleted_at IS NULL",
                (project_id,),
            ).fetchone()
        if project is None:
            raise KeyError(f"Project not found: {project_id}")
        if not project["model_storage_root"]:
            raise ValueError("Project model storage has not been configured.")

        root = Path(project["model_storage_root"]).resolve(strict=False)
        self._assert_no_symlinks(root)
        target = root.joinpath(*relative.parts)
        self._assert_no_symlinks(target)
        resolved = target.resolve(strict=False)
        if not self._is_within(resolved, root):
            raise ValueError("Model path escapes the project storage folder.")
        return resolved

    def _project_root(self, project_id: str) -> Path:
        with self.store.connect() as db:
            project = db.execute(
                "SELECT model_storage_root FROM projects "
                "WHERE project_id=? AND deleted_at IS NULL",
                (project_id,),
            ).fetchone()
        if project is None:
            raise KeyError(f"Project not found: {project_id}")
        if not project["model_storage_root"]:
            raise ValueError("Project model storage has not been configured.")
        root = Path(project["model_storage_root"]).resolve(strict=False)
        self._assert_no_symlinks(root)
        return root

    def taxonomy_sha256(self, project_id: str) -> str:
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT class_id,display_name,sort_order FROM classes "
                "WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
                (project_id,),
            ).fetchall()
        if not rows:
            raise ValueError("Project has no active taxonomy.")
        return hashlib.sha256(
            canonical_json([dict(row) for row in rows]).encode("utf-8")
        ).hexdigest()

    def prepare_registration(
        self,
        *,
        project_id: str,
        model_id: str,
        source_checkpoint: Path,
        expected_sha256: str,
        manifest: dict[str, Any],
    ) -> PreparedModel:
        return PreparedModel(
            self,
            project_id=project_id,
            model_id=model_id,
            source_checkpoint=source_checkpoint,
            expected_sha256=expected_sha256,
            manifest=manifest,
        )

    def reconcile_orphans(self) -> dict[str, int]:
        quarantine_root = self.app_root / "artifact_store" / "model_orphans"
        quarantined = 0
        with self.store.connect() as db:
            projects = db.execute(
                "SELECT project_id,model_storage_root FROM projects "
                "WHERE deleted_at IS NULL AND model_storage_root IS NOT NULL"
            ).fetchall()
            models = db.execute(
                "SELECT project_id,checkpoint_relpath FROM model_versions "
                "WHERE checkpoint_relpath IS NOT NULL"
            ).fetchall()
        referenced: dict[str, set[str]] = {}
        for model in models:
            first = PurePosixPath(model["checkpoint_relpath"]).parts[0]
            referenced.setdefault(model["project_id"], set()).add(first)
        for project in projects:
            root = Path(project["model_storage_root"]).resolve(strict=False)
            self._assert_no_symlinks(root)
            if not root.exists():
                continue
            for entry in list(root.iterdir()):
                if entry.name in referenced.get(project["project_id"], set()):
                    continue
                destination_dir = quarantine_root / project["project_id"]
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{entry.name}-{uuid.uuid4().hex}"
                shutil.move(str(entry), str(destination))
                quarantined += 1
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    self.store._append_audit(
                        db,
                        project_id=project["project_id"],
                        batch_id=None,
                        actor_type="system",
                        actor_id="model_storage_recovery",
                        tool_name="quarantine_orphan_model",
                        from_state=None,
                        to_state=None,
                        payload={"entry_name": entry.name, "quarantined_at": utc_now()},
                    )
        return {"quarantined": quarantined}
