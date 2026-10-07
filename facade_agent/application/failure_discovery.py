from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from ..core.permissions import ToolPolicy
from ..core.states import BATCH_TRANSITIONS, BatchState, ProjectState, require_transition
from ..storage import canonical_json, make_id, utc_now
from ..tools.registry import ToolDefinition, ToolRegistry
from .gpu_scheduler import GpuJobScheduler
from .local_training import (
    SubprocessWorkerRunner,
    TrainingCancelled,
    WorkerLaunchSpec,
    WorkerRunner,
)
from .training_jobs import FIXED_ZIP_TIME, json_bytes, sha256_file
from .worker_errors import classify_worker_error


DISCOVERY_WORKER_PROTOCOL_VERSION = 1
DISCOVERY_WORKER_MODULE = "facade_training_worker.champion_failure_discovery"
DISCOVERY_PROFILE_ID = "champion_failure_discovery_convnext_consensus_v1"
ACTIVE_RUN_STATUSES = {"queued", "running", "cancel_requested"}
TERMINAL_RUN_STATUSES = {"cancelled", "result_verified", "failed", "interrupted"}


class FailureDiscoveryService:
    """Govern one frozen-batch Champion screening and one failure-discovery pass."""

    def __init__(
        self,
        store: Any,
        environment_manager: Any,
        project_root: Path,
        root: Path,
        profile_path: Path,
        runner: WorkerRunner | None = None,
        scheduler: GpuJobScheduler | None = None,
    ) -> None:
        self.store = store
        self.environment = environment_manager
        self.project_root = project_root.resolve()
        self.root = root.resolve()
        self.profile_path = profile_path.resolve()
        self.export_root = (self.root / "exports" / "screening_jobs").resolve()
        self.run_root = (self.root / "artifact_store" / "screening_runs").resolve()
        self.result_root = (self.root / "artifact_store" / "screening_results").resolve()
        self.runner = runner or SubprocessWorkerRunner()
        self._requires_installed_worker = runner is None
        self.scheduler = scheduler or GpuJobScheduler(store)
        self._owns_scheduler = scheduler is None
        self.scheduler.register_handler("champion_failure_discovery", self)
        if self._owns_scheduler:
            self.scheduler.start()

    def _profile(self) -> dict[str, Any]:
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if profile.get("profile_id") != DISCOVERY_PROFILE_ID:
            raise ValueError("Unexpected Failure Discovery profile.")
        cluster = profile.get("clustering", {})
        if cluster.get("runs") != 3 or cluster.get("kmeans_n_init_per_run") != 20:
            raise ValueError("Failure Discovery must use the locked three-run consensus protocol.")
        if profile.get("threshold_retuning") is not False:
            raise ValueError("Maintenance-data threshold retuning is prohibited.")
        if profile.get("discovery_passes_per_frozen_batch") != 1:
            raise ValueError("Exactly one discovery pass is permitted per frozen batch.")
        return profile

    @staticmethod
    def _row_to_job(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        return item

    @staticmethod
    def _row_to_run(row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("gpu_devices_json", "preflight_json", "command_json", "progress_json", "summary_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM screening_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Screening job not found: {job_id}")
        return self._row_to_job(row)

    def list_jobs(self, project_id: str) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM screening_jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def _get_run(self, run_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM screening_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Screening run not found: {run_id}")
        return self._row_to_run(row)

    def get_run(self, run_id: str, event_after: int = 0) -> dict[str, Any]:
        run = self._get_run(run_id)
        log_path = Path(run["log_path"]).resolve()
        if self.run_root != log_path and self.run_root not in log_path.parents:
            raise PermissionError("Screening log is outside the managed run directory.")
        with self.store.connect() as db:
            events = []
            for row in db.execute(
                "SELECT event_id,kind,payload_json,created_at FROM screening_run_events "
                "WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT 500",
                (run_id, max(0, int(event_after))),
            ):
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                events.append(item)
        return {"run": run, "events": events}

    def list_runs(self, project_id: str, batch_id: str | None = None) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        query = "SELECT * FROM screening_runs WHERE project_id=?"
        parameters: list[Any] = [project_id]
        if batch_id:
            query += " AND batch_id=?"
            parameters.append(batch_id)
        query += " ORDER BY created_at DESC"
        with self.store.connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return [self._row_to_run(row) for row in rows]

    def list_slices(self, batch_id: str) -> list[dict[str, Any]]:
        self.store.get_maintenance_batch(batch_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT s.*,c.display_name AS class_name FROM failure_slices s "
                "JOIN classes c ON c.class_id=s.class_id WHERE s.batch_id=? "
                "ORDER BY c.sort_order,s.error_type,s.slice_id",
                (batch_id,),
            ).fetchall()
            output = []
            for row in rows:
                item = dict(row)
                item["members"] = [
                    dict(member)
                    for member in db.execute(
                        "SELECT m.image_id,m.membership_rank,i.filename FROM failure_slice_members m "
                        "JOIN images i ON i.image_id=m.image_id WHERE m.slice_id=? "
                        "ORDER BY m.membership_rank,m.image_id",
                        (row["slice_id"],),
                    )
                ]
                output.append(item)
        return output

    def _snapshot(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.get_maintenance_batch(batch_id)
        if batch["state"] != BatchState.MAINTENANCE_BATCH_FROZEN:
            raise PermissionError("Failure Discovery requires a frozen maintenance batch.")
        dataset = batch.get("dataset")
        if not dataset or dataset["role"] != "maintenance" or dataset["status"] != "frozen":
            raise PermissionError("The maintenance dataset is not frozen.")
        project = self.store.get_project(batch["project_id"])
        if project["state"] != ProjectState.SCREENING_READY or project.get("active_batch_id") != batch_id:
            raise PermissionError("The selected batch is not the active screening batch.")
        with self.store.connect() as db:
            model_row = db.execute(
                "SELECT m.* FROM active_models a JOIN model_versions m ON m.model_id=a.model_id "
                "WHERE a.project_id=?", (batch["project_id"],)
            ).fetchone()
            if model_row is None:
                raise PermissionError("An active Champion is required for screening.")
            version = db.execute(
                "SELECT * FROM label_versions WHERE dataset_id=? ORDER BY version_number DESC LIMIT 1",
                (dataset["dataset_id"],),
            ).fetchone()
            if version is None:
                raise ValueError("The frozen maintenance dataset has no label version.")
            classes = [
                dict(row) for row in db.execute(
                    "SELECT class_id,display_name,display_name_zh,display_name_en,description,sort_order "
                    "FROM classes WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
                    (batch["project_id"],),
                )
            ]
            images = []
            for row in db.execute(
                "SELECT i.image_id,i.filename,i.stored_path,i.sha256,i.size_bytes,i.mime_type,i.width,i.height,a.no_defect "
                "FROM images i JOIN image_annotations a ON a.image_id=i.image_id "
                "WHERE i.dataset_id=? AND a.status='complete' ORDER BY i.image_id",
                (dataset["dataset_id"],),
            ):
                image = dict(row)
                image["class_ids"] = [
                    item["class_id"] for item in db.execute(
                        "SELECT class_id FROM image_annotation_labels WHERE image_id=? ORDER BY class_id",
                        (row["image_id"],),
                    )
                ]
                images.append(image)
        model = dict(model_row)
        thresholds = json.loads(model.pop("thresholds_json"))
        if not isinstance(thresholds, dict) or set(thresholds) != {item["class_id"] for item in classes}:
            raise ValueError("Active Champion thresholds do not match the project taxonomy.")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1 for value in thresholds.values()):
            raise ValueError("Active Champion thresholds are invalid.")
        if len(images) != int(version["image_count"]):
            raise ValueError("Frozen label image count does not match the maintenance dataset.")
        return {
            "batch": batch, "dataset": dataset, "project": project, "model": model,
            "thresholds": {key: float(value) for key, value in thresholds.items()},
            "label_version": dict(version), "classes": classes, "images": images,
        }

    def _managed_file(self, raw_path: str, expected_sha256: str, root: Path) -> Path:
        path = Path(raw_path).resolve()
        if root != path and root not in path.parents:
            raise PermissionError("A frozen input is outside its managed directory.")
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"Frozen input checksum changed: {path.name}")
        return path

    @staticmethod
    def _zip_write_bytes(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
        info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100444 << 16
        archive.writestr(info, content)

    @staticmethod
    def _zip_write_file(archive: zipfile.ZipFile, name: str, source: Path) -> None:
        info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100444 << 16
        with archive.open(info, "w") as target, source.open("rb") as handle:
            shutil.copyfileobj(handle, target, length=1024 * 1024)

    def create_job(self, batch_id: str, actor_type: str = "human", actor_id: str = "local_engineer") -> dict[str, Any]:
        snapshot = self._snapshot(batch_id)
        profile = self._profile()
        profile_bytes = json_bytes(profile)
        checkpoint_root = Path(
            snapshot["project"].get("model_storage_root") or self.root / "models"
        ).resolve()
        checkpoint = self._managed_file(
            snapshot["model"]["checkpoint_path"], snapshot["model"]["checkpoint_sha256"],
            checkpoint_root,
        )
        fingerprint_payload = {
            "batch_id": batch_id,
            "dataset_id": snapshot["dataset"]["dataset_id"],
            "label_version_id": snapshot["label_version"]["label_version_id"],
            "labels_sha256": snapshot["label_version"]["labels_sha256"],
            "champion_model_id": snapshot["model"]["model_id"],
            "champion_sha256": snapshot["model"]["checkpoint_sha256"],
            "thresholds": snapshot["thresholds"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "images": [{"image_id": item["image_id"], "sha256": item["sha256"]} for item in snapshot["images"]],
        }
        content_fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode()).hexdigest()
        with self.store.connect() as db:
            existing = db.execute("SELECT * FROM screening_jobs WHERE batch_id=?", (batch_id,)).fetchone()
        if existing:
            job = self._row_to_job(existing)
            self.verify_job(job["job_id"])
            return {"job": job, "idempotent": True}
        job_id = f"screen_{content_fingerprint[:16]}"
        final_dir = self.export_root / job_id
        final_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = final_dir / f"{job_id}.zip"
        temporary = final_dir / f".{job_id}.tmp"
        champion_name = "champion/checkpoint.pt"
        manifest = {
            "schema_version": 1, "job_id": job_id, "kind": "champion_failure_discovery",
            "project_id": snapshot["project"]["project_id"], "batch_id": batch_id,
            "dataset_id": snapshot["dataset"]["dataset_id"],
            "label_version_id": snapshot["label_version"]["label_version_id"],
            "labels_sha256": snapshot["label_version"]["labels_sha256"],
            "champion_model_id": snapshot["model"]["model_id"],
            "champion_checkpoint_sha256": snapshot["model"]["checkpoint_sha256"],
            "discovery_profile_id": profile["profile_id"], "class_count": len(snapshot["classes"]),
            "image_count": len(snapshot["images"]), "content_fingerprint": content_fingerprint,
            "contract": "Use frozen Champion thresholds; perform one locked discovery pass; never retrain or deploy.",
        }
        champion_manifest = {
            "model_id": snapshot["model"]["model_id"], "checkpoint_file": champion_name,
            "checkpoint_sha256": snapshot["model"]["checkpoint_sha256"],
            "thresholds": snapshot["thresholds"], "training_profile_id": snapshot["model"]["training_profile_id"],
        }
        payloads = {
            "job_manifest.json": json_bytes(manifest), "classes.json": json_bytes(snapshot["classes"]),
            "champion.json": json_bytes(champion_manifest), "discovery_profile.json": profile_bytes,
        }
        labels = []
        image_sources: list[tuple[str, Path]] = []
        for image in snapshot["images"]:
            source = self._managed_file(image["stored_path"], image["sha256"], self.project_root)
            extension = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".bin"
            archive_name = f"images/{image['image_id']}{extension}"
            labels.append(canonical_json({
                "image_id": image["image_id"], "original_filename": image["filename"],
                "file": archive_name, "sha256": image["sha256"], "width": image["width"],
                "height": image["height"], "class_ids": image["class_ids"], "no_defect": bool(image["no_defect"]),
            }).encode() + b"\n")
            image_sources.append((archive_name, source))
        payloads["labels.jsonl"] = b"".join(labels)
        file_sources = [(champion_name, checkpoint), *image_sources]
        checksums = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
        checksums.update({name: sha256_file(source) for name, source in file_sources})
        payloads["checksums.json"] = json_bytes(checksums)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for name, content in sorted(payloads.items()):
                    self._zip_write_bytes(archive, name, content)
                for name, source in sorted(file_sources):
                    self._zip_write_file(archive, name, source)
            os.replace(temporary, bundle_path)
            bundle_sha = sha256_file(bundle_path)
            bundle_size = bundle_path.stat().st_size
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (batch_id,)).fetchone()
                if current is None or current["state"] != BatchState.MAINTENANCE_BATCH_FROZEN:
                    raise PermissionError("Maintenance batch changed while the immutable job was prepared.")
                db.execute(
                    "INSERT INTO screening_jobs(job_id,project_id,batch_id,dataset_id,label_version_id,champion_model_id,status,"
                    "discovery_profile_id,profile_json,content_fingerprint,bundle_path,bundle_sha256,bundle_size_bytes,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, snapshot["project"]["project_id"], batch_id, snapshot["dataset"]["dataset_id"],
                     snapshot["label_version"]["label_version_id"], snapshot["model"]["model_id"], "exported",
                     profile["profile_id"], canonical_json(profile), content_fingerprint, str(bundle_path), bundle_sha,
                     bundle_size, now, now),
                )
                db.execute(
                    "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (make_id("artifact"), snapshot["project"]["project_id"], batch_id,
                     "champion_failure_discovery_job", str(bundle_path), bundle_sha, bundle_size, now),
                )
                audit = self.store._append_audit(
                    db, project_id=snapshot["project"]["project_id"], batch_id=batch_id,
                    actor_type=actor_type, actor_id=actor_id, tool_name="create_failure_discovery_job",
                    from_state=BatchState.MAINTENANCE_BATCH_FROZEN, to_state=BatchState.MAINTENANCE_BATCH_FROZEN,
                    payload={"job_id": job_id, "model_id": snapshot["model"]["model_id"],
                             "bundle_sha256": bundle_sha, "thresholds_retuned": False, "discovery_passes": 1},
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            bundle_path.unlink(missing_ok=True)
            raise
        return {"job": self.get_job(job_id), "audit_event_sha256": audit, "idempotent": False}

    @staticmethod
    def _safe_names(archive: zipfile.ZipFile, maximum: int = 100_000) -> list[str]:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > maximum:
            raise ValueError("Bundle has duplicate or excessive entries.")
        total = 0
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Bundle contains an unsafe path.")
            info = archive.getinfo(name)
            if info.file_size > 8 * 1024**3:
                raise ValueError("A bundle member exceeds the 8 GiB safety limit.")
            total += info.file_size
        if total > 32 * 1024**3:
            raise ValueError("Bundle exceeds the 32 GiB uncompressed safety limit.")
        return names

    @staticmethod
    def _member_sha(archive: zipfile.ZipFile, name: str) -> str:
        digest = hashlib.sha256()
        with archive.open(name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        bundle = Path(job["bundle_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Screening bundle is outside the managed export directory.")
        if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
            raise ValueError("Screening bundle checksum verification failed.")
        with zipfile.ZipFile(bundle) as archive:
            names = self._safe_names(archive)
            required = {"job_manifest.json", "classes.json", "labels.jsonl", "champion.json", "discovery_profile.json", "checksums.json", "champion/checkpoint.pt"}
            if not required.issubset(names):
                raise ValueError("Screening bundle is missing a required payload.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Screening checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if not isinstance(expected, str) or len(expected) != 64 or self._member_sha(archive, name) != expected:
                    raise ValueError(f"Screening bundle member checksum failed: {name}")
            manifest = json.loads(archive.read("job_manifest.json"))
            profile = json.loads(archive.read("discovery_profile.json"))
            champion = json.loads(archive.read("champion.json"))
            if manifest.get("job_id") != job_id or manifest.get("content_fingerprint") != job["content_fingerprint"]:
                raise ValueError("Screening manifest does not match the database job.")
            if profile != job["profile"] or profile != self._profile():
                raise ValueError("Discovery preset differs from the locked official profile.")
            if champion.get("model_id") != job["champion_model_id"] or champion.get("checkpoint_sha256") != checksums["champion/checkpoint.pt"]:
                raise ValueError("Champion identity or checkpoint checksum changed.")
        return {"job": job, "manifest": manifest, "profile": profile, "champion": champion, "bundle_path": str(bundle)}

    @staticmethod
    def _command(bundle: Path, output: Path, run_id: str, world_size: int) -> tuple[str, ...]:
        return (
            sys.executable, "-m", "facade_training_worker.launcher", "--nproc", str(world_size),
            "--module", DISCOVERY_WORKER_MODULE, "--", "--protocol-version", str(DISCOVERY_WORKER_PROTOCOL_VERSION),
            "--bundle", str(bundle), "--output", str(output), "--run-id", run_id,
        )

    def start_batch(self, batch_id: str, actor_type: str = "human", actor_id: str = "local_engineer") -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can start Champion screening.")
        job = self.create_job(batch_id, actor_type, actor_id)["job"]
        verified = self.verify_job(job["job_id"])
        if job["status"] not in {"exported", "failed"}:
            raise PermissionError(f"Screening job cannot start from status {job['status']}.")
        if self._requires_installed_worker and importlib.util.find_spec(DISCOVERY_WORKER_MODULE) is None:
            raise ValueError(f"Locked Failure Discovery worker is not installed: {DISCOVERY_WORKER_MODULE}")
        preflight = self.environment.training_preflight("champion_failure_discovery")
        failures = [item["name"] for item in preflight["checks"] if not item["passed"]]
        if not preflight["passed"]:
            raise ValueError("Local GPU preflight failed: " + ", ".join(failures))
        run_id = make_id("screening_run")
        output = self.run_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        log_path = output / "worker.log"
        command = self._command(Path(verified["bundle_path"]), output, run_id, preflight["world_size"])
        now = utc_now()
        try:
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (batch_id,)).fetchone()
                if current is None or current["state"] != BatchState.MAINTENANCE_BATCH_FROZEN:
                    raise PermissionError("The batch is no longer ready for screening.")
                active = db.execute(
                    "SELECT run_id FROM screening_runs WHERE job_id=? AND status IN ('queued','running','cancel_requested')",
                    (job["job_id"],),
                ).fetchone()
                if active:
                    raise PermissionError(f"Screening job already has an active run: {active['run_id']}")
                db.execute(
                    "INSERT INTO screening_runs(run_id,job_id,project_id,batch_id,status,worker_protocol_version,gpu_devices_json,"
                    "world_size,preflight_json,command_json,progress_json,log_path,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, job["job_id"], job["project_id"], batch_id, "queued", DISCOVERY_WORKER_PROTOCOL_VERSION,
                     canonical_json(preflight["gpu_devices"]), preflight["world_size"], canonical_json(preflight),
                     canonical_json(list(command)), "{}", str(log_path), now, now),
                )
                db.execute("UPDATE screening_jobs SET status='awaiting_result',updated_at=? WHERE job_id=?", (now, job["job_id"]))
                audit = self.store._append_audit(
                    db, project_id=job["project_id"], batch_id=batch_id, actor_type=actor_type, actor_id=actor_id,
                    tool_name="start_champion_failure_discovery", from_state=BatchState.MAINTENANCE_BATCH_FROZEN,
                    to_state=BatchState.MAINTENANCE_BATCH_FROZEN,
                    payload={"job_id": job["job_id"], "run_id": run_id, "model_id": job["champion_model_id"],
                             "gpu_devices": preflight["gpu_devices"], "world_size": preflight["world_size"],
                             "thresholds_retuned": False, "automatic_reclustering": False},
                )
                queue_item = self.scheduler.enqueue_in_transaction(
                    db,
                    project_id=job["project_id"],
                    pipeline="champion_failure_discovery",
                    run_id=run_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                )
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
        self.scheduler.notify()
        return {"run": self._get_run(run_id), "queue": queue_item, "audit_event_sha256": audit}

    def _append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        if kind not in {"status", "progress", "log"}:
            kind = "log"
        safe = json.loads(canonical_json(payload))
        if "message" in safe:
            safe["message"] = str(safe["message"])[:4000]
        now = utc_now()
        run = self._get_run(run_id)
        log = Path(run["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        if kind == "log":
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"{now} {canonical_json(safe)}\n")
        with self.store.connect() as db:
            db.execute("INSERT INTO screening_run_events(run_id,kind,payload_json,created_at) VALUES(?,?,?,?)", (run_id, kind, canonical_json(safe), now))
            if kind == "progress":
                db.execute("UPDATE screening_runs SET progress_json=?,updated_at=? WHERE run_id=?", (canonical_json(safe), now, run_id))

    def _set_pid(self, run_id: str, pid: int) -> None:
        with self.store.connect() as db:
            db.execute("UPDATE screening_runs SET pid=?,updated_at=? WHERE run_id=?", (pid, utc_now(), run_id))

    def execute(self, run_id: str, cancellation: threading.Event) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("UPDATE screening_runs SET status='running',started_at=?,updated_at=? WHERE run_id=?", (now, now, run_id))
        self._append_event(run_id, "status", {"status": "running"})
        job = self.get_job(run["job_id"])
        spec = WorkerLaunchSpec(
            run_id=run_id, job_id=run["job_id"], bundle_path=Path(job["bundle_path"]),
            output_dir=Path(run["log_path"]).parent, gpu_devices=tuple(run["gpu_devices"]),
            world_size=run["world_size"], command=tuple(run["command"]),
        )
        try:
            result_path = self.runner.run(
                spec, lambda kind, payload: self._append_event(run_id, kind, payload), cancellation,
                lambda pid: self._set_pid(run_id, pid),
            )
            if cancellation.is_set():
                raise TrainingCancelled("Champion screening was cancelled by the engineer.")
            verified = self._verify_and_store_result(run_id, result_path)
            self._register_verified_result(run_id, verified)
            self._append_event(run_id, "status", {"status": "result_verified"})
        except TrainingCancelled as error:
            self._finish_failure(run_id, "cancelled", str(error), "cancel_failure_discovery")
        except Exception as error:
            self._finish_failure(run_id, "failed", str(error), "fail_failure_discovery")

    def status(self, run_id: str) -> str:
        return str(self._get_run(run_id)["status"])

    def cancel_queued(self, db: Any, run_id: str, actor_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM screening_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Screening run not found: {run_id}")
        if row["status"] != "queued":
            raise PermissionError(f"Screening run cannot be queue-cancelled from status {row['status']}.")
        message = "Queued Champion screening was cancelled before GPU execution."
        db.execute(
            "UPDATE screening_runs SET status='cancelled',error_message=?,finished_at=?,updated_at=? WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute(
            "UPDATE screening_jobs SET status='exported',updated_at=? WHERE job_id=?",
            (now, row["job_id"]),
        )
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="human",
            actor_id=actor_id, tool_name="cancel_queued_failure_discovery", from_state="queued",
            to_state="cancelled", payload={"run_id": run_id, "job_id": row["job_id"]},
        )

    def interrupt_abandoned(self, db: Any, run_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM screening_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] in TERMINAL_RUN_STATUSES:
            return
        message = "Agent restarted while Champion screening held the GPU lease."
        db.execute(
            "UPDATE screening_runs SET status='interrupted',error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute("UPDATE screening_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, row["job_id"]))
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="system",
            actor_id="global_gpu_scheduler", tool_name="interrupt_failure_discovery",
            from_state=row["status"], to_state="interrupted",
            payload={"run_id": run_id, "job_id": row["job_id"], "reason": message},
        )

    @staticmethod
    def _csv_rows(content: bytes) -> list[dict[str, str]]:
        text = content.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text, newline="")))

    def _validate_result(self, run_id: str, path: Path) -> dict[str, Any]:
        run = self._get_run(run_id)
        job_info = self.verify_job(run["job_id"])
        resolved = path.resolve()
        output = Path(run["log_path"]).parent.resolve()
        if output != resolved and output not in resolved.parents:
            raise PermissionError("Worker returned a result outside its managed run directory.")
        if not resolved.is_file():
            raise FileNotFoundError("Failure Discovery worker did not produce result_bundle.zip.")
        with zipfile.ZipFile(resolved) as archive:
            names = self._safe_names(archive, 100)
            required = {"result_manifest.json", "predictions.csv", "failure_slices.json", "failure_slice_members.csv", "checksums.json"}
            if not required.issubset(names):
                raise ValueError("Failure Discovery result is missing a required payload.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Result checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if not isinstance(expected, str) or len(expected) != 64 or self._member_sha(archive, name) != expected:
                    raise ValueError(f"Result checksum failed: {name}")
            manifest = json.loads(archive.read("result_manifest.json"))
            predictions = self._csv_rows(archive.read("predictions.csv"))
            slices = json.loads(archive.read("failure_slices.json"))
            members = self._csv_rows(archive.read("failure_slice_members.csv"))
        identities = (
            manifest.get("schema_version") == 1,
            manifest.get("worker_protocol_version") == DISCOVERY_WORKER_PROTOCOL_VERSION,
            manifest.get("job_id") == run["job_id"], manifest.get("run_id") == run_id,
            manifest.get("content_fingerprint") == job_info["job"]["content_fingerprint"],
            manifest.get("champion_model_id") == job_info["job"]["champion_model_id"],
            manifest.get("champion_checkpoint_sha256") == job_info["champion"]["checkpoint_sha256"],
            manifest.get("discovery_profile_id") == DISCOVERY_PROFILE_ID,
            manifest.get("thresholds") == job_info["champion"]["thresholds"],
            manifest.get("thresholds_retuned") is False,
            manifest.get("discovery_passes") == 1,
        )
        if not all(identities):
            raise ValueError("Result identity, Champion, thresholds, or protocol does not match the immutable job.")
        with zipfile.ZipFile(Path(job_info["bundle_path"])) as archive:
            classes = json.loads(archive.read("classes.json"))
            labels = [json.loads(line) for line in archive.read("labels.jsonl").decode().splitlines() if line]
        class_ids = [item["class_id"] for item in classes]
        image_labels = {item["image_id"]: set(item["class_ids"]) for item in labels}
        expected_pairs = {(image_id, class_id) for image_id in image_labels for class_id in class_ids}
        seen_pairs: set[tuple[str, str]] = set()
        failure_pairs: dict[tuple[str, str], str] = {}
        counts = {class_id: {key: 0 for key in ("TP", "TN", "FP", "FN")} for class_id in class_ids}
        for row in predictions:
            pair = (row.get("image_id", ""), row.get("class_id", ""))
            if pair not in expected_pairs or pair in seen_pairs:
                raise ValueError("Predictions contain an unknown or duplicate image/class pair.")
            seen_pairs.add(pair)
            try:
                label = int(row["label"]); predicted = int(row["predicted"])
                score = float(row["score"]); threshold = float(row["threshold"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Prediction fields are invalid.") from error
            expected_label = int(pair[1] in image_labels[pair[0]])
            frozen_threshold = float(job_info["champion"]["thresholds"][pair[1]])
            expected_predicted = int(score >= frozen_threshold)
            expected_error = "TP" if expected_label and expected_predicted else "FN" if expected_label else "FP" if expected_predicted else "TN"
            expected_severity = abs(score - frozen_threshold) if expected_error in {"FP", "FN"} else 0.0
            if label != expected_label or predicted != expected_predicted or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("A prediction conflicts with frozen labels or Champion threshold semantics.")
            if abs(threshold - frozen_threshold) > 1e-12 or row.get("error_type") != expected_error:
                raise ValueError("A prediction threshold or FP/FN designation is invalid.")
            try:
                severity = float(row["severity"])
            except (KeyError, ValueError) as error:
                raise ValueError("Prediction severity is invalid.") from error
            if not math.isfinite(severity) or abs(severity - expected_severity) > 1e-6:
                raise ValueError("Prediction severity does not equal the frozen-threshold margin.")
            counts[pair[1]][expected_error] += 1
            if expected_error in {"FP", "FN"}:
                failure_pairs[pair] = expected_error
        if seen_pairs != expected_pairs:
            raise ValueError("Result must contain exactly one prediction for every image/class pair.")
        if not isinstance(slices, list):
            raise ValueError("Failure slices must be a JSON array.")
        by_key: dict[str, dict[str, Any]] = {}
        minimum = int(job_info["profile"]["clustering"]["min_cluster_size"])
        for item in slices:
            key = str(item.get("slice_key", ""))
            if not key or key in by_key or len(key) > 200:
                raise ValueError("Failure slice keys must be unique and non-empty.")
            if item.get("class_id") not in class_ids or item.get("error_type") not in {"FP", "FN"}:
                raise ValueError("Failure slice class or error type is invalid.")
            support = item.get("support")
            consensus = item.get("consensus_score")
            if not isinstance(support, int) or support < minimum:
                raise ValueError("Failure slice support is below the locked minimum.")
            if not isinstance(consensus, (int, float)) or not float(job_info["profile"]["clustering"]["stable_member_coassociation"]) <= float(consensus) <= 1:
                raise ValueError("Failure slice consensus score is invalid.")
            by_key[key] = item
        grouped: dict[str, list[dict[str, str]]] = {key: [] for key in by_key}
        assigned: set[tuple[str, str]] = set()
        for member in members:
            key = member.get("slice_key", "")
            if key not in by_key:
                raise ValueError("A slice member references an unknown slice.")
            item = by_key[key]
            image_id = member.get("image_id", "")
            pair = (image_id, item["class_id"])
            if failure_pairs.get(pair) != item["error_type"] or pair in assigned:
                raise ValueError("A slice member is not a matching unique FP/FN record.")
            assigned.add(pair)
            grouped[key].append(member)
        for key, item in by_key.items():
            group = grouped[key]
            ranks = sorted(int(member["membership_rank"]) for member in group)
            if len(group) != item["support"] or ranks != list(range(1, len(group) + 1)):
                raise ValueError("Failure slice support or membership ranks are inconsistent.")
            member_ids = {member["image_id"] for member in group}
            if item.get("representative_image_id") not in member_ids:
                raise ValueError("Failure slice representative must be one of its members.")
        if manifest.get("prediction_rows") != len(predictions) or manifest.get("failure_records") != len(failure_pairs) or manifest.get("failure_slices") != len(slices):
            raise ValueError("Result manifest counts are inconsistent with verified payloads.")
        summary = {
            "images": len(image_labels), "classes": len(class_ids), "prediction_rows": len(predictions),
            "failure_records": len(failure_pairs), "failure_slices": len(slices), "per_class_confusion": counts,
            "thresholds": job_info["champion"]["thresholds"], "thresholds_retuned": False,
            "discovery_passes": 1, "automatic_reclustering": False,
        }
        return {"source": resolved, "manifest": manifest, "slices": slices, "members": grouped, "summary": summary}

    def _verify_and_store_result(self, run_id: str, path: Path) -> dict[str, Any]:
        validated = self._validate_result(run_id, path)
        run = self._get_run(run_id)
        destination_dir = self.result_root / run["job_id"] / run_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / "result_bundle.zip"
        temporary = destination_dir / ".result_bundle.tmp"
        try:
            shutil.copyfile(validated["source"], temporary)
            expected = sha256_file(validated["source"])
            if sha256_file(temporary) != expected:
                raise ValueError("Result changed while entering the artifact store.")
            os.replace(temporary, destination)
            destination.chmod(0o444)
        except Exception:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
        return {**validated, "path": str(destination), "sha256": expected}

    def _register_verified_result(self, run_id: str, verified: dict[str, Any]) -> None:
        run = self._get_run(run_id)
        job = self.get_job(run["job_id"])
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (run["batch_id"],)).fetchone()
            if batch is None or batch["state"] != BatchState.MAINTENANCE_BATCH_FROZEN:
                raise PermissionError("Batch state changed before the verified result could be registered.")
            inference = require_transition(batch["state"], "record_batch_inference", BATCH_TRANSITIONS, False)
            db.execute("UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?", (inference.target, now, run["batch_id"]))
            self.store._append_audit(
                db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                actor_id="locked_failure_discovery_worker", tool_name="record_batch_inference",
                from_state=batch["state"], to_state=inference.target,
                payload={"run_id": run_id, "model_id": job["champion_model_id"],
                         "prediction_rows": verified["summary"]["prediction_rows"], "thresholds_retuned": False},
            )
            discovery = require_transition(inference.target, "record_failure_discovery", BATCH_TRANSITIONS, False)
            for index, item in enumerate(verified["slices"], start=1):
                slice_id = f"slice_{run_id.removeprefix('screening_run_')}_{index:03d}"
                db.execute(
                    "INSERT INTO failure_slices(slice_id,batch_id,class_id,error_type,support,clustering_profile_id,status,created_at,"
                    "source_screening_run_id,source_slice_key,consensus_score,representative_image_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (slice_id, run["batch_id"], item["class_id"], item["error_type"], item["support"],
                     DISCOVERY_PROFILE_ID, "pending", now, run_id, item["slice_key"], float(item["consensus_score"]),
                     item["representative_image_id"]),
                )
                for member in verified["members"][item["slice_key"]]:
                    db.execute(
                        "INSERT INTO failure_slice_members(slice_id,image_id,membership_rank) VALUES(?,?,?)",
                        (slice_id, member["image_id"], int(member["membership_rank"])),
                    )
            db.execute("UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?", (discovery.target, now, run["batch_id"]))
            db.execute(
                "UPDATE screening_runs SET status='result_verified',result_bundle_path=?,result_bundle_sha256=?,summary_json=?,"
                "finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                (verified["path"], verified["sha256"], canonical_json(verified["summary"]), now, now, run_id),
            )
            db.execute("UPDATE screening_jobs SET status='result_verified',updated_at=? WHERE job_id=?", (now, run["job_id"]))
            db.execute(
                "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (make_id("artifact"), run["project_id"], run["batch_id"], "champion_failure_discovery_result",
                 verified["path"], verified["sha256"], Path(verified["path"]).stat().st_size, now),
            )
            self.store._append_audit(
                db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                actor_id="locked_failure_discovery_worker", tool_name="record_failure_discovery",
                from_state=inference.target, to_state=discovery.target,
                payload={"job_id": run["job_id"], "run_id": run_id, "result_bundle_sha256": verified["sha256"],
                         "failure_records": verified["summary"]["failure_records"], "failure_slices": verified["summary"]["failure_slices"],
                         "clustering_profile_id": DISCOVERY_PROFILE_ID, "discovery_passes": 1,
                         "expert_review_required": True},
            )

    def _finish_failure(self, run_id: str, status: str, message: str, action: str) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE screening_runs SET status=?,error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                (status, message[:4000], now, now, run_id),
            )
            db.execute("UPDATE screening_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, run["job_id"]))
            self.store._append_audit(
                db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                actor_id="locked_failure_discovery_worker", tool_name=action, from_state=run["status"], to_state=status,
                payload={"job_id": run["job_id"], "run_id": run_id, "error": message[:1000]},
            )
        self._append_event(run_id, "status", {"status": status, "error": classify_worker_error(message)})

    def cancel_run(self, run_id: str, actor_id: str = "local_engineer") -> dict[str, Any]:
        run = self._get_run(run_id)
        if run["status"] not in {"queued", "running"}:
            raise PermissionError(f"Screening run cannot be cancelled from status {run['status']}.")
        with self.store.connect() as db:
            queue_row = db.execute("SELECT queue_id FROM gpu_job_queue WHERE run_id=?", (run_id,)).fetchone()
        if queue_row is None:
            raise RuntimeError("Screening run is not attached to the global GPU queue.")
        queue_item = self.scheduler.cancel(queue_row["queue_id"], actor_id)
        return {"run": self._get_run(run_id), "queue": queue_item}

    def wait(self, run_id: str, timeout: float = 10) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        run = self._get_run(run_id)
        queue_terminal = {"completed", "failed", "cancelled", "interrupted"}
        while time.monotonic() < deadline:
            with self.store.connect() as db:
                queue_row = db.execute("SELECT status FROM gpu_job_queue WHERE run_id=?", (run_id,)).fetchone()
            if run["status"] in TERMINAL_RUN_STATUSES and (queue_row is None or queue_row["status"] in queue_terminal):
                break
            time.sleep(0.01)
            run = self._get_run(run_id)
        return run

    def shutdown(self, timeout: float = 10) -> None:
        if self._owns_scheduler:
            self.scheduler.shutdown(timeout)


def register_failure_discovery_tools(registry: ToolRegistry, service: FailureDiscoveryService) -> None:
    registry.register(ToolDefinition(
        "start_champion_failure_discovery",
        "After engineer confirmation, screen one frozen maintenance batch with the active Champion and run exactly one locked FP/FN consensus-discovery pass. No thresholds or clustering settings are accepted.",
        {"type": "object", "required": ["batch_id"], "properties": {"batch_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.start_batch(args["batch_id"], ctx["actor_type"], ctx["actor_id"]),
    ))
    registry.register(ToolDefinition(
        "cancel_champion_failure_discovery",
        "Safely stop the attached screening worker after engineer confirmation; the frozen batch and active Champion remain unchanged.",
        {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.cancel_run(args["run_id"], ctx["actor_id"]),
    ))
