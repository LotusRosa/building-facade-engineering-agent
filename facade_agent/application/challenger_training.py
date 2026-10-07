from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from ..core.permissions import ToolPolicy
from ..core.states import BATCH_TRANSITIONS, BatchState, ProjectState, require_transition
from ..storage import canonical_json, make_id, utc_now
from ..tools.registry import ToolDefinition, ToolRegistry
from .challenger_jobs import ChallengerJobService
from .gpu_scheduler import GpuJobScheduler
from .local_training import SubprocessWorkerRunner, TrainingCancelled, WorkerLaunchSpec, WorkerRunner
from .model_storage import ModelStorageService
from .training_jobs import json_bytes, sha256_file
from .worker_errors import classify_worker_error


CHALLENGER_WORKER_PROTOCOL_VERSION = 1
CHALLENGER_WORKER_MODULE = "facade_training_worker.challenger_update"
PRIMARY_SELECTION_METRIC = "development_macro_map"
ACTIVE_RUN_STATUSES = {"queued", "running", "cancel_requested"}
TERMINAL_RUN_STATUSES = {"cancelled", "result_verified", "failed", "interrupted"}


class ChallengerTrainingService:
    """Run and register a Challenger without changing the Active Champion."""

    def __init__(
        self,
        store: Any,
        jobs: ChallengerJobService,
        environment_manager: Any,
        root: Path,
        runner: WorkerRunner | None = None,
        model_storage: ModelStorageService | None = None,
        scheduler: GpuJobScheduler | None = None,
    ) -> None:
        self.store = store
        self.jobs = jobs
        self.environment = environment_manager
        self.root = root.resolve()
        self.run_root = (self.root / "artifact_store" / "challenger_runs").resolve()
        self.result_root = (self.root / "artifact_store" / "challenger_results").resolve()
        self.model_storage = model_storage or ModelStorageService(store, self.root)
        self.runner = runner or SubprocessWorkerRunner()
        self._requires_installed_worker = runner is None
        self._registration_lock = threading.Lock()
        self.scheduler = scheduler or GpuJobScheduler(store)
        self._owns_scheduler = scheduler is None
        self.scheduler.register_handler("challenger_update", self)
        if self._owns_scheduler:
            self.scheduler.start()

    @staticmethod
    def _row_to_run(row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("gpu_devices_json", "preflight_json", "command_json", "progress_json", "summary_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def _get_run(self, run_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM challenger_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Challenger run not found: {run_id}")
        return self._row_to_run(row)

    def get_run(self, run_id: str, event_after: int = 0) -> dict[str, Any]:
        run = self._get_run(run_id)
        log_path = Path(run["log_path"]).resolve()
        if self.run_root != log_path and self.run_root not in log_path.parents:
            raise PermissionError("Challenger log is outside the managed run directory.")
        with self.store.connect() as db:
            events = []
            for row in db.execute(
                "SELECT event_id,kind,payload_json,created_at FROM challenger_run_events "
                "WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT 500",
                (run_id, max(0, int(event_after))),
            ):
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                events.append(item)
        return {"run": run, "events": events}

    def list_runs(self, project_id: str, batch_id: str | None = None) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        sql = "SELECT * FROM challenger_runs WHERE project_id=?"
        parameters: list[Any] = [project_id]
        if batch_id:
            sql += " AND batch_id=?"
            parameters.append(batch_id)
        sql += " ORDER BY created_at DESC"
        with self.store.connect() as db:
            rows = db.execute(sql, parameters).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _command(bundle: Path, output_dir: Path, run_id: str, world_size: int) -> tuple[str, ...]:
        return (
            sys.executable, "-m", "facade_training_worker.launcher", "--nproc", str(world_size),
            "--module", CHALLENGER_WORKER_MODULE, "--",
            "--protocol-version", str(CHALLENGER_WORKER_PROTOCOL_VERSION),
            "--bundle", str(bundle), "--output", str(output_dir), "--run-id", run_id,
        )

    def _append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        kind = kind if kind in {"status", "progress", "log"} else "log"
        payload = json.loads(canonical_json(payload))
        if "message" in payload:
            payload["message"] = str(payload["message"])[:4000]
        now = utc_now()
        run = self._get_run(run_id)
        log_path = Path(run["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "log":
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now} {canonical_json(payload)}\n")
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO challenger_run_events(run_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
                (run_id, kind, canonical_json(payload), now),
            )
            if kind == "progress":
                db.execute("UPDATE challenger_runs SET progress_json=?,updated_at=? WHERE run_id=?", (canonical_json(payload), now, run_id))

    def _set_pid(self, run_id: str, pid: int) -> None:
        with self.store.connect() as db:
            db.execute("UPDATE challenger_runs SET pid=?,updated_at=? WHERE run_id=?", (pid, utc_now(), run_id))

    def _assert_snapshot(self, job_id: str) -> dict[str, Any]:
        verified = self.jobs.verify_bundle(job_id)
        job = verified["job"]
        batch = self.store.get_maintenance_batch(job["batch_id"])
        project = self.store.get_project(job["project_id"])
        if batch["state"] != BatchState.FAILURE_REVIEW_COMPLETED:
            raise PermissionError("Challenger training requires the frozen Failure Slice review state.")
        if project["state"] != ProjectState.SCREENING_READY or project.get("active_batch_id") != job["batch_id"]:
            raise PermissionError("The Challenger batch is not the active screening batch.")
        if job["status"] not in {"exported", "failed"}:
            raise PermissionError(f"Challenger job cannot start from status {job['status']}.")
        with self.store.connect() as db:
            parent = db.execute(
                "SELECT * FROM model_versions WHERE model_id=?", (job["parent_champion_model_id"],)
            ).fetchone()
            active = db.execute("SELECT model_id FROM active_models WHERE project_id=?", (job["project_id"],)).fetchone()
        if parent is None or parent["role"] != "champion" or active is None or active["model_id"] != job["parent_champion_model_id"]:
            raise PermissionError("The frozen parent is no longer the Active Champion.")
        return verified

    def start_job(self, job_id: str, actor_type: str = "human", actor_id: str = "local_engineer") -> dict[str, Any]:
        verified = self._assert_snapshot(job_id)
        if self._requires_installed_worker and importlib.util.find_spec(CHALLENGER_WORKER_MODULE) is None:
            raise ValueError(f"Locked Challenger worker is not installed: {CHALLENGER_WORKER_MODULE}")
        preflight = self.environment.training_preflight("challenger_update")
        failures = [item["name"] for item in preflight["checks"] if not item["passed"]]
        if not preflight["passed"]:
            raise ValueError("Local GPU preflight failed: " + ", ".join(failures))
        job = verified["job"]
        run_id = make_id("challenger_run")
        output_dir = self.run_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        log_path = output_dir / "worker.log"
        command = self._command(Path(verified["bundle_path"]), output_dir, run_id, preflight["world_size"])
        now = utc_now()
        try:
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                active = db.execute(
                    "SELECT run_id FROM challenger_runs WHERE job_id=? AND status IN ('queued','running','cancel_requested')",
                    (job_id,),
                ).fetchone()
                if active:
                    raise PermissionError(f"Challenger job already has an active run: {active['run_id']}")
                db.execute(
                    "INSERT INTO challenger_runs(run_id,job_id,project_id,batch_id,status,worker_protocol_version,"
                    "gpu_devices_json,world_size,preflight_json,command_json,progress_json,log_path,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, job_id, job["project_id"], job["batch_id"], "queued", CHALLENGER_WORKER_PROTOCOL_VERSION,
                     canonical_json(preflight["gpu_devices"]), preflight["world_size"], canonical_json(preflight),
                     canonical_json(list(command)), "{}", str(log_path), now, now),
                )
                db.execute("UPDATE challenger_jobs SET status='awaiting_result',updated_at=? WHERE job_id=?", (now, job_id))
                audit = self.store._append_audit(
                    db, project_id=job["project_id"], batch_id=job["batch_id"], actor_type=actor_type, actor_id=actor_id,
                    tool_name="start_challenger_training", from_state="exported", to_state="awaiting_result",
                    payload={"job_id": job_id, "run_id": run_id, "gpu_devices": preflight["gpu_devices"],
                             "world_size": preflight["world_size"], "worker_protocol_version": CHALLENGER_WORKER_PROTOCOL_VERSION,
                             "parent_champion_model_id": job["parent_champion_model_id"], "test_read": False},
                )
                queue_item = self.scheduler.enqueue_in_transaction(
                    db,
                    project_id=job["project_id"],
                    pipeline="challenger_update",
                    run_id=run_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        self.scheduler.notify()
        return {"run": self._get_run(run_id), "queue": queue_item, "audit_event_sha256": audit}

    def execute(self, run_id: str, cancellation: threading.Event) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("UPDATE challenger_runs SET status='running',started_at=?,updated_at=? WHERE run_id=?", (now, now, run_id))
        self._append_event(run_id, "status", {"status": "running"})
        spec = WorkerLaunchSpec(
            run_id=run_id, job_id=run["job_id"], bundle_path=Path(self.jobs.get_job(run["job_id"])["bundle_path"]),
            output_dir=Path(run["log_path"]).parent, gpu_devices=tuple(run["gpu_devices"]),
            world_size=run["world_size"], command=tuple(run["command"]),
        )
        try:
            result = self.runner.run(spec, lambda kind, payload: self._append_event(run_id, kind, payload), cancellation,
                                     lambda pid: self._set_pid(run_id, pid))
            if cancellation.is_set():
                raise TrainingCancelled("Challenger training was cancelled by the engineer.")
            verified = self._verify_and_store_result(run_id, result)
            finished = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE challenger_runs SET status='result_verified',result_bundle_path=?,result_bundle_sha256=?,summary_json=?,"
                    "finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                    (verified["path"], verified["sha256"], canonical_json(verified["summary"]), finished, finished, run_id),
                )
                db.execute("UPDATE challenger_jobs SET status='result_verified',updated_at=? WHERE job_id=?", (finished, run["job_id"]))
                self.store._append_audit(
                    db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                    actor_id="locked_challenger_worker", tool_name="verify_challenger_training_result",
                    from_state="awaiting_result", to_state="result_verified",
                    payload={"job_id": run["job_id"], "run_id": run_id, "result_bundle_sha256": verified["sha256"],
                             "selected_seed": verified["manifest"]["selected_seed"], "test_read": False},
                )
            self._append_event(run_id, "status", {"status": "result_verified"})
        except TrainingCancelled as error:
            self._finish_failure(run_id, "cancelled", str(error), "cancel_challenger_training")
        except Exception as error:
            self._finish_failure(run_id, "failed", str(error), "fail_challenger_training")

    def status(self, run_id: str) -> str:
        return str(self._get_run(run_id)["status"])

    def cancel_queued(self, db: Any, run_id: str, actor_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM challenger_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Challenger run not found: {run_id}")
        if row["status"] != "queued":
            raise PermissionError(f"Challenger run cannot be queue-cancelled from status {row['status']}.")
        message = "Queued Challenger training was cancelled before GPU execution."
        db.execute(
            "UPDATE challenger_runs SET status='cancelled',error_message=?,finished_at=?,updated_at=? WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute("UPDATE challenger_jobs SET status='exported',updated_at=? WHERE job_id=?", (now, row["job_id"]))
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="human",
            actor_id=actor_id, tool_name="cancel_queued_challenger_training", from_state="queued",
            to_state="cancelled", payload={"run_id": run_id, "job_id": row["job_id"]},
        )

    def interrupt_abandoned(self, db: Any, run_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM challenger_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] in TERMINAL_RUN_STATUSES:
            return
        message = "Agent restarted while Challenger training held the GPU lease."
        db.execute(
            "UPDATE challenger_runs SET status='interrupted',error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute("UPDATE challenger_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, row["job_id"]))
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="system",
            actor_id="global_gpu_scheduler", tool_name="interrupt_challenger_training",
            from_state=row["status"], to_state="interrupted",
            payload={"run_id": run_id, "job_id": row["job_id"], "reason": message},
        )

    @staticmethod
    def _safe_names(archive: zipfile.ZipFile) -> list[str]:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 100:
            raise ValueError("Challenger result has duplicate or excessive entries.")
        total = 0
        for name in names:
            path = Path(name)
            info = archive.getinfo(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or info.file_size > 8 * 1024**3:
                raise ValueError("Challenger result contains an unsafe artifact.")
            total += info.file_size
        if total > 32 * 1024**3:
            raise ValueError("Challenger result exceeds the uncompressed safety limit.")
        return names

    @staticmethod
    def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
        digest = hashlib.sha256()
        with archive.open(name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_result(self, run_id: str, path: Path, allowed_root: Path) -> dict[str, Any]:
        run = self._get_run(run_id)
        job_info = self.jobs.verify_bundle(run["job_id"])
        job = job_info["job"]
        resolved = path.resolve()
        root = allowed_root.resolve()
        if root != resolved and root not in resolved.parents:
            raise PermissionError("Challenger result is outside its managed directory.")
        if not resolved.is_file():
            raise FileNotFoundError("Challenger worker did not produce result_bundle.zip.")
        with zipfile.ZipFile(resolved) as archive:
            names = self._safe_names(archive)
            if not {"result_manifest.json", "checksums.json"}.issubset(names):
                raise ValueError("Challenger result is missing its manifest or checksums.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Challenger result checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if not isinstance(expected, str) or len(expected) != 64 or self._member_sha256(archive, name) != expected:
                    raise ValueError(f"Challenger result checksum failed: {name}")
            manifest = json.loads(archive.read("result_manifest.json"))
            profile_sha = hashlib.sha256(json_bytes(job_info["profile"])).hexdigest()
            identities = (
                manifest.get("schema_version") == 1,
                manifest.get("worker_protocol_version") == CHALLENGER_WORKER_PROTOCOL_VERSION,
                manifest.get("job_id") == run["job_id"], manifest.get("run_id") == run_id,
                manifest.get("content_fingerprint") == job["content_fingerprint"],
                manifest.get("profile_sha256") == profile_sha,
                manifest.get("parent_champion_model_id") == job["parent_champion_model_id"],
                manifest.get("epochs") == job_info["profile"]["epochs"], manifest.get("test_read") is False,
            )
            if not all(identities):
                raise ValueError("Challenger result identity, parent, epoch, or evaluation contract is invalid.")
            expected_draws = {}
            with zipfile.ZipFile(Path(job_info["bundle_path"])) as input_archive:
                classes = json.loads(input_archive.read("classes.json"))
                for seed in job_info["profile"]["seeds"]:
                    name = f"training_draws/seed_{seed}.jsonl"
                    expected_draws[str(seed)] = hashlib.sha256(input_archive.read(name)).hexdigest()
            if manifest.get("training_draws_sha256") != expected_draws:
                raise ValueError("Challenger Worker did not use the frozen per-seed draw manifests.")
            seeds = manifest.get("seed_results")
            expected_seeds = job_info["profile"]["seeds"]
            if not isinstance(seeds, list) or [item.get("seed") for item in seeds] != expected_seeds:
                raise ValueError("Challenger result must contain the three fixed seeds in preset order.")
            scores: list[tuple[float, int]] = []
            weight_files: set[str] = set()
            for item in seeds:
                weight = item.get("weights_file")
                weight_sha = item.get("weights_sha256")
                if not isinstance(weight, str) or not weight.startswith("weights/") or weight not in checksums or weight in weight_files:
                    raise ValueError("Challenger seed weights files are invalid or reused.")
                weight_files.add(weight)
                if weight_sha != checksums[weight]:
                    raise ValueError("Challenger seed weights SHA-256 does not match checksums.json.")
                score = item.get("metrics", {}).get(PRIMARY_SELECTION_METRIC)
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
                    raise ValueError("Challenger seed is missing a finite development selection metric.")
                scores.append((float(score), int(item["seed"])))
            selected = sorted(scores, key=lambda value: (-value[0], value[1]))[0][1]
            if manifest.get("selected_seed") != selected:
                raise ValueError("Challenger selected seed does not follow the locked development-metric rule.")
            class_ids = {item["class_id"] for item in classes}
            thresholds = manifest.get("thresholds")
            if not isinstance(thresholds, dict) or set(thresholds) != class_ids:
                raise ValueError("Challenger thresholds do not match the frozen taxonomy.")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1 for value in thresholds.values()):
                raise ValueError("Challenger thresholds must be numeric values from 0 to 1.")
            aggregate = manifest.get("aggregate_metrics")
            if not isinstance(aggregate, dict) or not aggregate or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
                for value in aggregate.values()
            ):
                raise ValueError("Challenger aggregate metrics must be finite numeric Worker results.")
        return {"source": resolved, "manifest": manifest, "summary": {"aggregate_metrics": aggregate, "selected_seed": selected, "test_read": False}}

    def _verify_and_store_result(self, run_id: str, path: Path) -> dict[str, Any]:
        run = self._get_run(run_id)
        validated = self._validate_result(run_id, path, Path(run["log_path"]).parent)
        destination_dir = self.result_root / run["job_id"] / run_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / "result_bundle.zip"
        temporary = destination_dir / ".result_bundle.tmp"
        try:
            shutil.copyfile(validated["source"], temporary)
            digest = sha256_file(validated["source"])
            if sha256_file(temporary) != digest:
                raise ValueError("Challenger result changed while entering the artifact store.")
            os.replace(temporary, destination)
            destination.chmod(0o444)
        except Exception:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
        return {**validated, "path": str(destination), "sha256": digest}

    def _finish_failure(self, run_id: str, status: str, message: str, action: str) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE challenger_runs SET status=?,error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?", (status, message[:4000], now, now, run_id))
            db.execute("UPDATE challenger_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, run["job_id"]))
            self.store._append_audit(
                db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                actor_id="locked_challenger_worker", tool_name=action, from_state=run["status"], to_state=status,
                payload={"job_id": run["job_id"], "run_id": run_id, "error": message[:1000]},
            )
        self._append_event(run_id, "status", {"status": status, "error": classify_worker_error(message)})

    def cancel_run(self, run_id: str, actor_id: str = "local_engineer") -> dict[str, Any]:
        run = self._get_run(run_id)
        if run["status"] not in {"queued", "running"}:
            raise PermissionError(f"Challenger run cannot be cancelled from status {run['status']}.")
        with self.store.connect() as db:
            queue_row = db.execute("SELECT queue_id FROM gpu_job_queue WHERE run_id=?", (run_id,)).fetchone()
        if queue_row is None:
            raise RuntimeError("Challenger run is not attached to the global GPU queue.")
        queue_item = self.scheduler.cancel(queue_row["queue_id"], actor_id)
        return {"run": self._get_run(run_id), "queue": queue_item}

    def register_challenger(self, run_id: str, actor_type: str = "human", actor_id: str = "local_engineer") -> dict[str, Any]:
        with self._registration_lock:
            return self._register_challenger(run_id, actor_type, actor_id)

    def _register_challenger(self, run_id: str, actor_type: str, actor_id: str) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can register a Challenger.")
        run = self._get_run(run_id)
        if run["status"] != "result_verified":
            raise PermissionError("Only a checksum-verified Challenger result can be registered.")
        job = self.jobs.get_job(run["job_id"])
        with self.store.connect() as db:
            existing = db.execute(
                "SELECT * FROM model_versions WHERE source_challenger_job_id=? AND role='challenger'",
                (run["job_id"],),
            ).fetchone()
        if existing:
            return {"model": self._model_dict(existing), "idempotent": True}
        if job["status"] != "result_verified" or sha256_file(Path(run["result_bundle_path"])) != run["result_bundle_sha256"]:
            raise ValueError("Verified Challenger result changed before registration.")
        verified = self._validate_result(run_id, Path(run["result_bundle_path"]), self.result_root)
        selected = next(item for item in verified["manifest"]["seed_results"] if item["seed"] == verified["manifest"]["selected_seed"])
        model_id = f"challenger_{run['result_bundle_sha256'][:16]}"
        source_root = self.root / "artifact_store" / "model_registration_sources"
        source_root.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        registration_manifest = {
            "schema_version": 1,
            "project_id": run["project_id"],
            "model_id": model_id,
            "parent_model_id": job["parent_champion_model_id"],
            "source_batch_id": run["batch_id"],
            "role": "challenger",
            "taxonomy_sha256": self.model_storage.taxonomy_sha256(run["project_id"]),
            "thresholds": verified["manifest"]["thresholds"],
            "training_profile_id": job["training_profile_id"],
            "source_challenger_job_id": run["job_id"],
            "source_challenger_run_id": run_id,
            "result_bundle_sha256": run["result_bundle_sha256"],
            "metrics": verified["manifest"]["aggregate_metrics"],
            "selected_seed": verified["manifest"]["selected_seed"],
            "registered_at": now,
        }
        with tempfile.TemporaryDirectory(dir=source_root) as temporary_dir:
            source_checkpoint = Path(temporary_dir) / Path(selected["weights_file"]).name
            with zipfile.ZipFile(Path(run["result_bundle_path"])) as archive, source_checkpoint.open("wb") as handle:
                with archive.open(selected["weights_file"]) as source:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
            with self.model_storage.prepare_registration(
                project_id=run["project_id"],
                model_id=model_id,
                source_checkpoint=source_checkpoint,
                expected_sha256=selected["weights_sha256"],
                manifest=registration_manifest,
            ) as prepared:
                published = prepared.publish()
                checkpoint = self.model_storage.resolve_model_file(
                    run["project_id"], published["checkpoint_relpath"]
                )
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    batch = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (run["batch_id"],)).fetchone()
                    active = db.execute("SELECT model_id FROM active_models WHERE project_id=?", (run["project_id"],)).fetchone()
                    if batch is None or batch["state"] != BatchState.FAILURE_REVIEW_COMPLETED:
                        raise PermissionError("Batch state changed before Challenger registration.")
                    if active is None or active["model_id"] != job["parent_champion_model_id"]:
                        raise PermissionError("Active Champion changed before Challenger registration.")
                    transition = require_transition(batch["state"], "record_challenger_training", BATCH_TRANSITIONS, True)
                    db.execute(
                        "INSERT INTO model_versions(model_id,project_id,parent_model_id,source_batch_id,role,"
                        "checkpoint_path,checkpoint_relpath,checkpoint_sha256,manifest_relpath,manifest_sha256,"
                        "thresholds_json,training_profile_id,created_at,result_bundle_sha256,metrics_json,"
                        "source_challenger_job_id,source_challenger_run_id) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (model_id, run["project_id"], job["parent_champion_model_id"], run["batch_id"], "challenger",
                         str(checkpoint), published["checkpoint_relpath"], selected["weights_sha256"],
                         published["manifest_relpath"], published["manifest_sha256"],
                         canonical_json(verified["manifest"]["thresholds"]), job["training_profile_id"], now,
                         run["result_bundle_sha256"], canonical_json(verified["manifest"]["aggregate_metrics"]),
                         run["job_id"], run_id),
                    )
                    db.execute("UPDATE challenger_jobs SET status='registered',updated_at=? WHERE job_id=?", (now, run["job_id"]))
                    db.execute("UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?", (transition.target, now, run["batch_id"]))
                    db.execute(
                        "UPDATE projects SET model_storage_locked_at=COALESCE(model_storage_locked_at,?),updated_at=? "
                        "WHERE project_id=?",
                        (now, now, run["project_id"]),
                    )
                    db.execute(
                        "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (make_id("artifact"), run["project_id"], run["batch_id"], "challenger_checkpoint", str(checkpoint),
                         selected["weights_sha256"], published["checkpoint_size_bytes"], now),
                    )
                    audit = self.store._append_audit(
                        db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type=actor_type, actor_id=actor_id,
                        tool_name="record_challenger_training", from_state=batch["state"], to_state=transition.target,
                        payload={"job_id": run["job_id"], "run_id": run_id, "model_id": model_id,
                                 "parent_champion_model_id": job["parent_champion_model_id"], "selected_seed": verified["manifest"]["selected_seed"],
                                 "checkpoint_sha256": selected["weights_sha256"], "manifest_sha256": published["manifest_sha256"],
                                 "result_bundle_sha256": run["result_bundle_sha256"],
                                 "active_champion_unchanged": True, "test_read": False},
                    )
                prepared.commit()
        return {"model": self.get_model(model_id), "audit_event_sha256": audit, "idempotent": False}

    @staticmethod
    def _model_dict(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["thresholds"] = json.loads(item.pop("thresholds_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM model_versions WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(f"Model not found: {model_id}")
        return self._model_dict(row)

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


def register_challenger_training_tools(registry: ToolRegistry, service: ChallengerTrainingService) -> None:
    registry.register(ToolDefinition(
        "start_challenger_training",
        "After engineer confirmation, run the frozen three-seed Challenger update on selected local NVIDIA GPUs. The caller cannot change sampling, epochs, seed, parent, or command.",
        {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.start_job(args["job_id"], ctx["actor_type"], ctx["actor_id"]),
    ))
    registry.register(ToolDefinition(
        "cancel_challenger_training",
        "Safely interrupt the attached Challenger Worker after engineer confirmation; the parent Champion remains active.",
        {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.cancel_run(args["run_id"], ctx["actor_id"]),
    ))
    registry.register(ToolDefinition(
        "register_challenger",
        "After separate engineer confirmation, register a checksum-verified Challenger for later evidence evaluation. This never deploys or activates it.",
        {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.register_challenger(args["run_id"], ctx["actor_type"], ctx["actor_id"]),
    ))
