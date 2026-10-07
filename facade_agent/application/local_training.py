from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..core.permissions import ToolPolicy
from ..core.states import ProjectState
from ..storage import canonical_json, make_id, utc_now
from ..tools.registry import ToolDefinition, ToolRegistry
from .gpu_scheduler import GpuJobScheduler
from .training_jobs import InitialChampionJobService, sha256_file
from .model_storage import ModelStorageService
from .worker_errors import classify_worker_error


WORKER_PROTOCOL_VERSION = 1
WORKER_MODULE = "facade_training_worker.initial_champion"
ACTIVE_RUN_STATUSES = {"queued", "running", "cancel_requested"}
TERMINAL_RUN_STATUSES = {"cancelled", "result_verified", "failed", "interrupted"}


@dataclass(frozen=True)
class WorkerLaunchSpec:
    run_id: str
    job_id: str
    bundle_path: Path
    output_dir: Path
    gpu_devices: tuple[int, ...]
    world_size: int
    command: tuple[str, ...]


class WorkerRunner(Protocol):
    def run(
        self,
        spec: WorkerLaunchSpec,
        emit: Callable[[str, dict[str, Any]], None],
        cancel_event: threading.Event,
        set_pid: Callable[[int], None],
    ) -> Path: ...


class TrainingCancelled(RuntimeError):
    pass


class SubprocessWorkerRunner:
    """Launch only the fixed Worker through the portable local GPU launcher."""

    @staticmethod
    def _environment(gpu_devices: tuple[int, ...]) -> dict[str, str]:
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (package_root, existing_pythonpath) if value
        )
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(value) for value in gpu_devices
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["USE_LIBUV"] = "0"
        environment["OMP_NUM_THREADS"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        return environment

    @staticmethod
    def _stop(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)

    def run(
        self,
        spec: WorkerLaunchSpec,
        emit: Callable[[str, dict[str, Any]], None],
        cancel_event: threading.Event,
        set_pid: Callable[[int], None],
    ) -> Path:
        environment = self._environment(spec.gpu_devices)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            list(spec.command),
            cwd=spec.output_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        set_pid(process.pid)
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line.rstrip("\r\n"))
            lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        result_path: Path | None = None
        stream_closed = False
        while process.poll() is None or not stream_closed:
            if cancel_event.is_set():
                self._stop(process)
                raise TrainingCancelled("Training was cancelled by the engineer.")
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                stream_closed = True
                continue
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                emit("log", {"message": line})
                continue
            kind = str(event.pop("event", "log"))
            if kind == "result" and event.get("result_bundle_path"):
                result_path = Path(str(event["result_bundle_path"]))
            emit("progress" if kind == "progress" else "log", event)
        code = process.wait()
        if cancel_event.is_set():
            raise TrainingCancelled("Training was cancelled by the engineer.")
        if code != 0:
            raise RuntimeError(f"GPU worker exited with code {code}.")
        return result_path or (spec.output_dir / "result_bundle.zip")


class LocalTrainingWorkerService:
    def __init__(
        self,
        store: Any,
        job_service: InitialChampionJobService,
        environment_manager: Any,
        root: Path,
        runner: WorkerRunner | None = None,
        model_storage: ModelStorageService | None = None,
        scheduler: GpuJobScheduler | None = None,
    ) -> None:
        self.store = store
        self.jobs = job_service
        self.environment = environment_manager
        self.root = root.resolve()
        self.run_root = (self.root / "artifact_store" / "training_runs").resolve()
        self.result_root = (self.root / "artifact_store" / "training_results").resolve()
        self.model_storage = model_storage or ModelStorageService(store, self.root)
        self.runner = runner or SubprocessWorkerRunner()
        self._registration_lock = threading.Lock()
        self.scheduler = scheduler or GpuJobScheduler(store)
        self._owns_scheduler = scheduler is None
        self.scheduler.register_handler("initial_champion", self)
        if self._owns_scheduler:
            self.scheduler.start()

    @staticmethod
    def _row_to_run(row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("gpu_devices_json", "preflight_json", "command_json", "progress_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def _get_run(self, run_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM training_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Training run not found: {run_id}")
        return self._row_to_run(row)

    def get_run(self, run_id: str, event_after: int = 0) -> dict[str, Any]:
        run = self._get_run(run_id)
        log_path = Path(run["log_path"])
        if self.run_root != log_path.resolve() and self.run_root not in log_path.resolve().parents:
            raise PermissionError("Training log is outside the managed run directory.")
        with self.store.connect() as db:
            events = [
                {**dict(row), "payload": json.loads(row["payload_json"])}
                for row in db.execute(
                    "SELECT event_id,kind,payload_json,created_at FROM training_run_events "
                    "WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT 500",
                    (run_id, max(0, int(event_after))),
                )
            ]
        for event in events:
            event.pop("payload_json", None)
        return {"run": run, "events": events}

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM training_runs WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def _command(bundle: Path, output_dir: Path, run_id: str, world_size: int) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "facade_training_worker.launcher",
            "--nproc",
            str(world_size),
            "--module",
            WORKER_MODULE,
            "--",
            "--protocol-version",
            str(WORKER_PROTOCOL_VERSION),
            "--bundle",
            str(bundle),
            "--output",
            str(output_dir),
            "--run-id",
            run_id,
        )

    def _append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        if kind not in {"status", "progress", "log"}:
            kind = "log"
        safe_payload = json.loads(canonical_json(payload))
        if "message" in safe_payload:
            safe_payload["message"] = str(safe_payload["message"])[:4000]
        now = utc_now()
        run = self._get_run(run_id)
        log_path = Path(run["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "log":
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now} {canonical_json(safe_payload)}\n")
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO training_run_events(run_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
                (run_id, kind, canonical_json(safe_payload), now),
            )
            if kind == "progress":
                db.execute(
                    "UPDATE training_runs SET progress_json=?,updated_at=? WHERE run_id=?",
                    (canonical_json(safe_payload), now, run_id),
                )

    def _set_pid(self, run_id: str, pid: int) -> None:
        with self.store.connect() as db:
            db.execute("UPDATE training_runs SET pid=?,updated_at=? WHERE run_id=?", (pid, utc_now(), run_id))

    def start_job(
        self,
        *,
        job_id: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        verified = self.jobs.verify_bundle(job_id)
        job = verified["job"]
        project = self.store.get_project(job["project_id"])
        if project["state"] != ProjectState.INITIAL_TRAINING_READY:
            raise PermissionError("The project is not waiting for Initial Champion training.")
        if job["status"] not in {"exported", "failed"}:
            raise PermissionError(f"Training job cannot start from status {job['status']}.")
        preflight = self.environment.training_preflight("initial_champion")
        failures = [item["name"] for item in preflight["checks"] if not item["passed"]]
        if not preflight["passed"]:
            raise ValueError("Local GPU preflight failed: " + ", ".join(failures))
        run_id = make_id("training_run")
        output_dir = self.run_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        log_path = output_dir / "worker.log"
        command = self._command(Path(verified["bundle_path"]), output_dir, run_id, preflight["world_size"])
        now = utc_now()
        try:
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                active = db.execute(
                    "SELECT run_id FROM training_runs WHERE job_id=? AND status IN ('queued','running','cancel_requested')",
                    (job_id,),
                ).fetchone()
                if active:
                    raise PermissionError(f"Training job already has an active run: {active['run_id']}")
                db.execute(
                    "INSERT INTO training_runs(run_id,job_id,project_id,status,worker_protocol_version,"
                    "gpu_devices_json,world_size,preflight_json,command_json,progress_json,log_path,"
                    "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id, job_id, job["project_id"], "queued", WORKER_PROTOCOL_VERSION,
                        canonical_json(preflight["gpu_devices"]), preflight["world_size"],
                        canonical_json(preflight), canonical_json(list(command)), "{}", str(log_path), now, now,
                    ),
                )
                db.execute("UPDATE training_jobs SET status='awaiting_result',updated_at=? WHERE job_id=?", (now, job_id))
                audit_sha256 = self.store._append_audit(
                    db,
                    project_id=job["project_id"],
                    batch_id=None,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name="start_initial_champion_training",
                    from_state=job["status"],
                    to_state="awaiting_result",
                    payload={
                        "job_id": job_id,
                        "run_id": run_id,
                        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
                        "gpu_devices": preflight["gpu_devices"],
                        "world_size": preflight["world_size"],
                        "profile_sha256": hashlib.sha256(
                            (json.dumps(verified["profile"], ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
                        ).hexdigest(),
                    },
                )
                queue_item = self.scheduler.enqueue_in_transaction(
                    db,
                    project_id=job["project_id"],
                    pipeline="initial_champion",
                    run_id=run_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        self.scheduler.notify()
        return {
            "run": self._get_run(run_id),
            "queue": queue_item,
            "audit_event_sha256": audit_sha256,
        }

    def execute(self, run_id: str, cancellation: threading.Event) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("UPDATE training_runs SET status='running',started_at=?,updated_at=? WHERE run_id=?", (now, now, run_id))
        self._append_event(run_id, "status", {"status": "running"})
        spec = WorkerLaunchSpec(
            run_id=run_id,
            job_id=run["job_id"],
            bundle_path=Path(self.jobs.get_job(run["job_id"])["bundle_path"]),
            output_dir=Path(run["log_path"]).parent,
            gpu_devices=tuple(run["gpu_devices"]),
            world_size=run["world_size"],
            command=tuple(run["command"]),
        )
        try:
            result_path = self.runner.run(
                spec,
                lambda kind, payload: self._append_event(run_id, kind, payload),
                cancellation,
                lambda pid: self._set_pid(run_id, pid),
            )
            if cancellation.is_set():
                raise TrainingCancelled("Training was cancelled by the engineer.")
            verified = self._verify_and_store_result(run_id, result_path)
            finished = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE training_runs SET status='result_verified',result_bundle_path=?,"
                    "result_bundle_sha256=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                    (verified["path"], verified["sha256"], finished, finished, run_id),
                )
                db.execute("UPDATE training_jobs SET status='result_verified',updated_at=? WHERE job_id=?", (finished, run["job_id"]))
                self.store._append_audit(
                    db,
                    project_id=run["project_id"],
                    batch_id=None,
                    actor_type="system",
                    actor_id="local_gpu_worker",
                    tool_name="verify_initial_training_result",
                    from_state="awaiting_result",
                    to_state="result_verified",
                    payload={
                        "job_id": run["job_id"], "run_id": run_id,
                        "result_bundle_sha256": verified["sha256"],
                        "selected_seed": verified["manifest"]["selected_seed"],
                    },
                )
            self._append_event(run_id, "status", {"status": "result_verified"})
        except TrainingCancelled as error:
            self._finish_failure(run_id, "cancelled", str(error), "cancel_initial_champion_training")
        except Exception as error:
            self._finish_failure(run_id, "failed", str(error), "fail_initial_champion_training")

    def status(self, run_id: str) -> str:
        return str(self._get_run(run_id)["status"])

    def cancel_queued(self, db: Any, run_id: str, actor_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,status FROM training_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Training run not found: {run_id}")
        if row["status"] != "queued":
            raise PermissionError(f"Training run cannot be queue-cancelled from status {row['status']}.")
        message = "Queued Initial Champion training was cancelled before GPU execution."
        db.execute(
            "UPDATE training_runs SET status='cancelled',error_message=?,finished_at=?,updated_at=? WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute(
            "UPDATE training_jobs SET status='exported',updated_at=? WHERE job_id=?",
            (now, row["job_id"]),
        )
        self.store._append_audit(
            db,
            project_id=row["project_id"],
            batch_id=None,
            actor_type="human",
            actor_id=actor_id,
            tool_name="cancel_queued_initial_champion_training",
            from_state="queued",
            to_state="cancelled",
            payload={"run_id": run_id, "job_id": row["job_id"]},
        )

    def interrupt_abandoned(self, db: Any, run_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,status FROM training_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] in TERMINAL_RUN_STATUSES:
            return
        message = "Agent restarted while Initial Champion training held the GPU lease."
        db.execute(
            "UPDATE training_runs SET status='interrupted',error_message=?,finished_at=?,updated_at=?,pid=NULL "
            "WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute(
            "UPDATE training_jobs SET status='failed',updated_at=? WHERE job_id=?",
            (now, row["job_id"]),
        )
        self.store._append_audit(
            db,
            project_id=row["project_id"],
            batch_id=None,
            actor_type="system",
            actor_id="global_gpu_scheduler",
            tool_name="interrupt_initial_training",
            from_state=row["status"],
            to_state="interrupted",
            payload={"run_id": run_id, "job_id": row["job_id"], "reason": message},
        )

    def _finish_failure(self, run_id: str, status: str, message: str, action: str) -> None:
        run = self._get_run(run_id)
        finished = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE training_runs SET status=?,error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                (status, message[:4000], finished, finished, run_id),
            )
            db.execute("UPDATE training_jobs SET status='failed',updated_at=? WHERE job_id=?", (finished, run["job_id"]))
            self.store._append_audit(
                db,
                project_id=run["project_id"],
                batch_id=None,
                actor_type="system",
                actor_id="local_gpu_worker",
                tool_name=action,
                from_state=run["status"],
                to_state=status,
                payload={"job_id": run["job_id"], "run_id": run_id, "error": message[:1000]},
            )
        self._append_event(run_id, "status", {"status": status, "error": classify_worker_error(message)})

    def cancel_run(self, run_id: str, actor_id: str = "local_engineer") -> dict[str, Any]:
        run = self._get_run(run_id)
        if run["status"] not in {"queued", "running"}:
            raise PermissionError(f"Training run cannot be cancelled from status {run['status']}.")
        with self.store.connect() as db:
            queue_row = db.execute(
                "SELECT queue_id FROM gpu_job_queue WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if queue_row is None:
            raise RuntimeError("Training run is not attached to the global GPU queue.")
        queue_item = self.scheduler.cancel(queue_row["queue_id"], actor_id)
        return {"run": self._get_run(run_id), "queue": queue_item}

    def wait(self, run_id: str, timeout: float = 10) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        run = self._get_run(run_id)
        queue_terminal = {"completed", "failed", "cancelled", "interrupted"}
        while time.monotonic() < deadline:
            with self.store.connect() as db:
                queue_row = db.execute(
                    "SELECT status FROM gpu_job_queue WHERE run_id=?", (run_id,)
                ).fetchone()
            if run["status"] in TERMINAL_RUN_STATUSES and (
                queue_row is None or queue_row["status"] in queue_terminal
            ):
                break
            time.sleep(0.01)
            run = self._get_run(run_id)
        return run

    def shutdown(self, timeout: float = 10) -> None:
        if self._owns_scheduler:
            self.scheduler.shutdown(timeout)

    @staticmethod
    def _safe_zip_names(archive: zipfile.ZipFile) -> list[str]:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 100:
            raise ValueError("Result bundle has duplicate or excessive entries.")
        total_size = 0
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Result bundle contains an unsafe path.")
            info = archive.getinfo(name)
            if info.file_size > 8 * 1024**3:
                raise ValueError("A result artifact exceeds the 8 GiB safety limit.")
            total_size += info.file_size
        if total_size > 32 * 1024**3:
            raise ValueError("Result bundle exceeds the 32 GiB uncompressed safety limit.")
        return names

    @staticmethod
    def _zip_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
        digest = hashlib.sha256()
        with archive.open(name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_result_bundle(self, run_id: str, path: Path) -> dict[str, Any]:
        run = self._get_run(run_id)
        job_info = self.jobs.verify_bundle(run["job_id"])
        job = job_info["job"]
        resolved = path.resolve()
        output_dir = Path(run["log_path"]).parent.resolve()
        if output_dir != resolved and output_dir not in resolved.parents:
            raise PermissionError("GPU worker returned a result outside its managed run directory.")
        if not resolved.is_file():
            raise FileNotFoundError("GPU worker did not produce result_bundle.zip.")
        with zipfile.ZipFile(resolved) as archive:
            names = self._safe_zip_names(archive)
            required = {"result_manifest.json", "checksums.json"}
            if not required.issubset(names):
                raise ValueError("Result bundle is missing its manifest or checksums.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Result checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError("Result bundle contains an invalid SHA-256 value.")
                if self._zip_member_sha256(archive, name) != expected:
                    raise ValueError(f"Result checksum failed: {name}")
            manifest = json.loads(archive.read("result_manifest.json"))
            if manifest.get("schema_version") != 1 or manifest.get("worker_protocol_version") != WORKER_PROTOCOL_VERSION:
                raise ValueError("Unsupported result or worker protocol version.")
            profile_sha = hashlib.sha256(
                (json.dumps(job_info["profile"], ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            ).hexdigest()
            identities = (
                manifest.get("job_id") == run["job_id"],
                manifest.get("run_id") == run_id,
                manifest.get("content_fingerprint") == job["content_fingerprint"],
                manifest.get("profile_sha256") == profile_sha,
            )
            if not all(identities):
                raise ValueError("Result manifest does not match the immutable training job.")
            expected_seeds = job_info["profile"]["seeds"]
            seed_results = manifest.get("seed_results")
            if not isinstance(seed_results, list) or [item.get("seed") for item in seed_results] != expected_seeds:
                raise ValueError("Result must contain the three fixed seeds in preset order.")
            primary_metric = job_info["profile"].get("primary_selection_metric")
            scores: list[tuple[float, int]] = []
            weight_files: set[str] = set()
            for item in seed_results:
                seed = item["seed"]
                weight_file = item.get("weights_file")
                weight_sha = item.get("weights_sha256")
                if not isinstance(weight_file, str) or not weight_file.startswith("weights/") or weight_file not in checksums:
                    raise ValueError(f"Seed {seed} has an invalid weights file.")
                if weight_file in weight_files:
                    raise ValueError("Every seed must have a distinct weights file.")
                weight_files.add(weight_file)
                if weight_sha != checksums[weight_file]:
                    raise ValueError(f"Seed {seed} weights SHA-256 does not match checksums.json.")
                value = item.get("metrics", {}).get(primary_metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"Seed {seed} is missing the finite primary selection metric.")
                scores.append((float(value), int(seed)))
            expected_selected = sorted(scores, key=lambda value: (-value[0], value[1]))[0][1]
            if manifest.get("selected_seed") != expected_selected:
                raise ValueError("Selected seed does not follow the locked development-metric rule.")
            classes = job_info["manifest"].get("class_count")
            with zipfile.ZipFile(Path(job["bundle_path"])) as job_archive:
                class_ids = [item["class_id"] for item in json.loads(job_archive.read("classes.json"))]
            thresholds = manifest.get("thresholds")
            if not isinstance(thresholds, dict) or set(thresholds) != set(class_ids) or len(class_ids) != classes:
                raise ValueError("Result thresholds do not match the frozen class taxonomy.")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1 for value in thresholds.values()):
                raise ValueError("Every class threshold must be a number from 0 to 1.")
            aggregate = manifest.get("aggregate_metrics")
            if not isinstance(aggregate, dict) or not aggregate:
                raise ValueError("Result bundle must contain aggregate metrics from the real worker.")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in aggregate.values()):
                raise ValueError("Aggregate metrics must contain only finite numeric values.")
        return {"manifest": manifest, "checksums": checksums, "source": resolved}

    def _verify_and_store_result(self, run_id: str, path: Path) -> dict[str, Any]:
        validated = self._validate_result_bundle(run_id, path)
        run = self._get_run(run_id)
        destination_dir = self.result_root / run["job_id"] / run_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / "result_bundle.zip"
        temporary = destination_dir / ".result_bundle.tmp"
        try:
            shutil.copyfile(validated["source"], temporary)
            expected = sha256_file(validated["source"])
            if sha256_file(temporary) != expected:
                raise ValueError("Result bundle changed while entering the artifact store.")
            os.replace(temporary, destination)
            destination.chmod(0o444)
        except Exception:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
        return {"path": str(destination), "sha256": expected, "manifest": validated["manifest"]}

    def register_initial_champion(
        self,
        *,
        run_id: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        with self._registration_lock:
            return self._register_initial_champion(
                run_id=run_id, actor_type=actor_type, actor_id=actor_id
            )

    def _register_initial_champion(
        self,
        *,
        run_id: str,
        actor_type: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can register the Initial Champion.")
        run = self._get_run(run_id)
        if run["status"] != "result_verified":
            raise PermissionError("Only a checksum-verified training result can be registered.")
        job = self.jobs.get_job(run["job_id"])
        with self.store.connect() as db:
            existing = db.execute(
                "SELECT * FROM model_versions WHERE source_training_job_id=? AND role='champion'",
                (run["job_id"],),
            ).fetchone()
        if existing:
            return {"model": self.get_model(existing["model_id"]), "idempotent": True}
        project = self.store.get_project(run["project_id"])
        if project["state"] != ProjectState.INITIAL_TRAINING_READY or job["status"] != "result_verified":
            raise PermissionError("Project and job states do not permit Initial Champion registration.")
        result_path = Path(run["result_bundle_path"])
        if sha256_file(result_path) != run["result_bundle_sha256"]:
            raise ValueError("Verified result bundle changed before registration.")
        validated = self._validate_stored_result(run_id, result_path)
        manifest = validated["manifest"]
        selected = next(item for item in manifest["seed_results"] if item["seed"] == manifest["selected_seed"])
        model_id = f"champion_{run['result_bundle_sha256'][:16]}"
        source_root = self.root / "artifact_store" / "model_registration_sources"
        source_root.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        registration_manifest = {
            "schema_version": 1,
            "project_id": run["project_id"],
            "model_id": model_id,
            "parent_model_id": None,
            "source_batch_id": None,
            "role": "champion",
            "taxonomy_sha256": self.model_storage.taxonomy_sha256(run["project_id"]),
            "thresholds": manifest["thresholds"],
            "training_profile_id": job["training_profile_id"],
            "source_training_job_id": run["job_id"],
            "source_training_run_id": run_id,
            "result_bundle_sha256": run["result_bundle_sha256"],
            "metrics": manifest["aggregate_metrics"],
            "selected_seed": manifest["selected_seed"],
            "registered_at": now,
        }
        with tempfile.TemporaryDirectory(dir=source_root) as temporary_dir:
            source_checkpoint = Path(temporary_dir) / Path(selected["weights_file"]).name
            with zipfile.ZipFile(result_path) as archive, source_checkpoint.open("wb") as handle:
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
                    db.execute(
                        "INSERT INTO model_versions(model_id,project_id,parent_model_id,source_batch_id,role,"
                        "checkpoint_path,checkpoint_relpath,checkpoint_sha256,manifest_relpath,manifest_sha256,"
                        "thresholds_json,training_profile_id,created_at,source_training_job_id,"
                        "source_training_run_id,result_bundle_sha256,metrics_json) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            model_id, run["project_id"], None, None, "champion", str(checkpoint),
                            published["checkpoint_relpath"], selected["weights_sha256"],
                            published["manifest_relpath"], published["manifest_sha256"],
                            canonical_json(manifest["thresholds"]), job["training_profile_id"], now,
                            run["job_id"], run_id, run["result_bundle_sha256"],
                            canonical_json(manifest["aggregate_metrics"]),
                        ),
                    )
                    db.execute(
                        "INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?) "
                        "ON CONFLICT(project_id) DO UPDATE SET model_id=excluded.model_id,updated_at=excluded.updated_at",
                        (run["project_id"], model_id, now),
                    )
                    db.execute("UPDATE training_jobs SET status='registered',updated_at=? WHERE job_id=?", (now, run["job_id"]))
                    db.execute(
                        "UPDATE projects SET state=?,model_storage_locked_at=COALESCE(model_storage_locked_at,?),"
                        "updated_at=? WHERE project_id=?",
                        (ProjectState.CHAMPION_READY, now, now, run["project_id"]),
                    )
                    db.execute(
                        "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (make_id("artifact"), run["project_id"], None, "initial_champion_checkpoint", str(checkpoint),
                         selected["weights_sha256"], published["checkpoint_size_bytes"], now),
                    )
                    audit_sha256 = self.store._append_audit(
                        db,
                        project_id=run["project_id"],
                        batch_id=None,
                        actor_type=actor_type,
                        actor_id=actor_id,
                        tool_name="register_initial_champion",
                        from_state=ProjectState.INITIAL_TRAINING_READY,
                        to_state=ProjectState.CHAMPION_READY,
                        payload={
                            "job_id": run["job_id"], "run_id": run_id, "model_id": model_id,
                            "selected_seed": manifest["selected_seed"],
                            "checkpoint_sha256": selected["weights_sha256"],
                            "manifest_sha256": published["manifest_sha256"],
                            "result_bundle_sha256": run["result_bundle_sha256"],
                        },
                    )
                prepared.commit()
        return {"model": self.get_model(model_id), "audit_event_sha256": audit_sha256, "idempotent": False}

    def _validate_stored_result(self, run_id: str, path: Path) -> dict[str, Any]:
        """Validate a result already moved from the run directory without relaxing content checks."""
        run = self._get_run(run_id)
        original = run["result_bundle_path"]
        expected_root = self.result_root.resolve()
        resolved = path.resolve()
        if expected_root != resolved and expected_root not in resolved.parents:
            raise PermissionError("Verified result is outside the managed artifact store.")
        # The full semantic checks already passed before the immutable move. Recheck every byte and manifest.
        with zipfile.ZipFile(resolved) as archive:
            names = self._safe_zip_names(archive)
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Stored result checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if self._zip_member_sha256(archive, name) != expected:
                    raise ValueError(f"Stored result checksum failed: {name}")
            manifest = json.loads(archive.read("result_manifest.json"))
        if manifest.get("run_id") != run_id or original != str(path):
            raise ValueError("Stored result identity changed before registration.")
        return {"manifest": manifest, "checksums": checksums}

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM model_versions WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(f"Model not found: {model_id}")
        item = dict(row)
        item["thresholds"] = json.loads(item.pop("thresholds_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item

    def list_models(self, project_id: str) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM model_versions WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["thresholds"] = json.loads(item.pop("thresholds_json"))
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result


def register_local_training_tools(registry: ToolRegistry, service: LocalTrainingWorkerService) -> None:
    registry.register(
        ToolDefinition(
            "cancel_initial_champion_training",
            "Safely interrupt the attached local GPU worker after explicit engineer confirmation.",
            {
                "type": "object",
                "required": ["run_id"],
                "properties": {"run_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
            lambda args, ctx: service.cancel_run(run_id=args["run_id"], actor_id=ctx["actor_id"]),
        )
    )
    registry.register(
        ToolDefinition(
            "start_initial_champion_training",
            "After engineer confirmation, run the immutable Initial Champion preset on the selected local NVIDIA GPUs. No preset fields are accepted from the caller.",
            {
                "type": "object",
                "required": ["job_id"],
                "properties": {"job_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True),
            lambda args, ctx: service.start_job(
                job_id=args["job_id"], actor_type=ctx["actor_type"], actor_id=ctx["actor_id"]
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "register_initial_champion",
            "Register a checksum-verified three-seed result as the first Champion after a separate engineer confirmation.",
            {
                "type": "object",
                "required": ["run_id"],
                "properties": {"run_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
            lambda args, ctx: service.register_initial_champion(
                run_id=args["run_id"], actor_type=ctx["actor_type"], actor_id=ctx["actor_id"]
            ),
        )
    )
