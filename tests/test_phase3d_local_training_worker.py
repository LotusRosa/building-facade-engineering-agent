from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from facade_agent.application.environment import EnvironmentManager
from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.application.model_storage import ModelStorageService
from facade_agent.application.local_training import (
    LocalTrainingWorkerService,
    SubprocessWorkerRunner,
    TrainingCancelled,
    register_local_training_tools,
)
from facade_agent.application.training_jobs import InitialChampionJobService
from facade_agent.storage import Store
from facade_agent.tools import build_phase1_registry


def ready_inventory(_: Path) -> dict:
    return {
        "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
        "python": {"version": "3.12", "executable": "python", "supported": True},
        "cpu": {"logical_cores": 16},
        "memory": {"total_bytes": 64 * 1024**3},
        "disk": {"root": "test", "free_bytes": 500 * 1024**3, "total_bytes": 1000 * 1024**3},
        "nvidia": {
            "available": True,
            "gpus": [
                {"index": 0, "name": "RTX 4090", "memory_total_mib": 24564, "driver_version": "test"},
                {"index": 1, "name": "RTX 4090", "memory_total_mib": 24564, "driver_version": "test"},
            ],
            "error": None,
        },
        "torch": {"installed": True, "version": "test", "cuda_available": True, "cuda_version": "test", "device_count": 2, "error": None},
        "training_runtime": {"module": "facade_training_worker.initial_champion", "available": True, "error": None},
        "capabilities": {"interface_ready": True, "supported_platform": True, "local_gpu_training_ready": True},
        "warnings": [],
    }


class ResultRunner:
    def __init__(self, invalid_selection: bool = False, invalid_checksum: bool = False) -> None:
        self.invalid_selection = invalid_selection
        self.invalid_checksum = invalid_checksum
        self.spec = None

    def run(self, spec, emit, cancel_event, set_pid):
        self.spec = spec
        set_pid(4242)
        emit("progress", {"seed": 20260921, "seed_index": 1, "seed_count": 3, "epoch": 1, "max_epochs": 40, "percent": 1})
        with zipfile.ZipFile(spec.bundle_path) as source:
            job_manifest = json.loads(source.read("job_manifest.json"))
            profile_bytes = source.read("training_profile.json")
            profile = json.loads(profile_bytes)
            classes = json.loads(source.read("classes.json"))
        seed_results = []
        files = {}
        scores = [0.71, 0.74, 0.69]
        for seed, score in zip(profile["seeds"], scores):
            name = f"weights/seed-{seed}.pt"
            content = f"real-worker-test-checkpoint-{seed}".encode()
            files[name] = content
            seed_results.append(
                {
                    "seed": seed,
                    "weights_file": name,
                    "weights_sha256": hashlib.sha256(content).hexdigest(),
                    "metrics": {profile["primary_selection_metric"]: score},
                }
            )
        manifest = {
            "schema_version": 1,
            "worker_protocol_version": 1,
            "job_id": spec.job_id,
            "run_id": spec.run_id,
            "content_fingerprint": job_manifest["content_fingerprint"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "seed_results": seed_results,
            "selected_seed": profile["seeds"][0] if self.invalid_selection else profile["seeds"][1],
            "thresholds": {item["class_id"]: 0.5 for item in classes},
            "aggregate_metrics": {"development_macro_map_mean": sum(scores) / len(scores)},
        }
        files["result_manifest.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
        if self.invalid_checksum:
            files[f"weights/seed-{profile['seeds'][0]}.pt"] += b"tampered-after-checksum"
        files["checksums.json"] = (json.dumps(checksums, sort_keys=True) + "\n").encode()
        result = spec.output_dir / "result_bundle.zip"
        with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        emit("log", {"message": "three real seed runs completed by test substitute"})
        return result


class BlockingRunner:
    def run(self, spec, emit, cancel_event, set_pid):
        set_pid(4243)
        emit("progress", {"seed": 20260921, "epoch": 1, "max_epochs": 40, "percent": 1})
        while not cancel_event.wait(0.01):
            pass
        raise TrainingCancelled("Training was cancelled by the engineer.")


class StatusObservingCancellation:
    def __init__(self, store: Store, queue_id: str) -> None:
        self.store = store
        self.queue_id = queue_id
        self.status_on_set = None

    def set(self) -> None:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT cancel_requested FROM gpu_job_queue WHERE queue_id=?",
                (self.queue_id,),
            ).fetchone()
        self.status_on_set = bool(row["cancel_requested"])


class Phase3DLocalTrainingWorkerTests(unittest.TestCase):
    def test_subprocess_runner_exposes_portable_package_root(self) -> None:
        environment = SubprocessWorkerRunner._environment((0, 2))
        package_root = str(
            Path(__file__).resolve().parents[1]
        )
        self.assertIn(package_root, environment["PYTHONPATH"].split(os.pathsep))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,2")
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        project_files = self.root / "projects"
        project_files.mkdir()
        self.store = Store(self.root / "agent.sqlite3")
        self.model_root = self.root / "project-models"
        self.model_root.mkdir()
        project = self.store.create_project(
            project_name="Facade",
            class_names=["crack", "spalling"],
            model_storage_root=str(self.model_root.resolve()),
        )["project"]
        self.project_id = project["project_id"]
        self.store.transition_project(project_id=self.project_id, action="confirm_taxonomy", confirmed=True)
        dataset = self.store.create_dataset(project_id=self.project_id, name="initial", role="initial_training")["dataset"]
        self.dataset_id = dataset["dataset_id"]
        classes = self.store.get_project(self.project_id)["classes"]
        for index in range(3):
            content = b"image" + bytes([index]) * 32
            path = project_files / f"image-{index}.png"
            path.write_bytes(content)
            image = self.store.register_image(
                dataset_id=self.dataset_id,
                filename=path.name,
                stored_path=str(path),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                mime_type="image/png",
                width=16,
                height=16,
                health_status="ok",
            )
            self.store.save_annotation(image_id=image["image_id"], class_ids=[classes[index % 2]["class_id"]], no_defect=False)
        self.store.validate_dataset(dataset_id=self.dataset_id)
        self.jobs = InitialChampionJobService(
            self.store,
            project_files,
            self.root / "exports" / "training_jobs",
            Path(__file__).resolve().parents[1] / "facade_agent" / "protocols" / "initial_champion_profile_v1.json",
        )
        self.job = self.jobs.create_job(project_id=self.project_id, dataset_id=self.dataset_id)["job"]
        self.environment = EnvironmentManager(
            self.store,
            self.root,
            ready_inventory,
            smoke_tester=lambda devices: {"passed": devices == [0], "devices": devices},
        )
        self.model_storage = ModelStorageService(self.store, self.root)
        self.services: list[LocalTrainingWorkerService] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.shutdown()
        self.temp.cleanup()

    def service(self, runner) -> LocalTrainingWorkerService:
        service = LocalTrainingWorkerService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            runner,
            self.model_storage,
        )
        self.services.append(service)
        return service

    def test_single_gpu_schedule_three_seed_verification_and_human_registration(self) -> None:
        runner = ResultRunner()
        service = self.service(runner)
        started = service.start_job(job_id=self.job["job_id"])
        run = service.wait(started["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "result_verified")
        self.assertEqual(run["gpu_devices"], [0])
        self.assertEqual(run["world_size"], 1)
        self.assertIn("facade_training_worker.launcher", runner.spec.command)
        self.assertEqual(runner.spec.command[runner.spec.command.index("--nproc") + 1], "1")
        self.assertEqual(run["progress"]["seed_count"], 3)
        self.assertEqual(hashlib.sha256(Path(run["result_bundle_path"]).read_bytes()).hexdigest(), run["result_bundle_sha256"])

        registry = build_phase1_registry(self.store)
        register_local_training_tools(registry, service)
        arguments = {"run_id": run["run_id"]}
        with self.assertRaises(PermissionError):
            registry.execute("register_initial_champion", arguments, actor_type="llm", actor_id="model", confirmed=True)
        registered = registry.execute(
            "register_initial_champion", arguments, actor_type="human", actor_id="engineer", confirmed=True
        )
        model = registered["model"]
        self.assertEqual(self.store.get_project(self.project_id)["state"], "CHAMPION_READY")
        self.assertEqual(hashlib.sha256(Path(model["checkpoint_path"]).read_bytes()).hexdigest(), model["checkpoint_sha256"])
        self.assertEqual(Path(model["checkpoint_relpath"]).name, "checkpoint.pt")
        manifest_path = self.model_storage.resolve_model_file(
            self.project_id, model["manifest_relpath"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_id"], model["model_id"])
        self.assertEqual(manifest["parent_model_id"], None)
        self.assertEqual(manifest["thresholds"], model["thresholds"])
        self.assertEqual(manifest["training_profile_id"], model["training_profile_id"])
        self.assertEqual(manifest["checkpoint_sha256"], model["checkpoint_sha256"])
        self.assertTrue(manifest["taxonomy_sha256"])
        self.assertIsNotNone(self.store.get_project(self.project_id)["model_storage_locked_at"])
        before = {
            item.relative_to(self.model_root).as_posix(): item.read_bytes()
            for item in self.model_root.rglob("*")
            if item.is_file()
        }
        repeated = registry.execute(
            "register_initial_champion",
            arguments,
            actor_type="human",
            actor_id="engineer",
            confirmed=True,
        )
        after = {
            item.relative_to(self.model_root).as_posix(): item.read_bytes()
            for item in self.model_root.rglob("*")
            if item.is_file()
        }
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(before, after)
        history = self.store.get_project_history(self.project_id)
        self.assertEqual(history["counts"]["training_runs"], 1)
        self.assertEqual(history["counts"]["models"], 1)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_invalid_seed_selection_fails_without_registering_a_model(self) -> None:
        service = self.service(ResultRunner(invalid_selection=True))
        started = service.start_job(job_id=self.job["job_id"])
        run = service.wait(started["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "failed")
        self.assertIn("Selected seed", run["error_message"])
        self.assertEqual(service.list_models(self.project_id), [])
        self.assertEqual(self.store.get_project(self.project_id)["state"], "INITIAL_TRAINING_READY")
        self.assertEqual(list(self.model_root.iterdir()), [])

    def test_tampered_weight_checksum_is_rejected(self) -> None:
        service = self.service(ResultRunner(invalid_checksum=True))
        started = service.start_job(job_id=self.job["job_id"])
        run = service.wait(started["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "failed")
        self.assertIn("Result checksum failed", run["error_message"])
        self.assertEqual(service.list_models(self.project_id), [])
        self.assertEqual(list(self.model_root.iterdir()), [])

    def test_database_failure_removes_published_model(self) -> None:
        service = self.service(ResultRunner())
        run = service.wait(
            service.start_job(job_id=self.job["job_id"])["run"]["run_id"],
            timeout=5,
        )
        self.assertEqual(run["status"], "result_verified")
        with patch.object(
            self.store,
            "_append_audit",
            side_effect=RuntimeError("simulated database transaction failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "transaction failure"):
                service.register_initial_champion(run_id=run["run_id"])
        self.assertEqual(list(self.model_root.iterdir()), [])
        self.assertEqual(service.list_models(self.project_id), [])
        self.assertEqual(
            self.store.get_project(self.project_id)["state"],
            "INITIAL_TRAINING_READY",
        )

    def test_engineer_cancellation_is_terminal_and_retry_safe(self) -> None:
        service = self.service(BlockingRunner())
        started = service.start_job(job_id=self.job["job_id"])
        run_id = started["run"]["run_id"]
        deadline = time.time() + 2
        while service.get_run(run_id)["run"]["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        registry = build_phase1_registry(self.store)
        register_local_training_tools(registry, service)
        with self.assertRaises(PermissionError):
            registry.execute(
                "cancel_initial_champion_training", {"run_id": run_id},
                actor_type="llm", actor_id="model", confirmed=True,
            )
        registry.execute(
            "cancel_initial_champion_training", {"run_id": run_id},
            actor_type="human", actor_id="engineer", confirmed=True,
        )
        run = service.wait(run_id, timeout=5)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(self.jobs.get_job(self.job["job_id"])["status"], "failed")
        self.assertEqual(self.store.get_project(self.project_id)["state"], "INITIAL_TRAINING_READY")

    def test_cancellation_request_is_persisted_before_worker_is_signalled(self) -> None:
        service = self.service(BlockingRunner())
        run_id = service.start_job(job_id=self.job["job_id"])["run"]["run_id"]
        deadline = time.time() + 2
        while service.get_run(run_id)["run"]["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)

        with self.store.connect() as db:
            queue_id = db.execute(
                "SELECT queue_id FROM gpu_job_queue WHERE run_id=?", (run_id,)
            ).fetchone()["queue_id"]
        with service.scheduler._lock:
            worker_cancellation = service.scheduler._current_cancellation
            observer = StatusObservingCancellation(self.store, queue_id)
            service.scheduler._current_cancellation = observer
        try:
            service.cancel_run(run_id)
            self.assertTrue(observer.status_on_set)
        finally:
            if worker_cancellation is not None:
                worker_cancellation.set()
            service.wait(run_id, timeout=5)

    def test_failed_preflight_never_launches_worker(self) -> None:
        environment = EnvironmentManager(
            self.store,
            self.root,
            lambda _: {**ready_inventory(self.root), "training_runtime": {"available": False, "error": "missing"}},
            smoke_tester=lambda devices: {"passed": True},
        )
        service = LocalTrainingWorkerService(self.store, self.jobs, environment, self.root, ResultRunner())
        self.services.append(service)
        with self.assertRaisesRegex(ValueError, "locked_training_runtime"):
            service.start_job(job_id=self.job["job_id"])
        self.assertEqual(service.list_runs(self.project_id), [])

    def test_startup_marks_orphaned_run_interrupted(self) -> None:
        now = "2026-09-14T00:00:00+00:00"
        run_id = "training_run_orphaned"
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO training_runs(run_id,job_id,project_id,status,worker_protocol_version,"
                "gpu_devices_json,world_size,preflight_json,command_json,progress_json,log_path,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, self.job["job_id"], self.project_id, "running", 1, "[0,1]", 2,
                    "{}", "[]", "{}", str(self.root / "artifact_store" / "training_runs" / run_id / "worker.log"), now, now,
                ),
            )
            db.execute("UPDATE training_jobs SET status='awaiting_result' WHERE job_id=?", (self.job["job_id"],))
            db.execute(
                "INSERT INTO gpu_job_queue(queue_id,project_id,pipeline,run_id,status,cancel_requested,"
                "lease_owner,lease_acquired_at,enqueued_at,started_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "gpu_queue_orphaned", self.project_id, "initial_champion", run_id,
                    "running", 0, "dead-agent", now, now, now, now,
                ),
            )
        service = self.service(ResultRunner())
        self.assertEqual(service.get_run(run_id)["run"]["status"], "interrupted")
        self.assertEqual(self.jobs.get_job(self.job["job_id"])["status"], "failed")
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_initial_training_start_enqueues_without_direct_thread(self) -> None:
        runner = ResultRunner()
        scheduler = GpuJobScheduler(self.store)
        service = LocalTrainingWorkerService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            runner,
            self.model_storage,
            scheduler=scheduler,
        )

        started = service.start_job(job_id=self.job["job_id"])

        self.assertIsNone(runner.spec)
        self.assertEqual(started["run"]["status"], "queued")
        self.assertEqual(started["queue"]["status"], "queued")
        self.assertEqual(started["queue"]["pipeline"], "initial_champion")
        self.assertEqual(started["queue"]["run_id"], started["run"]["run_id"])

    def test_initial_training_queued_cancel_restores_retryable_job(self) -> None:
        scheduler = GpuJobScheduler(self.store)
        service = LocalTrainingWorkerService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            ResultRunner(),
            self.model_storage,
            scheduler=scheduler,
        )
        started = service.start_job(job_id=self.job["job_id"])

        cancelled = service.cancel_run(started["run"]["run_id"])

        self.assertEqual(cancelled["run"]["status"], "cancelled")
        self.assertEqual(cancelled["queue"]["status"], "cancelled")
        self.assertEqual(self.jobs.get_job(self.job["job_id"])["status"], "exported")

    def test_initial_training_handler_updates_existing_result_states(self) -> None:
        runner = ResultRunner()
        scheduler = GpuJobScheduler(self.store)
        service = LocalTrainingWorkerService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            runner,
            self.model_storage,
            scheduler=scheduler,
        )
        started = service.start_job(job_id=self.job["job_id"])
        run_id = started["run"]["run_id"]

        service.execute(run_id, threading.Event())

        self.assertIsNotNone(runner.spec)
        self.assertEqual(service.status(run_id), "result_verified")
        self.assertEqual(self.jobs.get_job(self.job["job_id"])["status"], "result_verified")


if __name__ == "__main__":
    unittest.main()
