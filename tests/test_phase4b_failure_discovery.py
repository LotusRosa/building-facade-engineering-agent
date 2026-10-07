from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from facade_agent.application.environment import EnvironmentManager
from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.application.failure_discovery import (
    FailureDiscoveryService,
    register_failure_discovery_tools,
)
from facade_agent.application.local_training import TrainingCancelled
from facade_agent.core.states import BatchState, ProjectState
from facade_agent.storage import Store, utc_now
from facade_agent.tools import build_phase1_registry


def ready_inventory(_: Path) -> dict:
    return {
        "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
        "python": {"version": "3.12", "executable": "python", "supported": True},
        "cpu": {"logical_cores": 16}, "memory": {"total_bytes": 64 * 1024**3},
        "disk": {"root": "test", "free_bytes": 500 * 1024**3, "total_bytes": 1000 * 1024**3},
        "nvidia": {"available": True, "gpus": [
            {"index": 0, "name": "RTX 4090", "memory_total_mib": 24564, "driver_version": "test"},
            {"index": 1, "name": "RTX 4090", "memory_total_mib": 24564, "driver_version": "test"},
        ], "error": None},
        "torch": {"installed": True, "version": "test", "cuda_available": True, "cuda_version": "test", "device_count": 2, "error": None},
        "training_runtime": {"module": "test", "available": True, "error": None},
        "capabilities": {"interface_ready": True, "supported_platform": True, "local_gpu_training_ready": True},
        "warnings": [],
    }


def csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


class DiscoveryResultRunner:
    def __init__(self, wrong_threshold: bool = False) -> None:
        self.wrong_threshold = wrong_threshold
        self.spec = None

    def run(self, spec, emit, cancel_event, set_pid):
        self.spec = spec
        set_pid(9001)
        emit("progress", {"stage": "champion_inference", "images_completed": 6, "images_total": 6, "percent": 50})
        with zipfile.ZipFile(spec.bundle_path) as source:
            manifest = json.loads(source.read("job_manifest.json"))
            champion = json.loads(source.read("champion.json"))
            classes = json.loads(source.read("classes.json"))
            labels = [json.loads(line) for line in source.read("labels.jsonl").decode().splitlines()]
        prediction_rows = []
        failures = 0
        for label_row in labels:
            for index, class_row in enumerate(classes):
                class_id = class_row["class_id"]
                label = int(class_id in label_row["class_ids"])
                frozen = float(champion["thresholds"][class_id])
                score = 0.9 if index == 0 else 0.1
                predicted = int(score >= frozen)
                error = "TP" if label and predicted else "FN" if label else "FP" if predicted else "TN"
                if error in {"FP", "FN"}:
                    failures += 1
                prediction_rows.append({
                    "image_id": label_row["image_id"], "class_id": class_id, "label": label,
                    "score": score, "threshold": frozen, "predicted": predicted, "error_type": error,
                    "severity": abs(score - frozen) if error in {"FP", "FN"} else 0.0,
                })
        first_class = classes[0]["class_id"]
        member_rows = [
            {"slice_key": "crack-fp-001", "image_id": item["image_id"], "membership_rank": index}
            for index, item in enumerate(labels, start=1)
        ]
        slices = [{
            "slice_key": "crack-fp-001", "class_id": first_class, "error_type": "FP", "support": 6,
            "consensus_score": 0.8333333333333334, "representative_image_id": labels[0]["image_id"],
        }]
        thresholds = dict(champion["thresholds"])
        if self.wrong_threshold:
            thresholds[first_class] = 0.4
        result_manifest = {
            "schema_version": 1, "worker_protocol_version": 1, "job_id": spec.job_id, "run_id": spec.run_id,
            "content_fingerprint": manifest["content_fingerprint"], "champion_model_id": champion["model_id"],
            "champion_checkpoint_sha256": champion["checkpoint_sha256"],
            "discovery_profile_id": "champion_failure_discovery_convnext_consensus_v1",
            "thresholds": thresholds, "thresholds_retuned": False, "discovery_passes": 1,
            "prediction_rows": len(prediction_rows), "failure_records": failures, "failure_slices": 1,
        }
        files = {
            "result_manifest.json": (json.dumps(result_manifest, sort_keys=True) + "\n").encode(),
            "predictions.csv": csv_bytes(
                ["image_id", "class_id", "label", "score", "threshold", "predicted", "error_type", "severity"],
                prediction_rows,
            ),
            "failure_slices.json": (json.dumps(slices, sort_keys=True) + "\n").encode(),
            "failure_slice_members.csv": csv_bytes(["slice_key", "image_id", "membership_rank"], member_rows),
        }
        files["checksums.json"] = (json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in files.items()}, sort_keys=True) + "\n").encode()
        result = spec.output_dir / "result_bundle.zip"
        with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        emit("progress", {"stage": "consensus_clustering", "percent": 100})
        return result


class BlockingRunner:
    def run(self, spec, emit, cancel_event, set_pid):
        set_pid(9002)
        emit("progress", {"stage": "champion_inference", "percent": 1})
        while not cancel_event.wait(0.01):
            pass
        raise TrainingCancelled("Champion screening was cancelled by the engineer.")


class Phase4BFailureDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.projects = self.root / "projects"
        self.projects.mkdir()
        self.store = Store(self.root / "agent.sqlite3")
        project = self.store.create_project(project_name="Facade", class_names=["crack", "spalling"])["project"]
        self.project_id = project["project_id"]
        self.classes = project["classes"]
        self.store.transition_project(project_id=self.project_id, action="confirm_taxonomy", confirmed=True)
        model_dir = self.root / "models" / self.project_id / "fixture"
        model_dir.mkdir(parents=True)
        checkpoint = model_dir / "champion.pt"
        checkpoint.write_bytes(b"verified-champion-checkpoint")
        now = utc_now()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,checkpoint_sha256," 
                "thresholds_json,training_profile_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                ("champion-fixture", self.project_id, "champion", str(checkpoint),
                 hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 json.dumps({item["class_id"]: 0.5 for item in self.classes}), "fixture-v1", now),
            )
            db.execute("INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)", (self.project_id, "champion-fixture", now))
            db.execute("UPDATE projects SET state=?,updated_at=? WHERE project_id=?", (ProjectState.CHAMPION_READY, now, self.project_id))
        batch = self.store.create_maintenance_batch(project_id=self.project_id, batch_name="round-a")["maintenance_batch"]
        self.batch_id = batch["batch_id"]
        dataset = self.store.create_maintenance_dataset(batch_id=self.batch_id, name="round-a-images")["dataset"]
        for index in range(6):
            content = f"image-{index}".encode()
            path = self.projects / self.project_id / f"image-{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            image = self.store.register_image(
                dataset_id=dataset["dataset_id"], filename=path.name, stored_path=str(path),
                sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content), mime_type="image/png",
                width=16, height=16, health_status="ok",
            )
            self.store.save_annotation(image_id=image["image_id"], class_ids=[], no_defect=True)
        self.store.validate_dataset(dataset_id=dataset["dataset_id"])
        self.store.freeze_maintenance_batch(batch_id=self.batch_id, confirmed=True)
        self.environment = EnvironmentManager(
            self.store, self.root, ready_inventory,
            smoke_tester=lambda devices: {"passed": devices == [0], "devices": devices},
        )
        self.profile = Path(__file__).resolve().parents[1] / "facade_agent" / "protocols" / "failure_discovery_profile_v1.json"
        self.services: list[FailureDiscoveryService] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.shutdown()
        self.temp.cleanup()

    def service(self, runner) -> FailureDiscoveryService:
        service = FailureDiscoveryService(self.store, self.environment, self.projects, self.root, self.profile, runner)
        self.services.append(service)
        return service

    def test_frozen_threshold_inference_and_consensus_slices_are_verified(self) -> None:
        runner = DiscoveryResultRunner()
        service = self.service(runner)
        registry = build_phase1_registry(self.store)
        register_failure_discovery_tools(registry, service)
        with self.assertRaises(PermissionError):
            registry.execute(
                "start_champion_failure_discovery", {"batch_id": self.batch_id},
                actor_type="llm", actor_id="model", confirmed=True,
            )
        started = registry.execute(
            "start_champion_failure_discovery", {"batch_id": self.batch_id},
            actor_type="human", actor_id="engineer", confirmed=True,
        )
        run = service.wait(started["run"]["run_id"], 5)
        self.assertEqual(run["status"], "result_verified")
        self.assertEqual(run["gpu_devices"], [0])
        self.assertIn("facade_training_worker.launcher", runner.spec.command)
        self.assertEqual(runner.spec.command[runner.spec.command.index("--nproc") + 1], "1")
        self.assertEqual(run["summary"]["failure_records"], 6)
        self.assertFalse(run["summary"]["thresholds_retuned"])
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], BatchState.FAILURE_DISCOVERY_COMPLETED)
        slices = service.list_slices(self.batch_id)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0]["support"], 6)
        self.assertEqual(len(slices[0]["members"]), 6)
        history = self.store.get_project_history(self.project_id)
        self.assertEqual(history["counts"]["screening_runs"], 1)
        self.assertEqual(history["counts"]["failure_slices"], 1)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_discovery_accepts_champion_from_project_model_storage_folder(self) -> None:
        external_root = self.root / "user-selected-model-storage"
        external_checkpoint = external_root / "champion-fixture" / "checkpoint.pt"
        external_checkpoint.parent.mkdir(parents=True)
        with self.store.connect() as db:
            model = db.execute(
                "SELECT checkpoint_path FROM model_versions WHERE model_id=?",
                ("champion-fixture",),
            ).fetchone()
            Path(model["checkpoint_path"]).replace(external_checkpoint)
            db.execute(
                "UPDATE projects SET model_storage_root=?,model_storage_locked_at=? WHERE project_id=?",
                (str(external_root), utc_now(), self.project_id),
            )
            db.execute(
                "UPDATE model_versions SET checkpoint_path=? WHERE model_id=?",
                (str(external_checkpoint), "champion-fixture"),
            )

        service = self.service(DiscoveryResultRunner())
        started = service.start_batch(self.batch_id)
        run = service.wait(started["run"]["run_id"], 5)

        self.assertEqual(run["status"], "result_verified")
        self.assertEqual(
            self.store.get_maintenance_batch(self.batch_id)["state"],
            BatchState.FAILURE_DISCOVERY_COMPLETED,
        )

    def test_changed_threshold_is_rejected_and_batch_stays_frozen(self) -> None:
        service = self.service(DiscoveryResultRunner(wrong_threshold=True))
        started = service.start_batch(self.batch_id)
        run = service.wait(started["run"]["run_id"], 5)
        self.assertEqual(run["status"], "failed")
        self.assertIn("thresholds", run["error_message"])
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], BatchState.MAINTENANCE_BATCH_FROZEN)
        self.assertEqual(service.list_slices(self.batch_id), [])

    def test_cancellation_keeps_frozen_batch_retryable(self) -> None:
        service = self.service(BlockingRunner())
        started = service.start_batch(self.batch_id)
        run_id = started["run"]["run_id"]
        deadline = time.time() + 2
        while service.get_run(run_id)["run"]["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        service.cancel_run(run_id)
        run = service.wait(run_id, 5)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], BatchState.MAINTENANCE_BATCH_FROZEN)

    def test_discovery_start_waits_for_global_gpu_lease_and_queued_cancel_is_retryable(self) -> None:
        runner = DiscoveryResultRunner()
        scheduler = GpuJobScheduler(self.store)
        service = FailureDiscoveryService(
            self.store,
            self.environment,
            self.projects,
            self.root,
            self.profile,
            runner,
            scheduler=scheduler,
        )

        started = service.start_batch(self.batch_id)

        self.assertIsNone(runner.spec)
        self.assertEqual(started["queue"]["pipeline"], "champion_failure_discovery")
        self.assertEqual(started["queue"]["status"], "queued")
        cancelled = service.cancel_run(started["run"]["run_id"])
        self.assertEqual(cancelled["run"]["status"], "cancelled")
        self.assertEqual(cancelled["queue"]["status"], "cancelled")
        self.assertEqual(service.get_job(started["run"]["job_id"])["status"], "exported")
        self.assertEqual(
            self.store.get_maintenance_batch(self.batch_id)["state"],
            BatchState.MAINTENANCE_BATCH_FROZEN,
        )


if __name__ == "__main__":
    unittest.main()
