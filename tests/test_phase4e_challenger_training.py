from __future__ import annotations

import hashlib
import json
import time
import unittest
import zipfile
from pathlib import Path

from facade_agent.application.challenger_training import (
    ChallengerTrainingService,
    TrainingCancelled,
    register_challenger_training_tools,
)
from facade_agent.application.environment import EnvironmentManager
from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.application.model_storage import ModelStorageService
from facade_agent.tools import build_phase1_registry
from test_phase3d_local_training_worker import ready_inventory
import test_phase4d_challenger_jobs


class ChallengerResultRunner:
    """Test substitute only; it emits a protocol-complete, non-model result."""

    def __init__(self, invalid_selection: bool = False) -> None:
        self.invalid_selection = invalid_selection
        self.spec = None

    def run(self, spec, emit, cancel_event, set_pid):
        self.spec = spec
        set_pid(5252)
        emit("progress", {"seed": 20260921, "seed_index": 1, "seed_count": 3, "epoch": 1, "max_epochs": 5, "percent": 10})
        with zipfile.ZipFile(spec.bundle_path) as source:
            job = json.loads(source.read("job_manifest.json"))
            profile_bytes = source.read("training_profile.json")
            profile = json.loads(profile_bytes)
            classes = json.loads(source.read("classes.json"))
            draw_hashes = {
                str(seed): hashlib.sha256(source.read(f"training_draws/seed_{seed}.jsonl")).hexdigest()
                for seed in profile["seeds"]
            }
        scores = [0.70, 0.76, 0.72]
        files: dict[str, bytes] = {}
        seed_results = []
        for seed, score in zip(profile["seeds"], scores):
            name = f"weights/seed-{seed}.pt"
            content = f"test-substitute-challenger-weight-{seed}".encode()
            files[name] = content
            seed_results.append({
                "seed": seed,
                "weights_file": name,
                "weights_sha256": hashlib.sha256(content).hexdigest(),
                "metrics": {"development_macro_map": score},
            })
        manifest = {
            "schema_version": 1,
            "worker_protocol_version": 1,
            "job_id": spec.job_id,
            "run_id": spec.run_id,
            "content_fingerprint": job["content_fingerprint"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "parent_champion_model_id": job["parent_champion_model_id"],
            "epochs": profile["epochs"],
            "test_read": False,
            "training_draws_sha256": draw_hashes,
            "seed_results": seed_results,
            "selected_seed": profile["seeds"][0] if self.invalid_selection else profile["seeds"][1],
            "thresholds": {item["class_id"]: 0.5 for item in classes},
            "aggregate_metrics": {"development_macro_map_mean": sum(scores) / len(scores)},
        }
        files["result_manifest.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
        files["checksums.json"] = (json.dumps(checksums, sort_keys=True) + "\n").encode()
        output = spec.output_dir / "result_bundle.zip"
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        emit("log", {"message": "test substitute emitted all three locked Challenger seeds"})
        return output


class BlockingRunner:
    def run(self, spec, emit, cancel_event, set_pid):
        set_pid(5253)
        emit("progress", {"seed": 20260921, "seed_index": 1, "seed_count": 3, "epoch": 1, "max_epochs": 5, "percent": 10})
        while not cancel_event.wait(0.01):
            pass
        raise TrainingCancelled("Challenger training was cancelled by the engineer.")


class Phase4EChallengerTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = test_phase4d_challenger_jobs.Phase4DChallengerJobTests(
            "test_fixed_three_pool_bundle_is_reproducible_and_verified"
        )
        self.fixture.setUp()
        self.store = self.fixture.store
        self.root = self.fixture.root
        self.project_id = self.fixture.project_id
        self.batch_id = self.fixture.batch_id
        self.jobs = self.fixture.service
        self.job = self.jobs.create_job(batch_id=self.batch_id, actor_type="human", actor_id="engineer")["job"]
        self.model_root = self.root / "registered-challengers"
        self.model_root.mkdir()
        with self.store.connect() as db:
            db.execute(
                "UPDATE projects SET model_storage_root=? WHERE project_id=?",
                (str(self.model_root.resolve()), self.project_id),
            )
        self.model_storage = ModelStorageService(self.store, self.root)
        self.environment = EnvironmentManager(
            self.store, self.root, ready_inventory,
            smoke_tester=lambda devices: {"passed": devices == [0], "devices": devices},
        )
        self.services: list[ChallengerTrainingService] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.shutdown()
        self.fixture.tearDown()

    def service(self, runner) -> ChallengerTrainingService:
        service = ChallengerTrainingService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            runner,
            self.model_storage,
        )
        self.services.append(service)
        return service

    def active_model_id(self) -> str:
        with self.store.connect() as db:
            return db.execute("SELECT model_id FROM active_models WHERE project_id=?", (self.project_id,)).fetchone()["model_id"]

    def test_single_gpu_result_verification_and_separate_candidate_registration(self) -> None:
        runner = ChallengerResultRunner()
        service = self.service(runner)
        started = service.start_job(self.job["job_id"])
        run = service.wait(started["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "result_verified")
        self.assertEqual(run["gpu_devices"], [0])
        self.assertEqual(run["world_size"], 1)
        self.assertIn("facade_training_worker.launcher", runner.spec.command)
        self.assertEqual(runner.spec.command[runner.spec.command.index("--nproc") + 1], "1")
        self.assertEqual(run["progress"]["seed_count"], 3)
        self.assertEqual(hashlib.sha256(Path(run["result_bundle_path"]).read_bytes()).hexdigest(), run["result_bundle_sha256"])

        registry = build_phase1_registry(self.store)
        register_challenger_training_tools(registry, service)
        with self.assertRaises(PermissionError):
            registry.execute("register_challenger", {"run_id": run["run_id"]}, actor_type="llm", actor_id="model", confirmed=True)
        registered = registry.execute("register_challenger", {"run_id": run["run_id"]}, actor_type="human", actor_id="engineer", confirmed=True)
        model = registered["model"]
        self.assertEqual(model["role"], "challenger")
        self.assertEqual(model["parent_model_id"], "champion")
        self.assertEqual(self.active_model_id(), "champion")
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], "CHALLENGER_TRAINED")
        self.assertEqual(hashlib.sha256(Path(model["checkpoint_path"]).read_bytes()).hexdigest(), model["checkpoint_sha256"])
        manifest_path = self.model_storage.resolve_model_file(
            self.project_id, model["manifest_relpath"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["parent_model_id"], "champion")
        self.assertEqual(manifest["source_batch_id"], self.batch_id)
        self.assertEqual(manifest["thresholds"], model["thresholds"])
        self.assertEqual(manifest["checkpoint_sha256"], model["checkpoint_sha256"])
        before = {
            item.relative_to(self.model_root).as_posix(): item.read_bytes()
            for item in self.model_root.rglob("*")
            if item.is_file()
        }
        repeated = registry.execute(
            "register_challenger",
            {"run_id": run["run_id"]},
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
        self.assertEqual(history["counts"]["challenger_runs"], 1)
        self.assertEqual(history["counts"]["models"], 2)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_invalid_seed_selection_fails_without_changing_champion(self) -> None:
        service = self.service(ChallengerResultRunner(invalid_selection=True))
        run = service.wait(service.start_job(self.job["job_id"])["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "failed")
        self.assertIn("selected seed", run["error_message"].lower())
        self.assertEqual(self.active_model_id(), "champion")
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], "FAILURE_REVIEW_COMPLETED")
        self.assertEqual(list(self.model_root.iterdir()), [])

    def test_cancel_is_human_only_and_keeps_parent_active(self) -> None:
        service = self.service(BlockingRunner())
        run_id = service.start_job(self.job["job_id"])["run"]["run_id"]
        deadline = time.time() + 2
        while service.get_run(run_id)["run"]["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        registry = build_phase1_registry(self.store)
        register_challenger_training_tools(registry, service)
        with self.assertRaises(PermissionError):
            registry.execute("cancel_challenger_training", {"run_id": run_id}, actor_type="llm", actor_id="model", confirmed=True)
        registry.execute("cancel_challenger_training", {"run_id": run_id}, actor_type="human", actor_id="engineer", confirmed=True)
        self.assertEqual(service.wait(run_id, timeout=5)["status"], "cancelled")

    def test_challenger_start_waits_for_global_gpu_lease_and_queued_cancel_is_retryable(self) -> None:
        runner = ChallengerResultRunner()
        scheduler = GpuJobScheduler(self.store)
        service = ChallengerTrainingService(
            self.store,
            self.jobs,
            self.environment,
            self.root,
            runner,
            self.model_storage,
            scheduler=scheduler,
        )

        started = service.start_job(self.job["job_id"])

        self.assertIsNone(runner.spec)
        self.assertEqual(started["queue"]["pipeline"], "challenger_update")
        self.assertEqual(started["queue"]["status"], "queued")
        cancelled = service.cancel_run(started["run"]["run_id"])
        self.assertEqual(cancelled["run"]["status"], "cancelled")
        self.assertEqual(cancelled["queue"]["status"], "cancelled")
        self.assertEqual(self.jobs.get_job(self.job["job_id"])["status"], "exported")
        self.assertEqual(self.active_model_id(), "champion")
        self.assertEqual(self.active_model_id(), "champion")


if __name__ == "__main__":
    unittest.main()
