from __future__ import annotations

import hashlib
import json
import unittest
import zipfile
from pathlib import Path

import test_phase4e_challenger_training
from facade_agent.application.environment import EnvironmentManager
from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.application.model_evaluation import ModelEvaluationService, register_model_evaluation_tools
from facade_agent.tools import build_phase1_registry
from facade_training_worker.data import parse_classes, parse_image_records
from test_phase3d_local_training_worker import ready_inventory


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


class EvaluationResultRunner:
    def __init__(self, omit_prediction: bool = False) -> None:
        self.omit_prediction = omit_prediction
        self.spec = None

    def run(self, spec, emit, cancel_event, set_pid):
        self.spec = spec
        set_pid(6262)
        emit("progress", {"stage": "paired_inference", "images_completed": 1, "images_total": 4, "percent": 25})
        with zipfile.ZipFile(spec.bundle_path) as source:
            job = json.loads(source.read("job_manifest.json"))
            profile_bytes = source.read("evaluation_profile.json")
            labels = [json.loads(line) for line in source.read("holdout/labels.jsonl").splitlines()]
            classes = json.loads(source.read("holdout/classes.json"))
            models = json.loads(source.read("models.json"))
        class_ids = [item["class_id"] for item in classes]
        current_gate = job["current_gate"]
        core_safety = "core_safety"
        rows = []
        for index, label in enumerate(labels):
            truth = set(label["class_ids"])
            for role in ("champion", "challenger"):
                probabilities = {class_id: (0.9 if class_id in truth else 0.1) for class_id in class_ids}
                if role == "champion" and label["split"] == current_gate and index == 0:
                    probabilities[next(iter(truth))] = 0.1
                if role == "challenger" and label["split"] == core_safety and index == 2:
                    other = next(class_id for class_id in class_ids if class_id not in truth)
                    probabilities[other] = 0.9
                rows.append({
                    "image_id": label["image_id"],
                    "split": label["split"],
                    "model_id": models[role]["model_id"],
                    "probabilities": probabilities,
                })
        if self.omit_prediction:
            rows.pop()
        predictions = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
        files = {"predictions.jsonl": predictions}
        manifest = {
            "schema_version": 1,
            "worker_protocol_version": 1,
            "job_id": spec.job_id,
            "run_id": spec.run_id,
            "content_fingerprint": job["content_fingerprint"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "champion_model_id": models["champion"]["model_id"],
            "challenger_model_id": models["challenger"]["model_id"],
            "predictions_file": "predictions.jsonl",
            "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
            "test_read": False,
        }
        files["result_manifest.json"] = json_bytes(manifest)
        files["checksums.json"] = json_bytes({name: hashlib.sha256(content).hexdigest() for name, content in files.items()})
        result = spec.output_dir / "result_bundle.zip"
        with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        emit("log", {"message": "test substitute completed paired holdout inference"})
        return result


class Phase4FModelEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = test_phase4e_challenger_training.Phase4EChallengerTrainingTests(
            "test_single_gpu_result_verification_and_separate_candidate_registration"
        )
        self.fixture.setUp()
        challenger_service = self.fixture.service(test_phase4e_challenger_training.ChallengerResultRunner())
        run = challenger_service.wait(
            challenger_service.start_job(self.fixture.job["job_id"])["run"]["run_id"], timeout=5
        )
        challenger_service.register_challenger(run["run_id"], "human", "engineer")
        self.store = self.fixture.store
        self.root = self.fixture.root
        self.project_id = self.fixture.project_id
        self.batch_id = self.fixture.batch_id
        self.inbox = self.root / "inbox"
        self.inbox.mkdir(exist_ok=True)
        self.environment = EnvironmentManager(
            self.store, self.root, ready_inventory,
            smoke_tester=lambda devices: {"passed": devices == [0], "devices": devices},
        )
        self.profile = Path(__file__).resolve().parents[1] / "facade_agent" / "protocols" / "champion_challenger_evaluation_profile_v2.json"
        self.snapshot = self._make_snapshot("independent-evidence.zip")
        self.services: list[ModelEvaluationService] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.shutdown()
        self.fixture.tearDown()

    def _make_snapshot(
        self,
        name: str,
        overlap: bool = False,
        *,
        include_addition: bool = True,
        content_prefix: str = "round-a",
        first_override: tuple[str, bytes] | None = None,
    ) -> Path:
        project = self.store.get_project(self.project_id)
        classes = [{"class_id": item["class_id"], "display_name": item["display_name"]} for item in project["classes"]]
        payloads = {
            "evaluation_manifest.json": json_bytes({
                "schema_version": 3, "purpose": "champion_challenger_selection",
                "project_id": self.project_id, "batch_id": self.batch_id,
                "provided_splits": ["current_gate", "core_safety_addition"] if include_addition else ["current_gate"],
                "label_version_id": f"labels-{content_prefix}",
                "taxonomy_sha256": hashlib.sha256(json_bytes(classes)).hexdigest(),
                "final_test": False,
            }),
            "classes.json": json_bytes(classes),
        }
        rows = []
        splits = ("current_gate", "core_safety_addition") if include_addition else ("current_gate",)
        for split_index, split in enumerate(splits):
            for class_index, item in enumerate(classes):
                content = f"{content_prefix}-{split}-{item['class_id']}".encode()
                if overlap and split_index == 0 and class_index == 0:
                    with self.store.connect() as db:
                        stored = db.execute(
                            "SELECT i.stored_path FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id "
                            "WHERE d.project_id=? LIMIT 1", (self.project_id,),
                        ).fetchone()["stored_path"]
                    content = Path(stored).read_bytes()
                image_id = f"{content_prefix}-{split_index}-{class_index}"
                if first_override is not None and split_index == 0 and class_index == 0:
                    image_id, content = first_override
                image_file = f"images/{split_index}-{class_index}.png"
                payloads[image_file] = content
                rows.append({
                    "image_id": image_id, "split": split,
                    "image_file": image_file, "image_sha256": hashlib.sha256(content).hexdigest(),
                    "no_defect": False, "class_ids": [item["class_id"]],
                })
        payloads["labels.jsonl"] = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
        checksums = {path: hashlib.sha256(content).hexdigest() for path, content in payloads.items()}
        payloads["checksums.json"] = json_bytes(checksums)
        destination = self.inbox / name
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, content in sorted(payloads.items()):
                archive.writestr(path, content)
        return destination

    def service(self, runner) -> ModelEvaluationService:
        service = ModelEvaluationService(self.store, self.environment, self.root, self.profile, runner)
        self.services.append(service)
        return service

    def _prepare_next_round(self, batch_id: str, name: str) -> None:
        with self.store.connect() as db:
            champion = db.execute(
                "SELECT m.* FROM active_models a JOIN model_versions m ON m.model_id=a.model_id "
                "WHERE a.project_id=?",
                (self.project_id,),
            ).fetchone()
            db.execute(
                "INSERT INTO maintenance_batches(batch_id,project_id,name,state,created_at,updated_at) "
                "VALUES(?,?,?,'CHALLENGER_TRAINED','2026-09-20T00:00:00+00:00','2026-09-20T00:00:00+00:00')",
                (batch_id, self.project_id, name),
            )
            db.execute(
                "UPDATE projects SET active_batch_id=?,state='SCREENING_READY' WHERE project_id=?",
                (batch_id, self.project_id),
            )
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,parent_model_id,source_batch_id,role,checkpoint_path,"
                "checkpoint_sha256,thresholds_json,training_profile_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (f"challenger_{batch_id}", self.project_id, champion["model_id"], batch_id, "challenger",
                 champion["checkpoint_path"], champion["checkpoint_sha256"], champion["thresholds_json"],
                 "test_recursive_round", "2026-09-20T00:00:00+00:00"),
            )
        self.batch_id = batch_id

    @staticmethod
    def _first_job_image(job: dict) -> tuple[str, bytes]:
        with zipfile.ZipFile(job["bundle_path"]) as archive:
            labels = [json.loads(line) for line in archive.read("holdout/labels.jsonl").splitlines()]
            row = labels[0]
            return row["image_id"], archive.read(f"holdout/{row['image_file']}")

    def test_independent_paired_evidence_is_recomputed_and_opens_human_decision(self) -> None:
        runner = EvaluationResultRunner()
        service = self.service(runner)
        created = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")
        self.assertEqual(created["job"]["split_counts"], {"core_safety": 2, "current_gate": 2})
        self.assertEqual(
            [(cohort["origin_role"], cohort["status"]) for cohort in service.list_cohorts(self.project_id)],
            [("core_safety_seed", "pending"), ("current_gate", "pending")],
        )
        run = service.wait(service.start_job(created["job"]["job_id"], "human", "engineer")["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "result_verified")
        self.assertIn("facade_training_worker.launcher", runner.spec.command)
        self.assertEqual(runner.spec.command[runner.spec.command.index("--nproc") + 1], "1")
        evidence = service.get_evidence(self.batch_id)
        self.assertIsNotNone(evidence)
        self.assertIsNone(evidence["metrics"]["automatic_decision"])
        self.assertIsNone(evidence["metrics"]["default_decision"])
        self.assertEqual(evidence["metrics"]["decision_options"], ["retain", "promote"])
        new = evidence["metrics"]["current_gate"]
        historical = evidence["metrics"]["core_safety"]
        self.assertGreater(new["delta"]["macro_f1"], 0)
        self.assertLess(historical["delta"]["macro_f1"], 0)
        self.assertIn("macro_map", new["champion"])
        self.assertEqual(len(evidence["metrics"]["cohorts"]), 2)
        worst = evidence["metrics"]["worst_core_safety_cohort"]
        self.assertEqual(worst["origin_role"], "core_safety_seed")
        self.assertEqual(worst["evidence"], historical)
        self.assertFalse(evidence["metrics"]["test_read"])
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], "DECISION_PENDING")
        self.assertEqual(self.fixture.active_model_id(), "champion")
        self.assertIsNone(self.store.get_maintenance_batch(self.batch_id)["formal_decision"])
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_frozen_evaluation_bundle_matches_worker_image_record_contract(self) -> None:
        service = self.service(EvaluationResultRunner())
        job = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]

        with zipfile.ZipFile(job["bundle_path"]) as archive:
            classes = parse_classes(json.loads(archive.read("holdout/classes.json")))
            labels = [json.loads(line) for line in archive.read("holdout/labels.jsonl").splitlines()]
            adjusted = [
                {**row, "file": f"holdout/{row.get('file') or row['image_file']}"}
                for row in labels
            ]
            records = parse_image_records(adjusted, classes, archive.namelist())

        self.assertEqual(len(records), len(labels))
        self.assertTrue(all(record.sha256 for record in records))
        self.assertEqual({row["split"] for row in labels}, {"current_gate", "core_safety"})
        self.assertTrue(
            all(
                row.get("cohort_id") and row.get("source_round") and row.get("origin_role")
                for row in labels
            )
        )

    def test_snapshot_overlap_is_rejected_before_job_creation(self) -> None:
        overlapping = self._make_snapshot("overlap.zip", overlap=True)
        service = self.service(EvaluationResultRunner())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            service.create_job(self.batch_id, overlapping.name, "human", "engineer")
        self.assertEqual(service.list_jobs(self.project_id), [])

    def test_pending_cohort_overlap_is_rejected(self) -> None:
        service = self.service(EvaluationResultRunner())
        first = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        duplicate = self._first_job_image(first)
        self._prepare_next_round("batch_round_pending", "Pending overlap")
        overlapping = self._make_snapshot(
            "pending-overlap.zip", content_prefix="pending-overlap", first_override=duplicate
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            service.create_job(self.batch_id, overlapping.name, "human", "engineer")

    def test_active_cohort_overlap_is_rejected(self) -> None:
        service = self.service(EvaluationResultRunner())
        first = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        duplicate = self._first_job_image(first)
        with self.store.connect() as db:
            db.execute(
                "UPDATE evaluation_cohorts SET status='active',activated_at='2026-09-30T00:00:00+00:00' "
                "WHERE source_batch_id=?",
                (self.batch_id,),
            )
        self._prepare_next_round("batch_round_active", "Active overlap")
        overlapping = self._make_snapshot(
            "active-overlap.zip", content_prefix="active-overlap", first_override=duplicate
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            service.create_job(self.batch_id, overlapping.name, "human", "engineer")

    def test_final_test_inventory_overlap_is_rejected(self) -> None:
        content = b"sealed-final-test-image"
        digest = hashlib.sha256(content).hexdigest()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO final_test_image_inventory(project_id,image_id,content_sha256,"
                "source_artifact_id,created_at) VALUES(?,?,?,?,?)",
                (self.project_id, "final-image", digest, "sealed-final-v1", "2026-09-30T00:00:00+00:00"),
            )
        overlapping = self._make_snapshot(
            "final-overlap.zip", content_prefix="final-overlap", first_override=("different-id", content)
        )
        service = self.service(EvaluationResultRunner())
        with self.assertRaisesRegex(ValueError, "protected evaluation inventory"):
            service.create_job(self.batch_id, overlapping.name, "human", "engineer")

    def test_active_cohorts_plus_current_addition_form_one_core_safety_split(self) -> None:
        service = self.service(EvaluationResultRunner())
        first = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        with self.store.connect() as db:
            db.execute(
                "UPDATE evaluation_cohorts SET status='active',activated_at='2026-09-30T00:00:00+00:00' "
                "WHERE source_batch_id=?",
                (self.batch_id,),
            )

        self._prepare_next_round("batch_round_b", "Round B")
        round_b = self._make_snapshot(
            "round-b-evaluation.zip", content_prefix="round-b"
        )
        second = service.create_job(self.batch_id, round_b.name, "human", "engineer")["job"]
        self.assertEqual(
            second["split_counts"],
            {"current_gate": 2, "core_safety": 6},
        )
        with zipfile.ZipFile(second["bundle_path"]) as archive:
            labels = [json.loads(line) for line in archive.read("holdout/labels.jsonl").splitlines()]
        self.assertEqual({row["split"] for row in labels}, {"current_gate", "core_safety"})
        self.assertEqual(sum(row["split"] == "core_safety" for row in labels), 6)
        self.assertEqual(len({row["cohort_id"] for row in labels}), 4)
        self.assertEqual(
            [(cohort["origin_role"], cohort["status"]) for cohort in service.list_cohorts(self.project_id)],
            [
                ("core_safety_seed", "active"),
                ("core_safety_addition", "pending"),
                ("current_gate", "active"),
                ("current_gate", "pending"),
            ],
        )

    def test_incomplete_worker_predictions_fail_without_evidence_or_decision(self) -> None:
        service = self.service(EvaluationResultRunner(omit_prediction=True))
        job = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        run = service.wait(service.start_job(job["job_id"], "human", "engineer")["run"]["run_id"], timeout=5)
        self.assertEqual(run["status"], "failed")
        self.assertIn("exactly once", run["error_message"])
        self.assertIsNone(service.get_evidence(self.batch_id))
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], "CHALLENGER_TRAINED")

    def test_llm_cannot_freeze_or_start_evaluation(self) -> None:
        service = self.service(EvaluationResultRunner())
        registry = build_phase1_registry(self.store)
        register_model_evaluation_tools(registry, service)
        with self.assertRaises(PermissionError):
            registry.execute(
                "create_model_evaluation_job", {"batch_id": self.batch_id, "inbox_filename": self.snapshot.name},
                actor_type="llm", actor_id="model", confirmed=True,
            )

    def test_confirmed_promote_switches_active_model_and_closes_round_once(self) -> None:
        service = self.service(EvaluationResultRunner())
        job = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        service.wait(service.start_job(job["job_id"], "human", "engineer")["run"]["run_id"], timeout=5)
        evidence = service.get_evidence(self.batch_id)
        decision = self.store.record_batch_decision(
            batch_id=self.batch_id, decision="promote", reason="Independent evidence accepted by engineer.",
            confirmed=True, actor_type="human", actor_id="engineer",
        )
        self.assertEqual(decision["maintenance_batch"]["state"], "DEPLOYED")
        self.assertEqual(self.fixture.active_model_id(), evidence["challenger_model_id"])
        self.assertEqual(decision["round_completion"]["decision"], "promote")
        self.assertEqual(decision["round_completion"]["cumulative_core_safety_images"], 4)
        self.assertEqual(len(decision["activated_cohorts"]), 2)
        self.assertTrue(all(row["status"] == "active" for row in service.list_cohorts(self.project_id)))
        self.assertEqual(self.store.get_round_completion(self.batch_id), decision["round_completion"])
        with self.store.connect() as db:
            old_role = db.execute("SELECT role FROM model_versions WHERE model_id=?", (evidence["champion_model_id"],)).fetchone()["role"]
            new_role = db.execute("SELECT role FROM model_versions WHERE model_id=?", (evidence["challenger_model_id"],)).fetchone()["role"]
        self.assertEqual(old_role, "archived")
        self.assertEqual(new_role, "champion")
        with self.assertRaises(PermissionError):
            self.store.record_batch_decision(
                batch_id=self.batch_id, decision="retain", reason="Cannot overwrite.",
                confirmed=True, actor_type="human", actor_id="engineer",
            )

    def test_confirmed_retain_activates_both_cohorts_without_switching_champion(self) -> None:
        service = self.service(EvaluationResultRunner())
        job = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        service.wait(
            service.start_job(job["job_id"], "human", "engineer")["run"]["run_id"],
            timeout=5,
        )
        champion_before = self.fixture.active_model_id()
        decision = self.store.record_batch_decision(
            batch_id=self.batch_id,
            decision="retain",
            reason="Core Safety regression outweighs the Current Gate gain.",
            confirmed=True,
            actor_type="human",
            actor_id="engineer",
        )
        self.assertEqual(decision["maintenance_batch"]["state"], "HELD")
        self.assertEqual(self.fixture.active_model_id(), champion_before)
        self.assertEqual(decision["round_completion"]["decision"], "retain")
        self.assertEqual(
            {row["origin_role"] for row in decision["activated_cohorts"]},
            {"core_safety_seed", "current_gate"},
        )
        self.assertEqual(decision["round_completion"]["cumulative_core_safety_images"], 4)

    def test_failed_evaluation_leaves_both_round_cohorts_pending(self) -> None:
        service = self.service(EvaluationResultRunner(omit_prediction=True))
        job = service.create_job(self.batch_id, self.snapshot.name, "human", "engineer")["job"]
        run = service.wait(
            service.start_job(job["job_id"], "human", "engineer")["run"]["run_id"],
            timeout=5,
        )
        self.assertEqual(run["status"], "failed")
        self.assertEqual(
            {row["status"] for row in service.list_cohorts(self.project_id)},
            {"pending"},
        )
        self.assertIsNone(self.store.get_round_completion(self.batch_id))

    def test_evaluation_start_waits_for_global_gpu_lease_and_queued_cancel_is_retryable(self) -> None:
        runner = EvaluationResultRunner()
        scheduler = GpuJobScheduler(self.store)
        service = ModelEvaluationService(
            self.store,
            self.environment,
            self.root,
            self.profile,
            runner,
            scheduler=scheduler,
        )
        job = service.create_job(
            self.batch_id, self.snapshot.name, "human", "engineer"
        )["job"]

        started = service.start_job(job["job_id"], "human", "engineer")

        self.assertIsNone(runner.spec)
        self.assertEqual(
            started["queue"]["pipeline"], "champion_challenger_evaluation"
        )
        self.assertEqual(started["queue"]["status"], "queued")
        cancelled = service.cancel_run(started["run"]["run_id"], "engineer")
        self.assertEqual(cancelled["run"]["status"], "cancelled")
        self.assertEqual(cancelled["queue"]["status"], "cancelled")
        self.assertEqual(service.get_job(job["job_id"])["status"], "exported")
        self.assertEqual(
            {row["status"] for row in service.list_cohorts(self.project_id)},
            {"pending"},
        )


if __name__ == "__main__":
    unittest.main()
