from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from facade_agent.application.challenger_jobs import ChallengerJobService, register_challenger_job_tool
from facade_agent.application.failure_review import FailureReviewService
from facade_agent.core.states import BatchState, ProjectState
from facade_agent.storage import Store, canonical_json, utc_now
from facade_agent.tools import build_phase1_registry


class Phase4DChallengerJobTests(unittest.TestCase):
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
        initial = self.store.create_dataset(
            project_id=self.project_id, name="initial", role="initial_training"
        )["dataset"]
        self._add_images(initial["dataset_id"], "history", 4)
        self.store.validate_dataset(dataset_id=initial["dataset_id"])
        models = self.root / "models" / self.project_id
        models.mkdir(parents=True)
        checkpoint = models / "champion.pt"
        checkpoint.write_bytes(b"verified-active-champion")
        now = utc_now()
        thresholds = {item["class_id"]: 0.5 for item in self.classes}
        with self.store.connect() as db:
            db.execute("UPDATE datasets SET status='frozen' WHERE dataset_id=?", (initial["dataset_id"],))
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,checkpoint_sha256,"
                "thresholds_json,training_profile_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                ("champion", self.project_id, "champion", str(checkpoint),
                 hashlib.sha256(checkpoint.read_bytes()).hexdigest(), canonical_json(thresholds), "initial", now),
            )
            db.execute("INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)", (self.project_id, "champion", now))
            db.execute("UPDATE projects SET state=?,updated_at=? WHERE project_id=?", (ProjectState.CHAMPION_READY, now, self.project_id))
        self.batch_id = self.store.create_maintenance_batch(
            project_id=self.project_id, batch_name="round-a"
        )["maintenance_batch"]["batch_id"]
        current = self.store.create_maintenance_dataset(batch_id=self.batch_id, name="current")["dataset"]
        self.current_ids = self._add_images(current["dataset_id"], "current", 4)
        self.store.validate_dataset(dataset_id=current["dataset_id"])
        self.store.freeze_maintenance_batch(batch_id=self.batch_id, confirmed=True)
        with self.store.connect() as db:
            version = db.execute(
                "SELECT label_version_id FROM label_versions WHERE dataset_id=?", (current["dataset_id"],)
            ).fetchone()
            db.execute(
                "INSERT INTO screening_jobs(job_id,project_id,batch_id,dataset_id,label_version_id,"
                "champion_model_id,status,discovery_profile_id,profile_json,content_fingerprint,bundle_path,"
                "bundle_sha256,bundle_size_bytes,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,'result_verified','fixture','{}','screening-fingerprint','fixture.zip','0',0,?,?)",
                ("screening", self.project_id, self.batch_id, current["dataset_id"],
                 version["label_version_id"], "champion", now, now),
            )
        self.store.transition_batch(batch_id=self.batch_id, action="record_batch_inference", confirmed=False)
        self.store.transition_batch(batch_id=self.batch_id, action="record_failure_discovery", confirmed=False)
        with self.store.connect() as db:
            for number, image_id in enumerate(self.current_ids[:2], start=1):
                slice_id = f"slice-{number}"
                db.execute(
                    "INSERT INTO failure_slices(slice_id,batch_id,class_id,error_type,support,"
                    "clustering_profile_id,status,created_at,source_slice_key,consensus_score,representative_image_id) "
                    "VALUES(?,?,?,?,1,'fixture','pending',?,?,?,?)",
                    (slice_id, self.batch_id, self.classes[number - 1]["class_id"], "FP" if number == 1 else "FN",
                     now, f"source-{number}", 0.9, image_id),
                )
                db.execute(
                    "INSERT INTO failure_slice_members(slice_id,image_id,membership_rank) VALUES(?,?,1)",
                    (slice_id, image_id),
                )
        review = FailureReviewService(self.store)
        for slice_id in ("slice-1", "slice-2"):
            review.save_decision(
                slice_id=slice_id, decision="accept", excluded_image_ids=[],
                actor_type="human", actor_id="engineer",
            )
        review.freeze_review(
            batch_id=self.batch_id, confirmed=True, actor_type="human", actor_id="engineer"
        )
        profile = Path(__file__).resolve().parents[1] / "facade_agent" / "protocols" / "challenger_update_profile_v1.json"
        self.service = ChallengerJobService(
            self.store, self.projects, self.root / "exports" / "challenger_jobs", profile
        )

    def _add_images(self, dataset_id: str, prefix: str, count: int) -> list[str]:
        output = []
        for index in range(count):
            content = f"{prefix}-{index}".encode()
            path = self.projects / self.project_id / dataset_id / f"{prefix}-{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            image = self.store.register_image(
                dataset_id=dataset_id, filename=path.name, stored_path=str(path),
                sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content), mime_type="image/png",
                width=16, height=16, health_status="ok",
            )
            labels = [self.classes[index % len(self.classes)]["class_id"]]
            self.store.save_annotation(image_id=image["image_id"], class_ids=labels, no_defect=False)
            output.append(image["image_id"])
        return output

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fixed_three_pool_bundle_is_reproducible_and_verified(self) -> None:
        created = self.service.create_job(
            batch_id=self.batch_id, actor_type="human", actor_id="engineer"
        )
        self.assertFalse(created["idempotent"])
        job = created["job"]
        self.assertEqual(job["pool_counts"]["failure_member_records_available"], 2)
        self.assertEqual(job["pool_counts"]["remaining_new_images_available"], 2)
        self.assertEqual(job["pool_counts"]["history_replay_images_available"], 4)
        verified = self.service.verify_bundle(job["job_id"])
        self.assertEqual(verified["manifest"]["draws_per_seed_per_epoch"]["total"], 640)
        self.assertFalse(verified["manifest"]["test_read"])
        with zipfile.ZipFile(verified["bundle_path"]) as archive:
            for seed in (20260921, 20260922, 20260923):
                rows = [json.loads(line) for line in archive.read(f"training_draws/seed_{seed}.jsonl").splitlines()]
                self.assertEqual(len(rows), 640)
                self.assertEqual(sum(row["source_pool"] == "expert_confirmed_failure" for row in rows), 160)
                self.assertEqual(sum(row["source_pool"] == "remaining_new" for row in rows), 160)
                self.assertEqual(sum(row["source_pool"] == "history_replay" for row in rows), 320)
        again = self.service.create_job(
            batch_id=self.batch_id, actor_type="human", actor_id="engineer"
        )
        self.assertTrue(again["idempotent"])
        self.assertEqual(again["job"]["bundle_sha256"], job["bundle_sha256"])
        self.assertEqual(self.store.get_maintenance_batch(self.batch_id)["state"], BatchState.FAILURE_REVIEW_COMPLETED)
        self.assertEqual(self.store.get_project_history(self.project_id)["counts"]["challenger_jobs"], 1)
        self.assertTrue(self.store.verify_audit_chain()["valid"])

    def test_challenger_job_accepts_parent_from_project_model_storage_folder(self) -> None:
        external_root = self.root / "user-selected-model-storage"
        external_checkpoint = external_root / "champion" / "checkpoint.pt"
        external_checkpoint.parent.mkdir(parents=True)
        with self.store.connect() as db:
            model = db.execute(
                "SELECT checkpoint_path FROM model_versions WHERE model_id='champion'"
            ).fetchone()
            Path(model["checkpoint_path"]).replace(external_checkpoint)
            db.execute(
                "UPDATE projects SET model_storage_root=?,model_storage_locked_at=? WHERE project_id=?",
                (str(external_root), utc_now(), self.project_id),
            )
            db.execute(
                "UPDATE model_versions SET checkpoint_path=? WHERE model_id='champion'",
                (str(external_checkpoint),),
            )

        created = self.service.create_job(
            batch_id=self.batch_id, actor_type="human", actor_id="engineer"
        )

        self.assertEqual(created["job"]["parent_champion_model_id"], "champion")
        self.assertEqual(
            self.service.verify_bundle(created["job"]["job_id"])["manifest"]["parent_champion_model_id"],
            "champion",
        )

    def test_llm_cannot_create_or_parameterize_challenger_job(self) -> None:
        registry = build_phase1_registry(self.store)
        register_challenger_job_tool(registry, self.service)
        with self.assertRaises(PermissionError):
            registry.execute(
                "create_challenger_training_job", {"batch_id": self.batch_id},
                actor_type="llm", actor_id="model", confirmed=True,
            )
        with self.assertRaises(ValueError):
            registry.execute(
                "create_challenger_training_job", {"batch_id": self.batch_id, "epochs": 20},
                actor_type="human", actor_id="engineer", confirmed=True,
            )

    def test_tampered_bundle_is_rejected(self) -> None:
        job = self.service.create_job(
            batch_id=self.batch_id, actor_type="human", actor_id="engineer"
        )["job"]
        Path(job["bundle_path"]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.service.verify_bundle(job["job_id"])


if __name__ == "__main__":
    unittest.main()
