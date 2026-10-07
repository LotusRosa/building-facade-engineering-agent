from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from facade_agent.application.failure_review import FailureReviewService, register_failure_review_tools
from facade_agent.core.states import BatchState, ProjectState
from facade_agent.storage import Store, utc_now
from facade_agent.tools import build_phase1_registry


class Phase4CFailureReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "agent.sqlite3")
        project = self.store.create_project(project_name="Facade", class_names=["crack"])["project"]
        self.project_id = project["project_id"]
        self.class_id = project["classes"][0]["class_id"]
        self.store.transition_project(project_id=self.project_id, action="confirm_taxonomy", confirmed=True)
        checkpoint = self.root / "champion.pt"
        checkpoint.write_bytes(b"champion")
        now = utc_now()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO model_versions(model_id,project_id,role,checkpoint_path,checkpoint_sha256,"
                "thresholds_json,training_profile_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                ("champion", self.project_id, "champion", str(checkpoint),
                 hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 json.dumps({self.class_id: 0.5}), "fixture", now),
            )
            db.execute("INSERT INTO active_models(project_id,model_id,updated_at) VALUES(?,?,?)", (self.project_id, "champion", now))
            db.execute("UPDATE projects SET state=?,updated_at=? WHERE project_id=?", (ProjectState.CHAMPION_READY, now, self.project_id))
        self.batch_id = self.store.create_maintenance_batch(
            project_id=self.project_id, batch_name="round-a"
        )["maintenance_batch"]["batch_id"]
        dataset = self.store.create_maintenance_dataset(batch_id=self.batch_id, name="round-a-images")["dataset"]
        self.image_ids = []
        for index in range(3):
            content = f"image-{index}".encode()
            path = self.root / f"image-{index}.png"
            path.write_bytes(content)
            image = self.store.register_image(
                dataset_id=dataset["dataset_id"], filename=path.name, stored_path=str(path),
                sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content),
                mime_type="image/png", width=10, height=10, health_status="ok",
            )
            self.image_ids.append(image["image_id"])
            self.store.save_annotation(image_id=image["image_id"], class_ids=[], no_defect=True)
        self.store.validate_dataset(dataset_id=dataset["dataset_id"])
        self.store.freeze_maintenance_batch(batch_id=self.batch_id, confirmed=True)
        self.store.transition_batch(batch_id=self.batch_id, action="record_batch_inference", confirmed=False)
        self.store.transition_batch(batch_id=self.batch_id, action="record_failure_discovery", confirmed=False)
        with self.store.connect() as db:
            for number, error_type in enumerate(("FP", "FN"), start=1):
                slice_id = f"slice-{number}"
                db.execute(
                    "INSERT INTO failure_slices(slice_id,batch_id,class_id,error_type,support,"
                    "clustering_profile_id,status,created_at,source_slice_key,consensus_score,representative_image_id) "
                    "VALUES(?,?,?,?,?,'fixture','pending',?,?,?,?)",
                    (slice_id, self.batch_id, self.class_id, error_type, 3, now,
                     f"fixture-{number}", 0.9, self.image_ids[0]),
                )
                db.executemany(
                    "INSERT INTO failure_slice_members(slice_id,image_id,membership_rank) VALUES(?,?,?)",
                    [(slice_id, image_id, rank) for rank, image_id in enumerate(self.image_ids, start=1)],
                )
        self.service = FailureReviewService(self.store)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_single_expert_review_freezes_immutable_knowledge_version(self) -> None:
        with self.assertRaises(PermissionError):
            self.service.save_decision(
                slice_id="slice-1", decision="accept", excluded_image_ids=[],
                actor_type="llm", actor_id="model",
            )
        with self.assertRaises(ValueError):
            self.service.save_decision(
                slice_id="slice-1", decision="trim", excluded_image_ids=["outside"],
                actor_type="human", actor_id="engineer",
            )
        self.service.save_decision(
            slice_id="slice-1", decision="trim", excluded_image_ids=[self.image_ids[2]],
            actor_type="human", actor_id="engineer",
        )
        with self.assertRaises(PermissionError):
            self.service.freeze_review(
                batch_id=self.batch_id, confirmed=True, actor_type="human", actor_id="engineer"
            )
        self.service.save_decision(
            slice_id="slice-2", decision="accept", excluded_image_ids=[],
            actor_type="human", actor_id="engineer",
        )
        result = self.service.freeze_review(
            batch_id=self.batch_id, confirmed=True, actor_type="human", actor_id="engineer"
        )
        self.assertEqual(result["batch_state"], BatchState.FAILURE_REVIEW_COMPLETED)
        version = result["frozen_version"]
        self.assertEqual(version["eligible_slice_count"], 2)
        self.assertEqual(version["retained_member_records"], 5)
        self.assertEqual(version["retained_unique_images"], 3)
        self.assertEqual(len(version["review_sha256"]), 64)
        self.assertFalse(version["snapshot"]["labels_modified"])
        self.assertFalse(version["snapshot"]["reclustered"])
        self.assertEqual(
            {item["slice_id"]: item["status"] for item in result["slices"]},
            {"slice-1": "trimmed", "slice-2": "accepted"},
        )
        with self.assertRaises(PermissionError):
            self.service.save_decision(
                slice_id="slice-1", decision="accept", excluded_image_ids=[],
                actor_type="human", actor_id="engineer",
            )
        self.assertTrue(self.store.verify_audit_chain()["valid"])
        history = self.store.get_project_history(self.project_id)
        self.assertEqual(history["counts"]["failure_reviews"], 1)

    def test_review_tools_cannot_be_confirmed_by_llm(self) -> None:
        registry = build_phase1_registry(self.store)
        register_failure_review_tools(registry, self.service)
        with self.assertRaises(PermissionError):
            registry.execute(
                "save_failure_slice_review",
                {"slice_id": "slice-1", "decision": "accept", "excluded_image_ids": []},
                actor_type="llm", actor_id="model", confirmed=False,
            )
        with self.assertRaises(PermissionError):
            registry.execute(
                "freeze_failure_slice_review", {"batch_id": self.batch_id},
                actor_type="llm", actor_id="model", confirmed=True,
            )

    def test_all_rejected_is_a_valid_zero_eligible_review(self) -> None:
        for slice_id in ("slice-1", "slice-2"):
            self.service.save_decision(
                slice_id=slice_id, decision="reject", excluded_image_ids=[],
                actor_type="human", actor_id="engineer",
            )
        result = self.service.freeze_review(
            batch_id=self.batch_id, confirmed=True, actor_type="human", actor_id="engineer"
        )
        self.assertEqual(result["frozen_version"]["eligible_slice_count"], 0)
        self.assertEqual(result["frozen_version"]["retained_member_records"], 0)
        with self.assertRaises(PermissionError):
            self.store.record_batch_decision(
                batch_id=self.batch_id, decision="hold", reason="No stable slice",
                confirmed=False, actor_type="human", actor_id="engineer",
            )
        decision = self.store.record_batch_decision(
            batch_id=self.batch_id, decision="hold",
            reason="No stable failure slice is eligible for Challenger training.",
            confirmed=True, actor_type="human", actor_id="engineer",
        )
        self.assertEqual(decision["maintenance_batch"]["state"], BatchState.HELD)
        self.assertEqual(decision["maintenance_batch"]["formal_decision"], "hold")
        with self.store.connect() as db:
            active = db.execute(
                "SELECT model_id FROM active_models WHERE project_id=?", (self.project_id,)
            ).fetchone()
        self.assertEqual(active["model_id"], "champion")
        with self.assertRaises(PermissionError):
            self.store.record_batch_decision(
                batch_id=self.batch_id, decision="hold", reason="Duplicate",
                confirmed=True, actor_type="human", actor_id="engineer",
            )

    def test_human_can_hold_after_eligible_review_before_challenger_job(self) -> None:
        for slice_id in ("slice-1", "slice-2"):
            self.service.save_decision(
                slice_id=slice_id, decision="accept", excluded_image_ids=[],
                actor_type="human", actor_id="engineer",
            )
        self.service.freeze_review(
            batch_id=self.batch_id, confirmed=True, actor_type="human", actor_id="engineer"
        )
        decision = self.store.record_batch_decision(
            batch_id=self.batch_id, decision="hold",
            reason="Fixed Challenger pools cannot be constructed safely.",
            confirmed=True, actor_type="human", actor_id="engineer",
        )
        self.assertEqual(decision["maintenance_batch"]["state"], BatchState.HELD)


if __name__ == "__main__":
    unittest.main()
