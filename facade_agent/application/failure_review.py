from __future__ import annotations

import hashlib
import json
from typing import Any

from ..core.permissions import ToolPolicy
from ..core.states import BATCH_TRANSITIONS, BatchState, require_transition
from ..storage import Store, canonical_json, make_id, utc_now
from ..tools import ToolDefinition, ToolRegistry


DECISIONS = {"accept", "trim", "reject"}


class FailureReviewService:
    """Human-only review of verified failure slices.

    Reviews may filter discovery members but never change labels, thresholds,
    clustering output, or the active Champion.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def get_review(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.get_maintenance_batch(batch_id)
        if batch["state"] not in {
            BatchState.FAILURE_DISCOVERY_COMPLETED,
            BatchState.FAILURE_REVIEW_COMPLETED,
        }:
            raise PermissionError("Failure Slice review is not available for this batch state.")
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT s.*,c.display_name AS class_name,r.decision,r.excluded_json,"
                "r.reviewer_id,r.knowledge_version,r.updated_at AS reviewed_at "
                "FROM failure_slices s JOIN classes c ON c.class_id=s.class_id "
                "LEFT JOIN slice_reviews r ON r.slice_id=s.slice_id "
                "WHERE s.batch_id=? ORDER BY c.sort_order,s.error_type,s.slice_id",
                (batch_id,),
            ).fetchall()
            slices = []
            for row in rows:
                item = dict(row)
                excluded = json.loads(item.pop("excluded_json")) if item["excluded_json"] else []
                item["excluded_image_ids"] = excluded
                item["members"] = [
                    {
                        **dict(member),
                        "excluded": member["image_id"] in excluded,
                        "representative": member["image_id"] == item["representative_image_id"],
                    }
                    for member in db.execute(
                        "SELECT m.image_id,m.membership_rank,i.filename FROM failure_slice_members m "
                        "JOIN images i ON i.image_id=m.image_id WHERE m.slice_id=? "
                        "ORDER BY m.membership_rank,m.image_id",
                        (item["slice_id"],),
                    )
                ]
                slices.append(item)
            frozen = db.execute(
                "SELECT * FROM failure_review_versions WHERE batch_id=?", (batch_id,)
            ).fetchone()
        decision_counts = {name: 0 for name in sorted(DECISIONS)}
        for item in slices:
            if item["decision"]:
                decision_counts[item["decision"]] += 1
        version = dict(frozen) if frozen else None
        if version:
            version["snapshot"] = json.loads(version.pop("snapshot_json"))
        return {
            "batch_id": batch_id,
            "batch_state": batch["state"],
            "slices": slices,
            "summary": {
                "slice_count": len(slices),
                "reviewed_count": sum(decision_counts.values()),
                "remaining_count": len(slices) - sum(decision_counts.values()),
                "decision_counts": decision_counts,
                "complete": len(slices) == sum(decision_counts.values()),
                "labels_modified": False,
                "reclustered": False,
            },
            "frozen_version": version,
        }

    def save_decision(
        self,
        *,
        slice_id: str,
        decision: str,
        excluded_image_ids: list[str],
        actor_type: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can review Failure Slices.")
        normalized = str(decision).strip().lower()
        if normalized not in DECISIONS:
            raise ValueError("Decision must be Accept, Trim, or Reject.")
        exclusions = sorted({str(value).strip() for value in excluded_image_ids if str(value).strip()})
        if normalized != "trim" and exclusions:
            raise ValueError("Only Trim can exclude member images.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT s.*,b.project_id,b.state AS batch_state FROM failure_slices s "
                "JOIN maintenance_batches b ON b.batch_id=s.batch_id WHERE s.slice_id=?",
                (slice_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Failure Slice not found: {slice_id}")
            if row["batch_state"] != BatchState.FAILURE_DISCOVERY_COMPLETED:
                raise PermissionError("This Failure Slice review is not editable.")
            member_ids = {
                member["image_id"]
                for member in db.execute(
                    "SELECT image_id FROM failure_slice_members WHERE slice_id=?", (slice_id,)
                )
            }
            unknown = sorted(set(exclusions) - member_ids)
            if unknown:
                raise ValueError("Trim contains images outside this Failure Slice.")
            if normalized == "trim":
                if not exclusions:
                    raise ValueError("Trim must exclude at least one member image.")
                if len(exclusions) >= len(member_ids):
                    raise ValueError("Trim must retain at least one member image; use Reject otherwise.")
            reviewers = {
                item["reviewer_id"]
                for item in db.execute(
                    "SELECT r.reviewer_id FROM slice_reviews r JOIN failure_slices s "
                    "ON s.slice_id=r.slice_id WHERE s.batch_id=?",
                    (row["batch_id"],),
                )
            }
            if reviewers and reviewers != {actor_id}:
                raise PermissionError("This batch is governed by one domain expert reviewer.")
            now = utc_now()
            db.execute(
                "INSERT INTO slice_reviews(slice_id,decision,excluded_json,reviewer_id,knowledge_version,updated_at) "
                "VALUES(?,?,?,?,NULL,?) ON CONFLICT(slice_id) DO UPDATE SET "
                "decision=excluded.decision,excluded_json=excluded.excluded_json,"
                "reviewer_id=excluded.reviewer_id,knowledge_version=NULL,updated_at=excluded.updated_at",
                (slice_id, normalized, canonical_json(exclusions), actor_id, now),
            )
            digest = self.store._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=row["batch_id"],
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="save_failure_slice_review",
                from_state=row["batch_state"],
                to_state=row["batch_state"],
                payload={
                    "slice_id": slice_id,
                    "decision": normalized,
                    "excluded_image_ids": exclusions,
                    "labels_modified": False,
                    "reclustered": False,
                },
            )
        result = self.get_review(row["batch_id"])
        result["audit_event_sha256"] = digest
        return result

    def freeze_review(
        self,
        *,
        batch_id: str,
        confirmed: bool,
        actor_type: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can freeze a Failure Slice review.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            transition = require_transition(
                batch["state"], "freeze_failure_review", BATCH_TRANSITIONS, confirmed
            )
            rows = db.execute(
                "SELECT s.slice_id,s.class_id,s.error_type,s.source_slice_key,r.decision,"
                "r.excluded_json,r.reviewer_id FROM failure_slices s "
                "LEFT JOIN slice_reviews r ON r.slice_id=s.slice_id "
                "WHERE s.batch_id=? ORDER BY s.slice_id",
                (batch_id,),
            ).fetchall()
            missing = [row["slice_id"] for row in rows if row["decision"] is None]
            if missing:
                raise PermissionError(f"Review every Failure Slice before freezing ({len(missing)} remaining).")
            reviewers = {row["reviewer_id"] for row in rows if row["reviewer_id"]}
            if len(reviewers) > 1 or (reviewers and reviewers != {actor_id}):
                raise PermissionError("The frozen review must belong to one domain expert.")
            snapshot = []
            retained_records = 0
            retained_images: set[str] = set()
            eligible = 0
            for row in rows:
                member_ids = [
                    item["image_id"]
                    for item in db.execute(
                        "SELECT image_id FROM failure_slice_members WHERE slice_id=? "
                        "ORDER BY membership_rank,image_id",
                        (row["slice_id"],),
                    )
                ]
                excluded = json.loads(row["excluded_json"])
                retained = [] if row["decision"] == "reject" else [
                    image_id for image_id in member_ids if image_id not in excluded
                ]
                if row["decision"] in {"accept", "trim"}:
                    eligible += 1
                    retained_records += len(retained)
                    retained_images.update(retained)
                snapshot.append({
                    "slice_id": row["slice_id"],
                    "source_slice_key": row["source_slice_key"],
                    "class_id": row["class_id"],
                    "error_type": row["error_type"],
                    "decision": row["decision"],
                    "excluded_image_ids": excluded,
                    "retained_image_ids": retained,
                })
            frozen_payload = {
                "schema_version": 1,
                "batch_id": batch_id,
                "reviewer_id": actor_id,
                "single_expert": True,
                "labels_modified": False,
                "reclustered": False,
                "slices": snapshot,
            }
            review_sha256 = hashlib.sha256(canonical_json(frozen_payload).encode("utf-8")).hexdigest()
            review_version_id = make_id("failure-review")
            now = utc_now()
            db.execute(
                "INSERT INTO failure_review_versions(review_version_id,batch_id,reviewer_id,slice_count,"
                "eligible_slice_count,retained_member_records,retained_unique_images,review_sha256,"
                "snapshot_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    review_version_id, batch_id, actor_id, len(rows), eligible,
                    retained_records, len(retained_images), review_sha256,
                    canonical_json(frozen_payload), now,
                ),
            )
            for row in rows:
                status = {"accept": "accepted", "trim": "trimmed", "reject": "rejected"}[row["decision"]]
                db.execute("UPDATE failure_slices SET status=? WHERE slice_id=?", (status, row["slice_id"]))
            db.execute(
                "UPDATE slice_reviews SET knowledge_version=? WHERE slice_id IN "
                "(SELECT slice_id FROM failure_slices WHERE batch_id=?)",
                (review_version_id, batch_id),
            )
            db.execute(
                "UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?",
                (transition.target, now, batch_id),
            )
            digest = self.store._append_audit(
                db,
                project_id=batch["project_id"],
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="freeze_failure_review",
                from_state=batch["state"],
                to_state=transition.target,
                payload={
                    "review_version_id": review_version_id,
                    "review_sha256": review_sha256,
                    "slice_count": len(rows),
                    "eligible_slice_count": eligible,
                    "retained_member_records": retained_records,
                    "retained_unique_images": len(retained_images),
                    "labels_modified": False,
                    "reclustered": False,
                },
            )
        result = self.get_review(batch_id)
        result["audit_event_sha256"] = digest
        return result


def register_failure_review_tools(registry: ToolRegistry, service: FailureReviewService) -> None:
    registry.register(ToolDefinition(
        "save_failure_slice_review",
        "Record one human Accept, Trim, or Reject decision. Trim may only exclude members of that verified slice.",
        {
            "type": "object",
            "required": ["slice_id", "decision", "excluded_image_ids"],
            "properties": {
                "slice_id": {"type": "string", "minLength": 1},
                "decision": {"type": "string", "enum": ["accept", "trim", "reject"]},
                "excluded_image_ids": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        ToolPolicy(mutates_state=True, requires_confirmation=False, allowed_actor_types=("human",)),
        lambda args, ctx: service.save_decision(
            slice_id=args["slice_id"], decision=args["decision"],
            excluded_image_ids=args["excluded_image_ids"], actor_type=ctx["actor_type"],
            actor_id=ctx["actor_id"],
        ),
    ))
    registry.register(ToolDefinition(
        "freeze_failure_slice_review",
        "Freeze the complete single-expert Failure Slice review as an immutable knowledge version.",
        {
            "type": "object",
            "required": ["batch_id"],
            "properties": {"batch_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.freeze_review(
            batch_id=args["batch_id"], confirmed=ctx["confirmed"],
            actor_type=ctx["actor_type"], actor_id=ctx["actor_id"],
        ),
    ))
