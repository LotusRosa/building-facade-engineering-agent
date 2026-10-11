from __future__ import annotations

import hashlib
import json
from typing import Any

from ..core.permissions import ToolPolicy
from ..storage import canonical_json
from ..tools.registry import ToolDefinition, ToolRegistry
from .gpu_scheduler import public_queue_snapshot


class WorkflowSnapshotService:
    """One deterministic workflow view shared by conversation and Task Panel."""

    def __init__(self, store: Any, gpu_scheduler: Any | None = None) -> None:
        self.store = store
        self.gpu_scheduler = gpu_scheduler

    @staticmethod
    def _row(db: Any, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        row = db.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _action(
        action_id: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        interaction_mode: str,
        task_panel_target: str,
        *,
        requires_confirmation: bool = False,
        title: str,
    ) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "title": title,
            "tool_name": tool_name,
            "arguments": arguments,
            "interaction_mode": interaction_mode,
            "requires_confirmation": requires_confirmation,
            "task_panel_target": task_panel_target,
        }

    def get(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_project(project_id)
        gpu_queue = (
            public_queue_snapshot(self.gpu_scheduler.snapshot(project_id))
            if self.gpu_scheduler is not None
            else {"active": None, "waiting": [], "waiting_count": 0, "project_items": []}
        )
        batch_id = project.get("active_batch_id")
        with self.store.connect() as db:
            batch = self._row(db, "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)) if batch_id else None
            dataset = self._row(
                db,
                "SELECT * FROM datasets WHERE project_id=? AND "
                + ("batch_id=?" if batch else "role='initial_training'")
                + " ORDER BY created_at DESC LIMIT 1",
                (project_id, batch_id) if batch else (project_id,),
            )
            counts = db.execute(
                "SELECT count(*) AS total,sum(a.status='complete') AS labeled FROM images i "
                "JOIN image_annotations a ON a.image_id=i.image_id WHERE i.dataset_id=?",
                (dataset["dataset_id"] if dataset else "",),
            ).fetchone()
            annotation_progress = {
                "dataset_id": dataset["dataset_id"] if dataset else None,
                "dataset_status": dataset["status"] if dataset else None,
                "image_count": int(counts["total"] or 0),
                "labeled_count": int(counts["labeled"] or 0),
                "unlabeled_count": int(counts["total"] or 0) - int(counts["labeled"] or 0),
            }
            annotation_progress["complete"] = bool(annotation_progress["image_count"] and not annotation_progress["unlabeled_count"])
            training_job = self._row(db, "SELECT * FROM training_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            training_run = self._row(db, "SELECT * FROM training_runs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            screening_job = self._row(db, "SELECT * FROM screening_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            screening_run = self._row(db, "SELECT * FROM screening_runs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            challenger_job = self._row(db, "SELECT * FROM challenger_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            challenger_run = self._row(db, "SELECT * FROM challenger_runs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            evaluation_job = self._row(db, "SELECT * FROM evaluation_jobs WHERE project_id=? AND is_current=1 ORDER BY created_at DESC LIMIT 1", (project_id,))
            if batch:
                cohort_counts = {}
                history = db.execute(
                    "SELECT origin_role,image_count FROM evaluation_cohorts WHERE project_id=? AND status='active'",
                    (project_id,),
                ).fetchall()
                seed_required = not any(c["origin_role"] in {"core_safety_seed", "core_safety_addition"} for c in history)
                for cohort in ("current_gate", "core_safety") if seed_required else ("current_gate",):
                    row = db.execute(
                        "SELECT count(*) AS total,sum(annotation_status='complete') AS labeled "
                        "FROM evaluation_input_images WHERE batch_id=? AND cohort=?",
                        (batch_id, cohort),
                    ).fetchone()
                    cohort_counts[cohort] = {"image_count": int(row["total"] or 0), "labeled_count": int(row["labeled"] or 0)}
                annotation_progress["evaluation_cohorts"] = cohort_counts
                annotation_progress["core_safety_history_count"] = sum(int(c["image_count"]) for c in history)
                annotation_progress["core_safety_seed_required"] = seed_required
                if batch["state"] == "CHALLENGER_TRAINED":
                    total = sum(c["image_count"] for c in cohort_counts.values())
                    labeled = sum(c["labeled_count"] for c in cohort_counts.values())
                    annotation_progress = {
                        "kind": "evaluation", "cohorts": cohort_counts,
                        "core_safety_history_count": sum(int(c["image_count"]) for c in history),
                        "core_safety_seed_required": seed_required,
                        "image_count": total, "labeled_count": labeled, "unlabeled_count": total - labeled,
                        "dataset_status": "frozen" if evaluation_job and evaluation_job["batch_id"] == batch_id else "open",
                        "complete": all(c["image_count"] and c["labeled_count"] == c["image_count"] for c in cohort_counts.values()),
                    }
            evaluation_run = self._row(db, "SELECT * FROM evaluation_runs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            evidence = self._row(db, "SELECT evidence_id FROM evidence_reports WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            completion = self._row(db, "SELECT * FROM round_completion_summaries WHERE project_id=? ORDER BY completed_at DESC LIMIT 1", (project_id,))
            # Approval bookkeeping must not invalidate its own workflow proposal.
            # Actual data/state mutations still change this revision and reject stale confirmations.
            audit = self._row(db, "SELECT event_sha256 FROM audit_events WHERE project_id=? AND tool_name NOT IN ('request_human_confirmation','resolve_human_confirmation') ORDER BY event_id DESC LIMIT 1", (project_id,))

        state = batch["state"] if batch else project["state"]
        action: dict[str, Any] | None = None
        if gpu_queue["project_items"]:
            action = None
        elif batch and batch["state"] in {"DEPLOYED", "HELD"}:
            action = self._action("start_next_round", "create_maintenance_batch", {"project_id": project_id}, "task_panel_required", "maintenance-create", title="Start another maintenance round")
        elif batch and batch["state"] == "CREATED":
            action = self._action("import_maintenance_data", None, {"batch_id": batch_id}, "task_panel_required", "maintenance-import", title="Import and label maintenance data")
        elif batch and batch["state"] in {"MAINTENANCE_DATA_IMPORTED", "MAINTENANCE_LABELS_READY"}:
            if dataset and dataset['status'] == 'validated':
                action = self._action('freeze_maintenance_labels', 'freeze_maintenance_batch', {'batch_id': batch_id}, 'conversation_confirmable', 'maintenance-labels', requires_confirmation=True, title='Confirm and freeze maintenance labels')
            elif annotation_progress['complete']:
                action = self._action('validate_maintenance_labels', 'validate_dataset', {'dataset_id': dataset['dataset_id']}, 'conversation_confirmable', 'maintenance-labels', requires_confirmation=True, title='Validate complete maintenance annotations')
            else:
                action = self._action("review_maintenance_labels", None, {"batch_id": batch_id}, "task_panel_required", "maintenance-labels", title="Review maintenance labels")
        elif batch and batch["state"] == "MAINTENANCE_BATCH_FROZEN":
            action = self._action("start_failure_discovery", "start_champion_failure_discovery", {"batch_id": batch_id}, "conversation_confirmable", "screening-progress", requires_confirmation=True, title="Start Champion screening and failure discovery")
        elif batch and batch["state"] == "FAILURE_DISCOVERY_COMPLETED":
            action = self._action("review_failure_slices", None, {"batch_id": batch_id}, "task_panel_required", "failure-review", title="Complete expert Failure Slice review")
        elif batch and batch["state"] == "FAILURE_REVIEW_COMPLETED":
            if challenger_run and challenger_run["batch_id"] == batch_id and challenger_run["status"] == "result_verified":
                action = self._action("register_challenger", "register_challenger", {"run_id": challenger_run["run_id"]}, "conversation_confirmable", "challenger-progress", requires_confirmation=True, title="Register verified Challenger")
            elif challenger_job and challenger_job["batch_id"] == batch_id:
                action = self._action("start_challenger_training", "start_challenger_training", {"job_id": challenger_job["job_id"]}, "conversation_confirmable", "challenger-progress", requires_confirmation=True, title="Start single-GPU Challenger training")
            else:
                action = self._action("create_challenger_job", "create_challenger_training_job", {"batch_id": batch_id}, "conversation_confirmable", "challenger-job", requires_confirmation=True, title="Generate Challenger job")
        elif batch and batch["state"] == "CHALLENGER_TRAINED":
            if evaluation_job and evaluation_job["batch_id"] == batch_id:
                action = self._action("start_paired_evaluation", "start_model_evaluation", {"job_id": evaluation_job["job_id"]}, "conversation_confirmable", "evaluation-progress", requires_confirmation=True, title="Start Champion–Challenger evaluation")
            else:
                action = self._action("import_evaluation_snapshot", None, {"batch_id": batch_id}, "task_panel_required", "evaluation-import", title="Prepare independent evaluation folders")
        elif batch and batch["state"] == "DECISION_PENDING":
            action = self._action("record_engineer_decision", "record_deployment_decision", {"batch_id": batch_id}, "task_panel_required", "engineer-decision", requires_confirmation=True, title="Choose Retain or Promote")
        elif project["state"] == "PROJECT_CREATED":
            action = self._action("confirm_taxonomy", "confirm_taxonomy", {"project_id": project_id}, "conversation_confirmable", "project-setup", requires_confirmation=True, title="Confirm frozen classes")
        elif project["state"] == "TAXONOMY_CONFIRMED":
            action = self._action("import_initial_data", None, {"project_id": project_id}, "task_panel_required", "initial-import", title="Import and label initial data")
        elif project["state"] == "DATA_IMPORTED":
            if annotation_progress["complete"]:
                action = self._action("validate_initial_labels", "validate_dataset", {"dataset_id": dataset["dataset_id"]}, "conversation_confirmable", "initial-labels", requires_confirmation=True, title="Validate complete annotations and freeze the dataset")
            else:
                action = self._action("review_initial_labels", None, {"dataset_id": dataset["dataset_id"] if dataset else ""}, "task_panel_required", "initial-labels", title="Import a label table or complete image annotations and validate")
        elif project["state"] in {"DATA_VALIDATED", "ANNOTATION_READY"}:
            if training_job:
                action = self._action("start_initial_training", "start_initial_champion_training", {"job_id": training_job["job_id"]}, "conversation_confirmable", "initial-training", requires_confirmation=True, title="Start single-GPU Initial Champion training")
            else:
                action = self._action("prepare_initial_training", "create_initial_training_job", {"project_id": project_id, "dataset_id": dataset["dataset_id"] if dataset else ""}, "conversation_confirmable", "initial-training", requires_confirmation=True, title="Generate Initial Champion job")
        elif project["state"] == "INITIAL_TRAINING_READY" and training_run and training_run["status"] == "result_verified":
            action = self._action("register_initial_champion", "register_initial_champion", {"run_id": training_run["run_id"]}, "conversation_confirmable", "initial-training", requires_confirmation=True, title="Register Initial Champion")
        elif project["state"] in {"CHAMPION_READY", "SCREENING_READY"}:
            action = self._action("start_first_round", "create_maintenance_batch", {"project_id": project_id}, "task_panel_required", "maintenance-create", title="Create maintenance batch")

        identity = {
            "project_id": project_id,
            "project_state": project["state"],
            "batch_id": batch_id,
            "batch_state": batch["state"] if batch else None,
            "training_job": training_job["job_id"] if training_job else None,
            "training_run": training_run["run_id"] if training_run else None,
            "screening_job": screening_job["job_id"] if screening_job else None,
            "screening_run": screening_run["run_id"] if screening_run else None,
            "challenger_job": challenger_job["job_id"] if challenger_job else None,
            "challenger_run": challenger_run["run_id"] if challenger_run else None,
            "evaluation_job": evaluation_job["job_id"] if evaluation_job else None,
            "evaluation_run": evaluation_run["run_id"] if evaluation_run else None,
            "evidence": evidence["evidence_id"] if evidence else None,
            "completion": completion["batch_id"] if completion else None,
            "audit": audit["event_sha256"] if audit else None,
            "annotation_progress": annotation_progress,
            "dataset_updated_at": dataset["updated_at"] if dataset else None,
            "gpu_queue": [
                {
                    "queue_id": item.get("queue_id"),
                    "status": item.get("status"),
                    "queue_position": item.get("queue_position"),
                }
                for item in gpu_queue["project_items"]
            ],
        }
        workflow_version = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        if action is not None:
            action["workflow_version"] = workflow_version
        return {
            "project_id": project_id,
            "state": state,
            "workflow_version": workflow_version,
            "next_actions": [action] if action else [],
            "active_batch": batch,
            "round_completion": completion,
            "gpu_queue": gpu_queue,
            "annotation_progress": annotation_progress,
        }

    def request_task_panel_action(self, project_id: str, action_id: str) -> dict[str, Any]:
        snapshot = self.get(project_id)
        action = next((item for item in snapshot["next_actions"] if item["action_id"] == action_id), None)
        if action is None:
            raise PermissionError("The requested Task Panel action is no longer current.")
        return {
            "navigation": {
                "target": action["task_panel_target"],
                "action_id": action_id,
                "workflow_version": snapshot["workflow_version"],
            }
        }

    def enrich_tool_result(self, project_id: str | None, result: Any) -> Any:
        if (
            self.gpu_scheduler is None
            or not project_id
            or not isinstance(result, dict)
            or not isinstance(result.get("queue"), dict)
        ):
            return result
        queue_id = result["queue"].get("queue_id")
        snapshot = public_queue_snapshot(self.gpu_scheduler.snapshot(project_id))
        current = next(
            (
                item
                for item in ([snapshot["active"]] if snapshot["active"] else [])
                + snapshot["waiting"]
                if item.get("queue_id") == queue_id
            ),
            None,
        )
        if current is None:
            return result
        return {**result, "queue": current}


def register_workflow_tools(registry: ToolRegistry, service: WorkflowSnapshotService) -> None:
    registry.register(ToolDefinition(
        "get_annotation_status",
        "Read current image, annotated, and unannotated counts for the active dataset; never infer labels.",
        {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=False, requires_confirmation=False),
        lambda args, ctx: service.get(args["project_id"])["annotation_progress"],
    ))
    registry.register(ToolDefinition(
        "get_workflow_status",
        "Read the authoritative next governed workflow action for a project.",
        {"type": "object", "required": ["project_id"], "properties": {"project_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=False, requires_confirmation=False),
        lambda args, ctx: service.get(args["project_id"]),
    ))
    registry.register(ToolDefinition(
        "request_task_panel_action",
        "Open the current Task Panel target for an interaction that requires files, labels, paths, review, or a human-only decision.",
        {"type": "object", "required": ["project_id", "action_id"], "properties": {"project_id": {"type": "string", "minLength": 1}, "action_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=False, requires_confirmation=False),
        lambda args, ctx: service.request_task_panel_action(args["project_id"], args["action_id"]),
    ))
