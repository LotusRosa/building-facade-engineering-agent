from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from facade_training_worker.data import parse_classes, parse_image_records

from ..core.permissions import ToolPolicy
from ..core.states import BATCH_TRANSITIONS, BatchState, ProjectState, require_transition
from ..storage import canonical_json, make_id, utc_now
from ..tools.registry import ToolDefinition, ToolRegistry
from .gpu_scheduler import GpuJobScheduler
from .local_training import SubprocessWorkerRunner, TrainingCancelled, WorkerLaunchSpec, WorkerRunner
from .training_jobs import FIXED_ZIP_TIME, json_bytes, sha256_file
from .worker_errors import classify_worker_error


EVALUATION_WORKER_PROTOCOL_VERSION = 1
EVALUATION_WORKER_MODULE = "facade_training_worker.champion_challenger_evaluation"
CORE_GATE_KEY = "core_safety"
SOURCE_CURRENT_SPLIT = "current_gate"
SOURCE_CORE_SPLIT = "core_safety_addition"
SOURCE_CORE_SEED = "core_safety"
LOGICAL_CURRENT_SPLIT = "current_gate"
LOGICAL_CORE_SPLIT = "core_safety"
LEGACY_CURRENT_SPLIT = "new_scenario_holdout"
LEGACY_CORE_SPLIT = "historical_safety_holdout"
GATE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ACTIVE_RUN_STATUSES = {"queued", "running", "cancel_requested"}
TERMINAL_RUN_STATUSES = {"cancelled", "result_verified", "failed", "interrupted"}


class ModelEvaluationService:
    """Build and run leakage-checked paired Champion/Challenger evaluation jobs."""

    def __init__(
        self,
        store: Any,
        environment_manager: Any,
        root: Path,
        profile_path: Path,
        runner: WorkerRunner | None = None,
        scheduler: GpuJobScheduler | None = None,
    ) -> None:
        self.store = store
        self.environment = environment_manager
        self.root = root.resolve()
        self.profile_path = profile_path.resolve()
        self.inbox = (self.root / "inbox").resolve()
        self.export_root = (self.root / "exports" / "evaluation_jobs").resolve()
        self.run_root = (self.root / "artifact_store" / "evaluation_runs").resolve()
        self.result_root = (self.root / "artifact_store" / "evaluation_results").resolve()
        self.runner = runner or SubprocessWorkerRunner()
        self._requires_installed_worker = runner is None
        self.scheduler = scheduler or GpuJobScheduler(store)
        self._owns_scheduler = scheduler is None
        self.scheduler.register_handler("champion_challenger_evaluation", self)
        if self._owns_scheduler:
            self.scheduler.start()

    def _profile(self, required_splits: list[str] | None = None) -> dict[str, Any]:
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        policy = profile.get("required_split_policy")
        if not isinstance(policy, dict) or policy.get("history") != "all_active_cohorts_plus_current_core_safety_addition":
            raise ValueError("Evaluation profile does not contain the cumulative Core Safety policy.")
        if profile.get("automatic_deployment") is not False:
            raise ValueError("Evaluation profile must prohibit automatic deployment.")
        if required_splits is not None:
            if not required_splits or required_splits[0] == "overall" or len(required_splits) != len(set(required_splits)):
                raise ValueError("Evaluation Gate keys must be unique and non-empty.")
            profile["required_splits"] = list(required_splits)
        return profile

    @staticmethod
    def _row_to_job(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        item["split_counts"] = json.loads(item.pop("split_counts_json"))
        return item

    @staticmethod
    def _row_to_run(row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("gpu_devices_json", "preflight_json", "command_json", "progress_json", "summary_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def list_inbox(self) -> list[dict[str, Any]]:
        self.inbox.mkdir(parents=True, exist_ok=True)
        return [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(self.inbox.glob("*.zip"), key=lambda value: value.name.lower())
            if path.is_file()
        ]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM evaluation_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Evaluation job not found: {job_id}")
        return self._row_to_job(row)

    def list_jobs(self, project_id: str, batch_id: str | None = None) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        sql = "SELECT * FROM evaluation_jobs WHERE project_id=?"
        params: list[Any] = [project_id]
        if batch_id:
            sql += " AND batch_id=?"
            params.append(batch_id)
        sql += " ORDER BY created_at DESC"
        with self.store.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def _get_run(self, run_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Evaluation run not found: {run_id}")
        return self._row_to_run(row)

    def get_run(self, run_id: str, event_after: int = 0) -> dict[str, Any]:
        run = self._get_run(run_id)
        log_path = Path(run["log_path"]).resolve()
        if self.run_root != log_path and self.run_root not in log_path.parents:
            raise PermissionError("Evaluation log is outside the managed run directory.")
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT event_id,kind,payload_json,created_at FROM evaluation_run_events "
                "WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT 500",
                (run_id, max(0, int(event_after))),
            ).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            events.append(item)
        return {"run": run, "events": events}

    def list_runs(self, project_id: str, batch_id: str | None = None) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        sql = "SELECT * FROM evaluation_runs WHERE project_id=?"
        params: list[Any] = [project_id]
        if batch_id:
            sql += " AND batch_id=?"
            params.append(batch_id)
        sql += " ORDER BY created_at DESC"
        with self.store.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_evidence(self, batch_id: str) -> dict[str, Any] | None:
        self.store.get_maintenance_batch(batch_id)
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM evidence_reports WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item

    def list_gates(self, project_id: str, *, active_only: bool = False) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        sql = "SELECT * FROM evaluation_gates WHERE project_id=?"
        params: list[Any] = [project_id]
        if active_only:
            sql += " AND status='active'"
        sql += " ORDER BY CASE role WHEN 'core_safety' THEN 0 ELSE 1 END,created_at,gate_id"
        with self.store.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_cohorts(
        self, project_id: str, status: str | None = None
    ) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        if status is not None and status not in {"pending", "active"}:
            raise ValueError("Cohort status must be pending or active.")
        sql = "SELECT * FROM evaluation_cohorts WHERE project_id=?"
        params: list[Any] = [project_id]
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        sql += (
            " ORDER BY CASE origin_role WHEN 'core_safety_seed' THEN 0 "
            "WHEN 'core_safety_addition' THEN 1 ELSE 2 END,created_at,cohort_id"
        )
        with self.store.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _gate_content_sha256(rows: list[dict[str, Any]]) -> str:
        normalized = [
            {
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "no_defect": bool(row["no_defect"]),
                "class_ids": sorted(row["class_ids"]),
            }
            for row in rows
        ]
        normalized.sort(key=lambda item: (item["image_sha256"], item["image_id"]))
        return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_gate_key(value: Any) -> str:
        key = str(value or "").strip().lower()
        if not GATE_KEY_PATTERN.fullmatch(key) or key in {CORE_GATE_KEY, "overall", SOURCE_CURRENT_SPLIT}:
            raise ValueError(
                "current_gate_name must start with a letter, contain only lowercase letters, digits, or underscores, "
                "and must not use a reserved Gate name."
            )
        return key

    @staticmethod
    def _legacy_gate_key(batch_id: str) -> str:
        suffix = re.sub(r"[^a-z0-9]", "", batch_id.lower())[-32:] or "legacy"
        return f"round_{suffix}_gate"

    def _backfill_legacy_gates(self, project_id: str) -> None:
        """Register immutable Gate lineage for terminal jobs created before schema v2."""
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT j.job_id,j.batch_id,j.bundle_path,j.created_at,b.state "
                "FROM evaluation_jobs j JOIN maintenance_batches b ON b.batch_id=j.batch_id "
                "WHERE j.project_id=? AND b.state IN ('DEPLOYED','HELD') "
                "ORDER BY j.created_at,j.job_id",
                (project_id,),
            ).fetchall()
            known_batches = {
                row["source_batch_id"] for row in db.execute(
                    "SELECT source_batch_id FROM evaluation_gates WHERE project_id=? AND source_batch_id IS NOT NULL",
                    (project_id,),
                )
            }
            has_core = db.execute(
                "SELECT 1 FROM evaluation_gates WHERE project_id=? AND role='core_safety'",
                (project_id,),
            ).fetchone() is not None
        for row in rows:
            if row["batch_id"] in known_batches:
                continue
            bundle = Path(row["bundle_path"])
            if not bundle.is_file():
                raise ValueError("A legacy evaluation bundle required for Gate history is missing.")
            with zipfile.ZipFile(bundle) as archive:
                labels = self._jsonl(archive.read("holdout/labels.jsonl"))
            current_rows = [item for item in labels if item.get("split") == LEGACY_CURRENT_SPLIT]
            core_rows = [item for item in labels if item.get("split") == LEGACY_CORE_SPLIT]
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if not has_core and core_rows:
                    db.execute(
                        "INSERT OR IGNORE INTO evaluation_gates(gate_id,project_id,source_batch_id,gate_key,role,status,"
                        "source_evaluation_job_id,artifact_path,source_split,content_sha256,image_count,created_at,activated_at) "
                        "VALUES(?,?,?,?,?,'active',?,?,?,?,?,?,?)",
                        (make_id("gate"), project_id, None, CORE_GATE_KEY, "core_safety", row["job_id"],
                         str(bundle), LEGACY_CORE_SPLIT, self._gate_content_sha256(core_rows), len(core_rows),
                         row["created_at"], now),
                    )
                    has_core = True
                if current_rows:
                    db.execute(
                        "INSERT OR IGNORE INTO evaluation_gates(gate_id,project_id,source_batch_id,gate_key,role,status,"
                        "source_evaluation_job_id,artifact_path,source_split,content_sha256,image_count,created_at,activated_at) "
                        "VALUES(?,?,?,?,?,'active',?,?,?,?,?,?,?)",
                        (make_id("gate"), project_id, row["batch_id"], self._legacy_gate_key(row["batch_id"]),
                         "round_gate", row["job_id"], str(bundle), LEGACY_CURRENT_SPLIT,
                         self._gate_content_sha256(current_rows), len(current_rows), row["created_at"], now),
                    )
            known_batches.add(row["batch_id"])

    @staticmethod
    def _safe_names(archive: zipfile.ZipFile, max_entries: int = 100_000) -> list[str]:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > max_entries:
            raise ValueError("Archive contains duplicate or excessive entries.")
        total = 0
        for name in names:
            pure = Path(name)
            info = archive.getinfo(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name or info.file_size > 8 * 1024**3:
                raise ValueError("Archive contains an unsafe member.")
            total += info.file_size
        if total > 64 * 1024**3:
            raise ValueError("Archive exceeds the uncompressed safety limit.")
        return names

    @staticmethod
    def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
        digest = hashlib.sha256()
        with archive.open(name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _jsonl(payload: bytes) -> list[dict[str, Any]]:
        rows = []
        for line_number, raw in enumerate(payload.splitlines(), start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object.")
            rows.append(value)
        return rows

    def _snapshot_path(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".zip"):
            raise ValueError("Choose one ZIP file from the managed inbox.")
        path = (self.inbox / filename).resolve()
        if self.inbox != path and self.inbox not in path.parents:
            raise PermissionError("Evaluation snapshot escaped the managed inbox.")
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation snapshot not found in inbox: {filename}")
        return path

    def _load_registered_gate(self, gate: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        bundle = Path(gate["artifact_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Registered Gate artifact is outside the managed evaluation export directory.")
        if not bundle.is_file():
            raise ValueError(f"Registered Gate artifact is missing: {gate['gate_key']}")
        with zipfile.ZipFile(bundle) as archive:
            labels = self._jsonl(archive.read("holdout/labels.jsonl"))
            rows = [dict(row) for row in labels if row.get("split") == gate["source_split"]]
            if len(rows) != int(gate["image_count"]) or self._gate_content_sha256(rows) != gate["content_sha256"]:
                raise ValueError(f"Registered Gate content changed: {gate['gate_key']}")
            payloads = {row["image_file"]: archive.read(f"holdout/{row['image_file']}") for row in rows}
        return rows, payloads

    def _load_registered_cohort(
        self, cohort: dict[str, Any], *, include_images: bool = True
    ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        bundle = Path(cohort["artifact_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Registered cohort artifact is outside the managed evaluation export directory.")
        if not bundle.is_file():
            raise ValueError(f"Registered cohort artifact is missing: {cohort['cohort_id']}")
        with zipfile.ZipFile(bundle) as archive:
            labels = self._jsonl(archive.read("holdout/labels.jsonl"))
            rows = [
                dict(row)
                for row in labels
                if row.get("cohort_id") == cohort["cohort_id"]
                or (
                    row.get("cohort_id") is None
                    and row.get("split") == cohort["source_split"]
                )
            ]
            if (
                len(rows) != int(cohort["image_count"])
                or self._gate_content_sha256(rows) != cohort["content_sha256"]
            ):
                raise ValueError(f"Registered cohort content changed: {cohort['cohort_id']}")
            payloads = {
                row["image_file"]: archive.read(f"holdout/{row['image_file']}")
                for row in rows
            } if include_images else {}
        return rows, payloads

    def _validate_snapshot_legacy(self, batch_id: str, filename: str) -> dict[str, Any]:
        batch = self.store.get_maintenance_batch(batch_id)
        project = self.store.get_project(batch["project_id"])
        if batch["state"] != BatchState.CHALLENGER_TRAINED:
            raise PermissionError("Independent evaluation requires a registered Challenger.")
        if project["state"] != ProjectState.SCREENING_READY or project.get("active_batch_id") != batch_id:
            raise PermissionError("The evaluation batch is not the active screening batch.")
        self._backfill_legacy_gates(batch["project_id"])
        active_gates = self.list_gates(batch["project_id"], active_only=True)
        core_gates = [gate for gate in active_gates if gate["role"] == "core_safety"]
        if len(core_gates) > 1:
            raise ValueError("Project contains more than one Core Safety Gate.")
        source = self._snapshot_path(filename)
        with self.store.connect() as db:
            champion = db.execute(
                "SELECT m.* FROM active_models a JOIN model_versions m ON m.model_id=a.model_id WHERE a.project_id=?",
                (batch["project_id"],),
            ).fetchone()
            challenger = db.execute(
                "SELECT * FROM model_versions WHERE project_id=? AND source_batch_id=? AND role='challenger' "
                "ORDER BY created_at DESC LIMIT 1",
                (batch["project_id"], batch_id),
            ).fetchone()
            project_hashes = {
                row["sha256"] for row in db.execute(
                    "SELECT i.sha256 FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id WHERE d.project_id=?",
                    (batch["project_id"],),
                )
            }
        if champion is None or challenger is None or challenger["parent_model_id"] != champion["model_id"]:
            raise PermissionError("Evaluation requires the registered Challenger and its unchanged Active Champion parent.")
        for role, model in (("Champion", champion), ("Challenger", challenger)):
            checkpoint = Path(model["checkpoint_path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != model["checkpoint_sha256"]:
                raise ValueError(f"Registered {role} checkpoint is missing or changed before evaluation.")
            thresholds = json.loads(model["thresholds_json"])
            expected_threshold_ids = {item["class_id"] for item in project["classes"]}
            if set(thresholds) != expected_threshold_ids or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1
                for value in thresholds.values()
            ):
                raise ValueError(f"Registered {role} thresholds differ from the frozen taxonomy.")

        historical_hashes: set[str] = set()
        historical_ids: set[str] = set()
        for gate in active_gates:
            gate_rows, _ = self._load_registered_gate(gate)
            for row in gate_rows:
                if row["image_sha256"] in historical_hashes or row["image_id"] in historical_ids:
                    raise ValueError("Registered Core Safety cohorts overlap each other.")
                historical_hashes.add(row["image_sha256"])
                historical_ids.add(row["image_id"])

        with zipfile.ZipFile(source) as archive:
            names = self._safe_names(archive)
            required = {"evaluation_manifest.json", "classes.json", "labels.jsonl", "checksums.json"}
            if not required.issubset(names):
                raise ValueError("Evaluation snapshot is missing manifest, classes, labels, or checksums.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Evaluation snapshot checksum inventory is incomplete.")
            for name, digest in checksums.items():
                if not isinstance(digest, str) or len(digest) != 64 or self._member_sha256(archive, name) != digest:
                    raise ValueError(f"Evaluation snapshot checksum failed: {name}")
            manifest = json.loads(archive.read("evaluation_manifest.json"))
            common_identity = (
                manifest.get("purpose") == "deployment_decision_evidence"
                and manifest.get("project_id") == batch["project_id"]
                and manifest.get("batch_id") == batch_id
                and manifest.get("final_test") is False
            )
            schema_version = manifest.get("schema_version")
            if schema_version == 2:
                current_gate_key = self._validate_gate_key(manifest.get("current_gate_name"))
                expected_source_splits = [SOURCE_CURRENT_SPLIT]
                if not core_gates:
                    expected_source_splits.append(SOURCE_CORE_SPLIT)
                if manifest.get("provided_splits") != expected_source_splits:
                    raise ValueError(
                        "The first round must provide current_gate plus core_safety; later rounds provide only current_gate."
                    )
                source_to_gate = {SOURCE_CURRENT_SPLIT: current_gate_key}
                if not core_gates:
                    source_to_gate[SOURCE_CORE_SPLIT] = CORE_GATE_KEY
            elif schema_version == 1 and not active_gates:
                if manifest.get("required_splits") != [LEGACY_CURRENT_SPLIT, LEGACY_CORE_SPLIT]:
                    raise ValueError("Legacy evaluation snapshot split contract is invalid.")
                current_gate_key = self._legacy_gate_key(batch_id)
                expected_source_splits = [LEGACY_CURRENT_SPLIT, LEGACY_CORE_SPLIT]
                source_to_gate = {LEGACY_CURRENT_SPLIT: current_gate_key, LEGACY_CORE_SPLIT: CORE_GATE_KEY}
            else:
                raise ValueError("Use the accumulating Gate schema for this evaluation round.")
            if not common_identity:
                raise ValueError("Evaluation snapshot identity or cohort-isolation contract is invalid.")
            if any(gate["gate_key"] == current_gate_key for gate in active_gates):
                raise ValueError("The current Gate name was already used by an earlier round.")
            classes = json.loads(archive.read("classes.json"))
            expected_ids = [item["class_id"] for item in project["classes"]]
            if [item.get("class_id") for item in classes] != expected_ids:
                raise ValueError("Evaluation snapshot taxonomy differs from the frozen project taxonomy.")
            rows = self._jsonl(archive.read("labels.jsonl"))
            if not rows:
                raise ValueError("Evaluation snapshot has no labeled images.")
            ids: set[str] = set(historical_ids)
            hashes: set[str] = set(historical_hashes)
            source_split_counts = {name: 0 for name in expected_source_splits}
            positive_support = {split: {class_id: 0 for class_id in expected_ids} for split in expected_source_splits}
            for row in rows:
                image_id = row.get("image_id")
                split = row.get("split")
                image_file = row.get("image_file")
                image_sha = row.get("image_sha256")
                no_defect = row.get("no_defect")
                class_ids = row.get("class_ids")
                if not isinstance(image_id, str) or not image_id or image_id in ids:
                    raise ValueError("Evaluation image IDs must be unique across the Current Gate and Core Safety.")
                if split not in expected_source_splits or not isinstance(image_file, str) or not image_file.startswith("images/"):
                    raise ValueError("Evaluation label row has an invalid split or image path.")
                if image_file not in checksums or image_sha != checksums[image_file]:
                    raise ValueError("Evaluation image SHA-256 does not match checksums.json.")
                if image_sha in hashes or image_sha in project_hashes:
                    raise ValueError("Evaluation data overlaps training, maintenance, or Core Safety data.")
                if not isinstance(no_defect, bool) or not isinstance(class_ids, list):
                    raise ValueError("Evaluation labels must contain boolean no_defect and class_ids.")
                if len(class_ids) != len(set(class_ids)) or any(value not in expected_ids for value in class_ids):
                    raise ValueError("Evaluation labels contain duplicate or unknown classes.")
                if no_defect == bool(class_ids):
                    raise ValueError("Evaluation labels must choose No defect alone or one or more defect classes.")
                ids.add(image_id)
                hashes.add(image_sha)
                source_split_counts[split] += 1
                for class_id in class_ids:
                    positive_support[split][class_id] += 1
            if any(not count for count in source_split_counts.values()):
                raise ValueError("Every required Gate must contain images.")
            if any(not count for support in positive_support.values() for count in support.values()):
                raise ValueError("Every class needs positive support in every newly supplied Gate.")

        required_gate_keys = [current_gate_key] + [gate["gate_key"] for gate in active_gates]
        if not core_gates:
            required_gate_keys.append(CORE_GATE_KEY)
        split_counts = {
            source_to_gate[split]: count for split, count in source_split_counts.items()
        }
        split_counts.update({gate["gate_key"]: int(gate["image_count"]) for gate in active_gates})
        return {
            "batch": batch,
            "project": project,
            "champion": dict(champion),
            "challenger": dict(challenger),
            "source": source,
            "source_sha256": sha256_file(source),
            "manifest": manifest,
            "classes": classes,
            "labels": rows,
            "source_to_gate": source_to_gate,
            "current_gate_key": current_gate_key,
            "active_gates": active_gates,
            "required_gate_keys": required_gate_keys,
            "split_counts": split_counts,
        }

    def _validate_snapshot(self, batch_id: str, filename: str) -> dict[str, Any]:
        batch = self.store.get_maintenance_batch(batch_id)
        project = self.store.get_project(batch["project_id"])
        if batch["state"] != BatchState.CHALLENGER_TRAINED:
            raise PermissionError("Independent evaluation requires a registered Challenger.")
        if project["state"] != ProjectState.SCREENING_READY or project.get("active_batch_id") != batch_id:
            raise PermissionError("The evaluation batch is not the active screening batch.")

        source = self._snapshot_path(filename)
        all_cohorts = self.list_cohorts(batch["project_id"])
        active_cohorts = [cohort for cohort in all_cohorts if cohort["status"] == "active"]
        needs_core_seed = not any(c["origin_role"] in {"core_safety_seed", "core_safety_addition"} for c in active_cohorts)
        protected_cohorts = [
            cohort for cohort in all_cohorts if cohort["source_batch_id"] != batch_id
        ]
        with self.store.connect() as db:
            champion = db.execute(
                "SELECT m.* FROM active_models a JOIN model_versions m ON m.model_id=a.model_id "
                "WHERE a.project_id=?",
                (batch["project_id"],),
            ).fetchone()
            challenger = db.execute(
                "SELECT * FROM model_versions WHERE project_id=? AND source_batch_id=? AND role='challenger' "
                "ORDER BY created_at DESC LIMIT 1",
                (batch["project_id"], batch_id),
            ).fetchone()
            project_hashes = {
                row["sha256"]
                for row in db.execute(
                    "SELECT i.sha256 FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id "
                    "WHERE d.project_id=?",
                    (batch["project_id"],),
                )
            }
            final_test_ids = {
                row["image_id"]
                for row in db.execute(
                    "SELECT image_id FROM final_test_image_inventory WHERE project_id=?",
                    (batch["project_id"],),
                )
            }
            final_test_hashes = {
                row["content_sha256"]
                for row in db.execute(
                    "SELECT content_sha256 FROM final_test_image_inventory WHERE project_id=?",
                    (batch["project_id"],),
                )
            }
        if champion is None or challenger is None or challenger["parent_model_id"] != champion["model_id"]:
            raise PermissionError("Evaluation requires the registered Challenger and its unchanged Active Champion parent.")
        expected_threshold_ids = {item["class_id"] for item in project["classes"]}
        for role, model in (("Champion", champion), ("Challenger", challenger)):
            checkpoint = Path(model["checkpoint_path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != model["checkpoint_sha256"]:
                raise ValueError(f"Registered {role} checkpoint is missing or changed before evaluation.")
            thresholds = json.loads(model["thresholds_json"])
            if set(thresholds) != expected_threshold_ids or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= float(value) <= 1
                for value in thresholds.values()
            ):
                raise ValueError(f"Registered {role} thresholds differ from the frozen taxonomy.")

        cohort_ids: set[str] = set()
        cohort_hashes: set[str] = set()
        for cohort in protected_cohorts:
            rows, _ = self._load_registered_cohort(cohort)
            for row in rows:
                if row["image_id"] in cohort_ids or row["image_sha256"] in cohort_hashes:
                    raise ValueError("Registered evaluation cohorts overlap each other.")
                cohort_ids.add(row["image_id"])
                cohort_hashes.add(row["image_sha256"])

        with zipfile.ZipFile(source) as archive:
            names = self._safe_names(archive)
            required = {"evaluation_manifest.json", "classes.json", "labels.jsonl", "checksums.json"}
            if not required.issubset(names):
                raise ValueError("Evaluation snapshot is missing manifest, classes, labels, or checksums.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Evaluation snapshot checksum inventory is incomplete.")
            for name, digest in checksums.items():
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or self._member_sha256(archive, name) != digest
                ):
                    raise ValueError(f"Evaluation snapshot checksum failed: {name}")

            manifest = json.loads(archive.read("evaluation_manifest.json"))
            source_core_name = SOURCE_CORE_SEED if manifest.get("schema_version") == 4 else SOURCE_CORE_SPLIT
            expected_source_splits = [SOURCE_CURRENT_SPLIT] + ([source_core_name] if needs_core_seed else [])
            if not (
                manifest.get("schema_version") in {3, 4}
                and manifest.get("purpose") == "champion_challenger_selection"
                and manifest.get("project_id") == batch["project_id"]
                and manifest.get("batch_id") == batch_id
                and manifest.get("provided_splits") == expected_source_splits
                and manifest.get("final_test") is False
            ):
                raise ValueError("Evaluation snapshot identity or cohort-isolation contract is invalid: first evaluation requires Current Gate and initial Core Safety; later evaluations supply only Current Gate.")
            label_version_id = str(manifest.get("label_version_id") or "").strip()
            taxonomy_sha256 = str(manifest.get("taxonomy_sha256") or "").strip()
            if not label_version_id or len(taxonomy_sha256) != 64:
                raise ValueError("Evaluation snapshot label or taxonomy identity is missing.")

            classes = json.loads(archive.read("classes.json"))
            expected_ids = [item["class_id"] for item in project["classes"]]
            if [item.get("class_id") for item in classes] != expected_ids:
                raise ValueError("Evaluation snapshot taxonomy differs from the frozen project taxonomy.")
            if hashlib.sha256(json_bytes(classes)).hexdigest() != taxonomy_sha256:
                raise ValueError("Evaluation snapshot taxonomy SHA-256 is invalid.")

            rows = self._jsonl(archive.read("labels.jsonl"))
            if not rows:
                raise ValueError("Evaluation snapshot has no labeled images.")
            ids = set(cohort_ids)
            hashes = set(cohort_hashes)
            source_split_counts = {SOURCE_CURRENT_SPLIT: 0, SOURCE_CORE_SPLIT: 0}
            required_source_keys = [SOURCE_CURRENT_SPLIT] + ([SOURCE_CORE_SPLIT] if needs_core_seed else [])
            positive_support = {
                split: {class_id: 0 for class_id in expected_ids}
                for split in required_source_keys
            }
            for row in rows:
                image_id = row.get("image_id")
                raw_split = row.get("split")
                split = SOURCE_CORE_SPLIT if raw_split == source_core_name else raw_split
                image_file = row.get("image_file")
                image_sha = row.get("image_sha256")
                no_defect = row.get("no_defect")
                class_ids = row.get("class_ids")
                if image_id in final_test_ids or image_sha in final_test_hashes:
                    raise ValueError("Evaluation snapshot overlaps the protected evaluation inventory.")
                if not isinstance(image_id, str) or not image_id or image_id in ids:
                    raise ValueError("Evaluation image IDs overlap an active or pending evaluation cohort.")
                if raw_split not in expected_source_splits or not isinstance(image_file, str) or not image_file.startswith("images/"):
                    raise ValueError("Evaluation label row has an invalid split or image path.")
                if image_file not in checksums or image_sha != checksums[image_file]:
                    raise ValueError("Evaluation image SHA-256 does not match checksums.json.")
                if image_sha in hashes:
                    raise ValueError("Evaluation snapshot overlaps an active or pending evaluation cohort.")
                if image_sha in project_hashes:
                    raise ValueError("Evaluation snapshot overlaps project training or maintenance data.")
                if not isinstance(no_defect, bool) or not isinstance(class_ids, list):
                    raise ValueError("Evaluation labels must contain boolean no_defect and class_ids.")
                if len(class_ids) != len(set(class_ids)) or any(value not in expected_ids for value in class_ids):
                    raise ValueError("Evaluation labels contain duplicate or unknown classes.")
                if no_defect == bool(class_ids):
                    raise ValueError("Evaluation labels must choose No defect alone or one or more defect classes.")
                ids.add(image_id)
                hashes.add(image_sha)
                source_split_counts[split] += 1
                for class_id in class_ids:
                    positive_support[split][class_id] += 1
                row["split"] = split
            if any(not source_split_counts[key] for key in required_source_keys):
                raise ValueError("Every required new evaluation cohort must contain images.")
            if any(not count for support in positive_support.values() for count in support.values()):
                raise ValueError("Every class needs positive support in each required new evaluation cohort.")

        return {
            "batch": batch,
            "project": project,
            "champion": dict(champion),
            "challenger": dict(challenger),
            "source": source,
            "source_sha256": sha256_file(source),
            "manifest": manifest,
            "classes": classes,
            "labels": rows,
            "active_cohorts": active_cohorts,
            "needs_core_seed": needs_core_seed,
            "label_version_id": label_version_id,
            "taxonomy_sha256": taxonomy_sha256,
            "required_splits": [LOGICAL_CURRENT_SPLIT, LOGICAL_CORE_SPLIT],
            "split_counts": {
                LOGICAL_CURRENT_SPLIT: source_split_counts[SOURCE_CURRENT_SPLIT],
                LOGICAL_CORE_SPLIT: source_split_counts[SOURCE_CORE_SPLIT]
                + sum(int(cohort["image_count"]) for cohort in active_cohorts),
            },
        }

    @staticmethod
    def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
        info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100444 << 16
        archive.writestr(info, content)

    def _assemble_holdout_legacy(self, snapshot: dict[str, Any]) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
        payloads: dict[str, bytes] = {}
        assembled: list[dict[str, Any]] = []
        used_paths: set[str] = set()

        def append_rows(
            rows: list[dict[str, Any]], image_payloads: dict[str, bytes], gate_key: str
        ) -> None:
            for row in rows:
                source_path = row["image_file"]
                suffix = Path(source_path).suffix.lower() or ".bin"
                destination = f"images/{gate_key}/{row['image_sha256'][:24]}{suffix}"
                if destination in used_paths:
                    raise ValueError("Evaluation Gate image path collision detected.")
                content = image_payloads[source_path]
                if hashlib.sha256(content).hexdigest() != row["image_sha256"]:
                    raise ValueError("Evaluation Gate image changed during job assembly.")
                used_paths.add(destination)
                payloads[f"holdout/{destination}"] = content
                assembled.append({
                    **row,
                    "split": gate_key,
                    "image_file": destination,
                    "file": destination,
                    "sha256": row["image_sha256"],
                })

        with zipfile.ZipFile(snapshot["source"]) as archive:
            source_payloads = {row["image_file"]: archive.read(row["image_file"]) for row in snapshot["labels"]}
        for source_split, gate_key in snapshot["source_to_gate"].items():
            append_rows(
                [row for row in snapshot["labels"] if row["split"] == source_split],
                source_payloads,
                gate_key,
            )
        for gate in snapshot["active_gates"]:
            rows, image_payloads = self._load_registered_gate(gate)
            append_rows(rows, image_payloads, gate["gate_key"])

        assembled.sort(key=lambda row: (snapshot["required_gate_keys"].index(row["split"]), row["image_id"]))
        manifest = {
            "schema_version": 2,
            "purpose": "accumulating_gate_evaluation",
            "project_id": snapshot["project"]["project_id"],
            "batch_id": snapshot["batch"]["batch_id"],
            "current_gate": snapshot["current_gate_key"],
            "historical_gates": snapshot["required_gate_keys"][1:],
            "required_splits": snapshot["required_gate_keys"],
            "gate_data_used_for_training": False,
            "final_test": False,
        }
        payloads["holdout/evaluation_manifest.json"] = json_bytes(manifest)
        payloads["holdout/classes.json"] = json_bytes(snapshot["classes"])
        payloads["holdout/labels.jsonl"] = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in assembled
        )
        return payloads, assembled

    def _assemble_holdout(
        self, snapshot: dict[str, Any]
    ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
        payloads: dict[str, bytes] = {}
        assembled: list[dict[str, Any]] = []
        used_paths: set[str] = set()

        def append_rows(
            rows: list[dict[str, Any]],
            image_payloads: dict[str, bytes],
            cohort: dict[str, Any],
            logical_split: str,
        ) -> None:
            for row in rows:
                source_path = row["image_file"]
                suffix = Path(source_path).suffix.lower() or ".bin"
                destination = (
                    f"images/{logical_split}/{cohort['cohort_id']}/"
                    f"{row['image_sha256'][:24]}{suffix}"
                )
                if destination in used_paths:
                    raise ValueError("Evaluation cohort image path collision detected.")
                content = image_payloads[source_path]
                if hashlib.sha256(content).hexdigest() != row["image_sha256"]:
                    raise ValueError("Evaluation cohort image changed during job assembly.")
                used_paths.add(destination)
                payloads[f"holdout/{destination}"] = content
                assembled.append(
                    {
                        **row,
                        "split": logical_split,
                        "image_file": destination,
                        "file": destination,
                        "sha256": row["image_sha256"],
                        "cohort_id": cohort["cohort_id"],
                        "source_round": cohort["source_round_name"],
                        "origin_role": cohort["origin_role"],
                    }
                )

        with zipfile.ZipFile(snapshot["source"]) as archive:
            source_payloads = {
                row["image_file"]: archive.read(row["image_file"])
                for row in snapshot["labels"]
            }
        append_rows(
            [row for row in snapshot["labels"] if row["split"] == SOURCE_CURRENT_SPLIT],
            source_payloads,
            snapshot["new_cohorts"][SOURCE_CURRENT_SPLIT],
            LOGICAL_CURRENT_SPLIT,
        )
        if SOURCE_CORE_SPLIT in snapshot["new_cohorts"]:
            append_rows(
                [row for row in snapshot["labels"] if row["split"] == SOURCE_CORE_SPLIT],
                source_payloads,
                snapshot["new_cohorts"][SOURCE_CORE_SPLIT],
                LOGICAL_CORE_SPLIT,
            )
        for cohort in snapshot["active_cohorts"]:
            rows, image_payloads = self._load_registered_cohort(cohort)
            append_rows(rows, image_payloads, cohort, LOGICAL_CORE_SPLIT)

        split_order = {LOGICAL_CURRENT_SPLIT: 0, LOGICAL_CORE_SPLIT: 1}
        assembled.sort(
            key=lambda row: (
                split_order[row["split"]],
                row["cohort_id"],
                row["image_id"],
            )
        )
        manifest = {
            "schema_version": 3,
            "purpose": "champion_challenger_selection",
            "project_id": snapshot["project"]["project_id"],
            "batch_id": snapshot["batch"]["batch_id"],
            "required_splits": [LOGICAL_CURRENT_SPLIT, LOGICAL_CORE_SPLIT],
            "cohort_data_used_for_training": False,
            "final_test": False,
        }
        payloads["holdout/evaluation_manifest.json"] = json_bytes(manifest)
        payloads["holdout/classes.json"] = json_bytes(snapshot["classes"])
        payloads["holdout/labels.jsonl"] = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in assembled
        )
        return payloads, assembled

    def create_job(self, batch_id: str, inbox_filename: str, actor_type: str, actor_id: str) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can freeze an independent evaluation job.")
        snapshot = self._validate_snapshot(batch_id, inbox_filename)
        profile = self._profile(snapshot["required_splits"])
        profile_bytes = json_bytes(profile)
        fingerprint_payload = {
            "batch_id": batch_id,
            "snapshot_sha256": snapshot["source_sha256"],
            "cohort_lineage": [
                {
                    "cohort_id": cohort["cohort_id"],
                    "content_sha256": cohort["content_sha256"],
                }
                for cohort in snapshot["active_cohorts"]
            ],
            "required_splits": snapshot["required_splits"],
            "champion_sha256": snapshot["champion"]["checkpoint_sha256"],
            "challenger_sha256": snapshot["challenger"]["checkpoint_sha256"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        }
        fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode()).hexdigest()
        with self.store.connect() as db:
            existing = db.execute(
                "SELECT * FROM evaluation_jobs WHERE batch_id=? AND is_current=1",
                (batch_id,),
            ).fetchone()
        if existing:
            job = self._row_to_job(existing)
            if job["content_fingerprint"] != fingerprint:
                raise PermissionError("This batch already has a different immutable evaluation job.")
            self.verify_job(job["job_id"])
            return {"job": job, "idempotent": True}
        job_id = f"evaluation_{fingerprint[:16]}"
        snapshot["new_cohorts"] = {
            SOURCE_CURRENT_SPLIT: {
                "cohort_id": f"cohort_{fingerprint[:12]}_gate",
                "source_round_name": snapshot["batch"]["name"],
                "origin_role": "current_gate",
            },
        }
        if snapshot["needs_core_seed"]:
            snapshot["new_cohorts"][SOURCE_CORE_SPLIT] = {
                "cohort_id": f"cohort_{fingerprint[:12]}_safety",
                "source_round_name": snapshot["batch"]["name"],
                "origin_role": "core_safety_seed",
            }
        final_dir = self.export_root / job_id
        final_dir.mkdir(parents=True, exist_ok=False)
        bundle = final_dir / f"{job_id}.zip"
        temporary = final_dir / ".evaluation_job.tmp"
        payloads, assembled_rows = self._assemble_holdout(snapshot)
        model_manifest = {
            "champion": {
                "model_id": snapshot["champion"]["model_id"],
                "checkpoint_file": "models/champion.pt",
                "checkpoint_sha256": snapshot["champion"]["checkpoint_sha256"],
                "thresholds": json.loads(snapshot["champion"]["thresholds_json"]),
            },
            "challenger": {
                "model_id": snapshot["challenger"]["model_id"],
                "checkpoint_file": "models/challenger.pt",
                "checkpoint_sha256": snapshot["challenger"]["checkpoint_sha256"],
                "thresholds": json.loads(snapshot["challenger"]["thresholds_json"]),
            },
        }
        manifest = {
            "schema_version": 3,
            "job_id": job_id,
            "project_id": snapshot["project"]["project_id"],
            "batch_id": batch_id,
            "champion_model_id": snapshot["champion"]["model_id"],
            "challenger_model_id": snapshot["challenger"]["model_id"],
            "evaluation_profile_id": profile["profile_id"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "source_snapshot_sha256": snapshot["source_sha256"],
            "split_counts": snapshot["split_counts"],
            "current_gate": LOGICAL_CURRENT_SPLIT,
            "core_safety": LOGICAL_CORE_SPLIT,
            "required_splits": snapshot["required_splits"],
            "content_fingerprint": fingerprint,
            "test_read": False,
            "contract": "Evaluate both frozen models on the Current Gate and cumulative Core Safety; never train on cohort data, retune thresholds, or deploy automatically.",
        }
        payloads.update({
            "job_manifest.json": json_bytes(manifest),
            "evaluation_profile.json": profile_bytes,
            "models.json": json_bytes(model_manifest),
            "models/champion.pt": Path(snapshot["champion"]["checkpoint_path"]).read_bytes(),
            "models/challenger.pt": Path(snapshot["challenger"]["checkpoint_path"]).read_bytes(),
        })
        checksums = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
        payloads["checksums.json"] = json_bytes(checksums)
        try:
            with zipfile.ZipFile(temporary, "w") as archive:
                for name, content in sorted(payloads.items()):
                    self._write_member(archive, name, content)
            os.replace(temporary, bundle)
            bundle.chmod(0o444)
            self._preflight_bundle(bundle, expected_job_id=job_id, expected_fingerprint=fingerprint)
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (batch_id,)).fetchone()
                if current is None or current["state"] != BatchState.CHALLENGER_TRAINED:
                    raise PermissionError("Batch state changed while the evaluation job was prepared.")
                db.execute(
                    "INSERT INTO evaluation_jobs(job_id,project_id,batch_id,champion_model_id,challenger_model_id,status,"
                    "evaluation_profile_id,profile_json,source_snapshot_name,source_snapshot_sha256,split_counts_json,"
                    "content_fingerprint,bundle_path,bundle_sha256,bundle_size_bytes,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,'exported',?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, snapshot["project"]["project_id"], batch_id, snapshot["champion"]["model_id"],
                     snapshot["challenger"]["model_id"], profile["profile_id"], canonical_json(profile), inbox_filename,
                     snapshot["source_sha256"], canonical_json(snapshot["split_counts"]), fingerprint, str(bundle),
                     sha256_file(bundle), bundle.stat().st_size, now, now),
                )
                for source_split, cohort in snapshot["new_cohorts"].items():
                    cohort_rows = [
                        row for row in assembled_rows if row["cohort_id"] == cohort["cohort_id"]
                    ]
                    logical_split = (
                        LOGICAL_CURRENT_SPLIT
                        if source_split == SOURCE_CURRENT_SPLIT
                        else LOGICAL_CORE_SPLIT
                    )
                    db.execute(
                        "INSERT INTO evaluation_cohorts(cohort_id,project_id,source_batch_id,"
                        "source_evaluation_job_id,source_round_name,origin_role,status,artifact_path,"
                        "source_split,content_sha256,image_count,label_version_id,taxonomy_sha256,"
                        "created_at,activated_at) VALUES(?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,NULL)",
                        (
                            cohort["cohort_id"],
                            snapshot["project"]["project_id"],
                            batch_id,
                            job_id,
                            cohort["source_round_name"],
                            cohort["origin_role"],
                            str(bundle),
                            logical_split,
                            self._gate_content_sha256(cohort_rows),
                            len(cohort_rows),
                            snapshot["label_version_id"],
                            snapshot["taxonomy_sha256"],
                            now,
                        ),
                    )
                audit = self.store._append_audit(
                    db, project_id=snapshot["project"]["project_id"], batch_id=batch_id,
                    actor_type=actor_type, actor_id=actor_id, tool_name="create_model_evaluation_job",
                    from_state=BatchState.CHALLENGER_TRAINED, to_state=BatchState.CHALLENGER_TRAINED,
                    payload={"job_id": job_id, "snapshot_sha256": snapshot["source_sha256"],
                             "current_gate_cohort_id": snapshot["new_cohorts"][SOURCE_CURRENT_SPLIT]["cohort_id"],
                             "core_safety_cohort_id": snapshot["new_cohorts"].get(SOURCE_CORE_SPLIT, {}).get("cohort_id"),
                             "split_counts": snapshot["split_counts"],
                             "cohort_data_used_for_training": False, "test_read": False},
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            bundle.unlink(missing_ok=True)
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        return {"job": self.get_job(job_id), "audit_event_sha256": audit, "idempotent": False}

    def _preflight_bundle(
        self,
        bundle: Path,
        *,
        expected_job_id: str,
        expected_fingerprint: str,
    ) -> dict[str, Any]:
        with zipfile.ZipFile(bundle) as archive:
            names = self._safe_names(archive)
            manifest = json.loads(archive.read("job_manifest.json"))
            if (
                manifest.get("job_id") != expected_job_id
                or manifest.get("content_fingerprint") != expected_fingerprint
            ):
                raise ValueError("Worker preflight found a mismatched evaluation job manifest.")
            class_ids = parse_classes(json.loads(archive.read("holdout/classes.json")))
            labels = self._jsonl(archive.read("holdout/labels.jsonl"))
            adjusted = [
                {**row, "file": f"holdout/{row.get('file') or row.get('image_file', '')}"}
                for row in labels
            ]
            records = parse_image_records(adjusted, class_ids, names)
            if {row.get("split") for row in labels} != {
                LOGICAL_CURRENT_SPLIT,
                LOGICAL_CORE_SPLIT,
            }:
                raise ValueError("Worker preflight requires exactly current_gate and core_safety.")
            for row, record in zip(labels, records, strict=True):
                if (
                    not isinstance(row.get("cohort_id"), str)
                    or not row["cohort_id"]
                    or not isinstance(row.get("source_round"), str)
                    or not row["source_round"]
                    or row.get("origin_role")
                    not in {"core_safety_seed", "core_safety_addition", "current_gate"}
                ):
                    raise ValueError("Worker preflight requires complete cohort provenance.")
                if self._member_sha256(archive, record.member) != record.sha256:
                    raise ValueError(f"Worker preflight image checksum failed: {record.image_id}")
        return {
            "passed": True,
            "images": len(records),
            "splits": [LOGICAL_CURRENT_SPLIT, LOGICAL_CORE_SPLIT],
            "cohorts": sorted({row["cohort_id"] for row in labels}),
        }

    def preflight_job(self, job_id: str) -> dict[str, Any]:
        verified = self.verify_job(job_id)
        result = self._preflight_bundle(
            Path(verified["bundle_path"]),
            expected_job_id=job_id,
            expected_fingerprint=verified["job"]["content_fingerprint"],
        )
        return {**verified, "worker_preflight": result}

    def verify_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        bundle = Path(job["bundle_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Evaluation job is outside the managed export directory.")
        if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
            raise ValueError("Evaluation job bundle is missing or changed.")
        with zipfile.ZipFile(bundle) as archive:
            names = self._safe_names(archive)
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Evaluation job checksum inventory is incomplete.")
            for name, digest in checksums.items():
                if self._member_sha256(archive, name) != digest:
                    raise ValueError(f"Evaluation job member checksum failed: {name}")
            manifest = json.loads(archive.read("job_manifest.json"))
            profile = json.loads(archive.read("evaluation_profile.json"))
            models = json.loads(archive.read("models.json"))
            if manifest.get("job_id") != job_id or manifest.get("content_fingerprint") != job["content_fingerprint"]:
                raise ValueError("Evaluation job manifest does not match its database record.")
            base_profile = dict(profile)
            required_splits = base_profile.pop("required_splits", None)
            if (
                profile != job["profile"]
                or base_profile != self._profile()
                or manifest.get("test_read") is not False
            ):
                raise ValueError("Evaluation job profile or evaluation contract changed.")
            if not isinstance(required_splits, list) or set(required_splits) != set(manifest.get("split_counts", {})):
                raise ValueError("Evaluation job Gate lineage differs from the locked profile.")
            for role in ("champion", "challenger"):
                model = models[role]
                if model["checkpoint_sha256"] != checksums[model["checkpoint_file"]]:
                    raise ValueError("Evaluation model checkpoint hash differs from models.json.")
        return {"job": job, "bundle_path": str(bundle), "manifest": manifest, "profile": profile, "models": models}

    @staticmethod
    def _command(bundle: Path, output: Path, run_id: str, world_size: int) -> tuple[str, ...]:
        return (
            sys.executable, "-m", "facade_training_worker.launcher", "--nproc", str(world_size),
            "--module", EVALUATION_WORKER_MODULE, "--", "--protocol-version", str(EVALUATION_WORKER_PROTOCOL_VERSION),
            "--bundle", str(bundle), "--output", str(output), "--run-id", run_id,
        )

    def start_job(self, job_id: str, actor_type: str, actor_id: str) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can start model evaluation.")
        verified = self.preflight_job(job_id)
        job = verified["job"]
        batch = self.store.get_maintenance_batch(job["batch_id"])
        if batch["state"] != BatchState.CHALLENGER_TRAINED or job["status"] not in {"exported", "failed"}:
            raise PermissionError("Evaluation job is not startable from the current state.")
        with self.store.connect() as db:
            active = db.execute("SELECT model_id FROM active_models WHERE project_id=?", (job["project_id"],)).fetchone()
        if active is None or active["model_id"] != job["champion_model_id"]:
            raise PermissionError("Active Champion changed before evaluation.")
        if self._requires_installed_worker and importlib.util.find_spec(EVALUATION_WORKER_MODULE) is None:
            raise ValueError(f"Locked evaluation worker is not installed: {EVALUATION_WORKER_MODULE}")
        preflight = self.environment.training_preflight("champion_challenger_evaluation")
        if not preflight["passed"]:
            failures = [item["name"] for item in preflight["checks"] if not item["passed"]]
            raise ValueError("Local GPU preflight failed: " + ", ".join(failures))
        run_id = make_id("evaluation_run")
        output = self.run_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        log_path = output / "worker.log"
        command = self._command(Path(verified["bundle_path"]), output, run_id, preflight["world_size"])
        now = utc_now()
        try:
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute(
                    "SELECT 1 FROM evaluation_runs WHERE job_id=? AND status IN ('queued','running','cancel_requested')", (job_id,)
                ).fetchone():
                    raise PermissionError("Evaluation job already has an active run.")
                db.execute(
                    "INSERT INTO evaluation_runs(run_id,job_id,project_id,batch_id,status,worker_protocol_version,"
                    "gpu_devices_json,world_size,preflight_json,command_json,progress_json,log_path,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, job_id, job["project_id"], job["batch_id"], "queued", EVALUATION_WORKER_PROTOCOL_VERSION,
                     canonical_json(preflight["gpu_devices"]), preflight["world_size"], canonical_json(preflight),
                     canonical_json(list(command)), "{}", str(log_path), now, now),
                )
                db.execute("UPDATE evaluation_jobs SET status='awaiting_result',updated_at=? WHERE job_id=?", (now, job_id))
                audit = self.store._append_audit(
                    db, project_id=job["project_id"], batch_id=job["batch_id"], actor_type=actor_type,
                    actor_id=actor_id, tool_name="start_model_evaluation", from_state=BatchState.CHALLENGER_TRAINED,
                    to_state=BatchState.CHALLENGER_TRAINED,
                    payload={"job_id": job_id, "run_id": run_id, "gpu_devices": preflight["gpu_devices"],
                             "world_size": preflight["world_size"], "test_read": False},
                )
                queue_item = self.scheduler.enqueue_in_transaction(
                    db,
                    project_id=job["project_id"],
                    pipeline="champion_challenger_evaluation",
                    run_id=run_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                )
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
        self.scheduler.notify()
        return {"run": self._get_run(run_id), "queue": queue_item, "audit_event_sha256": audit}

    def _append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        kind = kind if kind in {"status", "progress", "log"} else "log"
        payload = json.loads(canonical_json(payload))
        if "message" in payload:
            payload["message"] = str(payload["message"])[:4000]
        now = utc_now()
        run = self._get_run(run_id)
        if kind == "log":
            path = Path(run["log_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now} {canonical_json(payload)}\n")
        with self.store.connect() as db:
            db.execute("INSERT INTO evaluation_run_events(run_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
                       (run_id, kind, canonical_json(payload), now))
            if kind == "progress":
                db.execute("UPDATE evaluation_runs SET progress_json=?,updated_at=? WHERE run_id=?",
                           (canonical_json(payload), now, run_id))

    def _set_pid(self, run_id: str, pid: int) -> None:
        with self.store.connect() as db:
            db.execute("UPDATE evaluation_runs SET pid=?,updated_at=? WHERE run_id=?", (pid, utc_now(), run_id))

    @staticmethod
    def _average_precision(rows: list[tuple[bool, float]]) -> float:
        positives = sum(truth for truth, _ in rows)
        if not positives:
            raise ValueError("Average precision requires positive support.")
        ordered = sorted(rows, key=lambda item: -item[1])
        true_positives = false_positives = 0
        previous_recall = 0.0
        average_precision = 0.0
        index = 0
        while index < len(ordered):
            score = ordered[index][1]
            group: list[tuple[bool, float]] = []
            while index < len(ordered) and ordered[index][1] == score:
                group.append(ordered[index])
                index += 1
            true_positives += sum(truth for truth, _ in group)
            false_positives += sum(not truth for truth, _ in group)
            recall = true_positives / positives
            precision = true_positives / (true_positives + false_positives)
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
        return average_precision

    @classmethod
    def _metrics(
        cls,
        rows: list[tuple[set[str], set[str], dict[str, float]]],
        class_ids: list[str],
    ) -> dict[str, Any]:
        per_class = {}
        total_tp = total_fp = total_fn = exact = 0
        for truth, predicted, _ in rows:
            exact += truth == predicted
        for class_id in class_ids:
            tp = sum(class_id in truth and class_id in predicted for truth, predicted, _ in rows)
            fp = sum(class_id not in truth and class_id in predicted for truth, predicted, _ in rows)
            fn = sum(class_id in truth and class_id not in predicted for truth, predicted, _ in rows)
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            average_precision = cls._average_precision([
                (class_id in truth, float(probabilities[class_id]))
                for truth, _, probabilities in rows
            ])
            per_class[class_id] = {
                "tp": tp, "fp": fp, "fn": fn, "precision": precision,
                "recall": recall, "f1": f1, "average_precision": average_precision,
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
        return {
            "image_count": len(rows),
            "macro_map": sum(value["average_precision"] for value in per_class.values()) / len(class_ids),
            "macro_f1": sum(value["f1"] for value in per_class.values()) / len(class_ids),
            "micro_f1": micro_f1, "exact_match_accuracy": exact / len(rows),
            "false_positive_count": total_fp, "false_negative_count": total_fn, "per_class": per_class,
        }

    def _validate_result(self, run_id: str, path: Path, allowed_root: Path) -> dict[str, Any]:
        run = self._get_run(run_id)
        job_info = self.verify_job(run["job_id"])
        job = job_info["job"]
        resolved = path.resolve()
        root = allowed_root.resolve()
        if root != resolved and root not in resolved.parents:
            raise PermissionError("Evaluation result is outside its managed directory.")
        if not resolved.is_file():
            raise FileNotFoundError("Evaluation Worker did not produce result_bundle.zip.")
        with zipfile.ZipFile(resolved) as archive:
            names = self._safe_names(archive, 1000)
            if not {"result_manifest.json", "predictions.jsonl", "checksums.json"}.issubset(names):
                raise ValueError("Evaluation result is missing manifest, predictions, or checksums.")
            checksums = json.loads(archive.read("checksums.json"))
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Evaluation result checksum inventory is incomplete.")
            for name, digest in checksums.items():
                if self._member_sha256(archive, name) != digest:
                    raise ValueError(f"Evaluation result checksum failed: {name}")
            manifest = json.loads(archive.read("result_manifest.json"))
            identities = (
                manifest.get("schema_version") == 1,
                manifest.get("worker_protocol_version") == EVALUATION_WORKER_PROTOCOL_VERSION,
                manifest.get("job_id") == run["job_id"], manifest.get("run_id") == run_id,
                manifest.get("content_fingerprint") == job["content_fingerprint"],
                manifest.get("profile_sha256") == hashlib.sha256(json_bytes(job["profile"])).hexdigest(),
                manifest.get("champion_model_id") == job["champion_model_id"],
                manifest.get("challenger_model_id") == job["challenger_model_id"],
                manifest.get("predictions_file") == "predictions.jsonl",
                manifest.get("predictions_sha256") == checksums["predictions.jsonl"],
                manifest.get("test_read") is False,
            )
            if not all(identities):
                raise ValueError("Evaluation result identity, model, or evaluation contract is invalid.")
            predictions = self._jsonl(archive.read("predictions.jsonl"))
        with zipfile.ZipFile(Path(job_info["bundle_path"])) as bundle:
            labels = self._jsonl(bundle.read("holdout/labels.jsonl"))
            classes = json.loads(bundle.read("holdout/classes.json"))
            models = json.loads(bundle.read("models.json"))
        class_ids = [item["class_id"] for item in classes]
        truths = {row["image_id"]: row for row in labels}
        expected_keys = {(image_id, model_id) for image_id in truths for model_id in (job["champion_model_id"], job["challenger_model_id"])}
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in predictions:
            key = (row.get("image_id"), row.get("model_id"))
            probabilities = row.get("probabilities")
            if key not in expected_keys or key in observed or row.get("split") != truths[key[0]]["split"]:
                raise ValueError("Evaluation predictions contain missing, duplicate, unknown, or mis-split rows.")
            if not isinstance(probabilities, dict) or set(probabilities) != set(class_ids):
                raise ValueError("Evaluation probability keys differ from the frozen taxonomy.")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in probabilities.values()):
                raise ValueError("Evaluation probabilities must be finite values from 0 to 1.")
            observed[key] = row
        if set(observed) != expected_keys:
            raise ValueError("Evaluation predictions do not cover every image-model pair exactly once.")
        thresholds = {
            job["champion_model_id"]: models["champion"]["thresholds"],
            job["challenger_model_id"]: models["challenger"]["thresholds"],
        }
        required_splits = list(job["profile"].get("required_splits") or [])
        if not required_splits or set(required_splits) != {row["split"] for row in labels}:
            raise ValueError("Evaluation result Gate lineage differs from the frozen job profile.")
        def group_evidence(ids: list[str]) -> dict[str, Any]:
            model_rows: dict[str, list[tuple[set[str], set[str], dict[str, float]]]] = {}
            correctness: dict[str, dict[str, bool]] = {}
            for model_id in thresholds:
                pairs = []
                correctness[model_id] = {}
                for image_id in ids:
                    truth = set(truths[image_id]["class_ids"])
                    probs = observed[(image_id, model_id)]["probabilities"]
                    predicted = {class_id for class_id in class_ids if float(probs[class_id]) >= float(thresholds[model_id][class_id])}
                    pairs.append((truth, predicted, {key: float(value) for key, value in probs.items()}))
                    correctness[model_id][image_id] = truth == predicted
                model_rows[model_id] = pairs
            champion_metrics = self._metrics(model_rows[job["champion_model_id"]], class_ids)
            challenger_metrics = self._metrics(model_rows[job["challenger_model_id"]], class_ids)
            transitions = {"both_correct": 0, "champion_only_correct": 0, "challenger_only_correct": 0, "both_wrong": 0}
            for image_id in ids:
                champion_ok = correctness[job["champion_model_id"]][image_id]
                challenger_ok = correctness[job["challenger_model_id"]][image_id]
                key = "both_correct" if champion_ok and challenger_ok else "champion_only_correct" if champion_ok else "challenger_only_correct" if challenger_ok else "both_wrong"
                transitions[key] += 1
            affected_images = {
                "champion_only_correct": [
                    image_id for image_id in ids
                    if correctness[job["champion_model_id"]][image_id]
                    and not correctness[job["challenger_model_id"]][image_id]
                ],
                "challenger_only_correct": [
                    image_id for image_id in ids
                    if not correctness[job["champion_model_id"]][image_id]
                    and correctness[job["challenger_model_id"]][image_id]
                ],
            }
            return {
                "champion": champion_metrics, "challenger": challenger_metrics,
                "delta": {
                    "macro_map": challenger_metrics["macro_map"] - champion_metrics["macro_map"],
                    "macro_f1": challenger_metrics["macro_f1"] - champion_metrics["macro_f1"],
                    "micro_f1": challenger_metrics["micro_f1"] - champion_metrics["micro_f1"],
                    "exact_match_accuracy": challenger_metrics["exact_match_accuracy"] - champion_metrics["exact_match_accuracy"],
                    "false_positive_count": challenger_metrics["false_positive_count"] - champion_metrics["false_positive_count"],
                    "false_negative_count": challenger_metrics["false_negative_count"] - champion_metrics["false_negative_count"],
                },
                "paired_transitions": transitions,
                "affected_image_ids": affected_images,
            }
        current_ids = [
            image_id for image_id, truth in truths.items()
            if truth["split"] == LOGICAL_CURRENT_SPLIT
        ]
        core_ids = [
            image_id for image_id, truth in truths.items()
            if truth["split"] == LOGICAL_CORE_SPLIT
        ]
        current_evidence = group_evidence(current_ids)
        core_evidence = group_evidence(core_ids)
        cohort_evidence: dict[str, dict[str, Any]] = {}
        for cohort_id in sorted({str(row["cohort_id"]) for row in labels}):
            cohort_rows = [row for row in labels if row["cohort_id"] == cohort_id]
            cohort_ids = [row["image_id"] for row in cohort_rows]
            first = cohort_rows[0]
            cohort_evidence[cohort_id] = {
                "cohort_id": cohort_id,
                "source_round": first["source_round"],
                "origin_role": first["origin_role"],
                "logical_split": first["split"],
                "image_count": len(cohort_rows),
                "evidence": group_evidence(cohort_ids),
            }
        core_cohorts = [
            item for item in cohort_evidence.values()
            if item["logical_split"] == LOGICAL_CORE_SPLIT
        ]
        worst_core = min(
            core_cohorts,
            key=lambda item: (
                item["evidence"]["challenger"]["exact_match_accuracy"],
                item["evidence"]["challenger"]["macro_f1"],
                item["evidence"]["challenger"]["macro_map"],
                item["cohort_id"],
            ),
        )
        summary: dict[str, Any] = {
            "schema_version": 3,
            "current_gate": current_evidence,
            "core_safety": core_evidence,
            "cohorts": cohort_evidence,
            "worst_core_safety_cohort": worst_core,
            "overall": group_evidence(sorted(truths)),
            "splits": {
                LOGICAL_CURRENT_SPLIT: current_evidence,
                LOGICAL_CORE_SPLIT: core_evidence,
            },
            "test_read": False,
            "automatic_decision": None,
            "default_decision": None,
            "decision_options": ["retain", "promote"],
        }
        return {"source": resolved, "summary": summary, "manifest": manifest}

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
                raise ValueError("Evaluation result changed while entering the artifact store.")
            os.replace(temporary, destination)
            destination.chmod(0o444)
        except Exception:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
        return {**validated, "path": str(destination), "sha256": digest}

    def execute(self, run_id: str, cancellation: threading.Event) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("UPDATE evaluation_runs SET status='running',started_at=?,updated_at=? WHERE run_id=?", (now, now, run_id))
        self._append_event(run_id, "status", {"status": "running"})
        job = self.get_job(run["job_id"])
        spec = WorkerLaunchSpec(run_id=run_id, job_id=run["job_id"], bundle_path=Path(job["bundle_path"]),
                                output_dir=Path(run["log_path"]).parent, gpu_devices=tuple(run["gpu_devices"]),
                                world_size=run["world_size"], command=tuple(run["command"]))
        try:
            result = self.runner.run(spec, lambda kind, payload: self._append_event(run_id, kind, payload), cancellation,
                                     lambda pid: self._set_pid(run_id, pid))
            if cancellation.is_set():
                raise TrainingCancelled("Model evaluation was cancelled by the engineer.")
            verified = self._verify_and_store_result(run_id, result)
            finished = utc_now()
            evidence_id = make_id("evidence")
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                batch = db.execute("SELECT state FROM maintenance_batches WHERE batch_id=?", (run["batch_id"],)).fetchone()
                active = db.execute("SELECT model_id FROM active_models WHERE project_id=?", (run["project_id"],)).fetchone()
                if batch is None or batch["state"] != BatchState.CHALLENGER_TRAINED:
                    raise PermissionError("Batch state changed before evidence registration.")
                if active is None or active["model_id"] != job["champion_model_id"]:
                    raise PermissionError("Active Champion changed before evidence registration.")
                evaluated = require_transition(batch["state"], "record_challenger_evaluation", BATCH_TRANSITIONS, False)
                pending = require_transition(evaluated.target, "open_human_decision", BATCH_TRANSITIONS, False)
                db.execute(
                    "INSERT INTO evidence_reports(evidence_id,project_id,batch_id,champion_model_id,challenger_model_id,"
                    "metrics_json,artifact_path,artifact_sha256,created_at,source_evaluation_job_id,source_evaluation_run_id,evaluation_profile_id) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (evidence_id, run["project_id"], run["batch_id"], job["champion_model_id"], job["challenger_model_id"],
                     canonical_json(verified["summary"]), verified["path"], verified["sha256"], finished,
                     run["job_id"], run_id, job["evaluation_profile_id"]),
                )
                db.execute("UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?", (pending.target, finished, run["batch_id"]))
                db.execute("UPDATE evaluation_jobs SET status='registered',updated_at=? WHERE job_id=?", (finished, run["job_id"]))
                db.execute(
                    "UPDATE evaluation_runs SET status='result_verified',result_bundle_path=?,result_bundle_sha256=?,summary_json=?,"
                    "finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                    (verified["path"], verified["sha256"], canonical_json(verified["summary"]), finished, finished, run_id),
                )
                self.store._append_audit(
                    db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                    actor_id="locked_evaluation_worker", tool_name="record_challenger_evaluation",
                    from_state=batch["state"], to_state=evaluated.target,
                    payload={"job_id": run["job_id"], "run_id": run_id, "evidence_id": evidence_id,
                             "result_bundle_sha256": verified["sha256"], "test_read": False},
                )
                self.store._append_audit(
                    db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                    actor_id="evidence_registry", tool_name="open_human_decision",
                    from_state=evaluated.target, to_state=pending.target,
                    payload={"evidence_id": evidence_id, "automatic_decision": None},
                )
            self._append_event(run_id, "status", {"status": "result_verified", "evidence_id": evidence_id})
        except TrainingCancelled as error:
            self._finish_failure(run_id, "cancelled", str(error), "cancel_model_evaluation")
        except Exception as error:
            self._finish_failure(run_id, "failed", str(error), "fail_model_evaluation")

    def status(self, run_id: str) -> str:
        return str(self._get_run(run_id)["status"])

    def cancel_queued(self, db: Any, run_id: str, actor_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM evaluation_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Evaluation run not found: {run_id}")
        if row["status"] != "queued":
            raise PermissionError(f"Evaluation run cannot be queue-cancelled from status {row['status']}.")
        message = "Queued Champion–Challenger evaluation was cancelled before GPU execution."
        db.execute(
            "UPDATE evaluation_runs SET status='cancelled',error_message=?,finished_at=?,updated_at=? WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute("UPDATE evaluation_jobs SET status='exported',updated_at=? WHERE job_id=?", (now, row["job_id"]))
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="human",
            actor_id=actor_id, tool_name="cancel_queued_model_evaluation", from_state="queued",
            to_state="cancelled", payload={"run_id": run_id, "job_id": row["job_id"]},
        )

    def interrupt_abandoned(self, db: Any, run_id: str, now: str) -> None:
        row = db.execute(
            "SELECT run_id,job_id,project_id,batch_id,status FROM evaluation_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] in TERMINAL_RUN_STATUSES:
            return
        message = "Agent restarted while model evaluation held the GPU lease."
        db.execute(
            "UPDATE evaluation_runs SET status='interrupted',error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
            (message, now, now, run_id),
        )
        db.execute("UPDATE evaluation_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, row["job_id"]))
        self.store._append_audit(
            db, project_id=row["project_id"], batch_id=row["batch_id"], actor_type="system",
            actor_id="global_gpu_scheduler", tool_name="interrupt_model_evaluation",
            from_state=row["status"], to_state="interrupted",
            payload={"run_id": run_id, "job_id": row["job_id"], "reason": message},
        )

    def _finish_failure(self, run_id: str, status: str, message: str, action: str) -> None:
        run = self._get_run(run_id)
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE evaluation_runs SET status=?,error_message=?,finished_at=?,updated_at=?,pid=NULL WHERE run_id=?",
                       (status, message[:4000], now, now, run_id))
            db.execute("UPDATE evaluation_jobs SET status='failed',updated_at=? WHERE job_id=?", (now, run["job_id"]))
            self.store._append_audit(
                db, project_id=run["project_id"], batch_id=run["batch_id"], actor_type="system",
                actor_id="locked_evaluation_worker", tool_name=action,
                from_state=run["status"], to_state=status,
                payload={"job_id": run["job_id"], "run_id": run_id, "error": message[:1000]},
            )
        self._append_event(run_id, "status", {"status": status, "error": classify_worker_error(message)})

    def cancel_run(self, run_id: str, actor_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        if run["status"] not in {"queued", "running"}:
            raise PermissionError(f"Evaluation run cannot be cancelled from status {run['status']}.")
        with self.store.connect() as db:
            queue_row = db.execute("SELECT queue_id FROM gpu_job_queue WHERE run_id=?", (run_id,)).fetchone()
        if queue_row is None:
            raise RuntimeError("Evaluation run is not attached to the global GPU queue.")
        queue_item = self.scheduler.cancel(queue_row["queue_id"], actor_id)
        return {"run": self._get_run(run_id), "queue": queue_item}

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


def register_model_evaluation_tools(registry: ToolRegistry, service: ModelEvaluationService) -> None:
    registry.register(ToolDefinition(
        "create_model_evaluation_job",
        "Freeze a leakage-checked Champion/Challenger evaluation job for the fresh Current Gate; the backend automatically assembles cumulative Core Safety.",
        {"type": "object", "required": ["batch_id", "inbox_filename"], "properties": {
            "batch_id": {"type": "string", "minLength": 1},
            "inbox_filename": {"type": "string", "minLength": 1},
        }, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.create_job(args["batch_id"], args["inbox_filename"], ctx["actor_type"], ctx["actor_id"]),
    ))
    registry.register(ToolDefinition(
        "start_model_evaluation",
        "Run both frozen models on the fresh Current Gate and cumulative Core Safety. The Worker cannot retune thresholds, select, or deploy.",
        {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.start_job(args["job_id"], ctx["actor_type"], ctx["actor_id"]),
    ))
    registry.register(ToolDefinition(
        "cancel_model_evaluation",
        "Safely interrupt the attached evaluation Worker without changing either model or opening a decision.",
        {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string", "minLength": 1}}, "additionalProperties": False},
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.cancel_run(args["run_id"], ctx["actor_id"]),
    ))
