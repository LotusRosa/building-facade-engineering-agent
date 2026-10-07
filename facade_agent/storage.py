from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.states import (
    BATCH_TRANSITIONS,
    PROJECT_TRANSITIONS,
    BatchState,
    ProjectState,
    require_transition,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class ClosingConnection(sqlite3.Connection):
    """Commit or roll back on context exit, then release the Windows file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class Store:
    """SQLite repository with a compatibility snapshot for the current MVP UI."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.migrations = (
            Path(__file__).resolve().parent
            / "adapters"
            / "storage"
            / "migrations"
        )
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, factory=ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            applied = {
                row["version"]
                for row in db.execute("SELECT version FROM schema_migrations")
            }
            for migration in sorted(self.migrations.glob("*.sql")):
                if migration.name in applied:
                    continue
                sql = migration.read_text(encoding="utf-8")
                digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                db.executescript(sql)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,sha256) VALUES(?,?,?)",
                    (migration.name, utc_now(), digest),
                )
            self._initialize_legacy_tables(db)
            db.execute(
                "INSERT OR IGNORE INTO agent_settings(id,updated_at) VALUES(1,?)",
                (utc_now(),),
            )
            db.execute("PRAGMA optimize")

    @staticmethod
    def _initialize_legacy_tables(db: sqlite3.Connection) -> None:
        db.execute(
            "CREATE TABLE IF NOT EXISTS runtime ("
            "id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL, "
            "project_name TEXT NOT NULL DEFAULT '', "
            "classes_json TEXT NOT NULL DEFAULT '[]', "
            "metadata_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, "
            "action TEXT NOT NULL, from_state TEXT NOT NULL, to_state TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, event_sha256 TEXT NOT NULL UNIQUE)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS reviews ("
            "slice_id TEXT PRIMARY KEY, decision TEXT NOT NULL, "
            "excluded_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT OR IGNORE INTO runtime(id,state,updated_at) VALUES(1,'EMPTY',?)",
            (utc_now(),),
        )

    @staticmethod
    def _project_dict(db: sqlite3.Connection, project_id: str) -> dict[str, Any]:
        row = db.execute(
            "SELECT * FROM projects WHERE project_id=? AND deleted_at IS NULL", (project_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        classes = db.execute(
            "SELECT class_id,display_name,display_name_zh,display_name_en,"
            "description,sort_order FROM classes "
            "WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
            (project_id,),
        ).fetchall()
        batches = db.execute(
            "SELECT b.batch_id,b.name,b.state,b.formal_decision,b.decision_reason,"
            "b.created_at,b.updated_at,d.dataset_id,d.status AS dataset_status "
            "FROM maintenance_batches b LEFT JOIN datasets d ON d.batch_id=b.batch_id "
            "WHERE b.project_id=? ORDER BY b.created_at",
            (project_id,),
        ).fetchall()
        result = dict(row)
        result["classes"] = [dict(item) for item in classes]
        result["maintenance_batches"] = [dict(item) for item in batches]
        return result


    def _append_audit(
        self,
        db: sqlite3.Connection,
        *,
        project_id: str | None,
        batch_id: str | None,
        actor_type: str,
        actor_id: str,
        tool_name: str,
        from_state: str | None,
        to_state: str | None,
        payload: dict[str, Any],
    ) -> str:
        previous = db.execute(
            "SELECT event_sha256 FROM audit_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_sha256"] if previous else None
        body = {
            "occurred_at": utc_now(),
            "project_id": project_id,
            "batch_id": batch_id,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "tool_name": tool_name,
            "from_state": from_state,
            "to_state": to_state,
            "payload": payload,
            "previous_event_sha256": previous_hash,
        }
        digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        db.execute(
            "INSERT INTO audit_events("
            "occurred_at,project_id,batch_id,actor_type,actor_id,tool_name,"
            "from_state,to_state,payload_json,previous_event_sha256,event_sha256"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                body["occurred_at"],
                project_id,
                batch_id,
                actor_type,
                actor_id,
                tool_name,
                from_state,
                to_state,
                canonical_json(payload),
                previous_hash,
                digest,
            ),
        )
        return digest

    def verify_audit_chain(self) -> dict[str, Any]:
        previous_hash: str | None = None
        checked = 0
        with self.connect() as db:
            rows = db.execute("SELECT * FROM audit_events ORDER BY event_id").fetchall()
        for row in rows:
            body = {
                "occurred_at": row["occurred_at"],
                "project_id": row["project_id"],
                "batch_id": row["batch_id"],
                "actor_type": row["actor_type"],
                "actor_id": row["actor_id"],
                "tool_name": row["tool_name"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "payload": json.loads(row["payload_json"]),
                "previous_event_sha256": row["previous_event_sha256"],
            }
            expected = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            if row["previous_event_sha256"] != previous_hash or row["event_sha256"] != expected:
                return {
                    "valid": False,
                    "checked_events": checked,
                    "failed_event_id": row["event_id"],
                }
            previous_hash = row["event_sha256"]
            checked += 1
        return {"valid": True, "checked_events": checked, "head_sha256": previous_hash}

    def record_labeled_bundle_import(
        self,
        *,
        dataset_id: str,
        bundle_filename: str,
        bundle_sha256: str,
        agent_role: str,
        source_split: str,
        image_count: int,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dataset = db.execute(
                "SELECT project_id,batch_id,status FROM datasets WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            if dataset is None:
                raise KeyError(f"Dataset not found: {dataset_id}")
            if dataset["status"] not in {"validated", "frozen"}:
                raise PermissionError("Bundle provenance is recorded only after validation.")
            digest = self._append_audit(
                db,
                project_id=dataset["project_id"],
                batch_id=dataset["batch_id"],
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="import_verified_labeled_bundle",
                from_state=None,
                to_state=None,
                payload={
                    "dataset_id": dataset_id,
                    "bundle_filename": str(bundle_filename),
                    "bundle_sha256": str(bundle_sha256),
                    "agent_role": str(agent_role),
                    "source_split": str(source_split),
                    "image_count": int(image_count),
                },
            )
        return {"audit_event_sha256": digest}

    def create_project(
        self,
        *,
        project_name: str,
        class_names: list[str],
        model_storage_root: str | None = None,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        name = str(project_name).strip()
        normalized = [str(value).strip() for value in class_names if str(value).strip()]
        if not name or not normalized:
            raise ValueError("Project name and at least one class are required.")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("Class names must be unique within a project.")

        project_id = make_id("project")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO projects(project_id,name,state,model_storage_root,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (
                    project_id,
                    name,
                    ProjectState.PROJECT_CREATED,
                    model_storage_root,
                    now,
                    now,
                ),
            )
            for index, class_name in enumerate(normalized, start=1):
                db.execute(
                    "INSERT INTO classes("
                    "class_id,project_id,display_name,sort_order,created_at"
                    ") VALUES(?,?,?,?,?)",
                    (f"{project_id}_class_{index:03d}", project_id, class_name, index, now),
                )
            db.execute(
                "UPDATE agent_settings SET active_project_id=?,updated_at=? WHERE id=1",
                (project_id, now),
            )
            digest = self._append_audit(
                db,
                project_id=project_id,
                batch_id=None,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="create_project",
                from_state=None,
                to_state=ProjectState.PROJECT_CREATED,
                payload={
                    "project_name": name,
                    "class_names": normalized,
                    **(
                        {
                            "model_storage_root_sha256": hashlib.sha256(
                                model_storage_root.encode("utf-8")
                            ).hexdigest(),
                            "model_storage_leaf": Path(model_storage_root).name,
                        }
                        if model_storage_root
                        else {}
                    ),
                },
            )
            result = self._project_dict(db, project_id)
        return {"project": result, "audit_event_sha256": digest}

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT project_id,name,state,active_batch_id,model_storage_root,"
                "model_storage_locked_at,created_at,updated_at "
                "FROM projects WHERE deleted_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as db:
            return self._project_dict(db, project_id)

    def delete_project(
        self,
        *,
        project_id: str,
        confirmation_name: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        """Remove project data while retaining the minimum audit-chain tombstone."""
        if actor_type != "human":
            raise PermissionError("Only a human can delete a project.")

        managed_paths: list[str] = []
        counts: dict[str, int] = {}
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            project = db.execute(
                "SELECT project_id,name,state,model_storage_root FROM projects "
                "WHERE project_id=? AND deleted_at IS NULL",
                (project_id,),
            ).fetchone()
            if project is None:
                raise KeyError(f"Project not found: {project_id}")
            if str(confirmation_name) != project["name"]:
                raise PermissionError("Type the complete project name to confirm deletion.")

            for table in (
                "training_runs",
                "screening_runs",
                "challenger_runs",
                "evaluation_runs",
            ):
                active = db.execute(
                    f"SELECT run_id,status FROM {table} WHERE project_id=? "
                    "AND status IN ('queued','running','cancel_requested') LIMIT 1",
                    (project_id,),
                ).fetchone()
                if active is not None:
                    raise PermissionError(
                        f"Project has an active run ({active['run_id']}: {active['status']}). "
                        "Stop it before deleting the project."
                    )

            for table, column in (
                ("images", "stored_path"),
                ("training_jobs", "bundle_path"),
                ("screening_jobs", "bundle_path"),
                ("challenger_jobs", "bundle_path"),
                ("evaluation_jobs", "bundle_path"),
                ("training_runs", "log_path"),
                ("training_runs", "result_bundle_path"),
                ("screening_runs", "log_path"),
                ("screening_runs", "result_bundle_path"),
                ("challenger_runs", "log_path"),
                ("challenger_runs", "result_bundle_path"),
                ("evaluation_runs", "log_path"),
                ("evaluation_runs", "result_bundle_path"),
                ("model_versions", "checkpoint_path"),
                ("evidence_reports", "artifact_path"),
                ("evaluation_gates", "artifact_path"),
                ("evaluation_cohorts", "artifact_path"),
            ):
                if table == "images":
                    query = (
                        f"SELECT {column} AS path FROM images i JOIN datasets d "
                        "ON d.dataset_id=i.dataset_id WHERE d.project_id=?"
                    )
                else:
                    query = f"SELECT {column} AS path FROM {table} WHERE project_id=?"
                managed_paths.extend(
                    row["path"]
                    for row in db.execute(query, (project_id,)).fetchall()
                    if row["path"]
                )

            self._append_audit(
                db,
                project_id=project_id,
                batch_id=None,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="delete_project",
                from_state=project["state"],
                to_state="DELETED",
                payload={"project_name": project["name"], "confirmation": "exact_name"},
            )

            db.execute(
                "UPDATE model_versions SET parent_model_id=NULL,source_batch_id=NULL,"
                "source_training_job_id=NULL,source_training_run_id=NULL,"
                "source_challenger_job_id=NULL,source_challenger_run_id=NULL "
                "WHERE project_id=?",
                (project_id,),
            )
            db.execute(
                "UPDATE agent_settings SET active_project_id=NULL,updated_at=? "
                "WHERE active_project_id=?",
                (now, project_id),
            )

            delete_steps = (
                ("final_test_image_inventory", "project_id=?"),
                ("round_completion_summaries", "project_id=?"),
                ("evaluation_cohorts", "project_id=?"),
                ("evaluation_gates", "project_id=?"),
                ("evidence_reports", "project_id=?"),
                ("evaluation_run_events", "run_id IN (SELECT run_id FROM evaluation_runs WHERE project_id=?)"),
                ("evaluation_runs", "project_id=?"),
                ("evaluation_jobs", "project_id=?"),
                ("failure_slice_members", "slice_id IN (SELECT s.slice_id FROM failure_slices s JOIN maintenance_batches b ON b.batch_id=s.batch_id WHERE b.project_id=?)"),
                ("slice_reviews", "slice_id IN (SELECT s.slice_id FROM failure_slices s JOIN maintenance_batches b ON b.batch_id=s.batch_id WHERE b.project_id=?)"),
                ("failure_slices", "batch_id IN (SELECT batch_id FROM maintenance_batches WHERE project_id=?)"),
                ("challenger_run_events", "run_id IN (SELECT run_id FROM challenger_runs WHERE project_id=?)"),
                ("challenger_runs", "project_id=?"),
                ("challenger_jobs", "project_id=?"),
                ("failure_review_versions", "batch_id IN (SELECT batch_id FROM maintenance_batches WHERE project_id=?)"),
                ("screening_run_events", "run_id IN (SELECT run_id FROM screening_runs WHERE project_id=?)"),
                ("screening_runs", "project_id=?"),
                ("screening_jobs", "project_id=?"),
                ("training_run_events", "run_id IN (SELECT run_id FROM training_runs WHERE project_id=?)"),
                ("training_runs", "project_id=?"),
                ("training_jobs", "project_id=?"),
                ("active_models", "project_id=?"),
                ("model_versions", "project_id=?"),
                ("image_annotation_labels", "image_id IN (SELECT i.image_id FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id WHERE d.project_id=?)"),
                ("image_annotations", "image_id IN (SELECT i.image_id FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id WHERE d.project_id=?)"),
                ("label_versions", "dataset_id IN (SELECT dataset_id FROM datasets WHERE project_id=?)"),
                ("images", "dataset_id IN (SELECT dataset_id FROM datasets WHERE project_id=?)"),
                ("datasets", "project_id=?"),
                ("deployment_decisions", "batch_id IN (SELECT batch_id FROM maintenance_batches WHERE project_id=?)"),
                ("pending_tool_calls", "project_id=?"),
                ("llm_messages", "project_id=?"),
                ("tool_runs", "project_id=?"),
                ("artifact_refs", "project_id=?"),
                ("classes", "project_id=?"),
            )
            for table, predicate in delete_steps:
                cursor = db.execute(f"DELETE FROM {table} WHERE {predicate}", (project_id,))
                counts[table] = max(0, cursor.rowcount)

            db.execute(
                "UPDATE projects SET state='DELETED',active_batch_id=NULL,deleted_at=?,updated_at=? "
                "WHERE project_id=?",
                (now, now, project_id),
            )
            runtime = db.execute("SELECT metadata_json FROM runtime WHERE id=1").fetchone()
            runtime_project_id = None
            if runtime is not None:
                try:
                    runtime_project_id = json.loads(runtime["metadata_json"]).get("project_id")
                except (TypeError, ValueError):
                    runtime_project_id = None
            if runtime_project_id == project_id:
                db.execute(
                    "UPDATE runtime SET state='EMPTY',project_name='',classes_json='[]',"
                    "metadata_json='{}',updated_at=? WHERE id=1",
                    (now,),
                )

        file_result = self._quarantine_project_files(
            project_id,
            managed_paths,
            project_model_root=project["model_storage_root"],
        )
        return {
            "project_id": project_id,
            "project_name": project["name"],
            "deleted": True,
            "records_deleted": counts,
            "audit_tombstone_retained": True,
            **file_result,
        }

    def _quarantine_project_files(
        self,
        project_id: str,
        stored_paths: list[str],
        *,
        project_model_root: str | None = None,
    ) -> dict[str, Any]:
        root = self.path.parent.parent.resolve()
        candidates: list[tuple[Path, bool]] = [
            (root / "projects" / project_id, False),
            (root / "models" / project_id, False),
        ]
        candidates.extend((Path(value), False) for value in stored_paths)
        if project_model_root:
            candidates.append((Path(project_model_root), True))
        resolved: list[tuple[Path, bool]] = []
        failures: list[str] = []
        for candidate, allow_external in candidates:
            try:
                path = candidate.resolve()
            except (OSError, ValueError):
                continue
            if path == root:
                failures.append(f"Agent root was not moved: {path}")
                continue
            is_internal = path == root or root in path.parents
            if not is_internal:
                if not allow_external:
                    continue
                if path == Path(path.anchor) or path == root or path in root.parents:
                    failures.append(f"Unsafe external model root was not moved: {path}")
                    continue
                current = Path(path.anchor)
                unsafe_link = False
                for part in path.parts[1:]:
                    current /= part
                    if current.is_symlink():
                        unsafe_link = True
                        break
                if unsafe_link:
                    failures.append(f"Symlinked external model root was not moved: {path}")
                    continue
            if path.exists() and not any(
                parent == path or parent in path.parents for parent, _ in resolved
            ):
                resolved = [
                    item for item in resolved if path not in item[0].parents
                ]
                resolved.append((path, not is_internal))

        trash = root / "trash" / "deleted_projects" / f"{project_id}_{uuid.uuid4().hex[:8]}"
        moved: list[str] = []
        for source, is_external in sorted(
            resolved, key=lambda item: len(item[0].parts)
        ):
            try:
                relative = (
                    Path("external_model_storage") / source.name
                    if is_external
                    else source.relative_to(root)
                )
                destination = trash / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append(str(relative))
            except OSError as error:
                failures.append(f"{source}: {error}")
        return {
            "quarantine_path": str(trash) if moved else None,
            "paths_quarantined": moved,
            "file_cleanup_warnings": failures,
        }


    def transition_project(
        self,
        *,
        project_id: str,
        action: str,
        confirmed: bool,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state FROM projects WHERE project_id=?", (project_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Project not found: {project_id}")
            transition = require_transition(
                row["state"], action, PROJECT_TRANSITIONS, confirmed
            )
            now = utc_now()
            db.execute(
                "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                (transition.target, now, project_id),
            )
            digest = self._append_audit(
                db,
                project_id=project_id,
                batch_id=None,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name=action,
                from_state=row["state"],
                to_state=transition.target,
                payload=payload or {},
            )
            result = self._project_dict(db, project_id)
        return {"project": result, "audit_event_sha256": digest}

    def create_maintenance_batch(
        self,
        *,
        project_id: str,
        batch_name: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        name = str(batch_name).strip()
        if not name:
            raise ValueError("Maintenance batch name is required.")
        now = utc_now()
        batch_id = make_id("batch")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            project = db.execute(
                "SELECT state,active_batch_id FROM projects WHERE project_id=?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise KeyError(f"Project not found: {project_id}")
            if project["state"] not in {
                ProjectState.CHAMPION_READY,
                ProjectState.SCREENING_READY,
            }:
                raise PermissionError("An active Champion is required before maintenance.")
            active_model = db.execute(
                "SELECT model_id FROM active_models WHERE project_id=?", (project_id,)
            ).fetchone()
            if active_model is None:
                raise PermissionError("An active Champion model record is required before maintenance.")
            if project["active_batch_id"]:
                active = db.execute(
                    "SELECT state FROM maintenance_batches WHERE batch_id=?",
                    (project["active_batch_id"],),
                ).fetchone()
                if active and active["state"] not in {BatchState.DEPLOYED, BatchState.HELD}:
                    raise PermissionError("Finish the active maintenance batch first.")
            db.execute(
                "INSERT INTO maintenance_batches("
                "batch_id,project_id,name,state,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?)",
                (batch_id, project_id, name, BatchState.CREATED, now, now),
            )
            db.execute(
                "UPDATE projects SET active_batch_id=?,updated_at=? WHERE project_id=?",
                (batch_id, now, project_id),
            )
            digest = self._append_audit(
                db,
                project_id=project_id,
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="create_maintenance_batch",
                from_state=None,
                to_state=BatchState.CREATED,
                payload={"batch_name": name},
            )
            result = dict(
                db.execute(
                    "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
            )
        return {"maintenance_batch": result, "audit_event_sha256": digest}

    def create_maintenance_dataset(
        self,
        *,
        batch_id: str,
        name: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        dataset_name = str(name).strip()
        if not dataset_name:
            raise ValueError("Dataset name is required.")
        dataset_id = make_id("dataset")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            transition = require_transition(
                batch["state"],
                "record_maintenance_data_import",
                BATCH_TRANSITIONS,
                False,
            )
            existing = db.execute(
                "SELECT dataset_id FROM datasets WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if existing is not None:
                raise PermissionError("This maintenance batch already has a dataset.")
            project = db.execute(
                "SELECT state,active_batch_id FROM projects WHERE project_id=?",
                (batch["project_id"],),
            ).fetchone()
            if project is None or project["active_batch_id"] != batch_id:
                raise PermissionError("The maintenance batch is not active for this project.")
            if project["state"] not in {
                ProjectState.CHAMPION_READY,
                ProjectState.SCREENING_READY,
            }:
                raise PermissionError("An active Champion is required before maintenance data import.")
            db.execute(
                "INSERT INTO datasets(dataset_id,project_id,batch_id,name,role,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    batch["project_id"],
                    batch_id,
                    dataset_name,
                    "maintenance",
                    now,
                    now,
                ),
            )
            db.execute(
                "UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?",
                (transition.target, now, batch_id),
            )
            digest = self._append_audit(
                db,
                project_id=batch["project_id"],
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="create_maintenance_dataset",
                from_state=batch["state"],
                to_state=transition.target,
                payload={"dataset_id": dataset_id, "dataset_name": dataset_name},
            )
        return {
            "dataset": self.get_dataset(dataset_id),
            "maintenance_batch": self.get_maintenance_batch(batch_id),
            "audit_event_sha256": digest,
        }

    def get_maintenance_batch(self, batch_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            dataset = db.execute(
                "SELECT dataset_id FROM datasets WHERE batch_id=?", (batch_id,)
            ).fetchone()
        result = dict(row)
        result["dataset"] = self.get_dataset(dataset["dataset_id"]) if dataset else None
        return result

    def list_maintenance_batches(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self.connect() as db:
            ids = [
                row["batch_id"]
                for row in db.execute(
                    "SELECT batch_id FROM maintenance_batches WHERE project_id=? "
                    "ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
        return [self.get_maintenance_batch(batch_id) for batch_id in ids]

    def transition_batch(
        self,
        *,
        batch_id: str,
        action: str,
        confirmed: bool,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            transition = require_transition(
                row["state"], action, BATCH_TRANSITIONS, confirmed
            )
            now = utc_now()
            db.execute(
                "UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?",
                (transition.target, now, batch_id),
            )
            digest = self._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name=action,
                from_state=row["state"],
                to_state=transition.target,
                payload=payload or {},
            )
            result = dict(
                db.execute(
                    "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
            )
        return {"maintenance_batch": result, "audit_event_sha256": digest}

    def get_round_completion(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM round_completion_summaries WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["final_test_read"] = bool(result["final_test_read"])
        return result

    def record_batch_decision(
        self,
        *,
        batch_id: str,
        decision: str,
        reason: str,
        confirmed: bool,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        requested_decision = str(decision).strip().lower()
        decision_aliases = {
            "retain": "hold",
            "promote": "deploy",
            "hold": "hold",
            "deploy": "deploy",
        }
        normalized_decision = decision_aliases.get(requested_decision)
        normalized_reason = str(reason).strip()
        if normalized_decision is None:
            raise ValueError("Decision must be retain or promote.")
        external_decision = "promote" if normalized_decision == "deploy" else "retain"
        if not normalized_reason:
            raise ValueError("A human decision reason is required.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            if row["formal_decision"] is not None:
                raise PermissionError("This maintenance batch already has a final decision.")
            early_hold = False
            review = None
            if normalized_decision == "hold" and row["state"] == BatchState.FAILURE_REVIEW_COMPLETED:
                review = db.execute(
                    "SELECT review_version_id,eligible_slice_count,review_sha256 "
                    "FROM failure_review_versions WHERE batch_id=? ORDER BY created_at DESC LIMIT 1",
                    (batch_id,),
                ).fetchone()
                if review is None:
                    raise PermissionError("A frozen Failure Slice review is required before holding this batch.")
                challenger = db.execute(
                    "SELECT job_id FROM challenger_jobs WHERE batch_id=? LIMIT 1", (batch_id,)
                ).fetchone()
                if challenger is not None:
                    raise PermissionError("This batch already has a frozen Challenger job and cannot use the pre-training Hold path.")
                early_hold = True
            action = (
                "deploy_challenger"
                if normalized_decision == "deploy"
                else "hold_without_challenger" if early_hold else "hold_challenger"
            )
            transition = require_transition(
                row["state"], action, BATCH_TRANSITIONS, confirmed
            )
            if actor_type != "human":
                raise PermissionError("Only a human engineer can make the final decision.")
            evidence = db.execute(
                "SELECT evidence_id,champion_model_id,challenger_model_id,artifact_sha256 "
                "FROM evidence_reports WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            if evidence is None and not early_hold:
                raise PermissionError("A verified Champion-Challenger evidence report is required for the final decision.")
            active_before = db.execute(
                "SELECT model_id FROM active_models WHERE project_id=?", (row["project_id"],)
            ).fetchone()
            if active_before is None:
                raise PermissionError("This project has no active Champion.")
            champion_model_id = active_before["model_id"] if early_hold else evidence["champion_model_id"]
            if active_before["model_id"] != champion_model_id:
                raise PermissionError("The evaluated Champion is no longer active.")
            now = utc_now()
            decision_id = make_id("decision")
            db.execute(
                "INSERT INTO deployment_decisions("
                "decision_id,batch_id,decision,reason,actor_id,decided_at"
                ") VALUES(?,?,?,?,?,?)",
                (decision_id, batch_id, normalized_decision, normalized_reason, actor_id, now),
            )
            db.execute(
                "UPDATE maintenance_batches SET state=?,formal_decision=?,"
                "decision_reason=?,updated_at=? WHERE batch_id=?",
                (transition.target, normalized_decision, normalized_reason, now, batch_id),
            )
            activated_cohorts: list[dict[str, Any]] = []
            cumulative_core_safety_images = 0
            current_gate_cohort = None
            core_safety_cohort = None
            if not early_hold:
                pending_cohorts = [
                    dict(item)
                    for item in db.execute(
                        "SELECT * FROM evaluation_cohorts "
                        "WHERE project_id=? AND source_batch_id=? AND status='pending' "
                        "ORDER BY origin_role,cohort_id",
                        (row["project_id"], batch_id),
                    ).fetchall()
                ]
                if len(pending_cohorts) not in {0, 2}:
                    raise ValueError(
                        "A completed evaluation round must activate exactly its Current Gate and Core Safety cohort."
                    )
                if pending_cohorts:
                    current = [item for item in pending_cohorts if item["origin_role"] == "current_gate"]
                    safety = [
                        item for item in pending_cohorts
                        if item["origin_role"] in {"core_safety_seed", "core_safety_addition"}
                    ]
                    if len(current) != 1 or len(safety) != 1:
                        raise ValueError(
                            "A completed evaluation round has invalid cohort roles."
                        )
                    current_gate_cohort = current[0]
                    core_safety_cohort = safety[0]
                    db.execute(
                        "UPDATE evaluation_cohorts SET status='active',activated_at=? "
                        "WHERE project_id=? AND source_batch_id=? AND status='pending'",
                        (now, row["project_id"], batch_id),
                    )
                    activated_cohorts = [
                        dict(item)
                        for item in db.execute(
                            "SELECT * FROM evaluation_cohorts WHERE project_id=? AND source_batch_id=? "
                            "ORDER BY origin_role,cohort_id",
                            (row["project_id"], batch_id),
                        ).fetchall()
                    ]
                cumulative_core_safety_images = int(
                    db.execute(
                        "SELECT COALESCE(SUM(image_count),0) AS total FROM evaluation_cohorts "
                        "WHERE project_id=? AND status='active'",
                        (row["project_id"],),
                    ).fetchone()["total"]
                )
            activated_gate = None
            if not early_hold:
                pending_gates = db.execute(
                    "SELECT gate_id,gate_key FROM evaluation_gates "
                    "WHERE project_id=? AND source_batch_id=? AND role='round_gate' AND status='pending'",
                    (row["project_id"], batch_id),
                ).fetchall()
                if len(pending_gates) > 1:
                    raise ValueError("A maintenance round cannot activate more than one current Gate.")
                if pending_gates:
                    activated_gate = dict(pending_gates[0])
                    db.execute(
                        "UPDATE evaluation_gates SET status='active',activated_at=? WHERE gate_id=?",
                        (now, activated_gate["gate_id"]),
                    )
            active_after = champion_model_id
            if normalized_decision == "deploy":
                challenger = db.execute(
                    "SELECT role FROM model_versions WHERE model_id=? AND project_id=?",
                    (evidence["challenger_model_id"], row["project_id"]),
                ).fetchone()
                if challenger is None or challenger["role"] != "challenger":
                    raise PermissionError("The evaluated Challenger is not deployable from the model registry.")
                db.execute("UPDATE model_versions SET role='archived' WHERE model_id=?", (evidence["champion_model_id"],))
                db.execute("UPDATE model_versions SET role='champion' WHERE model_id=?", (evidence["challenger_model_id"],))
                db.execute(
                    "UPDATE active_models SET model_id=?,updated_at=? WHERE project_id=?",
                    (evidence["challenger_model_id"], now, row["project_id"]),
                )
                active_after = evidence["challenger_model_id"]
            round_completion = None
            if not early_hold and current_gate_cohort is not None and core_safety_cohort is not None:
                db.execute(
                    "INSERT INTO round_completion_summaries("
                    "batch_id,project_id,decision,reason,previous_champion_model_id,"
                    "active_champion_model_id,challenger_model_id,current_gate_cohort_id,"
                    "core_safety_cohort_id,cumulative_core_safety_images,evidence_id,"
                    "final_test_read,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,0,?)",
                    (
                        batch_id,
                        row["project_id"],
                        external_decision,
                        normalized_reason,
                        champion_model_id,
                        active_after,
                        evidence["challenger_model_id"],
                        current_gate_cohort["cohort_id"],
                        core_safety_cohort["cohort_id"],
                        cumulative_core_safety_images,
                        evidence["evidence_id"],
                        now,
                    ),
                )
                round_completion = dict(
                    db.execute(
                        "SELECT * FROM round_completion_summaries WHERE batch_id=?",
                        (batch_id,),
                    ).fetchone()
                )
                round_completion["final_test_read"] = bool(round_completion["final_test_read"])
            digest = self._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name=action,
                from_state=row["state"],
                to_state=transition.target,
                payload={
                    "decision": external_decision if not early_hold else "hold_before_evaluation",
                    "reason": normalized_reason,
                    "champion_model_id": champion_model_id,
                    "challenger_model_id": evidence["challenger_model_id"] if evidence else None,
                    "evidence_sha256": evidence["artifact_sha256"] if evidence else None,
                    "review_version_id": review["review_version_id"] if review else None,
                    "review_sha256": review["review_sha256"] if review else None,
                    "early_hold_reason": (
                        "no_eligible_failure_slices"
                        if early_hold and int(review["eligible_slice_count"]) == 0
                        else "human_hold_before_challenger" if early_hold else None
                    ),
                    "active_model_before": active_before["model_id"],
                    "active_model_after": active_after,
                    "activated_gate_id": activated_gate["gate_id"] if activated_gate else None,
                    "activated_gate_key": activated_gate["gate_key"] if activated_gate else None,
                    "activated_cohort_ids": [item["cohort_id"] for item in activated_cohorts],
                    "cumulative_core_safety_images": cumulative_core_safety_images,
                    "final_test_read": False,
                },
            )
            result = dict(
                db.execute(
                    "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
            )
        return {
            "decision_id": decision_id,
            "maintenance_batch": result,
            "activated_cohorts": activated_cohorts,
            "round_completion": round_completion,
            "audit_event_sha256": digest,
        }


    def snapshot(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runtime WHERE id=1").fetchone()
            events = db.execute(
                "SELECT id,occurred_at,action,from_state,to_state,event_sha256 "
                "FROM events ORDER BY id DESC LIMIT 30"
            ).fetchall()
            reviews = db.execute(
                "SELECT * FROM reviews ORDER BY slice_id"
            ).fetchall()
            setting = db.execute(
                "SELECT active_project_id FROM agent_settings WHERE id=1"
            ).fetchone()
            active_project = None
            if setting and setting["active_project_id"]:
                active_project = self._project_dict(db, setting["active_project_id"])
        return {
            "state": row["state"],
            "project_name": row["project_name"],
            "classes": json.loads(row["classes_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "updated_at": row["updated_at"],
            "project_id": active_project["project_id"] if active_project else None,
            "active_project": active_project,
            "projects": self.list_projects(),
            "audit_chain": self.verify_audit_chain(),
            "events": [dict(item) for item in events],
            "reviews": [
                {
                    "slice_id": item["slice_id"],
                    "decision": item["decision"],
                    "excluded": json.loads(item["excluded_json"]),
                    "updated_at": item["updated_at"],
                }
                for item in reviews
            ],
        }

    def transition(self, action: str, target: str, payload: dict) -> dict[str, Any]:
        now = utc_now()
        modern_result: dict[str, Any] | None = None
        if action == "create_project":
            modern_result = self.create_project(
                project_name=str(payload.get("project_name", "")),
                class_names=list(payload.get("classes", [])),
            )
        elif action == "confirm_classes":
            with self.connect() as lookup:
                runtime = lookup.execute(
                    "SELECT metadata_json FROM runtime WHERE id=1"
                ).fetchone()
                project_id = json.loads(runtime["metadata_json"]).get("project_id")
            if project_id:
                self.transition_project(
                    project_id=project_id,
                    action="confirm_taxonomy",
                    confirmed=True,
                )

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runtime WHERE id=1").fetchone()
            metadata = json.loads(row["metadata_json"])
            project_name = row["project_name"]
            classes = json.loads(row["classes_json"])
            if action == "create_project":
                project_name = modern_result["project"]["name"]
                classes = [
                    item["display_name"]
                    for item in modern_result["project"]["classes"]
                ]
                metadata["project_id"] = modern_result["project"]["project_id"]
            metadata.update(payload.get("metadata", {}))
            body = {
                "occurred_at": now,
                "action": action,
                "from_state": row["state"],
                "to_state": target,
                "payload": payload,
            }
            digest = hashlib.sha256(
                canonical_json(body).encode("utf-8")
            ).hexdigest()
            db.execute(
                "UPDATE runtime SET state=?,project_name=?,classes_json=?,"
                "metadata_json=?,updated_at=? WHERE id=1",
                (
                    target,
                    project_name,
                    json.dumps(classes, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                ),
            )
            db.execute(
                "INSERT INTO events("
                "occurred_at,action,from_state,to_state,payload_json,event_sha256"
                ") VALUES(?,?,?,?,?,?)",
                (
                    now,
                    action,
                    row["state"],
                    target,
                    json.dumps(payload, ensure_ascii=False),
                    digest,
                ),
            )
        return self.snapshot()

    def save_review(
        self, slice_id: str, decision: str, excluded: list[str]
    ) -> dict[str, Any]:
        if decision not in {"accept", "trim", "reject"}:
            raise ValueError("Review decision must be Accept, Trim, or Reject.")
        if not str(slice_id).strip():
            raise ValueError("A failure slice identifier is required.")
        if decision != "trim" and excluded:
            raise ValueError("Only Trim can contain excluded member identifiers.")
        with self.connect() as db:
            state = db.execute(
                "SELECT state FROM runtime WHERE id=1"
            ).fetchone()["state"]
            if state != "FAILURE_SLICES_READY":
                raise PermissionError("Failure Slice review is not available yet.")
            db.execute(
                "INSERT INTO reviews(slice_id,decision,excluded_json,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(slice_id) DO UPDATE SET "
                "decision=excluded.decision,excluded_json=excluded.excluded_json,"
                "updated_at=excluded.updated_at",
                (slice_id, decision, json.dumps(excluded), utc_now()),
            )
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        """Reset only the compatibility demo; governed project records remain intact."""
        with self.connect() as db:
            db.execute("DELETE FROM reviews")
            db.execute("DELETE FROM events")
            db.execute(
                "UPDATE runtime SET state='EMPTY',project_name='',classes_json='[]',"
                "metadata_json='{}',updated_at=? WHERE id=1",
                (utc_now(),),
            )
            db.execute(
                "UPDATE agent_settings SET active_project_id=NULL,updated_at=? WHERE id=1",
                (utc_now(),),
            )
        return self.snapshot()




    def create_dataset(
        self,
        *,
        project_id: str,
        name: str,
        role: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        dataset_name = str(name).strip()
        dataset_role = str(role).strip()
        if not dataset_name:
            raise ValueError("Dataset name is required.")
        if dataset_role not in {"initial_training", "screening", "maintenance"}:
            raise ValueError("Unsupported dataset role.")
        dataset_id = make_id("dataset")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            project = db.execute(
                "SELECT state FROM projects WHERE project_id=?", (project_id,)
            ).fetchone()
            if project is None:
                raise KeyError(f"Project not found: {project_id}")
            if project["state"] != ProjectState.TAXONOMY_CONFIRMED:
                raise PermissionError(
                    "Confirm the project taxonomy before importing a dataset."
                )
            db.execute(
                "INSERT INTO datasets(dataset_id,project_id,name,role,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (dataset_id, project_id, dataset_name, dataset_role, now, now),
            )
            db.execute(
                "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                (ProjectState.DATA_IMPORTED, now, project_id),
            )
            digest = self._append_audit(
                db,
                project_id=project_id,
                batch_id=None,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="import_dataset",
                from_state=project["state"],
                to_state=ProjectState.DATA_IMPORTED,
                payload={
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "role": dataset_role,
                },
            )
        return {
            "dataset": self.get_dataset(dataset_id),
            "audit_event_sha256": digest,
        }

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Dataset not found: {dataset_id}")
            stats = db.execute(
                "SELECT count(*) AS images,"
                "sum(CASE WHEN health_status='ok' THEN 1 ELSE 0 END) AS readable "
                "FROM images WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            labeled = db.execute(
                "SELECT count(*) FROM image_annotations a JOIN images i "
                "ON i.image_id=a.image_id WHERE i.dataset_id=? AND a.status='complete'",
                (dataset_id,),
            ).fetchone()[0]
            version = db.execute(
                "SELECT label_version_id,version_number,labels_sha256,image_count,created_at "
                "FROM label_versions WHERE dataset_id=? "
                "ORDER BY version_number DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
        result = dict(row)
        result["image_count"] = int(stats["images"] or 0)
        result["readable_count"] = int(stats["readable"] or 0)
        result["labeled_count"] = int(labeled)
        result["label_version"] = dict(version) if version else None
        return result

    def list_datasets(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            ids = [
                row["dataset_id"]
                for row in db.execute(
                    "SELECT dataset_id FROM datasets WHERE project_id=? "
                    "ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
        return [self.get_dataset(dataset_id) for dataset_id in ids]

    def register_image(
        self,
        *,
        dataset_id: str,
        filename: str,
        stored_path: str,
        sha256: str,
        size_bytes: int,
        mime_type: str,
        width: int | None,
        height: int | None,
        health_status: str,
    ) -> dict[str, Any]:
        if health_status not in {"ok", "corrupt", "unsupported"}:
            raise ValueError("Unsupported image health status.")
        image_id = make_id("image")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dataset = db.execute(
                "SELECT status FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if dataset is None:
                raise KeyError(f"Dataset not found: {dataset_id}")
            if dataset["status"] != "open":
                raise PermissionError("The dataset is no longer open for uploads.")
            db.execute(
                "INSERT INTO images("
                "image_id,dataset_id,filename,stored_path,sha256,size_bytes,"
                "mime_type,width,height,health_status,imported_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    image_id,
                    dataset_id,
                    filename,
                    stored_path,
                    sha256,
                    int(size_bytes),
                    mime_type,
                    width,
                    height,
                    health_status,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO image_annotations(image_id,updated_at) VALUES(?,?)",
                (image_id, now),
            )
            db.execute(
                "UPDATE datasets SET updated_at=? WHERE dataset_id=?",
                (now, dataset_id),
            )
        return {
            "image_id": image_id,
            "filename": filename,
            "sha256": sha256,
            "health_status": health_status,
            "width": width,
            "height": height,
        }

    def list_images(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT i.*,a.no_defect,a.status AS annotation_status,a.updated_at "
                "AS annotation_updated_at FROM images i JOIN image_annotations a "
                "ON a.image_id=i.image_id WHERE i.dataset_id=? "
                "ORDER BY i.imported_at,i.filename",
                (dataset_id,),
            ).fetchall()
            output = []
            for row in rows:
                item = dict(row)
                item["class_ids"] = [
                    value["class_id"]
                    for value in db.execute(
                        "SELECT class_id FROM image_annotation_labels "
                        "WHERE image_id=? ORDER BY class_id",
                        (row["image_id"],),
                    )
                ]
                output.append(item)
        return output

    def list_project_image_hashes(self, project_id: str) -> set[str]:
        self.get_project(project_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT i.sha256 FROM images i "
                "JOIN datasets d ON d.dataset_id=i.dataset_id "
                "WHERE d.project_id=?",
                (project_id,),
            ).fetchall()
        return {str(row["sha256"]) for row in rows}


    def get_image(self, image_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT i.*,a.no_defect,a.status AS annotation_status,"
                "a.updated_at AS annotation_updated_at FROM images i "
                "JOIN image_annotations a ON a.image_id=i.image_id "
                "WHERE i.image_id=?",
                (image_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Image not found: {image_id}")
            result = dict(row)
            result["class_ids"] = [
                item["class_id"]
                for item in db.execute(
                    "SELECT class_id FROM image_annotation_labels "
                    "WHERE image_id=? ORDER BY class_id",
                    (image_id,),
                )
            ]
        return result

    def save_annotation(
        self,
        *,
        image_id: str,
        class_ids: list[str],
        no_defect: bool,
    ) -> dict[str, Any]:
        selected = sorted({str(value).strip() for value in class_ids if str(value).strip()})
        if no_defect and selected:
            raise ValueError("No defect cannot be combined with defect classes.")
        if not no_defect and not selected:
            raise ValueError("Select at least one defect class or No defect.")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT i.dataset_id,d.project_id,d.status FROM images i "
                "JOIN datasets d ON d.dataset_id=i.dataset_id WHERE i.image_id=?",
                (image_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Image not found: {image_id}")
            if row["status"] != "open":
                raise PermissionError("The validated dataset is read-only.")
            valid_ids = {
                item["class_id"]
                for item in db.execute(
                    "SELECT class_id FROM classes WHERE project_id=? AND active=1",
                    (row["project_id"],),
                )
            }
            unknown = sorted(set(selected) - valid_ids)
            if unknown:
                raise ValueError(f"Classes do not belong to this project: {', '.join(unknown)}")
            db.execute(
                "DELETE FROM image_annotation_labels WHERE image_id=?", (image_id,)
            )
            db.executemany(
                "INSERT INTO image_annotation_labels(image_id,class_id) VALUES(?,?)",
                [(image_id, class_id) for class_id in selected],
            )
            db.execute(
                "UPDATE image_annotations SET no_defect=?,status='complete',updated_at=? "
                "WHERE image_id=?",
                (int(bool(no_defect)), now, image_id),
            )
            db.execute(
                "UPDATE datasets SET updated_at=? WHERE dataset_id=?",
                (now, row["dataset_id"]),
            )
        return self.get_image(image_id)

    def validate_dataset(
        self,
        *,
        dataset_id: str,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dataset = db.execute(
                "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if dataset is None:
                raise KeyError(f"Dataset not found: {dataset_id}")
            project = db.execute(
                "SELECT state FROM projects WHERE project_id=?", (dataset["project_id"],)
            ).fetchone()
            image_count = db.execute(
                "SELECT count(*) FROM images WHERE dataset_id=?", (dataset_id,)
            ).fetchone()[0]
            unreadable = db.execute(
                "SELECT count(*) FROM images WHERE dataset_id=? AND health_status!='ok'",
                (dataset_id,),
            ).fetchone()[0]
            missing = db.execute(
                "SELECT count(*) FROM images i JOIN image_annotations a "
                "ON a.image_id=i.image_id WHERE i.dataset_id=? AND a.status!='complete'",
                (dataset_id,),
            ).fetchone()[0]
            duplicate_rows = db.execute(
                "SELECT sha256,count(*) AS n FROM images WHERE dataset_id=? "
                "GROUP BY sha256 HAVING count(*)>1 ORDER BY sha256",
                (dataset_id,),
            ).fetchall()
            historical_duplicate_rows: list[sqlite3.Row] = []
            if dataset["role"] == "maintenance":
                historical_duplicate_rows = db.execute(
                    "SELECT DISTINCT current.sha256 FROM images current "
                    "JOIN images prior ON prior.sha256=current.sha256 "
                    "JOIN datasets prior_dataset ON prior_dataset.dataset_id=prior.dataset_id "
                    "WHERE current.dataset_id=? AND prior.dataset_id!=current.dataset_id "
                    "AND prior_dataset.project_id=? ORDER BY current.sha256",
                    (dataset_id, dataset["project_id"]),
                ).fetchall()
            no_defect_count = db.execute(
                "SELECT count(*) FROM images i JOIN image_annotations a "
                "ON a.image_id=i.image_id WHERE i.dataset_id=? AND a.no_defect=1",
                (dataset_id,),
            ).fetchone()[0]
            classes = db.execute(
                "SELECT class_id,display_name FROM classes WHERE project_id=? "
                "AND active=1 ORDER BY sort_order,class_id",
                (dataset["project_id"],),
            ).fetchall()
            per_class = []
            for class_row in classes:
                positives = db.execute(
                    "SELECT count(*) FROM image_annotation_labels l JOIN images i "
                    "ON i.image_id=l.image_id WHERE i.dataset_id=? AND l.class_id=?",
                    (dataset_id, class_row["class_id"]),
                ).fetchone()[0]
                per_class.append({
                    "class_id": class_row["class_id"],
                    "class_name": class_row["display_name"],
                    "positive_images": positives,
                    "negative_images": max(0, image_count - positives),
                })
            duplicate_groups = len(duplicate_rows)
            historical_duplicate_images = len(historical_duplicate_rows)
            valid = bool(
                image_count > 0
                and unreadable == 0
                and missing == 0
                and duplicate_groups == 0
                and historical_duplicate_images == 0
            )
            problems: list[dict[str, Any]] = []
            warnings: list[dict[str, Any]] = []
            if image_count == 0:
                problems.append({"code": "empty_dataset", "count": 0})
            if unreadable:
                problems.append({"code": "unreadable_images", "count": unreadable})
            if missing:
                problems.append({"code": "missing_annotations", "count": missing})
            if duplicate_groups:
                problems.append({
                    "code": "exact_duplicate_groups",
                    "count": duplicate_groups,
                    "hashes": [row["sha256"] for row in duplicate_rows],
                })
            if historical_duplicate_images:
                problems.append({
                    "code": "historical_duplicate_images",
                    "count": historical_duplicate_images,
                    "hashes": [row["sha256"] for row in historical_duplicate_rows],
                })
            for item in per_class:
                if item["positive_images"] < 10:
                    warnings.append({
                        "code": "low_positive_count",
                        "class_id": item["class_id"],
                        "class_name": item["class_name"],
                        "count": item["positive_images"],
                    })
            report = {
                "dataset_id": dataset_id,
                "valid": valid,
                "images": image_count,
                "readable_images": image_count - unreadable,
                "unreadable_images": unreadable,
                "missing_annotations": missing,
                "exact_duplicate_groups": duplicate_groups,
                "historical_duplicate_images": historical_duplicate_images,
                "no_defect_images": no_defect_count,
                "per_class": per_class,
                "problems": problems,
                "warnings": warnings,
            }
            if valid and dataset["status"] == "open":
                batch = None
                batch_transition = None
                if dataset["role"] == "maintenance":
                    if not dataset["batch_id"]:
                        raise PermissionError("Maintenance datasets must belong to a maintenance batch.")
                    batch = db.execute(
                        "SELECT * FROM maintenance_batches WHERE batch_id=?",
                        (dataset["batch_id"],),
                    ).fetchone()
                    if batch is None:
                        raise KeyError(f"Maintenance batch not found: {dataset['batch_id']}")
                    batch_transition = require_transition(
                        batch["state"],
                        "record_maintenance_labels_ready",
                        BATCH_TRANSITIONS,
                        False,
                    )
                elif project["state"] != ProjectState.DATA_IMPORTED:
                    raise PermissionError("Dataset validation is only allowed after data import.")
                label_rows = [
                    dict(row)
                    for row in db.execute(
                        "SELECT i.image_id,a.no_defect,group_concat(l.class_id,',') AS classes "
                        "FROM images i JOIN image_annotations a ON a.image_id=i.image_id "
                        "LEFT JOIN image_annotation_labels l ON l.image_id=i.image_id "
                        "WHERE i.dataset_id=? GROUP BY i.image_id,a.no_defect ORDER BY i.image_id",
                        (dataset_id,),
                    )
                ]
                labels_sha256 = hashlib.sha256(
                    canonical_json(label_rows).encode("utf-8")
                ).hexdigest()
                label_version_id = make_id("labels")
                db.execute(
                    "INSERT INTO label_versions(label_version_id,dataset_id,version_number,"
                    "labels_sha256,image_count,created_at) VALUES(?,?,?,?,?,?)",
                    (label_version_id, dataset_id, 1, labels_sha256, image_count, utc_now()),
                )
                now = utc_now()
                db.execute(
                    "UPDATE datasets SET status='validated',updated_at=? WHERE dataset_id=?",
                    (now, dataset_id),
                )
                if batch is not None and batch_transition is not None:
                    from_state = batch["state"]
                    to_state = batch_transition.target
                    batch_id = batch["batch_id"]
                    db.execute(
                        "UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?",
                        (to_state, now, batch_id),
                    )
                else:
                    from_state = ProjectState.DATA_IMPORTED
                    to_state = ProjectState.DATA_VALIDATED
                    batch_id = None
                    db.execute(
                        "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                        (to_state, now, dataset["project_id"]),
                    )
                digest = self._append_audit(
                    db,
                    project_id=dataset["project_id"],
                    batch_id=batch_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name="validate_dataset",
                    from_state=from_state,
                    to_state=to_state,
                    payload={**report, "label_version_id": label_version_id, "labels_sha256": labels_sha256},
                )
                report["label_version_id"] = label_version_id
                report["labels_sha256"] = labels_sha256
                report["audit_event_sha256"] = digest
            elif valid and dataset["status"] in {"validated", "frozen"}:
                version = db.execute(
                    "SELECT label_version_id,labels_sha256 FROM label_versions "
                    "WHERE dataset_id=? ORDER BY version_number DESC LIMIT 1",
                    (dataset_id,),
                ).fetchone()
                if version:
                    report.update(dict(version))
        return report

    def freeze_maintenance_batch(
        self,
        *,
        batch_id: str,
        confirmed: bool,
        actor_type: str = "human",
        actor_id: str = "local_engineer",
    ) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can freeze maintenance labels.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM maintenance_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(f"Maintenance batch not found: {batch_id}")
            transition = require_transition(
                batch["state"], "freeze_maintenance_batch", BATCH_TRANSITIONS, confirmed
            )
            dataset = db.execute(
                "SELECT * FROM datasets WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if dataset is None or dataset["role"] != "maintenance":
                raise PermissionError("The maintenance dataset is missing.")
            if dataset["status"] != "validated":
                raise PermissionError("Run deterministic data checks before freezing labels.")
            version = db.execute(
                "SELECT * FROM label_versions WHERE dataset_id=? "
                "ORDER BY version_number DESC LIMIT 1",
                (dataset["dataset_id"],),
            ).fetchone()
            if version is None:
                raise PermissionError("The validated maintenance label version is missing.")
            label_rows = [
                dict(row)
                for row in db.execute(
                    "SELECT i.image_id,a.no_defect,group_concat(l.class_id,',') AS classes "
                    "FROM images i JOIN image_annotations a ON a.image_id=i.image_id "
                    "LEFT JOIN image_annotation_labels l ON l.image_id=i.image_id "
                    "WHERE i.dataset_id=? GROUP BY i.image_id,a.no_defect ORDER BY i.image_id",
                    (dataset["dataset_id"],),
                )
            ]
            current_sha256 = hashlib.sha256(
                canonical_json(label_rows).encode("utf-8")
            ).hexdigest()
            if current_sha256 != version["labels_sha256"]:
                raise ValueError("Maintenance labels changed after validation.")
            project = db.execute(
                "SELECT state,active_batch_id FROM projects WHERE project_id=?",
                (batch["project_id"],),
            ).fetchone()
            if project is None or project["active_batch_id"] != batch_id:
                raise PermissionError("The maintenance batch is not active for this project.")
            now = utc_now()
            db.execute(
                "UPDATE datasets SET status='frozen',updated_at=? WHERE dataset_id=?",
                (now, dataset["dataset_id"]),
            )
            db.execute(
                "UPDATE maintenance_batches SET state=?,updated_at=? WHERE batch_id=?",
                (transition.target, now, batch_id),
            )
            db.execute(
                "UPDATE projects SET state=?,updated_at=? WHERE project_id=?",
                (ProjectState.SCREENING_READY, now, batch["project_id"]),
            )
            digest = self._append_audit(
                db,
                project_id=batch["project_id"],
                batch_id=batch_id,
                actor_type=actor_type,
                actor_id=actor_id,
                tool_name="freeze_maintenance_batch",
                from_state=batch["state"],
                to_state=transition.target,
                payload={
                    "dataset_id": dataset["dataset_id"],
                    "label_version_id": version["label_version_id"],
                    "labels_sha256": version["labels_sha256"],
                    "image_count": version["image_count"],
                },
            )
        return {
            "maintenance_batch": self.get_maintenance_batch(batch_id),
            "project": self.get_project(batch["project_id"]),
            "audit_event_sha256": digest,
        }

    def record_llm_message(
        self,
        *,
        project_id: str | None,
        role: str,
        content: str,
        provider: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError("Unsupported LLM message role.")
        message_id = make_id("message")
        created_at = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO llm_messages(message_id,project_id,role,content,provider,"
                "model,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    message_id,
                    project_id,
                    role,
                    str(content),
                    provider,
                    model,
                    canonical_json(metadata or {}),
                    created_at,
                ),
            )
        return {"message_id": message_id, "created_at": created_at}

    def list_llm_messages(
        self, project_id: str | None, limit: int = 20
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 100))
        with self.connect() as db:
            if project_id:
                rows = db.execute(
                    "SELECT * FROM llm_messages WHERE project_id=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (project_id, bounded),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM llm_messages WHERE project_id IS NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
        output = []
        for row in reversed(rows):
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            output.append(item)
        return output

    def delete_llm_messages(self, project_id: str | None) -> int:
        """Delete one local conversation without deleting its project or audit data."""
        if project_id is not None:
            self.get_project(project_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if project_id is None:
                cursor = db.execute(
                    "DELETE FROM llm_messages WHERE project_id IS NULL"
                )
            else:
                cursor = db.execute(
                    "DELETE FROM llm_messages WHERE project_id=?", (project_id,)
                )
            deleted_count = int(cursor.rowcount)
            db.commit()
        return deleted_count

    def move_llm_message_to_project(self, message_id: str, project_id: str) -> None:
        """Attach a starter-chat message to the project it just created."""
        self.get_project(project_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                "UPDATE llm_messages SET project_id=? "
                "WHERE message_id=? AND project_id IS NULL",
                (project_id, message_id),
            )
            if cursor.rowcount != 1:
                db.rollback()
                raise ValueError("Only a global starter-chat message can be moved.")
            db.commit()

    def create_pending_tool_call(
        self,
        *,
        project_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        provider: str,
        model: str,
        workflow_version: str | None = None,
    ) -> dict[str, Any]:
        pending_id = make_id("approval")
        requested_at = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO pending_tool_calls(pending_id,project_id,tool_name,"
                "arguments_json,provider,model,status,requested_at,workflow_version) "
                "VALUES(?,?,?,?,?,?,'pending',?,?)",
                (
                    pending_id,
                    project_id,
                    tool_name,
                    canonical_json(arguments),
                    provider,
                    model,
                    requested_at,
                    workflow_version,
                ),
            )
            digest = self._append_audit(
                db,
                project_id=project_id,
                batch_id=None,
                actor_type="llm",
                actor_id=model,
                tool_name="request_human_confirmation",
                from_state=None,
                to_state=None,
                payload={
                    "pending_id": pending_id,
                    "proposed_tool": tool_name,
                    "arguments": arguments,
                    "workflow_version": workflow_version,
                },
            )
        return {
            "pending_id": pending_id,
            "project_id": project_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "pending",
            "requested_at": requested_at,
            "workflow_version": workflow_version,
            "audit_event_sha256": digest,
        }

    def get_pending_tool_call(self, pending_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM pending_tool_calls WHERE pending_id=?", (pending_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Pending tool call not found: {pending_id}")
        item = dict(row)
        item["arguments"] = json.loads(item.pop("arguments_json"))
        item["result"] = json.loads(item["result_json"]) if item["result_json"] else None
        item.pop("result_json")
        return item

    def resolve_pending_tool_call(
        self,
        *,
        pending_id: str,
        status: str,
        resolved_by: str,
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if status not in {"approved", "rejected", "failed"}:
            raise ValueError("Unsupported pending-call resolution.")
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT project_id,status,tool_name FROM pending_tool_calls "
                "WHERE pending_id=?",
                (pending_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Pending tool call not found: {pending_id}")
            if row["status"] != "pending":
                raise PermissionError("This tool call has already been resolved.")
            db.execute(
                "UPDATE pending_tool_calls SET status=?,resolved_at=?,resolved_by=?,"
                "result_json=? WHERE pending_id=?",
                (status, now, resolved_by, canonical_json(result) if result is not None else None, pending_id),
            )
            digest = self._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=None,
                actor_type="human",
                actor_id=resolved_by,
                tool_name="resolve_human_confirmation",
                from_state=None,
                to_state=None,
                payload={
                    "pending_id": pending_id,
                    "proposed_tool": row["tool_name"],
                    "resolution": status,
                },
            )
        resolved = self.get_pending_tool_call(pending_id)
        resolved["audit_event_sha256"] = digest
        return resolved

    def record_tool_run(
        self,
        *,
        project_id: str | None,
        tool_name: str,
        status: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
    ) -> str:
        run_id = make_id("run")
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO tool_runs(run_id,project_id,batch_id,tool_name,status,"
                "request_json,response_json,started_at,finished_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    project_id,
                    None,
                    tool_name,
                    status,
                    canonical_json(request),
                    canonical_json(response) if response is not None else None,
                    now,
                    now,
                ),
            )
        return run_id

    def get_project_history(self, project_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        with self.connect() as db:
            audits = [
                dict(row)
                for row in db.execute(
                    "SELECT event_id,occurred_at,actor_type,actor_id,tool_name,"
                    "from_state,to_state,payload_json,event_sha256 "
                    "FROM audit_events WHERE project_id=? ORDER BY event_id DESC",
                    (project_id,),
                )
            ]
            messages = [
                dict(row)
                for row in db.execute(
                    "SELECT message_id,role,content,provider,model,metadata_json,created_at "
                    "FROM llm_messages WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            tool_runs = [
                dict(row)
                for row in db.execute(
                    "SELECT run_id,tool_name,status,request_json,response_json,started_at,finished_at "
                    "FROM tool_runs WHERE project_id=? ORDER BY started_at DESC",
                    (project_id,),
                )
            ]
            approvals = [
                dict(row)
                for row in db.execute(
                    "SELECT pending_id,tool_name,arguments_json,status,provider,model,"
                    "requested_at,resolved_at,resolved_by FROM pending_tool_calls "
                    "WHERE project_id=? ORDER BY requested_at DESC",
                    (project_id,),
                )
            ]
            label_versions = [
                dict(row)
                for row in db.execute(
                    "SELECT v.label_version_id,v.dataset_id,v.version_number,v.labels_sha256,"
                    "v.image_count,v.created_at,d.name AS dataset_name FROM label_versions v "
                    "JOIN datasets d ON d.dataset_id=v.dataset_id "
                    "WHERE d.project_id=? ORDER BY v.created_at DESC",
                    (project_id,),
                )
            ]
            training_jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT job_id,dataset_id,kind,status,training_profile_id,bundle_sha256,"
                    "bundle_size_bytes,created_at,updated_at FROM training_jobs "
                    "WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            maintenance_batches = [
                dict(row)
                for row in db.execute(
                    "SELECT batch_id,name,state,formal_decision,decision_reason,"
                    "created_at,updated_at FROM maintenance_batches "
                    "WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            training_runs = [
                dict(row)
                for row in db.execute(
                    "SELECT run_id,job_id,status,gpu_devices_json,world_size,progress_json,"
                    "result_bundle_sha256,error_message,created_at,started_at,finished_at,updated_at "
                    "FROM training_runs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            models = [
                dict(row)
                for row in db.execute(
                    "SELECT model_id,role,checkpoint_sha256,training_profile_id,"
                    "source_training_job_id,source_training_run_id,result_bundle_sha256,metrics_json,created_at "
                    "FROM model_versions WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            screening_jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT job_id,batch_id,status,champion_model_id,discovery_profile_id,bundle_sha256,"
                    "created_at,updated_at FROM screening_jobs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            screening_runs = [
                dict(row)
                for row in db.execute(
                    "SELECT run_id,job_id,batch_id,status,gpu_devices_json,world_size,progress_json,summary_json,"
                    "result_bundle_sha256,error_message,created_at,started_at,finished_at,updated_at "
                    "FROM screening_runs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            failure_slices = [
                dict(row)
                for row in db.execute(
                    "SELECT slice_id,batch_id,class_id,error_type,support,clustering_profile_id,status,"
                    "source_screening_run_id,consensus_score,created_at FROM failure_slices "
                    "WHERE batch_id IN (SELECT batch_id FROM maintenance_batches WHERE project_id=?) "
                    "ORDER BY created_at DESC,slice_id",
                    (project_id,),
                )
            ]
            failure_review_versions = [
                dict(row)
                for row in db.execute(
                    "SELECT review_version_id,batch_id,reviewer_id,slice_count,eligible_slice_count,"
                    "retained_member_records,retained_unique_images,review_sha256,created_at "
                    "FROM failure_review_versions WHERE batch_id IN "
                    "(SELECT batch_id FROM maintenance_batches WHERE project_id=?) "
                    "ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            challenger_jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT job_id,batch_id,review_version_id,parent_champion_model_id,status,"
                    "training_profile_id,pool_counts_json,bundle_sha256,bundle_size_bytes,created_at,updated_at "
                    "FROM challenger_jobs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            challenger_runs = [
                dict(row)
                for row in db.execute(
                    "SELECT run_id,job_id,batch_id,status,gpu_devices_json,world_size,progress_json,summary_json,"
                    "result_bundle_sha256,error_message,created_at,started_at,finished_at,updated_at "
                    "FROM challenger_runs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            evaluation_jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT job_id,batch_id,champion_model_id,challenger_model_id,status,evaluation_profile_id,"
                    "source_snapshot_sha256,split_counts_json,bundle_sha256,created_at,updated_at "
                    "FROM evaluation_jobs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            evaluation_gates = [
                dict(row)
                for row in db.execute(
                    "SELECT gate_id,source_batch_id,gate_key,role,status,source_evaluation_job_id,"
                    "content_sha256,image_count,created_at,activated_at "
                    "FROM evaluation_gates WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            evaluation_cohorts = [
                dict(row)
                for row in db.execute(
                    "SELECT cohort_id,source_batch_id,source_evaluation_job_id,source_round_name,"
                    "origin_role,status,content_sha256,image_count,label_version_id,taxonomy_sha256,"
                    "created_at,activated_at FROM evaluation_cohorts WHERE project_id=? "
                    "ORDER BY created_at DESC,cohort_id",
                    (project_id,),
                )
            ]
            round_completions = [
                dict(row)
                for row in db.execute(
                    "SELECT batch_id,decision,reason,previous_champion_model_id,active_champion_model_id,"
                    "challenger_model_id,current_gate_cohort_id,core_safety_cohort_id,"
                    "cumulative_core_safety_images,evidence_id,final_test_read,completed_at "
                    "FROM round_completion_summaries WHERE project_id=? ORDER BY completed_at DESC",
                    (project_id,),
                )
            ]
            evaluation_runs = [
                dict(row)
                for row in db.execute(
                    "SELECT run_id,job_id,batch_id,status,gpu_devices_json,world_size,progress_json,summary_json,"
                    "result_bundle_sha256,error_message,created_at,started_at,finished_at,updated_at "
                    "FROM evaluation_runs WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            evidence_reports = [
                dict(row)
                for row in db.execute(
                    "SELECT evidence_id,batch_id,champion_model_id,challenger_model_id,metrics_json,artifact_sha256,"
                    "evaluation_profile_id,created_at FROM evidence_reports WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                )
            ]
            deployment_decisions = [
                dict(row)
                for row in db.execute(
                    "SELECT d.decision_id,d.batch_id,d.decision,d.reason,d.actor_id,d.decided_at "
                    "FROM deployment_decisions d JOIN maintenance_batches b ON b.batch_id=d.batch_id "
                    "WHERE b.project_id=? ORDER BY d.decided_at DESC",
                    (project_id,),
                )
            ]
        for item in audits:
            item["payload"] = json.loads(item.pop("payload_json"))
        for item in messages:
            item["metadata"] = json.loads(item.pop("metadata_json"))
        for item in tool_runs:
            item["request"] = json.loads(item.pop("request_json"))
            raw_response = item.pop("response_json")
            item["response"] = json.loads(raw_response) if raw_response else None
        for item in approvals:
            item["arguments"] = json.loads(item.pop("arguments_json"))
        for item in training_runs:
            item["gpu_devices"] = json.loads(item.pop("gpu_devices_json"))
            item["progress"] = json.loads(item.pop("progress_json"))
        for item in models:
            item["metrics"] = json.loads(item.pop("metrics_json"))
        for item in screening_runs:
            item["gpu_devices"] = json.loads(item.pop("gpu_devices_json"))
            item["progress"] = json.loads(item.pop("progress_json"))
            item["summary"] = json.loads(item.pop("summary_json"))
        for item in challenger_jobs:
            item["pool_counts"] = json.loads(item.pop("pool_counts_json"))
        for item in challenger_runs:
            item["gpu_devices"] = json.loads(item.pop("gpu_devices_json"))
            item["progress"] = json.loads(item.pop("progress_json"))
            item["summary"] = json.loads(item.pop("summary_json"))
        for item in evaluation_jobs:
            item["split_counts"] = json.loads(item.pop("split_counts_json"))
        for item in evaluation_runs:
            item["gpu_devices"] = json.loads(item.pop("gpu_devices_json"))
            item["progress"] = json.loads(item.pop("progress_json"))
            item["summary"] = json.loads(item.pop("summary_json"))
        for item in evidence_reports:
            item["metrics"] = json.loads(item.pop("metrics_json"))
        timeline: list[dict[str, Any]] = []
        timeline.extend({"kind": "audit", "at": item["occurred_at"], "data": item} for item in audits)
        timeline.extend({"kind": "message", "at": item["created_at"], "data": item} for item in messages)
        timeline.extend({"kind": "tool_run", "at": item["started_at"], "data": item} for item in tool_runs)
        timeline.extend({"kind": "approval", "at": item["requested_at"], "data": item} for item in approvals)
        timeline.extend({"kind": "label_version", "at": item["created_at"], "data": item} for item in label_versions)
        timeline.extend({"kind": "maintenance_batch", "at": item["created_at"], "data": item} for item in maintenance_batches)
        timeline.extend({"kind": "training_job", "at": item["created_at"], "data": item} for item in training_jobs)
        timeline.extend({"kind": "training_run", "at": item["created_at"], "data": item} for item in training_runs)
        timeline.extend({"kind": "model", "at": item["created_at"], "data": item} for item in models)
        timeline.extend({"kind": "screening_job", "at": item["created_at"], "data": item} for item in screening_jobs)
        timeline.extend({"kind": "screening_run", "at": item["created_at"], "data": item} for item in screening_runs)
        timeline.extend({"kind": "failure_slice", "at": item["created_at"], "data": item} for item in failure_slices)
        timeline.extend(
            {"kind": "failure_review", "at": item["created_at"], "data": item}
            for item in failure_review_versions
        )
        timeline.extend(
            {"kind": "challenger_job", "at": item["created_at"], "data": item}
            for item in challenger_jobs
        )
        timeline.extend(
            {"kind": "challenger_run", "at": item["created_at"], "data": item}
            for item in challenger_runs
        )
        timeline.extend({"kind": "evaluation_job", "at": item["created_at"], "data": item} for item in evaluation_jobs)
        timeline.extend({"kind": "evaluation_gate", "at": item["activated_at"] or item["created_at"], "data": item} for item in evaluation_gates)
        timeline.extend({"kind": "evaluation_cohort", "at": item["activated_at"] or item["created_at"], "data": item} for item in evaluation_cohorts)
        timeline.extend({"kind": "round_completion", "at": item["completed_at"], "data": item} for item in round_completions)
        timeline.extend({"kind": "evaluation_run", "at": item["created_at"], "data": item} for item in evaluation_runs)
        timeline.extend({"kind": "evidence", "at": item["created_at"], "data": item} for item in evidence_reports)
        timeline.extend({"kind": "decision", "at": item["decided_at"], "data": item} for item in deployment_decisions)
        timeline.sort(key=lambda item: item["at"], reverse=True)
        return {
            "project_id": project_id,
            "counts": {
                "audit_events": len(audits),
                "messages": len(messages),
                "tool_runs": len(tool_runs),
                "approvals": len(approvals),
                "label_versions": len(label_versions),
                "maintenance_batches": len(maintenance_batches),
                "training_jobs": len(training_jobs),
                "training_runs": len(training_runs),
                "models": len(models),
                "screening_jobs": len(screening_jobs),
                "screening_runs": len(screening_runs),
                "failure_slices": len(failure_slices),
                "failure_reviews": len(failure_review_versions),
                "challenger_jobs": len(challenger_jobs),
                "challenger_runs": len(challenger_runs),
                "evaluation_jobs": len(evaluation_jobs),
                "evaluation_gates": len(evaluation_gates),
                "evaluation_cohorts": len(evaluation_cohorts),
                "round_completions": len(round_completions),
                "evaluation_runs": len(evaluation_runs),
                "evidence_reports": len(evidence_reports),
                "deployment_decisions": len(deployment_decisions),
            },
            "timeline": timeline,
        }
