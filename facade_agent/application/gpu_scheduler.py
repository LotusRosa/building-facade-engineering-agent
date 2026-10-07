from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Protocol

from ..storage import make_id, utc_now


PIPELINES = {
    "initial_champion",
    "champion_failure_discovery",
    "challenger_update",
    "champion_challenger_evaluation",
}
ACTIVE_STATUSES = ("starting", "running")
PUBLIC_QUEUE_FIELDS = {
    "queue_id",
    "project_id",
    "project_name",
    "pipeline",
    "run_id",
    "status",
    "cancel_requested",
    "error_code",
    "error_message",
    "queue_position",
    "enqueued_at",
    "started_at",
    "finished_at",
    "updated_at",
}


def public_queue_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {key: item.get(key) for key in PUBLIC_QUEUE_FIELDS if key in item}


def public_queue_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "active": public_queue_item(snapshot.get("active")),
        "waiting": [public_queue_item(item) for item in snapshot.get("waiting", [])],
        "waiting_count": int(snapshot.get("waiting_count", 0)),
        "project_items": [
            public_queue_item(item) for item in snapshot.get("project_items", [])
        ],
    }


class ScheduledGpuHandler(Protocol):
    def execute(self, run_id: str, cancellation: threading.Event) -> None: ...

    def status(self, run_id: str) -> str: ...

    def cancel_queued(
        self, db: sqlite3.Connection, run_id: str, actor_id: str, now: str
    ) -> None: ...

    def interrupt_abandoned(
        self, db: sqlite3.Connection, run_id: str, now: str
    ) -> None: ...


class GpuJobScheduler:
    """Persistent admission control for all managed single-GPU work."""

    def __init__(self, store: Any, lease_owner: str | None = None) -> None:
        self.store = store
        self.lease_owner = lease_owner or f"agent-{uuid.uuid4().hex[:16]}"
        self._handlers: dict[str, ScheduledGpuHandler] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_queue_id: str | None = None
        self._current_cancellation: threading.Event | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _item(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["cancel_requested"] = bool(item["cancel_requested"])
        return item

    def enqueue_in_transaction(
        self,
        db: sqlite3.Connection,
        *,
        project_id: str,
        pipeline: str,
        run_id: str,
        actor_type: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if pipeline not in PIPELINES:
            raise ValueError(f"Unsupported GPU pipeline: {pipeline}")
        existing = db.execute(
            "SELECT * FROM gpu_job_queue WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing is not None:
            return self._item(existing) or {}
        self.store.get_project(project_id)
        queue_id = make_id("gpu_queue")
        now = utc_now()
        db.execute(
            "INSERT INTO gpu_job_queue("
            "queue_id,project_id,pipeline,run_id,status,cancel_requested,"
            "enqueued_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (queue_id, project_id, pipeline, run_id, "queued", 0, now, now),
        )
        self.store._append_audit(
            db,
            project_id=project_id,
            batch_id=None,
            actor_type=actor_type,
            actor_id=actor_id,
            tool_name="enqueue_gpu_job",
            from_state=None,
            to_state="queued",
            payload={"queue_id": queue_id, "pipeline": pipeline, "run_id": run_id},
        )
        row = db.execute(
            "SELECT * FROM gpu_job_queue WHERE queue_id=?", (queue_id,)
        ).fetchone()
        return self._item(row) or {}

    def claim_next(self) -> dict[str, Any] | None:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute(
                "SELECT queue_id FROM gpu_job_queue "
                "WHERE status IN ('starting','running') LIMIT 1"
            ).fetchone()
            if active is not None:
                return None
            row = db.execute(
                "SELECT * FROM gpu_job_queue WHERE status='queued' "
                "ORDER BY enqueued_at,queue_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utc_now()
            db.execute(
                "UPDATE gpu_job_queue SET status='starting',lease_owner=?,"
                "lease_acquired_at=?,started_at=COALESCE(started_at,?),updated_at=? "
                "WHERE queue_id=? AND status='queued'",
                (self.lease_owner, now, now, now, row["queue_id"]),
            )
            claimed = db.execute(
                "SELECT * FROM gpu_job_queue WHERE queue_id=?", (row["queue_id"],)
            ).fetchone()
            self.store._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=None,
                actor_type="system",
                actor_id=self.lease_owner,
                tool_name="claim_gpu_job",
                from_state="queued",
                to_state="starting",
                payload={
                    "queue_id": row["queue_id"],
                    "pipeline": row["pipeline"],
                    "run_id": row["run_id"],
                },
            )
            return self._item(claimed)

    def snapshot(self, project_id: str | None = None) -> dict[str, Any]:
        if project_id is not None:
            self.store.get_project(project_id)
        with self.store.connect() as db:
            active_row = db.execute(
                "SELECT q.*,p.name AS project_name FROM gpu_job_queue q "
                "JOIN projects p ON p.project_id=q.project_id "
                "WHERE q.status IN ('starting','running') LIMIT 1"
            ).fetchone()
            waiting_rows = db.execute(
                "SELECT q.*,p.name AS project_name FROM gpu_job_queue q "
                "JOIN projects p ON p.project_id=q.project_id "
                "WHERE q.status='queued' ORDER BY q.enqueued_at,q.queue_id"
            ).fetchall()
        active = self._item(active_row)
        waiting: list[dict[str, Any]] = []
        for position, row in enumerate(waiting_rows, 1):
            item = self._item(row) or {}
            item["queue_position"] = position
            waiting.append(item)
        project_items = [
            item
            for item in ([active] if active else []) + waiting
            if item["project_id"] == project_id
        ] if project_id is not None else []
        return {
            "active": active,
            "waiting": waiting,
            "waiting_count": len(waiting),
            "project_items": project_items,
        }

    def get_item(self, queue_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT * FROM gpu_job_queue WHERE queue_id=?", (queue_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"GPU queue item not found: {queue_id}")
        return self._item(row) or {}

    def register_handler(self, pipeline: str, handler: ScheduledGpuHandler) -> None:
        if pipeline not in PIPELINES:
            raise ValueError(f"Unsupported GPU pipeline: {pipeline}")
        if pipeline in self._handlers:
            raise ValueError(f"GPU pipeline handler already registered: {pipeline}")
        self._handlers[pipeline] = handler

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._recover_abandoned()
            self._thread = threading.Thread(
                target=self._loop,
                name="facade-global-gpu-scheduler",
                daemon=True,
            )
            self._thread.start()
        self.notify()

    def shutdown(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            cancellation = self._current_cancellation
            thread = self._thread
        if cancellation is not None:
            cancellation.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    def cancel(self, queue_id: str, actor_id: str) -> dict[str, Any]:
        handler: ScheduledGpuHandler | None = None
        should_signal = False
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM gpu_job_queue WHERE queue_id=?", (queue_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"GPU queue item not found: {queue_id}")
            if row["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return self._item(row) or {}
            now = utc_now()
            handler = self._handlers.get(row["pipeline"])
            if row["status"] == "queued":
                if handler is not None:
                    handler.cancel_queued(db, row["run_id"], actor_id, now)
                db.execute(
                    "UPDATE gpu_job_queue SET status='cancelled',cancel_requested=1,"
                    "finished_at=?,updated_at=? WHERE queue_id=? AND status='queued'",
                    (now, now, queue_id),
                )
                to_state = "cancelled"
            else:
                db.execute(
                    "UPDATE gpu_job_queue SET cancel_requested=1,updated_at=? "
                    "WHERE queue_id=?",
                    (now, queue_id),
                )
                to_state = row["status"]
                should_signal = True
            self.store._append_audit(
                db,
                project_id=row["project_id"],
                batch_id=None,
                actor_type="human",
                actor_id=actor_id,
                tool_name="cancel_gpu_job",
                from_state=row["status"],
                to_state=to_state,
                payload={
                    "queue_id": queue_id,
                    "pipeline": row["pipeline"],
                    "run_id": row["run_id"],
                },
            )
        if should_signal:
            with self._lock:
                cancellation = (
                    self._current_cancellation
                    if self._current_queue_id == queue_id
                    else None
                )
            if cancellation is not None:
                cancellation.set()
        self.notify()
        return self.get_item(queue_id)

    def _recover_abandoned(self) -> None:
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM gpu_job_queue WHERE status IN ('starting','running') "
                "ORDER BY enqueued_at,queue_id"
            ).fetchall()
            if not rows:
                return
            db.execute("BEGIN IMMEDIATE")
            now = utc_now()
            for row in rows:
                handler = self._handlers.get(row["pipeline"])
                if handler is not None:
                    handler.interrupt_abandoned(db, row["run_id"], now)
                db.execute(
                    "UPDATE gpu_job_queue SET status='interrupted',"
                    "error_code='GPU_QUEUE_INTERRUPTED',"
                    "error_message='Agent restarted while this GPU task held the execution lease.',"
                    "finished_at=?,updated_at=?,lease_owner=NULL,lease_acquired_at=NULL "
                    "WHERE queue_id=?",
                    (now, now, row["queue_id"]),
                )
                self.store._append_audit(
                    db,
                    project_id=row["project_id"],
                    batch_id=None,
                    actor_type="system",
                    actor_id=self.lease_owner,
                    tool_name="interrupt_gpu_job_after_restart",
                    from_state=row["status"],
                    to_state="interrupted",
                    payload={
                        "queue_id": row["queue_id"],
                        "pipeline": row["pipeline"],
                        "run_id": row["run_id"],
                    },
                )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.claim_next()
            except sqlite3.OperationalError:
                # A test workspace or removable project directory can disappear
                # while the daemon thread is winding down. Exit cleanly in that
                # case; transient SQLite errors keep the queue alive and retry.
                if self._stop.is_set() or not Path(self.store.path).parent.exists():
                    return
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if item is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            self._execute_item(item)

    def _execute_item(self, item: dict[str, Any]) -> None:
        handler = self._handlers.get(item["pipeline"])
        if handler is None:
            self._finish_item(
                item,
                "failed",
                error_code="GPU_QUEUE_HANDLER_MISSING",
                error_message=f"No handler is registered for pipeline {item['pipeline']}.",
            )
            return
        cancellation = threading.Event()
        with self._lock:
            self._current_queue_id = item["queue_id"]
            self._current_cancellation = cancellation
        try:
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE gpu_job_queue SET status='running',updated_at=? "
                    "WHERE queue_id=? AND status='starting'",
                    (now, item["queue_id"]),
                )
            handler.execute(item["run_id"], cancellation)
            status = handler.status(item["run_id"])
            terminal = self._queue_terminal_status(status)
            self._finish_item(item, terminal)
        except Exception as error:
            self._finish_item(
                item,
                "failed",
                error_code="WORKER_FAILED",
                error_message=str(error)[:1000],
            )
        finally:
            with self._lock:
                self._current_queue_id = None
                self._current_cancellation = None
            self.notify()

    @staticmethod
    def _queue_terminal_status(run_status: str) -> str:
        if run_status in {"result_verified", "completed"}:
            return "completed"
        if run_status == "cancelled":
            return "cancelled"
        if run_status == "interrupted":
            return "interrupted"
        return "failed"

    def _finish_item(
        self,
        item: dict[str, Any],
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM gpu_job_queue WHERE queue_id=?", (item["queue_id"],)
            ).fetchone()
            if row is None or row["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return
            db.execute(
                "UPDATE gpu_job_queue SET status=?,error_code=?,error_message=?,"
                "finished_at=?,updated_at=?,lease_owner=NULL,lease_acquired_at=NULL "
                "WHERE queue_id=?",
                (status, error_code, error_message, now, now, item["queue_id"]),
            )
            self.store._append_audit(
                db,
                project_id=item["project_id"],
                batch_id=None,
                actor_type="system",
                actor_id=self.lease_owner,
                tool_name="finish_gpu_job",
                from_state=row["status"],
                to_state=status,
                payload={
                    "queue_id": item["queue_id"],
                    "pipeline": item["pipeline"],
                    "run_id": item["run_id"],
                    "error_code": error_code,
                },
            )
