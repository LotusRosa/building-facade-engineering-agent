from __future__ import annotations

import http.client
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.storage import Store


def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for scheduler state.")


class FakeHandler:
    def __init__(self, *, block_first: bool = False, fail_runs: set[str] | None = None) -> None:
        self.block_first = block_first
        self.fail_runs = fail_runs or set()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.executed: list[str] = []
        self.statuses: dict[str, str] = {}
        self.interrupted: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def execute(self, run_id: str, cancellation: threading.Event) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed.append(run_id)
        self.statuses[run_id] = "running"
        self.entered.set()
        try:
            if self.block_first and len(self.executed) == 1:
                while not self.release.wait(0.01):
                    if cancellation.is_set():
                        self.statuses[run_id] = "cancelled"
                        return
            if cancellation.is_set():
                self.statuses[run_id] = "cancelled"
            elif run_id in self.fail_runs:
                self.statuses[run_id] = "failed"
            else:
                self.statuses[run_id] = "result_verified"
        finally:
            with self._lock:
                self.active -= 1

    def status(self, run_id: str) -> str:
        return self.statuses.get(run_id, "queued")

    def cancel_queued(self, db, run_id: str, actor_id: str, now: str) -> None:
        self.statuses[run_id] = "cancelled"

    def interrupt_abandoned(self, db, run_id: str, now: str) -> None:
        self.interrupted.append(run_id)
        self.statuses[run_id] = "interrupted"


class GpuJobQueueRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "agent.sqlite3")
        self.first_project = self.store.create_project(
            project_name="North facade", class_names=["crack"]
        )["project"]
        self.second_project = self.store.create_project(
            project_name="South facade", class_names=["spalling"]
        )["project"]
        self.scheduler = GpuJobScheduler(self.store, lease_owner="test-process")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def enqueue(self, project_id: str, pipeline: str, run_id: str) -> dict:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self.scheduler.enqueue_in_transaction(
                db,
                project_id=project_id,
                pipeline=pipeline,
                run_id=run_id,
                actor_type="human",
                actor_id="engineer",
            )

    def test_queue_migration_enforces_one_active_lease(self) -> None:
        first = self.enqueue(self.first_project["project_id"], "initial_champion", "run_a")
        second = self.enqueue(self.second_project["project_id"], "challenger_update", "run_b")

        with self.store.connect() as db:
            db.execute("UPDATE gpu_job_queue SET status='running' WHERE queue_id=?", (first["queue_id"],))
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE gpu_job_queue SET status='starting' WHERE queue_id=?", (second["queue_id"],))

    def test_enqueue_is_idempotent_by_run_id(self) -> None:
        first = self.enqueue(self.first_project["project_id"], "initial_champion", "run_same")
        second = self.enqueue(self.first_project["project_id"], "initial_champion", "run_same")

        self.assertEqual(first["queue_id"], second["queue_id"])
        with self.store.connect() as db:
            count = db.execute(
                "SELECT COUNT(*) FROM gpu_job_queue WHERE run_id='run_same'"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_duplicate_start_surfaces_reuse_same_queue_row(self) -> None:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            chat = self.scheduler.enqueue_in_transaction(
                db,
                project_id=self.first_project["project_id"],
                pipeline="initial_champion",
                run_id="run_shared_surface",
                actor_type="llm",
                actor_id="conversation",
            )
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            panel = self.scheduler.enqueue_in_transaction(
                db,
                project_id=self.first_project["project_id"],
                pipeline="initial_champion",
                run_id="run_shared_surface",
                actor_type="human",
                actor_id="task_panel",
            )

        self.assertEqual(chat["queue_id"], panel["queue_id"])
        with self.store.connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM gpu_job_queue WHERE run_id=?",
                    ("run_shared_surface",),
                ).fetchone()[0],
                1,
            )

    def test_claim_next_is_fifo_across_projects(self) -> None:
        first = self.enqueue(self.first_project["project_id"], "initial_champion", "run_first")
        self.enqueue(self.second_project["project_id"], "champion_failure_discovery", "run_second")

        claimed = self.scheduler.claim_next()

        self.assertEqual(claimed["queue_id"], first["queue_id"])
        self.assertEqual(claimed["status"], "starting")
        self.assertEqual(claimed["lease_owner"], "test-process")
        self.assertIsNone(self.scheduler.claim_next())

    def test_snapshot_reports_global_active_and_project_position(self) -> None:
        first = self.enqueue(self.first_project["project_id"], "initial_champion", "run_active")
        self.enqueue(self.second_project["project_id"], "challenger_update", "run_waiting_1")
        self.enqueue(self.first_project["project_id"], "champion_challenger_evaluation", "run_waiting_2")
        claimed = self.scheduler.claim_next()
        self.assertEqual(claimed["queue_id"], first["queue_id"])

        snapshot = self.scheduler.snapshot(self.first_project["project_id"])

        self.assertEqual(snapshot["active"]["run_id"], "run_active")
        self.assertEqual(snapshot["waiting_count"], 2)
        self.assertEqual(
            [(item["run_id"], item["queue_position"]) for item in snapshot["waiting"]],
            [("run_waiting_1", 1), ("run_waiting_2", 2)],
        )
        self.assertEqual(
            [item["run_id"] for item in snapshot["project_items"]],
            ["run_active", "run_waiting_2"],
        )


class GpuJobSchedulerExecutionTests(GpuJobQueueRepositoryTests):
    def tearDown(self) -> None:
        self.scheduler.shutdown(timeout=1)
        super().tearDown()

    def queue_status(self, run_id: str) -> str:
        with self.store.connect() as db:
            return db.execute(
                "SELECT status FROM gpu_job_queue WHERE run_id=?", (run_id,)
            ).fetchone()[0]

    def queue_item(self, run_id: str) -> dict:
        with self.store.connect() as db:
            return dict(db.execute(
                "SELECT * FROM gpu_job_queue WHERE run_id=?", (run_id,)
            ).fetchone())

    def test_two_concurrent_enqueues_run_one_handler_at_a_time(self) -> None:
        handler = FakeHandler(block_first=True)
        self.scheduler.register_handler("initial_champion", handler)
        self.enqueue(self.first_project["project_id"], "initial_champion", "run_one")
        self.enqueue(self.second_project["project_id"], "initial_champion", "run_two")

        self.scheduler.start()
        self.assertTrue(handler.entered.wait(1))
        self.assertEqual(handler.executed, ["run_one"])
        self.assertEqual(handler.max_active, 1)
        handler.release.set()
        wait_until(lambda: self.queue_status("run_two") == "completed")

        self.assertEqual(handler.executed, ["run_one", "run_two"])
        self.assertEqual(handler.max_active, 1)

    def test_completion_failure_and_running_cancellation_advance_fifo(self) -> None:
        handler = FakeHandler(block_first=True, fail_runs={"run_fail"})
        self.scheduler.register_handler("initial_champion", handler)
        first = self.enqueue(self.first_project["project_id"], "initial_champion", "run_cancel")
        self.enqueue(self.second_project["project_id"], "initial_champion", "run_fail")
        self.enqueue(self.first_project["project_id"], "initial_champion", "run_complete")
        self.scheduler.start()
        self.assertTrue(handler.entered.wait(1))

        self.scheduler.cancel(first["queue_id"], "engineer")
        wait_until(lambda: self.queue_status("run_complete") == "completed")

        self.assertEqual(self.queue_status("run_cancel"), "cancelled")
        self.assertEqual(self.queue_status("run_fail"), "failed")
        self.assertEqual(self.queue_status("run_complete"), "completed")

    def test_cancel_queued_never_calls_execute(self) -> None:
        handler = FakeHandler(block_first=True)
        self.scheduler.register_handler("initial_champion", handler)
        self.enqueue(self.first_project["project_id"], "initial_champion", "run_active")
        queued = self.enqueue(self.second_project["project_id"], "initial_champion", "run_never")
        self.scheduler.start()
        self.assertTrue(handler.entered.wait(1))

        cancelled = self.scheduler.cancel(queued["queue_id"], "engineer")
        handler.release.set()
        wait_until(lambda: self.queue_status("run_active") == "completed")

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertNotIn("run_never", handler.executed)
        self.assertEqual(handler.status("run_never"), "cancelled")

    def test_cancel_race_with_previous_completion_does_not_launch_cancelled_item(self) -> None:
        handler = FakeHandler(block_first=True)
        self.scheduler.register_handler("initial_champion", handler)
        self.enqueue(self.first_project["project_id"], "initial_champion", "run_active")
        queued = self.enqueue(self.second_project["project_id"], "initial_champion", "run_race")
        self.scheduler.start()
        self.assertTrue(handler.entered.wait(1))

        self.scheduler.cancel(queued["queue_id"], "engineer")
        handler.release.set()
        wait_until(lambda: self.queue_status("run_active") == "completed")

        self.assertEqual(self.queue_status("run_race"), "cancelled")
        self.assertNotIn("run_race", handler.executed)

    def test_missing_handler_fails_item_and_advances(self) -> None:
        handler = FakeHandler()
        self.scheduler.register_handler("challenger_update", handler)
        self.enqueue(self.first_project["project_id"], "initial_champion", "run_missing")
        self.enqueue(self.second_project["project_id"], "challenger_update", "run_valid")

        self.scheduler.start()
        wait_until(lambda: self.queue_status("run_valid") == "completed")

        missing = self.queue_item("run_missing")
        self.assertEqual(missing["status"], "failed")
        self.assertEqual(missing["error_code"], "GPU_QUEUE_HANDLER_MISSING")
        self.assertEqual(handler.executed, ["run_valid"])

    def test_restart_interrupts_abandoned_lease_and_preserves_waiters(self) -> None:
        handler = FakeHandler()
        self.scheduler.register_handler("initial_champion", handler)
        abandoned = self.enqueue(self.first_project["project_id"], "initial_champion", "run_abandoned")
        self.enqueue(self.second_project["project_id"], "initial_champion", "run_waiting")
        with self.store.connect() as db:
            db.execute(
                "UPDATE gpu_job_queue SET status='running',lease_owner='old-process' WHERE queue_id=?",
                (abandoned["queue_id"],),
            )

        self.scheduler.start()
        wait_until(lambda: self.queue_status("run_waiting") == "completed")

        self.assertEqual(self.queue_status("run_abandoned"), "interrupted")
        self.assertEqual(handler.interrupted, ["run_abandoned"])
        self.assertEqual(handler.executed, ["run_waiting"])


class FakeQueueApi:
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str]] = []

    @staticmethod
    def _item() -> dict:
        return {
            "queue_id": "gpu_queue_public",
            "project_id": "project_public",
            "project_name": "Facade A",
            "pipeline": "challenger_update",
            "run_id": "challenger_run_public",
            "status": "running",
            "cancel_requested": False,
            "queue_position": None,
            "enqueued_at": "2026-10-03T00:00:00+00:00",
            "started_at": "2026-10-03T00:00:01+00:00",
            "updated_at": "2026-10-03T00:00:02+00:00",
            "lease_owner": "must-not-leak",
            "checkpoint_sha256": "a" * 64,
            "worker_path": r"C:\secret\worker.py",
        }

    def snapshot(self, project_id=None) -> dict:
        item = self._item()
        return {
            "active": item,
            "waiting": [],
            "waiting_count": 0,
            "project_items": [item] if project_id == "project_public" else [],
        }

    def get_item(self, queue_id: str) -> dict:
        if queue_id != "gpu_queue_public":
            raise KeyError(queue_id)
        return self._item()

    def cancel(self, queue_id: str, actor_id: str) -> dict:
        self.cancelled.append((queue_id, actor_id))
        return {**self._item(), "status": "cancelled", "cancel_requested": True}


class GpuQueueHTTPTests(unittest.TestCase):
    @staticmethod
    def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        from facade_agent import server

        httpd = server.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=httpd.handle_request, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if encoded is not None else {}
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        thread.join(timeout=5)
        httpd.server_close()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", errors="replace")}
        return response.status, payload

    def test_queue_api_exposes_bounded_state_and_confirmed_cancellation(self) -> None:
        from facade_agent import server

        fake = FakeQueueApi()
        with patch.object(server, "GPU_SCHEDULER", fake, create=True):
            status, snapshot = self.request(
                "GET", "/api/gpu-queue?project_id=project_public"
            )
            item_status, item = self.request(
                "GET", "/api/gpu-queue/item?queue_id=gpu_queue_public"
            )
            denied_status, _ = self.request(
                "POST", "/api/gpu-queue/cancel", {"queue_id": "gpu_queue_public"}
            )
            cancel_status, cancelled = self.request(
                "POST",
                "/api/gpu-queue/cancel",
                {
                    "queue_id": "gpu_queue_public",
                    "actor_id": "engineer",
                    "confirmed": True,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(item_status, 200)
        self.assertEqual(denied_status, 400)
        self.assertEqual(cancel_status, 200)
        serialized = json.dumps([snapshot, item, cancelled])
        self.assertNotIn("must-not-leak", serialized)
        self.assertNotIn("C:\\secret", serialized)
        self.assertNotIn("a" * 64, serialized)
        self.assertEqual(fake.cancelled, [("gpu_queue_public", "engineer")])

    def test_queue_api_and_registry_exclude_final_test(self) -> None:
        from facade_agent import server

        names = {item["name"] for item in server.TOOLS.describe()}
        self.assertFalse(any("final_test" in name for name in names))


if __name__ == "__main__":
    unittest.main()
