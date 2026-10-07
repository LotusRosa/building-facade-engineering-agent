from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL, project_name TEXT NOT NULL DEFAULT '', classes_json TEXT NOT NULL DEFAULT '[]', metadata_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, action TEXT NOT NULL, from_state TEXT NOT NULL, to_state TEXT NOT NULL, payload_json TEXT NOT NULL, event_sha256 TEXT NOT NULL UNIQUE)")
            db.execute("CREATE TABLE IF NOT EXISTS reviews (slice_id TEXT PRIMARY KEY, decision TEXT NOT NULL, excluded_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at)")
            db.execute("INSERT OR IGNORE INTO runtime(id,state,updated_at) VALUES(1,'EMPTY',?)", (utc_now(),))
            db.execute("PRAGMA optimize")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def snapshot(self) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runtime WHERE id=1").fetchone()
            events = db.execute("SELECT id,occurred_at,action,from_state,to_state,event_sha256 FROM events ORDER BY id DESC LIMIT 30").fetchall()
            reviews = db.execute("SELECT * FROM reviews ORDER BY slice_id").fetchall()
        return {
            "state": row["state"],
            "project_name": row["project_name"],
            "classes": json.loads(row["classes_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "updated_at": row["updated_at"],
            "events": [dict(item) for item in events],
            "reviews": [{"slice_id": item["slice_id"], "decision": item["decision"], "excluded": json.loads(item["excluded_json"]), "updated_at": item["updated_at"]} for item in reviews],
        }

    def transition(self, action: str, target: str, payload: dict) -> dict:
        now = utc_now()
        with self.connect() as db:
            row = db.execute("SELECT * FROM runtime WHERE id=1").fetchone()
            metadata = json.loads(row["metadata_json"])
            project_name = row["project_name"]
            classes = json.loads(row["classes_json"])
            if action == "create_project":
                project_name = str(payload.get("project_name", "")).strip()
                classes = [str(value).strip() for value in payload.get("classes", []) if str(value).strip()]
                if not project_name or not classes:
                    raise ValueError("项目名称和至少一个缺陷类别不能为空。")
            metadata.update(payload.get("metadata", {}))
            body = {"occurred_at": now, "action": action, "from_state": row["state"], "to_state": target, "payload": payload}
            digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            db.execute("UPDATE runtime SET state=?,project_name=?,classes_json=?,metadata_json=?,updated_at=? WHERE id=1", (target, project_name, json.dumps(classes, ensure_ascii=False), json.dumps(metadata, ensure_ascii=False), now))
            db.execute("INSERT INTO events(occurred_at,action,from_state,to_state,payload_json,event_sha256) VALUES(?,?,?,?,?,?)", (now, action, row["state"], target, json.dumps(payload, ensure_ascii=False), digest))
        return self.snapshot()

    def save_review(self, slice_id: str, decision: str, excluded: list[str]) -> dict:
        if decision not in {"accept", "trim", "reject"}:
            raise ValueError("审核结论只能是 Accept、Trim 或 Reject。")
        with self.connect() as db:
            state = db.execute("SELECT state FROM runtime WHERE id=1").fetchone()["state"]
            if state != "FAILURE_SLICES_READY":
                raise PermissionError("当前还不能审核 Failure Slice。")
            db.execute("INSERT INTO reviews(slice_id,decision,excluded_json,updated_at) VALUES(?,?,?,?) ON CONFLICT(slice_id) DO UPDATE SET decision=excluded.decision,excluded_json=excluded.excluded_json,updated_at=excluded.updated_at", (slice_id, decision, json.dumps(excluded), utc_now()))
        return self.snapshot()

    def reset(self) -> dict:
        with self.connect() as db:
            db.execute("DELETE FROM reviews")
            db.execute("DELETE FROM events")
            db.execute("UPDATE runtime SET state='EMPTY',project_name='',classes_json='[]',metadata_json='{}',updated_at=? WHERE id=1", (utc_now(),))
        return self.snapshot()

