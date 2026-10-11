"""Human-friendly evaluation folders; the Worker ZIP stays an internal contract."""
from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path
from typing import Any

from ..storage import canonical_json, make_id, utc_now
from .annotation_tables import class_headers, parse_labels, write_table
from .image_io import probe_image, safe_filename
from .model_evaluation import SOURCE_CURRENT_SPLIT, SOURCE_CORE_SEED
from .training_jobs import json_bytes

COHORTS = (SOURCE_CURRENT_SPLIT, SOURCE_CORE_SEED)


class EvaluationInputService:
    def __init__(self, store: Any, root: Path, evaluation: Any) -> None:
        self.store, self.root, self.evaluation = store, root.resolve(), evaluation
        self.lock = threading.RLock()

    def _batch(self, batch_id: str, *, editable: bool = False, db: Any = None) -> dict:
        batch = self.store.get_maintenance_batch(batch_id)
        if editable:
            if self.store.get_project(batch["project_id"]).get("active_batch_id") != batch_id:
                raise PermissionError("Only the active maintenance round accepts evaluation input.")
            if batch["state"] not in {"CREATED", "MAINTENANCE_DATA_IMPORTED", "MAINTENANCE_LABELS_READY", "MAINTENANCE_BATCH_FROZEN", "FAILURE_DISCOVERY_COMPLETED", "FAILURE_REVIEW_COMPLETED", "CHALLENGER_TRAINED"}:
                raise PermissionError("This round no longer accepts evaluation input.")
            if db is None:
                with self.store.connect() as connection:
                    exists = connection.execute("SELECT 1 FROM evaluation_jobs WHERE batch_id=? AND is_current=1", (batch_id,)).fetchone()
            else:
                exists = db.execute("SELECT 1 FROM evaluation_jobs WHERE batch_id=? AND is_current=1", (batch_id,)).fetchone()
            if exists:
                raise PermissionError("The evaluation job is already frozen; input is read-only.")
        return batch

    @staticmethod
    def _cohort(cohort: str) -> None:
        if cohort not in COHORTS:
            raise ValueError("Choose current_gate or the initial core_safety seed.")

    def _history(self, project_id: str) -> list[dict]:
        return [c for c in self.evaluation.list_cohorts(project_id) if c["status"] == "active"]

    def _required(self, batch: dict) -> tuple[str, ...]:
        seeded = any(c["origin_role"] in {"core_safety_seed", "core_safety_addition"} for c in self._history(batch["project_id"]))
        return (SOURCE_CURRENT_SPLIT,) if seeded else COHORTS

    def _editable_cohort(self, batch: dict, cohort: str) -> None:
        self._cohort(cohort)
        if cohort not in self._required(batch):
            raise PermissionError("Core Safety is already established; this round supplies only Current Gate. Train remains separate.")

    def images(self, batch_id: str, cohort: str | None = None, *, db: Any = None) -> list[dict]:
        if cohort is not None:
            self._cohort(cohort)
        if db is None:
            with self.store.connect() as connection:
                return self.images(batch_id, cohort, db=connection)
        rows = db.execute("SELECT * FROM evaluation_input_images WHERE batch_id=?" + (" AND cohort=?" if cohort else "") + " ORDER BY cohort,filename_key", (batch_id, cohort) if cohort else (batch_id,)).fetchall()
        return [dict(row) | {"class_ids": json.loads(row["class_ids_json"])} for row in rows]

    def snapshot(self, batch_id: str) -> dict:
        with self.lock:
            batch = self._batch(batch_id)
            images = self.images(batch_id)
            history = self._history(batch["project_id"])
            required = self._required(batch)
            counts = {c: {"image_count": sum(i["cohort"] == c for i in images), "labeled_count": sum(i["cohort"] == c and i["annotation_status"] == "complete" for i in images)} for c in required}
            lineage = sorted((c["cohort_id"], c["content_sha256"]) for c in history)
            token = hashlib.sha256((self.store.annotation_revision(images) + canonical_json(lineage)).encode()).hexdigest()
            return {"batch_id": batch_id, "project_id": batch["project_id"], "cohorts": counts,
                    "required_cohorts": list(required), "core_safety_seed_required": SOURCE_CORE_SEED in required,
                    "core_safety_history_count": sum(int(c["image_count"]) for c in history),
                    "preview_token": token, "complete": all(v["image_count"] and v["image_count"] == v["labeled_count"] for v in counts.values()),
                    "images": [{k: v for k, v in i.items() if k not in {"stored_path", "sha256", "class_ids_json", "filename_key"}} for i in images]}

    def upload(self, batch_id: str, cohort: str, filename: str, content: bytes) -> dict:
        self._cohort(cohort)
        filename = safe_filename(filename)
        if "/" in filename or "\\" in filename or Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Select JPEG/PNG images with plain filenames.")
        probe = probe_image(content, filename)
        if probe.health_status != "ok":
            raise ValueError(f"Unreadable or unsupported evaluation image: {filename}")
        digest = hashlib.sha256(content).hexdigest()
        with self.lock, self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = self._batch(batch_id, editable=True, db=db)
            self._editable_cohort(batch, cohort)
            existing = db.execute("SELECT * FROM evaluation_input_images WHERE batch_id=? AND cohort=? AND filename_key=?", (batch_id, cohort, filename.casefold())).fetchone()
            if existing:
                if existing["sha256"] != digest:
                    raise ValueError("A different image already uses this filename. Rename the new image.")
                self.image_content(existing["image_id"])
                return {"image_id": existing["image_id"], "filename": existing["filename"], "idempotent": True}
            if db.execute("SELECT 1 FROM evaluation_input_images WHERE batch_id=? AND sha256=?", (batch_id, digest)).fetchone():
                raise ValueError("An evaluation image is duplicated across cohorts or filenames.")
            for historical in self._history(batch["project_id"]):
                historical_rows, _ = self.evaluation._load_registered_cohort(historical, include_images=False)
                if digest in {row["image_sha256"] for row in historical_rows}:
                    raise ValueError("Evaluation images must not overlap existing Core Safety or historical Gates.")
            if db.execute("SELECT 1 FROM images i JOIN datasets d ON d.dataset_id=i.dataset_id WHERE d.project_id=? AND i.sha256=?", (batch["project_id"], digest)).fetchone():
                raise ValueError("Evaluation images must not overlap training or maintenance images.")
            image_id = make_id("eval_image")
            directory = (self.root / "projects" / batch["project_id"] / "evaluation_input" / batch_id / cohort).resolve()
            project_root = (self.root / "projects").resolve()
            if project_root not in directory.parents:
                raise PermissionError("Evaluation path is outside the managed project directory.")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{image_id}{Path(filename).suffix.lower()}"
            try:
                with path.open("xb") as handle:
                    handle.write(content)
                db.execute("INSERT INTO evaluation_input_images(image_id,project_id,batch_id,cohort,filename,filename_key,stored_path,sha256,width,height,annotation_updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (image_id, batch["project_id"], batch_id, cohort, filename, filename.casefold(), str(path), digest, probe.width, probe.height, utc_now()))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return {"image_id": image_id, "filename": filename, "idempotent": False}

    def image_content(self, image_id: str) -> tuple[bytes, str]:
        with self.store.connect() as db:
            image = db.execute("SELECT * FROM evaluation_input_images WHERE image_id=?", (image_id,)).fetchone()
        if image is None:
            raise KeyError("Evaluation image not found.")
        path = Path(image["stored_path"]).resolve()
        if (self.root / "projects").resolve() not in path.parents:
            raise PermissionError("Evaluation image path is outside the project directory.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != image["sha256"]:
            raise ValueError("Evaluation image has changed on disk.")
        return content, probe_image(content, image["filename"]).mime_type

    def export(self, batch_id: str, cohort: str, format: str, *, template: bool = False) -> bytes:
        batch = self._batch(batch_id)
        classes = self.store.get_project(batch["project_id"])["classes"]
        rows = [["filename", *class_headers(classes), "no_defect"]]
        for image in self.images(batch_id, cohort):
            values = [int(c["class_id"] in image["class_ids"]) for c in classes] + [int(image["no_defect"])] if not template and image["annotation_status"] == "complete" else [""] * (len(classes) + 1)
            rows.append([image["filename"], *values])
        return write_table(rows, format)

    def import_table(self, batch_id: str, cohort: str, content: bytes, filename: str, *, confirmed: bool = False, preview_token: str = "") -> dict:
        self._cohort(cohort)
        with self.lock, self.store.connect() as db:
            # The write lock protects both the preview revision and all applied rows.
            if confirmed:
                db.execute("BEGIN IMMEDIATE")
            batch = self._batch(batch_id, editable=True, db=db)
            self._editable_cohort(batch, cohort)
            classes = self.store.get_project(batch["project_id"])["classes"]
            images = self.images(batch_id, cohort, db=db)
            parsed = parse_labels(content, filename, classes, images)
            token = hashlib.sha256(content + self.store.annotation_revision(images).encode()).hexdigest()
            if confirmed:
                if preview_token != token:
                    raise PermissionError("Evaluation labels changed. Preview the table again.")
                self._apply(db, batch, parsed["entries"])
            return {k: v for k, v in parsed.items() if k != "entries"} | {"applied": confirmed, "preview_token": token}

    def _apply(self, db: Any, batch: dict, entries: list[dict]) -> None:
        now = utc_now()
        for entry in entries:
            db.execute("UPDATE evaluation_input_images SET class_ids_json=?,no_defect=?,annotation_status='complete',annotation_updated_at=? WHERE image_id=? AND batch_id=?", (canonical_json(entry["class_ids"]), int(entry["no_defect"]), now, entry["image_id"], batch["batch_id"]))
        self.store._append_audit(db, project_id=batch["project_id"], batch_id=batch["batch_id"], actor_type="human", actor_id="browser_engineer", tool_name="save_evaluation_labels", from_state=batch["state"], to_state=batch["state"], payload={"image_count": len(entries), "labels_sha256": hashlib.sha256(canonical_json(entries).encode()).hexdigest()})

    def save_label(self, image_id: str, class_ids: list[str], no_defect: bool) -> dict:
        if not isinstance(no_defect, bool) or not isinstance(class_ids, list) or no_defect == bool(class_ids) or len(set(class_ids)) != len(class_ids):
            raise ValueError("Choose one or more defects or No defect, exclusively.")
        with self.lock, self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            image = db.execute("SELECT * FROM evaluation_input_images WHERE image_id=?", (image_id,)).fetchone()
            if image is None:
                raise KeyError("Evaluation image not found.")
            batch = self._batch(image["batch_id"], editable=True, db=db)
            self._editable_cohort(batch, image["cohort"])
            classes = self.store.get_project(batch["project_id"])["classes"]
            if set(class_ids) - {c["class_id"] for c in classes}:
                raise ValueError("Evaluation labels differ from the project taxonomy.")
            self._apply(db, batch, [{"image_id": image_id, "class_ids": class_ids, "no_defect": no_defect}])
        return {"image_id": image_id, "saved": True}

    def freeze(self, batch_id: str, *, confirmed: bool, preview_token: str) -> dict:
        if not confirmed:
            raise PermissionError("Human confirmation is required before freezing evaluation input.")
        with self.lock:
            batch = self._batch(batch_id, editable=True)
            if batch["state"] != "CHALLENGER_TRAINED":
                raise PermissionError("Evaluation job preparation requires the registered Challenger; early annotations are retained.")
            snapshot = self.snapshot(batch_id)
            if preview_token != snapshot["preview_token"]:
                raise PermissionError("Evaluation input changed. Review it again before confirming.")
            if not snapshot["complete"]:
                raise ValueError("Complete annotations in every required independent evaluation cohort first.")
            classes = [{"class_id": c["class_id"], "display_name": c["display_name"]} for c in self.store.get_project(batch["project_id"])["classes"]]
            rows = self.images(batch_id)
            if any(i["cohort"] not in snapshot["required_cohorts"] for i in rows):
                raise ValueError("Unexpected evaluation cohort: only the initial round accepts a Core Safety seed.")
            labels = [{"image_id": i["image_id"], "split": i["cohort"], "image_file": f"images/{i['image_id']}{Path(i['filename']).suffix.lower()}", "image_sha256": i["sha256"], "class_ids": i["class_ids"], "no_defect": bool(i["no_defect"])} for i in rows]
            files = {
                "classes.json": json_bytes(classes),
                "labels.jsonl": b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in labels),
                "evaluation_manifest.json": json_bytes({"schema_version": 4, "purpose": "champion_challenger_selection", "project_id": batch["project_id"], "batch_id": batch_id, "provided_splits": snapshot["required_cohorts"], "label_version_id": f"folder-labels-{snapshot['preview_token'][:16]}", "taxonomy_sha256": hashlib.sha256(json_bytes(classes)).hexdigest(), "final_test": False}),
            }
            inbox = self.evaluation.inbox
            inbox.mkdir(parents=True, exist_ok=True)
            name = f"folder-evaluation-{snapshot['preview_token'][:24]}.zip"
            path = inbox / name
            temporary = inbox / f"{make_id('evaluation_draft')}.tmp"
            # The legacy backend repeats checksum, taxonomy, support and leakage checks.
            checksums = {n: hashlib.sha256(value).hexdigest() for n, value in files.items()}
            try:
                self._write_bundle(temporary, files, rows, labels, checksums)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return self.evaluation.create_job(batch_id, name, "human", "browser_engineer")

    def _write_bundle(self, path: Path, files: dict, rows: list, labels: list, checksums: dict) -> None:
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
            for n, value in files.items():
                self.evaluation._write_member(archive, n, value)
            for image, label in zip(rows, labels):
                content, _ = self.image_content(image["image_id"])
                self.evaluation._write_member(archive, label["image_file"], content)
                checksums[label["image_file"]] = hashlib.sha256(content).hexdigest()
            self.evaluation._write_member(archive, "checksums.json", json_bytes(checksums))
