from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.permissions import ToolPolicy
from ..core.states import BatchState, ProjectState
from ..storage import Store, canonical_json, make_id, utc_now
from ..tools import ToolDefinition, ToolRegistry
from .training_jobs import FIXED_ZIP_TIME, json_bytes, sha256_file


PROFILE_ID = "challenger_convnext_tiny_768_balanced_replay_v2"
POOL_NAMES = ("expert_confirmed_failure", "remaining_new", "history_replay")


class ChallengerJobService:
    """Build a frozen, reproducible Challenger update job without training it."""

    def __init__(
        self,
        store: Store,
        project_root: Path,
        export_root: Path,
        profile_path: Path,
    ) -> None:
        self.store = store
        self.project_root = project_root.resolve()
        self.export_root = export_root.resolve()
        self.profile_path = profile_path.resolve()

    def _profile(self) -> dict[str, Any]:
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if profile.get("profile_id") != PROFILE_ID:
            raise ValueError("Unexpected Challenger update profile.")
        seeds = profile.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Challenger profile must contain exactly three distinct seeds.")
        if any(not isinstance(seed, int) for seed in seeds):
            raise ValueError("Challenger seeds must be integers.")
        draws = profile.get("draws_per_epoch", {})
        if any(int(draws.get(name, -1)) <= 0 for name in POOL_NAMES):
            raise ValueError("Every fixed Challenger source pool must have positive draws.")
        if sum(int(draws[name]) for name in POOL_NAMES) != int(draws.get("total", -1)):
            raise ValueError("Challenger draw counts do not sum to the locked total.")
        if int(profile.get("epochs", 0)) != 5:
            raise ValueError("The formal Challenger update preset must use five epochs.")
        return profile

    @staticmethod
    def _row_to_job(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        item["pool_counts"] = json.loads(item.pop("pool_counts_json"))
        return item

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM challenger_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Challenger job not found: {job_id}")
        return self._row_to_job(row)

    def list_jobs(self, project_id: str, batch_id: str | None = None) -> list[dict[str, Any]]:
        self.store.get_project(project_id)
        query = "SELECT * FROM challenger_jobs WHERE project_id=?"
        parameters: list[Any] = [project_id]
        if batch_id:
            query += " AND batch_id=?"
            parameters.append(batch_id)
        query += " ORDER BY created_at DESC"
        with self.store.connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return [self._row_to_job(row) for row in rows]

    def _managed_file(
        self,
        raw_path: str,
        expected_sha256: str,
        kind: str,
        managed_root: str | None = None,
    ) -> Path:
        path = Path(raw_path).resolve()
        allowed_roots = [self.project_root]
        if kind == "checkpoint":
            allowed_roots.append(self.project_root.parent / "models")
            allowed_roots.append(self.project_root.parent / "runs")
            if managed_root:
                allowed_roots.append(Path(managed_root).resolve())
        if not any(root == path or root in path.parents for root in allowed_roots):
            raise PermissionError(f"A {kind} is outside the managed local workspace.")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{kind.capitalize()} checksum changed: {path.name}")
        return path

    def _dataset_snapshot(self, db: Any, dataset: Any) -> dict[str, Any]:
        version = db.execute(
            "SELECT * FROM label_versions WHERE dataset_id=? ORDER BY version_number DESC LIMIT 1",
            (dataset["dataset_id"],),
        ).fetchone()
        if version is None:
            raise ValueError(f"Frozen dataset has no label version: {dataset['dataset_id']}")
        hash_rows = [
            dict(row)
            for row in db.execute(
                "SELECT i.image_id,a.no_defect,group_concat(l.class_id,',') AS classes "
                "FROM images i JOIN image_annotations a ON a.image_id=i.image_id "
                "LEFT JOIN image_annotation_labels l ON l.image_id=i.image_id "
                "WHERE i.dataset_id=? GROUP BY i.image_id,a.no_defect ORDER BY i.image_id",
                (dataset["dataset_id"],),
            )
        ]
        current_hash = hashlib.sha256(canonical_json(hash_rows).encode("utf-8")).hexdigest()
        if current_hash != version["labels_sha256"]:
            raise ValueError(f"Labels changed after freezing dataset: {dataset['dataset_id']}")
        images = []
        for row in db.execute(
            "SELECT i.image_id,i.filename,i.stored_path,i.sha256,i.size_bytes,i.mime_type,"
            "i.width,i.height,a.no_defect FROM images i JOIN image_annotations a "
            "ON a.image_id=i.image_id WHERE i.dataset_id=? AND a.status='complete' "
            "ORDER BY i.image_id",
            (dataset["dataset_id"],),
        ):
            image = dict(row)
            image["dataset_id"] = dataset["dataset_id"]
            image["dataset_name"] = dataset["name"]
            image["dataset_role"] = dataset["role"]
            image["class_ids"] = [
                label["class_id"]
                for label in db.execute(
                    "SELECT class_id FROM image_annotation_labels WHERE image_id=? ORDER BY class_id",
                    (image["image_id"],),
                )
            ]
            images.append(image)
        if len(images) != int(version["image_count"]):
            raise ValueError("Frozen label version image count does not match its dataset.")
        return {"dataset": dict(dataset), "label_version": dict(version), "images": images}

    def _load_snapshot(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.get_maintenance_batch(batch_id)
        if batch["state"] != BatchState.FAILURE_REVIEW_COMPLETED:
            raise PermissionError("Freeze the complete Failure Slice review before preparing a Challenger job.")
        project = self.store.get_project(batch["project_id"])
        if project["state"] != ProjectState.SCREENING_READY or project.get("active_batch_id") != batch_id:
            raise PermissionError("The reviewed maintenance batch is not the active governed batch.")
        with self.store.connect() as db:
            review = db.execute(
                "SELECT * FROM failure_review_versions WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if review is None:
                raise ValueError("The frozen Failure Slice review version is missing.")
            review_payload = json.loads(review["snapshot_json"])
            actual_review_hash = hashlib.sha256(canonical_json(review_payload).encode("utf-8")).hexdigest()
            if actual_review_hash != review["review_sha256"]:
                raise ValueError("Frozen Failure Slice review checksum verification failed.")
            if int(review["eligible_slice_count"]) <= 0:
                raise PermissionError("The expert rejected every Failure Slice; no Challenger update job is justified.")
            current_dataset = db.execute(
                "SELECT * FROM datasets WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if current_dataset is None or current_dataset["status"] != "frozen":
                raise PermissionError("The current maintenance dataset is not frozen.")
            current = self._dataset_snapshot(db, current_dataset)
            champion = db.execute(
                "SELECT m.* FROM active_models a JOIN model_versions m ON m.model_id=a.model_id "
                "WHERE a.project_id=?", (batch["project_id"],)
            ).fetchone()
            if champion is None or champion["role"] != "champion":
                raise PermissionError("An Active Champion is required.")
            discovery = db.execute(
                "SELECT champion_model_id FROM screening_jobs WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if discovery is None or discovery["champion_model_id"] != champion["model_id"]:
                raise PermissionError("The Active Champion differs from the model used to discover these slices.")
            classes = [
                dict(row)
                for row in db.execute(
                    "SELECT class_id,display_name,display_name_zh,display_name_en,description,sort_order "
                    "FROM classes WHERE project_id=? AND active=1 ORDER BY sort_order,class_id",
                    (batch["project_id"],),
                )
            ]
            history_datasets = db.execute(
                "SELECT d.* FROM datasets d LEFT JOIN maintenance_batches b ON b.batch_id=d.batch_id "
                "WHERE d.project_id=? AND d.status='frozen' AND d.dataset_id<>? AND "
                "(d.role='initial_training' OR (d.role='maintenance' AND b.state IN ('DEPLOYED','HELD'))) "
                "ORDER BY d.created_at,d.dataset_id",
                (batch["project_id"], current_dataset["dataset_id"]),
            ).fetchall()
            history = [self._dataset_snapshot(db, dataset) for dataset in history_datasets]
            valid_members = {
                (row["slice_id"], row["image_id"])
                for row in db.execute(
                    "SELECT s.slice_id,m.image_id FROM failure_slices s JOIN failure_slice_members m "
                    "ON m.slice_id=s.slice_id WHERE s.batch_id=?", (batch_id,)
                )
            }
        current_by_id = {item["image_id"]: item for item in current["images"]}
        failure_records = []
        for slice_row in review_payload["slices"]:
            if slice_row["decision"] not in {"accept", "trim"}:
                continue
            for image_id in slice_row["retained_image_ids"]:
                if (slice_row["slice_id"], image_id) not in valid_members or image_id not in current_by_id:
                    raise ValueError("Frozen review contains an invalid retained member.")
                failure_records.append({
                    "slice_id": slice_row["slice_id"],
                    "source_slice_key": slice_row["source_slice_key"],
                    "class_id": slice_row["class_id"],
                    "error_type": slice_row["error_type"],
                    "image_id": image_id,
                    "dataset_id": current["dataset"]["dataset_id"],
                })
        if len(failure_records) != int(review["retained_member_records"]):
            raise ValueError("Frozen review retained-member count is inconsistent.")
        retained_ids = {row["image_id"] for row in failure_records}
        remaining = [item for item in current["images"] if item["image_id"] not in retained_ids]
        history_images = [image for dataset in history for image in dataset["images"]]
        if not failure_records:
            raise PermissionError("The frozen expert review contains no eligible failure members.")
        if not remaining:
            raise PermissionError("The fixed update protocol requires at least one remaining-new image.")
        if not history_images:
            raise PermissionError("The fixed update protocol requires a frozen historical replay pool.")
        champion_item = dict(champion)
        champion_item["thresholds"] = json.loads(champion_item.pop("thresholds_json"))
        champion_item["metrics"] = json.loads(champion_item.pop("metrics_json", "{}"))
        return {
            "project": project,
            "batch": batch,
            "review": dict(review),
            "review_payload": review_payload,
            "classes": classes,
            "champion": champion_item,
            "current": current,
            "failure_records": failure_records,
            "remaining_new": remaining,
            "history_datasets": history,
            "history_images": history_images,
        }

    @staticmethod
    def _hash_order(items: list[Any], seed: int, cycle: int, identity) -> list[Any]:
        return sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{seed}:{cycle}:{identity(item)}".encode("utf-8")
            ).hexdigest(),
        )

    def _cyclic_draws(self, items: list[dict[str, Any]], count: int, seed: int, identity) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        cycle = 0
        while len(output) < count:
            output.extend(self._hash_order(items, seed, cycle, identity)[: count - len(output)])
            cycle += 1
        return output

    def _failure_draws(self, records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[record["slice_id"]].append(record)
        output: list[dict[str, Any]] = []
        cycle = 0
        while len(output) < count:
            slice_ids = self._hash_order(list(groups), seed, cycle, lambda value: value)
            for slice_id in slice_ids:
                members = self._hash_order(groups[slice_id], seed, cycle, lambda item: item["image_id"])
                output.append(members[cycle % len(members)])
                if len(output) == count:
                    break
            cycle += 1
        return output

    @staticmethod
    def _zip_write_bytes(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
        info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100444 << 16
        archive.writestr(info, content)

    def create_job(
        self,
        *,
        batch_id: str,
        actor_type: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if actor_type != "human":
            raise PermissionError("Only a human engineer can freeze a Challenger training job.")
        snapshot = self._load_snapshot(batch_id)
        profile = self._profile()
        profile_bytes = json_bytes(profile)
        checkpoint = self._managed_file(
            snapshot["champion"]["checkpoint_path"],
            snapshot["champion"]["checkpoint_sha256"],
            "checkpoint",
            snapshot["project"].get("model_storage_root"),
        )
        all_pool_images = {
            image["image_id"]: image
            for image in (
                snapshot["current"]["images"] + snapshot["history_images"]
            )
        }
        pool_manifest = {
            "schema_version": 1,
            "expert_confirmed_failure": snapshot["failure_records"],
            "remaining_new": [
                {"image_id": item["image_id"], "dataset_id": item["dataset_id"]}
                for item in snapshot["remaining_new"]
            ],
            "history_replay": [
                {"image_id": item["image_id"], "dataset_id": item["dataset_id"]}
                for item in snapshot["history_images"]
            ],
            "history_datasets": [
                {
                    "dataset_id": item["dataset"]["dataset_id"],
                    "name": item["dataset"]["name"],
                    "role": item["dataset"]["role"],
                    "label_version_id": item["label_version"]["label_version_id"],
                    "labels_sha256": item["label_version"]["labels_sha256"],
                    "image_count": item["label_version"]["image_count"],
                }
                for item in snapshot["history_datasets"]
            ],
        }
        pool_counts = {
            "failure_member_records_available": len(snapshot["failure_records"]),
            "failure_unique_images_available": len({item["image_id"] for item in snapshot["failure_records"]}),
            "remaining_new_images_available": len(snapshot["remaining_new"]),
            "history_replay_images_available": len(snapshot["history_images"]),
        }
        draw_payloads: dict[str, bytes] = {}
        selected_image_ids: set[str] = set()
        for seed in profile["seeds"]:
            selected = {
                "expert_confirmed_failure": self._failure_draws(
                    snapshot["failure_records"], profile["draws_per_epoch"]["expert_confirmed_failure"], seed
                ),
                "remaining_new": self._cyclic_draws(
                    snapshot["remaining_new"], profile["draws_per_epoch"]["remaining_new"], seed,
                    lambda item: item["image_id"],
                ),
                "history_replay": self._cyclic_draws(
                    snapshot["history_images"], profile["draws_per_epoch"]["history_replay"], seed,
                    lambda item: item["image_id"],
                ),
            }
            rows = []
            draw_index = 0
            for pool_name in POOL_NAMES:
                for item in selected[pool_name]:
                    draw_index += 1
                    image_id = item["image_id"]
                    selected_image_ids.add(image_id)
                    row = {
                        "draw_index": draw_index,
                        "source_pool": pool_name,
                        "image_id": image_id,
                        "dataset_id": item["dataset_id"],
                    }
                    if pool_name == "expert_confirmed_failure":
                        row.update({key: item[key] for key in ("slice_id", "source_slice_key", "class_id", "error_type")})
                    rows.append(row)
            draw_payloads[f"training_draws/seed_{seed}.jsonl"] = b"".join(
                canonical_json(row).encode("utf-8") + b"\n" for row in rows
            )
        image_fingerprint = [
            {
                "image_id": image_id,
                "dataset_id": all_pool_images[image_id]["dataset_id"],
                "sha256": all_pool_images[image_id]["sha256"],
                "class_ids": all_pool_images[image_id]["class_ids"],
                "no_defect": bool(all_pool_images[image_id]["no_defect"]),
            }
            for image_id in sorted(all_pool_images)
        ]
        fingerprint_payload = {
            "project_id": snapshot["project"]["project_id"],
            "batch_id": batch_id,
            "review_version_id": snapshot["review"]["review_version_id"],
            "review_sha256": snapshot["review"]["review_sha256"],
            "parent_champion_model_id": snapshot["champion"]["model_id"],
            "parent_checkpoint_sha256": snapshot["champion"]["checkpoint_sha256"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "pool_manifest": pool_manifest,
            "images": image_fingerprint,
            "draws": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(draw_payloads.items())},
        }
        content_fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()
        with self.store.connect() as db:
            existing = db.execute("SELECT * FROM challenger_jobs WHERE batch_id=?", (batch_id,)).fetchone()
        if existing is not None:
            job = self._row_to_job(existing)
            if job["content_fingerprint"] != content_fingerprint:
                raise PermissionError("This batch already has a different immutable Challenger job.")
            bundle = Path(job["bundle_path"])
            if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
                raise ValueError("Existing Challenger bundle is missing or changed.")
            return {"job": job, "idempotent": True}
        job_id = f"challenger_{content_fingerprint[:16]}"
        final_dir = self.export_root / job_id
        final_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = final_dir / f"{job_id}.zip"
        temporary_path = final_dir / f".{job_id}.tmp"
        parent = {
            "model_id": snapshot["champion"]["model_id"],
            "checkpoint_file": "parent_champion/checkpoint.pt",
            "checkpoint_sha256": snapshot["champion"]["checkpoint_sha256"],
            "thresholds": snapshot["champion"]["thresholds"],
            "training_profile_id": snapshot["champion"]["training_profile_id"],
        }
        manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "kind": "challenger_update",
            "project_id": snapshot["project"]["project_id"],
            "project_name": snapshot["project"]["name"],
            "batch_id": batch_id,
            "review_version_id": snapshot["review"]["review_version_id"],
            "review_sha256": snapshot["review"]["review_sha256"],
            "parent_champion_model_id": snapshot["champion"]["model_id"],
            "parent_checkpoint_sha256": snapshot["champion"]["checkpoint_sha256"],
            "training_profile_id": profile["profile_id"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "pool_counts": pool_counts,
            "draws_per_seed_per_epoch": profile["draws_per_epoch"],
            "epochs": profile["epochs"],
            "seeds": profile["seeds"],
            "selected_unique_images": len(selected_image_ids),
            "content_fingerprint": content_fingerprint,
            "contract": "Verify every checksum and use only the frozen per-seed draws. No LLM or UI parameter overrides are allowed.",
            "test_read": False,
        }
        payloads: dict[str, bytes] = {
            "job_manifest.json": json_bytes(manifest),
            "classes.json": json_bytes(snapshot["classes"]),
            "training_profile.json": profile_bytes,
            "parent_champion.json": json_bytes(parent),
            "failure_review.json": json_bytes(snapshot["review_payload"]),
            "pool_manifest.json": json_bytes(pool_manifest),
            **draw_payloads,
        }
        image_sources: list[tuple[str, Path]] = []
        label_lines = []
        for image_id in sorted(selected_image_ids):
            image = all_pool_images[image_id]
            source = self._managed_file(image["stored_path"], image["sha256"], "image")
            extension = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".bin"
            archive_name = f"images/{image_id}{extension}"
            label_lines.append(canonical_json({
                "image_id": image_id,
                "dataset_id": image["dataset_id"],
                "dataset_name": image["dataset_name"],
                "original_filename": image["filename"],
                "file": archive_name,
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "class_ids": image["class_ids"],
                "no_defect": bool(image["no_defect"]),
            }).encode("utf-8") + b"\n")
            image_sources.append((archive_name, source))
        payloads["labels.jsonl"] = b"".join(label_lines)
        file_sources = image_sources + [("parent_champion/checkpoint.pt", checkpoint)]
        checksums = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())}
        checksums.update({name: sha256_file(source) for name, source in file_sources})
        payloads["checksums.json"] = json_bytes(checksums)
        try:
            with zipfile.ZipFile(temporary_path, "w") as archive:
                for name in sorted(payloads):
                    self._zip_write_bytes(archive, name, payloads[name])
                for name, source in sorted(file_sources):
                    self._zip_write_bytes(archive, name, source.read_bytes())
            os.replace(temporary_path, bundle_path)
            bundle_sha256 = sha256_file(bundle_path)
            bundle_size = bundle_path.stat().st_size
            now = utc_now()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute(
                    "SELECT state FROM maintenance_batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
                if current is None or current["state"] != BatchState.FAILURE_REVIEW_COMPLETED:
                    raise PermissionError("Batch state changed while the Challenger bundle was prepared.")
                db.execute(
                    "INSERT INTO challenger_jobs(job_id,project_id,batch_id,review_version_id,"
                    "parent_champion_model_id,status,training_profile_id,profile_json,pool_counts_json,"
                    "manifest_sha256,content_fingerprint,bundle_path,bundle_sha256,bundle_size_bytes,"
                    "created_at,updated_at) VALUES(?,?,?,?,?,'exported',?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id, snapshot["project"]["project_id"], batch_id,
                        snapshot["review"]["review_version_id"], snapshot["champion"]["model_id"],
                        profile["profile_id"], canonical_json(profile), canonical_json(pool_counts),
                        hashlib.sha256(payloads["job_manifest.json"]).hexdigest(), content_fingerprint,
                        str(bundle_path), bundle_sha256, bundle_size, now, now,
                    ),
                )
                db.execute(
                    "INSERT INTO artifact_refs(artifact_id,project_id,batch_id,kind,path,sha256,size_bytes,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (make_id("artifact"), snapshot["project"]["project_id"], batch_id,
                     "challenger_training_bundle", str(bundle_path), bundle_sha256, bundle_size, now),
                )
                audit_sha256 = self.store._append_audit(
                    db,
                    project_id=snapshot["project"]["project_id"],
                    batch_id=batch_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    tool_name="create_challenger_training_job",
                    from_state=BatchState.FAILURE_REVIEW_COMPLETED,
                    to_state=BatchState.FAILURE_REVIEW_COMPLETED,
                    payload={
                        "job_id": job_id,
                        "review_version_id": snapshot["review"]["review_version_id"],
                        "parent_champion_model_id": snapshot["champion"]["model_id"],
                        "training_profile_id": profile["profile_id"],
                        "pool_counts": pool_counts,
                        "draws_per_seed_per_epoch": profile["draws_per_epoch"],
                        "bundle_sha256": bundle_sha256,
                        "bundle_size_bytes": bundle_size,
                        "test_read": False,
                    },
                )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            bundle_path.unlink(missing_ok=True)
            raise
        return {"job": self.get_job(job_id), "audit_event_sha256": audit_sha256, "idempotent": False}

    def verify_bundle(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        bundle = Path(job["bundle_path"]).resolve()
        if self.export_root != bundle and self.export_root not in bundle.parents:
            raise PermissionError("Challenger bundle is outside the managed export directory.")
        if not bundle.is_file() or sha256_file(bundle) != job["bundle_sha256"]:
            raise ValueError("Challenger bundle checksum verification failed.")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or len(names) > 100_000:
                raise ValueError("Challenger bundle has duplicate or excessive entries.")
            for name in names:
                pure = Path(name)
                if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                    raise ValueError("Challenger bundle contains an unsafe path.")
            profile = json.loads(archive.read("training_profile.json"))
            manifest = json.loads(archive.read("job_manifest.json"))
            checksums = json.loads(archive.read("checksums.json"))
            required = {
                "job_manifest.json", "classes.json", "training_profile.json", "parent_champion.json",
                "parent_champion/checkpoint.pt", "failure_review.json", "pool_manifest.json",
                "labels.jsonl", "checksums.json",
                *{f"training_draws/seed_{seed}.jsonl" for seed in profile.get("seeds", [])},
            }
            if not required.issubset(names):
                raise ValueError("Challenger bundle is missing a required payload.")
            if set(checksums) != set(names) - {"checksums.json"}:
                raise ValueError("Challenger checksum inventory is incomplete.")
            for name, expected in checksums.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Challenger bundle member checksum failed: {name}")
            if profile != job["profile"] or profile != self._profile():
                raise ValueError("Challenger preset differs from the locked official profile.")
            if manifest.get("job_id") != job_id or manifest.get("content_fingerprint") != job["content_fingerprint"]:
                raise ValueError("Challenger manifest does not match its database record.")
            parent = json.loads(archive.read("parent_champion.json"))
            if parent["model_id"] != job["parent_champion_model_id"]:
                raise ValueError("Parent Champion identity mismatch.")
            if hashlib.sha256(archive.read(parent["checkpoint_file"])).hexdigest() != parent["checkpoint_sha256"]:
                raise ValueError("Parent Champion checkpoint checksum mismatch.")
            labels = {
                row["image_id"]: row
                for row in (
                    json.loads(line) for line in archive.read("labels.jsonl").decode("utf-8").splitlines() if line
                )
            }
            expected_counts = profile["draws_per_epoch"]
            for seed in profile["seeds"]:
                rows = [
                    json.loads(line)
                    for line in archive.read(f"training_draws/seed_{seed}.jsonl").decode("utf-8").splitlines()
                    if line
                ]
                if len(rows) != expected_counts["total"]:
                    raise ValueError("A frozen Challenger draw manifest has the wrong total.")
                if [row["draw_index"] for row in rows] != list(range(1, len(rows) + 1)):
                    raise ValueError("Challenger draw indices are not contiguous.")
                for pool_name in POOL_NAMES:
                    if sum(row["source_pool"] == pool_name for row in rows) != expected_counts[pool_name]:
                        raise ValueError(f"Challenger draw count changed for {pool_name}.")
                if any(row["image_id"] not in labels for row in rows):
                    raise ValueError("A Challenger draw references an image without a frozen label.")
        return {"job": job, "manifest": manifest, "profile": profile, "bundle_path": str(bundle)}


def register_challenger_job_tool(registry: ToolRegistry, service: ChallengerJobService) -> None:
    registry.register(ToolDefinition(
        "create_challenger_training_job",
        "After engineer confirmation, freeze the reviewed failure, remaining-new, and history-replay pools into the locked 160/160/320 three-seed Challenger job. No parameters are accepted.",
        {
            "type": "object",
            "required": ["batch_id"],
            "properties": {"batch_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
        ToolPolicy(mutates_state=True, requires_confirmation=True, allowed_actor_types=("human",)),
        lambda args, ctx: service.create_job(
            batch_id=args["batch_id"], actor_type=ctx["actor_type"], actor_id=ctx["actor_id"]
        ),
    ))
