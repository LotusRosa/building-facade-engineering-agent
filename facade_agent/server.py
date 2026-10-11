from __future__ import annotations

import atexit
import hashlib
import json
import mimetypes
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .application.agent_runtime import AgentRuntime
from .application.annotation_tables import AnnotationTableService, MAX_BYTES
from .application.challenger_jobs import ChallengerJobService, register_challenger_job_tool
from .application.challenger_training import ChallengerTrainingService, register_challenger_training_tools
from .application.environment import EnvironmentManager
from .application.evaluation_input import EvaluationInputService
from .application.failure_discovery import FailureDiscoveryService, register_failure_discovery_tools
from .application.failure_review import FailureReviewService, register_failure_review_tools
from .application.image_io import probe_image, safe_filename
from .application.local_training import LocalTrainingWorkerService, register_local_training_tools
from .application.gpu_scheduler import GpuJobScheduler, public_queue_item, public_queue_snapshot
from .application.model_evaluation import ModelEvaluationService, register_model_evaluation_tools
from .application.model_storage import (
    ModelStorageService,
    backfill_legacy_model_storage,
)
from .application.sample_dataset import SampleDatasetImportService, receive_bundle
from .application.training_jobs import InitialChampionJobService, register_initial_training_tool
from .application.workflow import WorkflowSnapshotService, register_workflow_tools
from .adapters.llm import LLMConnectionError, LLMManager
from .storage import Store
from .tools import build_phase1_registry


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
STATIC = PACKAGE / "static"
PROJECT_FILES = ROOT / "projects"
BUNDLE_INBOX = ROOT / "inbox" / "labeled_bundles"
STORE = Store(ROOT / "data" / "facade_agent.sqlite3")
with STORE.connect() as startup_db:
    LEGACY_MODEL_STORAGE = backfill_legacy_model_storage(startup_db, ROOT)
ENVIRONMENT = EnvironmentManager(STORE, ROOT)
MODEL_STORAGE = ModelStorageService(STORE, ROOT)
ORPHAN_MODEL_STORAGE = MODEL_STORAGE.reconcile_orphans()
TOOLS = build_phase1_registry(STORE, MODEL_STORAGE)
GPU_SCHEDULER = GpuJobScheduler(STORE)
TRAINING_JOBS = InitialChampionJobService(
    STORE,
    PROJECT_FILES,
    ROOT / "exports" / "training_jobs",
    PACKAGE / "protocols" / "initial_champion_profile_v1.json",
)
register_initial_training_tool(TOOLS, TRAINING_JOBS)
LOCAL_TRAINING = LocalTrainingWorkerService(
    STORE, TRAINING_JOBS, ENVIRONMENT, ROOT, model_storage=MODEL_STORAGE, scheduler=GPU_SCHEDULER
)
register_local_training_tools(TOOLS, LOCAL_TRAINING)
FAILURE_DISCOVERY = FailureDiscoveryService(
    STORE,
    ENVIRONMENT,
    PROJECT_FILES,
    ROOT,
    PACKAGE / "protocols" / "failure_discovery_profile_v1.json",
    scheduler=GPU_SCHEDULER,
)
register_failure_discovery_tools(TOOLS, FAILURE_DISCOVERY)
FAILURE_REVIEW = FailureReviewService(STORE)
register_failure_review_tools(TOOLS, FAILURE_REVIEW)
CHALLENGER_JOBS = ChallengerJobService(
    STORE,
    PROJECT_FILES,
    ROOT / "exports" / "challenger_jobs",
    PACKAGE / "protocols" / "challenger_update_profile_v1.json",
)
register_challenger_job_tool(TOOLS, CHALLENGER_JOBS)
CHALLENGER_TRAINING = ChallengerTrainingService(
    STORE, CHALLENGER_JOBS, ENVIRONMENT, ROOT, model_storage=MODEL_STORAGE, scheduler=GPU_SCHEDULER
)
register_challenger_training_tools(TOOLS, CHALLENGER_TRAINING)
MODEL_EVALUATION = ModelEvaluationService(
    STORE,
    ENVIRONMENT,
    ROOT,
    PACKAGE / "protocols" / "champion_challenger_evaluation_profile_v2.json",
    scheduler=GPU_SCHEDULER,
)
register_model_evaluation_tools(TOOLS, MODEL_EVALUATION)
GPU_SCHEDULER.start()
atexit.register(GPU_SCHEDULER.shutdown)
WORKFLOW = WorkflowSnapshotService(STORE, GPU_SCHEDULER)
register_workflow_tools(TOOLS, WORKFLOW)
LLM = LLMManager()
AGENT = AgentRuntime(STORE, TOOLS, LLM, WORKFLOW)
SAMPLE_DATASETS = SampleDatasetImportService(STORE, ROOT)
ANNOTATION_TABLES = AnnotationTableService(STORE)
EVALUATION_INPUT = EvaluationInputService(STORE, ROOT, MODEL_EVALUATION)

class Handler(BaseHTTPRequestHandler):
    server_version = "FacadeAgent/2.2"

    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_bytes(self, body: bytes, mime_type: str, filename: str | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("请求过大。")
        return json.loads(self.rfile.read(length) or b"{}")

    def read_bytes(self, limit: int = 100_000_000) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("图片内容为空。")
        if length > limit:
            raise ValueError("单张图片不能超过100 MB。")
        return self.rfile.read(length)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path.startswith("/api/evaluation/input/"):
            try:
                if path == "/api/evaluation/input/status":
                    self.send_json({"ok": True, "result": EVALUATION_INPUT.snapshot(query.get("batch_id", [""])[0])})
                elif path == "/api/evaluation/input/image":
                    content, mime = EVALUATION_INPUT.image_content(query.get("image_id", [""])[0])
                    self.send_bytes(content, mime)
                elif path == "/api/evaluation/input/table":
                    format = query.get("format", ["xlsx"])[0]
                    content = EVALUATION_INPUT.export(query.get("batch_id", [""])[0], query.get("cohort", [""])[0], format, template=query.get("template", ["false"])[0] == "true")
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if format == "xlsx" else "text/csv; charset=utf-8"
                    self.send_bytes(content, mime, f"labels.{format}")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, PermissionError, KeyError, OSError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/annotations/table":
            try:
                format = query.get("format", ["xlsx"])[0]
                content = ANNOTATION_TABLES.export(
                    query.get("project_id", [""])[0], query.get("dataset_id", [None])[0],
                    format, template=query.get("template", ["false"])[0] == "true",
                )
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if format == "xlsx" else "text/csv; charset=utf-8"
                self.send_bytes(content, mime, f"labels.{format}")
            except (ValueError, KeyError, PermissionError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/state":
            projects = STORE.list_projects()
            active = STORE.get_project(projects[0]["project_id"]) if projects else None
            self.send_json({"ok": True, "snapshot": STORE.snapshot(), "projects": projects, "active_project": active})
            return
        if path == "/api/projects":
            self.send_json({"ok": True, "projects": STORE.list_projects()})
            return
        if path == "/api/project":
            self.send_json({"ok": True, "project": STORE.get_project(query.get("project_id", [""])[0])})
            return
        if path == "/api/project/history":
            self.send_json({"ok": True, "history": STORE.get_project_history(query.get("project_id", [""])[0])})
            return
        if path == "/api/workflow":
            self.send_json({"ok": True, "workflow": WORKFLOW.get(query.get("project_id", [""])[0])})
            return
        if path == "/api/gpu-queue":
            project_id = query.get("project_id", [None])[0]
            self.send_json({"ok": True, "gpu_queue": public_queue_snapshot(GPU_SCHEDULER.snapshot(project_id))})
            return
        if path == "/api/gpu-queue/item":
            queue_id = query.get("queue_id", [""])[0]
            if not queue_id:
                self.send_json({"ok": False, "error": "queue_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "item": public_queue_item(GPU_SCHEDULER.get_item(queue_id))})
            return
        if path == "/api/training/jobs":
            self.send_json({"ok": True, "jobs": TRAINING_JOBS.list_jobs(query.get("project_id", [""])[0])})
            return
        if path == "/api/training/jobs/download":
            job = TRAINING_JOBS.get_job(query.get("job_id", [""])[0])
            bundle = Path(job["bundle_path"]).resolve()
            export_root = (ROOT / "exports" / "training_jobs").resolve()
            if export_root != bundle and export_root not in bundle.parents:
                raise PermissionError("Training bundle is outside the managed export directory.")
            content = bundle.read_bytes()
            if hashlib.sha256(content).hexdigest() != job["bundle_sha256"]:
                raise ValueError("Training bundle checksum verification failed.")
            self.send_bytes(content, "application/zip", bundle.name)
            return
        if path == "/api/training/runs":
            self.send_json({"ok": True, "runs": LOCAL_TRAINING.list_runs(query.get("project_id", [""])[0])})
            return
        if path == "/api/training/run":
            after = int(query.get("event_after", ["0"])[0])
            self.send_json({"ok": True, **LOCAL_TRAINING.get_run(query.get("run_id", [""])[0], after)})
            return
        if path == "/api/models":
            self.send_json({"ok": True, "models": LOCAL_TRAINING.list_models(query.get("project_id", [""])[0])})
            return
        if path == "/api/maintenance/batches":
            self.send_json({
                "ok": True,
                "maintenance_batches": STORE.list_maintenance_batches(
                    query.get("project_id", [""])[0]
                ),
            })
            return
        if path == "/api/maintenance/batch":
            self.send_json({
                "ok": True,
                "maintenance_batch": STORE.get_maintenance_batch(
                    query.get("batch_id", [""])[0]
                ),
            })
            return
        if path == "/api/screening/jobs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "jobs": FAILURE_DISCOVERY.list_jobs(project_id)})
            return
        if path == "/api/screening/runs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({
                "ok": True,
                "runs": FAILURE_DISCOVERY.list_runs(
                    project_id, query.get("batch_id", [None])[0]
                ),
            })
            return
        if path == "/api/screening/run":
            run_id = query.get("run_id", [""])[0]
            if not run_id:
                self.send_json({"ok": False, "error": "run_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            after = int(query.get("event_after", ["0"])[0])
            self.send_json({"ok": True, **FAILURE_DISCOVERY.get_run(run_id, after)})
            return
        if path == "/api/failure-slices":
            batch_id = query.get("batch_id", [""])[0]
            if not batch_id:
                self.send_json({"ok": False, "error": "batch_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "failure_slices": FAILURE_DISCOVERY.list_slices(batch_id)})
            return
        if path == "/api/failure-review":
            batch_id = query.get("batch_id", [""])[0]
            if not batch_id:
                self.send_json({"ok": False, "error": "batch_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "review": FAILURE_REVIEW.get_review(batch_id)})
            return
        if path == "/api/challenger/jobs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({
                "ok": True,
                "jobs": CHALLENGER_JOBS.list_jobs(project_id, query.get("batch_id", [None])[0]),
            })
            return
        if path == "/api/challenger/jobs/download":
            job_id = query.get("job_id", [""])[0]
            if not job_id:
                self.send_json({"ok": False, "error": "job_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            verified = CHALLENGER_JOBS.verify_bundle(job_id)
            bundle = Path(verified["bundle_path"])
            self.send_bytes(bundle.read_bytes(), "application/zip", bundle.name)
            return
        if path == "/api/challenger/runs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({
                "ok": True,
                "runs": CHALLENGER_TRAINING.list_runs(
                    project_id, query.get("batch_id", [None])[0]
                ),
            })
            return
        if path == "/api/challenger/run":
            run_id = query.get("run_id", [""])[0]
            if not run_id:
                self.send_json({"ok": False, "error": "run_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            after = int(query.get("event_after", ["0"])[0])
            self.send_json({"ok": True, **CHALLENGER_TRAINING.get_run(run_id, after)})
            return
        if path == "/api/evaluation/inbox":
            self.send_json({"ok": True, "files": MODEL_EVALUATION.list_inbox()})
            return
        if path == "/api/evaluation/gates":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "gates": MODEL_EVALUATION.list_gates(project_id)})
            return
        if path == "/api/evaluation/jobs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "jobs": MODEL_EVALUATION.list_jobs(project_id, query.get("batch_id", [None])[0])})
            return
        if path == "/api/evaluation/jobs/download":
            verified = MODEL_EVALUATION.verify_job(query.get("job_id", [""])[0])
            bundle = Path(verified["bundle_path"])
            self.send_bytes(bundle.read_bytes(), "application/zip", bundle.name)
            return
        if path == "/api/evaluation/runs":
            project_id = query.get("project_id", [""])[0]
            if not project_id:
                self.send_json({"ok": False, "error": "project_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "runs": MODEL_EVALUATION.list_runs(project_id, query.get("batch_id", [None])[0])})
            return
        if path == "/api/evaluation/run":
            run_id = query.get("run_id", [""])[0]
            if not run_id:
                self.send_json({"ok": False, "error": "run_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, **MODEL_EVALUATION.get_run(run_id, int(query.get("event_after", ["0"])[0]))})
            return
        if path == "/api/evaluation/evidence":
            batch_id = query.get("batch_id", [""])[0]
            if not batch_id:
                self.send_json({"ok": False, "error": "batch_id is required."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "evidence": MODEL_EVALUATION.get_evidence(batch_id)})
            return
        if path == "/api/datasets":
            self.send_json({"ok": True, "datasets": STORE.list_datasets(query.get("project_id", [""])[0])})
            return
        if path == "/api/images":
            self.send_json({"ok": True, "images": STORE.list_images(query.get("dataset_id", [""])[0])})
            return
        if path == "/api/images/content":
            image = STORE.get_image(query.get("image_id", [""])[0])
            stored = Path(image["stored_path"]).resolve()
            project_root = PROJECT_FILES.resolve()
            if project_root != stored and project_root not in stored.parents:
                raise PermissionError("Image path is outside the managed project directory.")
            self.send_bytes(stored.read_bytes(), image["mime_type"])
            return
        if path == "/api/tools":
            self.send_json({"ok": True, "tools": TOOLS.describe()})
            return
        if path == "/api/llm/status":
            self.send_json({"ok": True, "llm": LLM.status()})
            return
        if path == "/api/llm/presets":
            self.send_json({"ok": True, "presets": LLM.presets()})
            return
        if path == "/api/environment/status":
            self.send_json({"ok": True, "environment": ENVIRONMENT.status()})
            return
        if path == "/api/agent/messages":
            project_id = query.get("project_id", [None])[0]
            self.send_json({"ok": True, "messages": STORE.list_llm_messages(project_id, 50)})
            return
        if path == "/api/audit/verify":
            self.send_json({"ok": True, "audit_chain": STORE.verify_audit_chain()})
            return
        if path == "/api/protocol":
            protocol = json.loads((PACKAGE / "protocols" / "agent_protocol_v2.json").read_text(encoding="utf-8"))
            self.send_json({"ok": True, "protocol": protocol})
            return
        if path == "/":
            path = "/index.html"
        candidate = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"/api/evaluation/input/upload", "/api/evaluation/input/table/import"}:
                query = parse_qs(parsed.query)
                batch_id, cohort = query.get("batch_id", [""])[0], query.get("cohort", [""])[0]
                if parsed.path.endswith("/upload"):
                    if query.get("confirmed", ["false"])[0] != "true":
                        raise PermissionError("Confirm folder import before receiving evaluation images.")
                    result = EVALUATION_INPUT.upload(batch_id, cohort, query.get("filename", [""])[0], self.read_bytes())
                else:
                    result = EVALUATION_INPUT.import_table(batch_id, cohort, self.read_bytes(limit=MAX_BYTES), query.get("filename", [""])[0], confirmed=query.get("confirmed", ["false"])[0] == "true", preview_token=query.get("preview_token", [""])[0])
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/annotations/table/import":
                query = parse_qs(parsed.query)
                result = ANNOTATION_TABLES.import_table(
                    query.get("dataset_id", [""])[0], self.read_bytes(limit=MAX_BYTES),
                    query.get("filename", [""])[0],
                    confirmed=query.get("confirmed", ["false"])[0] == "true",
                    preview_token=query.get("preview_token", [""])[0],
                )
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/labeled-bundles/import":
                query = parse_qs(parsed.query)
                if query.get("confirmed", [""])[0].casefold() != "true":
                    raise PermissionError(
                        "Explicit engineer confirmation is required before bundle import."
                    )
                target = query.get("target", [""])[0]
                if target not in {"initial", "maintenance"}:
                    raise ValueError("Bundle target must be initial or maintenance.")
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("Bundle Content-Length is invalid.") from exc
                temporary = BUNDLE_INBOX / f"{uuid.uuid4().hex}.zip"
                try:
                    receive_bundle(
                        self.rfile,
                        content_length=content_length,
                        destination=temporary,
                    )
                    if target == "initial":
                        result = SAMPLE_DATASETS.import_initial_bundle(
                            project_id=query.get("project_id", [""])[0],
                            bundle_path=temporary,
                            actor_id="browser_engineer",
                            confirmed=True,
                        )
                    else:
                        result = SAMPLE_DATASETS.import_maintenance_bundle(
                            batch_id=query.get("batch_id", [""])[0],
                            bundle_path=temporary,
                            actor_id="browser_engineer",
                            confirmed=True,
                        )
                finally:
                    temporary.unlink(missing_ok=True)
                self.send_json({"ok": True, "result": result}, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/datasets/upload":
                query = parse_qs(parsed.query)
                dataset_id = query.get("dataset_id", [""])[0]
                filename = safe_filename(query.get("filename", [""])[0])
                dataset = STORE.get_dataset(dataset_id)
                if dataset["status"] != "open":
                    raise PermissionError("The dataset is read-only.")
                content = self.read_bytes()
                probe = probe_image(content, filename)
                if probe.health_status != "ok":
                    raise ValueError("The selected image is unreadable or unsupported; use a valid JPEG/PNG.")
                digest = hashlib.sha256(content).hexdigest()
                existing = next((i for i in STORE.list_images(dataset_id) if i["filename"].casefold() == filename.casefold()), None)
                if existing:
                    if existing["sha256"] != digest:
                        raise ValueError("A different image already uses this filename. Rename the new image.")
                    if hashlib.sha256(Path(existing["stored_path"]).read_bytes()).hexdigest() != digest:
                        raise ValueError("The previously imported image changed on disk.")
                    self.send_json({"ok": True, "image": existing, "idempotent": True})
                    return
                if any(i["sha256"] == digest for i in STORE.list_images(dataset_id)):
                    raise ValueError("This image content is already imported under another filename.")
                folder = PROJECT_FILES / dataset["project_id"] / "datasets" / dataset_id / "images"
                folder.mkdir(parents=True, exist_ok=True)
                stored = folder / f"{digest[:12]}__{filename}"
                if stored.exists():
                    raise ValueError("该图片已经存在于本数据集中。")
                with stored.open("xb") as handle:
                    handle.write(content)
                try:
                    image = STORE.register_image(
                        dataset_id=dataset_id,
                        filename=filename,
                        stored_path=str(stored.resolve()),
                        sha256=digest,
                        size_bytes=len(content),
                        mime_type=probe.mime_type,
                        width=probe.width,
                        height=probe.height,
                        health_status=probe.health_status,
                    )
                except Exception:
                    stored.unlink(missing_ok=True)
                    raise
                self.send_json({"ok": True, "image": image}, HTTPStatus.CREATED)
                return

            body = self.read_json()
            if parsed.path == "/api/evaluation/input/save":
                result = EVALUATION_INPUT.save_label(str(body.get("image_id", "")), body.get("class_ids", []), body.get("no_defect"))
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/evaluation/input/freeze":
                result = EVALUATION_INPUT.freeze(str(body.get("batch_id", "")), confirmed=body.get("confirmed") is True, preview_token=str(body.get("preview_token", "")))
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/gpu-queue/cancel":
                if not bool(body.get("confirmed")):
                    raise PermissionError("Explicit engineer confirmation is required before queue cancellation.")
                queue_id = str(body.get("queue_id", ""))
                if not queue_id:
                    raise ValueError("queue_id is required.")
                item = GPU_SCHEDULER.cancel(
                    queue_id, str(body.get("actor_id", "local_engineer"))
                )
                self.send_json({"ok": True, "item": public_queue_item(item)})
                return
            if parsed.path == "/api/llm/test":
                self.send_json({"ok": True, **LLM.test_connection(body)})
                return
            if parsed.path == "/api/llm/configure":
                self.send_json({"ok": True, "llm": LLM.configure(body)})
                return
            if parsed.path == "/api/environment/configure":
                self.send_json({"ok": True, "environment": ENVIRONMENT.configure(body)})
                return
            if parsed.path == "/api/environment/bootstrap":
                self.send_json({"ok": True, "environment": ENVIRONMENT.bootstrap(confirmed=body.get("confirmed") is True)})
                return
            if parsed.path == "/api/agent/messages/delete":
                raw_project_id = body.get("project_id")
                project_id = str(raw_project_id) if raw_project_id else None
                deleted_count = STORE.delete_llm_messages(project_id)
                self.send_json(
                    {
                        "ok": True,
                        "project_id": project_id,
                        "deleted_count": deleted_count,
                    }
                )
                return
            if parsed.path == "/api/agent/chat":
                result = AGENT.chat(
                    text=str(body.get("text", "")),
                    language=body.get("language"),
                    project_id=str(body["project_id"]) if body.get("project_id") else None,
                    selected_image_id=str(body["selected_image_id"]) if body.get("selected_image_id") else None,
                )
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/agent/confirm":
                if not isinstance(body.get('approved'), bool):
                    raise ValueError('Approval must be a JSON boolean.')
                result = AGENT.resolve_confirmation(
                    pending_id=str(body.get("pending_id", "")),
                    approved=body['approved'],
                    actor_id=str(body.get("actor_id", "local_engineer")),
                    language=body.get("language"),
                )
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/tool/run":
                result = TOOLS.execute(
                    str(body.get("tool", "")),
                    dict(body.get("arguments", {})),
                    actor_type=str(body.get("actor_type", "human")),
                    actor_id=str(body.get("actor_id", "local_engineer")),
                    confirmed=body.get("confirmed") is True,
                )
                project_id = body.get("project_id")
                if not project_id and isinstance(result, dict):
                    run = result.get("run")
                    if isinstance(run, dict):
                        project_id = run.get("project_id")
                result = WORKFLOW.enrich_tool_result(
                    str(project_id) if project_id else None, result
                )
                self.send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/annotations/save":
                if not isinstance(body.get("class_ids"), list) or not isinstance(body.get("no_defect"), bool):
                    raise ValueError("Labels require a class_ids array and boolean no_defect.")
                result = STORE.save_annotation(
                    image_id=str(body.get("image_id", "")),
                    class_ids=body["class_ids"],
                    no_defect=body["no_defect"],
                )
                self.send_json({"ok": True, "image": result})
                return
            if parsed.path == "/api/datasets/validate":
                result = STORE.validate_dataset(dataset_id=str(body.get("dataset_id", "")))
                self.send_json({"ok": True, "report": result})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except LLMConnectionError as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_GATEWAY)
        except KeyError as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.NOT_FOUND)
        except (ValueError, PermissionError) as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"ok": False, "error": f"服务器错误：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)
