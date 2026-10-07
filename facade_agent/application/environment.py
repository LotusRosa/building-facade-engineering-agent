from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ..storage import canonical_json, utc_now


def _memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def _nvidia_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "gpus": [], "error": "nvidia-smi not found"}
    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        gpus = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "driver_version": parts[3],
                    }
                )
        return {"available": bool(gpus), "gpus": gpus, "error": None}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "gpus": [], "error": str(error)}


def _torch_inventory() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "version": None, "cuda_available": False, "cuda_version": None, "device_count": 0, "error": None}
    code = (
        "import json,torch;print(json.dumps({"
        "'version':torch.__version__,'cuda_available':torch.cuda.is_available(),"
        "'cuda_version':torch.version.cuda,'device_count':torch.cuda.device_count()}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        return {"installed": True, "error": None, **result}
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        return {"installed": True, "version": None, "cuda_available": False, "cuda_version": None, "device_count": 0, "error": str(error)}


def _training_runtime_inventory() -> dict[str, Any]:
    """Probe the separately managed, version-locked training runtime."""
    module = "facade_training_worker.initial_champion"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "facade_training_worker.diagnostics",
                "--json",
                "--required-gpus",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        return {
            "module": module,
            "available": bool(result.get("available")),
            "ready": bool(result.get("ready")),
            "worker_version": result.get("worker_version"),
            "phase": result.get("phase"),
            "minimum_visible_gpus": result.get("minimum_visible_gpus"),
            "pipelines": result.get("pipelines", {}),
            "pipeline_readiness": result.get("pipeline_readiness", {}),
            "issues": result.get("issues", []),
            "error": None,
        }
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        return {
            "module": module,
            "available": False,
            "ready": False,
            "issues": ["Locked Worker diagnostics could not run."],
            "error": str(error),
        }


def gpu_smoke_test(gpu_devices: list[int]) -> dict[str, Any]:
    """Allocate and synchronize one tensor on every selected visible CUDA device."""
    if not gpu_devices:
        return {"passed": False, "error": "No GPU devices were selected."}
    code = (
        "import json,torch;results=[]\n"
        "for index in range(torch.cuda.device_count()):\n"
        " torch.cuda.set_device(index);x=torch.ones(1024,device='cuda');"
        " y=(x*x).sum();torch.cuda.synchronize();results.append(float(y.item()))\n"
        "print(json.dumps({'device_count':torch.cuda.device_count(),'results':results}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_devices)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        passed = int(result["device_count"]) == len(gpu_devices)
        return {"passed": passed, "details": result, "error": None if passed else "CUDA visible-device count mismatch."}
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def detect_environment(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    nvidia = _nvidia_inventory()
    torch = _torch_inventory()
    training_runtime = _training_runtime_inventory()
    python_ok = sys.version_info >= (3, 11)
    supported_platform = platform.system() in {"Windows", "Linux"} and platform.machine().lower() in {"amd64", "x86_64"}
    local_gpu_ready = bool(
        supported_platform
        and nvidia["available"]
        and torch["cuda_available"]
        and training_runtime.get("ready", training_runtime.get("available", False))
    )
    warnings: list[str] = []
    if not supported_platform:
        warnings.append("Only 64-bit Windows and Linux are supported.")
    if not python_ok:
        warnings.append("Python 3.11 or newer is required by this development build.")
    if not local_gpu_ready:
        warnings.append("Local NVIDIA training is unavailable. Training remains blocked until the driver and managed CUDA-enabled PyTorch runtime pass validation.")
    if training_runtime.get("available") and not training_runtime.get("ready", False):
        warnings.extend(str(item) for item in training_runtime.get("issues", []))
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_ok,
        },
        "cpu": {"logical_cores": os.cpu_count() or 1},
        "memory": {"total_bytes": _memory_bytes()},
        "disk": {"root": str(root.resolve()), "free_bytes": disk.free, "total_bytes": disk.total},
        "nvidia": nvidia,
        "torch": torch,
        "training_runtime": training_runtime,
        "capabilities": {
            "interface_ready": python_ok and supported_platform,
            "supported_platform": supported_platform,
            "local_gpu_training_ready": python_ok and local_gpu_ready,
        },
        "warnings": warnings,
    }


class EnvironmentManager:
    MODES = {"auto", "local_gpu"}

    def __init__(
        self,
        store: Any,
        root: Path,
        detector: Callable[[Path], dict[str, Any]] = detect_environment,
        smoke_tester: Callable[[list[int]], dict[str, Any]] = gpu_smoke_test,
    ) -> None:
        self.store = store
        self.root = root.resolve()
        self.detector = detector
        self.smoke_tester = smoke_tester
        with self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO compute_settings(id,updated_at) VALUES(1,?)",
                (utc_now(),),
            )

    def _settings(self) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM compute_settings WHERE id=1").fetchone()
        result = dict(row)
        result["gpu_devices"] = json.loads(result.pop("gpu_devices_json"))
        return result

    @staticmethod
    def _effective_mode(configured: str, inventory: dict[str, Any]) -> str:
        if configured != "auto":
            return configured
        if inventory["capabilities"]["local_gpu_training_ready"]:
            return "local_gpu"
        return "unsupported"

    def status(self) -> dict[str, Any]:
        inventory = self.detector(self.root)
        settings = self._settings()
        settings["effective_mode"] = self._effective_mode(settings["configured_mode"], inventory)
        setup_script = self.root / ("setup_windows_gpu.ps1" if os.name == "nt" else "setup_linux_gpu.sh")
        return {
            "settings": settings,
            "inventory": inventory,
            "bootstrap": {
                "available": setup_script.is_file(),
                "script": setup_script.name,
                "requires_confirmation": True,
                "installs_into_user_runtime": True,
                "downloads_pretrained_weights": True,
            },
        }

    def bootstrap(self, *, confirmed: bool) -> dict[str, Any]:
        """Provision missing runtime dependencies only after an explicit confirmation."""
        if not confirmed:
            raise PermissionError("Explicit engineer confirmation is required before runtime setup.")
        script = self.root / ("setup_windows_gpu.ps1" if os.name == "nt" else "setup_linux_gpu.sh")
        if not script.is_file():
            raise FileNotFoundError(f"Runtime setup script is missing: {script.name}")
        if os.name == "nt":
            command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        else:
            command = ["bash", str(script)]
        completed = subprocess.run(command, cwd=str(self.root), capture_output=True, text=True, timeout=1800, check=False)
        result = self.status()
        result["bootstrap_result"] = {
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode != 0:
            raise RuntimeError(f"Runtime setup failed with exit code {completed.returncode}.")
        return result

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "auto"))
        if mode not in self.MODES:
            raise ValueError("Unsupported compute mode.")
        inventory = self.detector(self.root)
        available_gpu_ids = {int(item["index"]) for item in inventory["nvidia"]["gpus"]}
        gpu_devices = sorted({int(value) for value in payload.get("gpu_devices", [])})
        if any(value not in available_gpu_ids for value in gpu_devices):
            raise ValueError("A selected GPU is not available on this computer.")
        if mode == "local_gpu":
            if not inventory["capabilities"]["local_gpu_training_ready"]:
                raise ValueError("Local GPU mode requires an NVIDIA driver and CUDA-enabled PyTorch in this runtime.")
            if not gpu_devices:
                gpu_devices = sorted(available_gpu_ids)[:1]
            if len(gpu_devices) != 1:
                raise ValueError("This release supports one GPU at a time; leave selection empty to use the first available GPU.")
        cpu_threads = int(payload.get("cpu_threads", 0))
        logical = int(inventory["cpu"]["logical_cores"])
        if cpu_threads < 0 or cpu_threads > logical:
            raise ValueError(f"CPU threads must be between 0 and {logical}.")
        now = utc_now()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = db.execute("SELECT * FROM compute_settings WHERE id=1").fetchone()
            db.execute(
                "UPDATE compute_settings SET configured_mode=?,gpu_devices_json=?,cpu_threads=?,updated_at=? WHERE id=1",
                (mode, canonical_json(gpu_devices), cpu_threads, now),
            )
            audit_sha256 = self.store._append_audit(
                db,
                project_id=None,
                batch_id=None,
                actor_type="human",
                actor_id="local_engineer",
                tool_name="configure_compute_environment",
                from_state=before["configured_mode"],
                to_state=mode,
                payload={"gpu_devices": gpu_devices, "cpu_threads": cpu_threads},
            )
        result = self.status()
        result["audit_event_sha256"] = audit_sha256
        return result

    def training_preflight(self, pipeline: str | None = None) -> dict[str, Any]:
        """Return an auditable, non-mutating gate report for the local GPU worker."""
        inventory = self.detector(self.root)
        settings = self._settings()
        available = sorted(int(item["index"]) for item in inventory["nvidia"].get("gpus", []))
        selected = (sorted(int(value) for value in settings["gpu_devices"]) or available)[:1]
        checks = [
            {"name": "supported_platform", "passed": bool(inventory["capabilities"].get("supported_platform")), "detail": inventory["platform"]},
            {"name": "supported_python", "passed": bool(inventory["python"].get("supported")), "detail": inventory["python"].get("version")},
            {"name": "nvidia_inventory", "passed": bool(inventory["nvidia"].get("available")), "detail": inventory["nvidia"].get("error")},
            {"name": "cuda_pytorch", "passed": bool(inventory["torch"].get("cuda_available")), "detail": inventory["torch"].get("version")},
            {"name": "selected_gpu_inventory", "passed": bool(selected) and set(selected).issubset(available), "detail": {"selected": selected, "available": available}},
            {
                "name": "locked_training_runtime",
                "passed": bool(
                    inventory.get("training_runtime", {}).get(
                        "ready", inventory.get("training_runtime", {}).get("available")
                    )
                ),
                "detail": inventory.get("training_runtime", {}),
            },
        ]
        if pipeline is not None:
            runtime = inventory.get("training_runtime", {})
            pipelines = runtime.get("pipelines")
            readiness = runtime.get("pipeline_readiness")
            if isinstance(readiness, dict) and pipeline in readiness:
                enabled = readiness[pipeline].get("ready") is True
                pipeline_detail = readiness[pipeline]
            else:
                enabled = (
                    pipelines.get(pipeline) is True
                    if isinstance(pipelines, dict)
                    else bool(runtime.get("ready", runtime.get("available")))
                )
                pipeline_detail = {"pipeline": pipeline, "enabled": enabled}
            checks.append(
                {
                    "name": "worker_pipeline_enabled",
                    "passed": enabled,
                    "detail": pipeline_detail,
                }
            )
        mode = self._effective_mode(settings["configured_mode"], inventory)
        checks.append({"name": "local_gpu_mode", "passed": mode == "local_gpu", "detail": mode})
        if all(item["passed"] for item in checks):
            smoke = self.smoke_tester(selected)
        else:
            smoke = {"passed": False, "error": "Skipped because an earlier preflight check failed."}
        checks.append({"name": "gpu_smoke_test", "passed": bool(smoke.get("passed")), "detail": smoke})
        return {
            "passed": all(item["passed"] for item in checks),
            "checked_at": utc_now(),
            "gpu_devices": selected,
            "world_size": 1 if selected else 0,
            "pipeline": pipeline,
            "checks": checks,
            "inventory": {
                "platform": inventory["platform"],
                "python": inventory["python"],
                "nvidia": inventory["nvidia"],
                "torch": inventory["torch"],
                "training_runtime": inventory.get("training_runtime", {}),
            },
        }



