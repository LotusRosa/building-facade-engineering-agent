from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from importlib import resources
from typing import Any


def load_worker_manifest() -> dict[str, Any]:
    text = resources.files("facade_training_worker").joinpath("worker_manifest.json").read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("Invalid facade_training_worker manifest.")
    return value


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _dependency_versions(expected: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for package, locked in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        matches = actual is not None and _base_version(actual) == locked
        result[package] = {"locked": locked, "actual": actual, "matches": matches}
        if not matches:
            issues.append(f"Dependency {package} must be {locked}; found {actual or 'missing'}.")
    return result, issues


def collect_runtime_report(required_gpus: int | None = None, pipeline: str | None = None) -> dict[str, Any]:
    manifest = load_worker_manifest()
    required = max(int(required_gpus or 1), int(manifest["minimum_visible_gpus"]))
    issues: list[str] = []
    expected_python = str(manifest["python_major_minor"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        issues.append(f"Python must be {expected_python}.x; found {platform.python_version()}.")

    dependencies, dependency_issues = _dependency_versions(manifest["dependencies"])
    issues.extend(dependency_issues)
    cuda: dict[str, Any] = {
        "available": False,
        "version": None,
        "device_count": 0,
        "devices": [],
        "distributed_available": False,
        "backends": [],
    }
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["version"] = torch.version.cuda
        cuda["device_count"] = int(torch.cuda.device_count())
        distributed = getattr(torch, "distributed", None)
        cuda["distributed_available"] = bool(distributed and distributed.is_available())
        if cuda["distributed_available"]:
            if getattr(distributed, "is_nccl_available", lambda: False)():
                cuda["backends"].append("nccl")
            if getattr(distributed, "is_gloo_available", lambda: False)():
                cuda["backends"].append("gloo")
        for index in range(cuda["device_count"]):
            properties = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
    except Exception as error:  # Runtime probing must report, never disguise, import/driver failures.
        cuda["probe_error"] = f"{type(error).__name__}: {error}"

    if not cuda["available"]:
        issues.append("CUDA-enabled PyTorch is not available.")
    if cuda["device_count"] < required:
        issues.append(f"At least {required} visible CUDA GPU is required; found {cuda['device_count']}.")
    # Production runs are intentionally single-process/single-GPU. Distributed
    # backends may be present, but they are not a readiness requirement.

    minimum_memory = int(manifest["minimum_gpu_memory_gib"]) * 1024**3
    minimum_capability = tuple(int(value) for value in manifest["minimum_compute_capability"])
    for device in cuda["devices"]:
        if int(device["memory_bytes"]) < minimum_memory:
            issues.append(f"GPU {device['index']} has less than {manifest['minimum_gpu_memory_gib']} GiB memory.")
        if tuple(device["compute_capability"]) < minimum_capability:
            issues.append(f"GPU {device['index']} compute capability is below {minimum_capability[0]}.{minimum_capability[1]}.")

    pipeline_readiness: dict[str, dict[str, Any]] = {}
    for name, enabled in manifest["pipelines"].items():
        pipeline_issues: list[str] = []
        assets: dict[str, Any] = {}
        if enabled is not True:
            pipeline_issues.append(f"Production pipeline is not enabled: {name}.")
        if name == "initial_champion" and enabled is True:
            assets["convnext_tiny_imagenet1k_v1"] = {
                "managed_by": "torchvision",
                "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                "readiness_gate": False,
            }
        pipeline_readiness[name] = {
            "enabled": enabled is True,
            "ready": enabled is True and not pipeline_issues,
            "issues": pipeline_issues,
            "assets": assets,
        }
    if pipeline is not None:
        if pipeline not in pipeline_readiness:
            issues.append(f"Unknown governed Worker pipeline: {pipeline}.")
        else:
            issues.extend(pipeline_readiness[pipeline]["issues"])

    return {
        "schema_version": 1,
        "worker_version": manifest["worker_version"],
        "worker_protocol_version": manifest["worker_protocol_version"],
        "phase": manifest["phase"],
        "available": True,
        "ready": not issues,
        "minimum_visible_gpus": required,
        "requested_pipeline": pipeline,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "required_major_minor": expected_python,
            "matches": actual_python == expected_python,
        },
        "dependencies": dependencies,
        "cuda": cuda,
        "pipelines": manifest["pipelines"],
        "pipeline_readiness": pipeline_readiness,
        "issues": issues,
    }
