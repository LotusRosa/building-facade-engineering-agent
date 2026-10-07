from __future__ import annotations

from typing import Any


def classify_worker_error(message: str) -> dict[str, Any]:
    """Convert raw worker text into a stable, user-facing error envelope."""
    text = str(message).strip()
    lowered = text.casefold()
    if "cuda" in lowered or "gpu" in lowered or "nvidia" in lowered:
        code, retryable = "RUNTIME_GPU_UNREADY", True
    elif "checksum" in lowered or "sha" in lowered or "hash" in lowered:
        code, retryable = "BUNDLE_INTEGRITY_FAILED", False
    elif "outside its managed directory" in lowered or "managed" in lowered:
        code, retryable = "MANAGED_PATH_INVALID", False
    elif "cancel" in lowered or "interrupt" in lowered:
        code, retryable = "WORKER_INTERRUPTED", True
    else:
        code, retryable = "WORKER_FAILED", True
    return {"code": code, "message": text[:4000], "retryable": retryable}
