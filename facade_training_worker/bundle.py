from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MAX_ENTRIES = 200_000
MAX_MEMBER_BYTES = 16 * 1024**3
MAX_TOTAL_BYTES = 512 * 1024**3


class BundleValidationError(ValueError):
    pass


def _sha256_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_names(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise BundleValidationError("Bundle contains duplicate ZIP member names.")
    if len(names) > MAX_ENTRIES:
        raise BundleValidationError("Bundle contains too many ZIP members.")
    total = 0
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        unix_mode = info.external_attr >> 16
        if (
            not name
            or name.startswith(("/", "\\"))
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(unix_mode)
        ):
            raise BundleValidationError(f"Unsafe ZIP member path: {name!r}")
        if info.is_dir():
            raise BundleValidationError("Explicit directory entries are not permitted in governed bundles.")
        if info.file_size > MAX_MEMBER_BYTES:
            raise BundleValidationError(f"ZIP member exceeds the 16 GiB limit: {name}")
        total += info.file_size
    if total > MAX_TOTAL_BYTES:
        raise BundleValidationError("Bundle exceeds the 512 GiB uncompressed limit.")
    return names


@dataclass(frozen=True)
class VerifiedBundle:
    path: Path
    names: tuple[str, ...]
    manifest: dict[str, Any]
    checksums: dict[str, str]

    def read_bytes(self, name: str) -> bytes:
        if name not in self.names:
            raise KeyError(name)
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(name)

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_bytes(name))

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(self.read_bytes(name).decode("utf-8").splitlines(), 1):
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BundleValidationError(f"{name} line {number} is not a JSON object.")
            rows.append(value)
        return rows


def verify_bundle(
    path: Path,
    *,
    required_members: Iterable[str],
    expected_kind: str | None,
) -> VerifiedBundle:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise BundleValidationError("Worker bundle is not a regular file.")
    try:
        with zipfile.ZipFile(resolved) as archive:
            names = _safe_names(archive)
            required = set(required_members) | {"job_manifest.json", "checksums.json"}
            missing = sorted(required - set(names))
            if missing:
                raise BundleValidationError("Bundle is missing required members: " + ", ".join(missing))
            checksums = json.loads(archive.read("checksums.json"))
            if not isinstance(checksums, dict) or set(checksums) != set(names) - {"checksums.json"}:
                raise BundleValidationError("checksums.json does not exactly cover the bundle payload.")
            for name, expected in sorted(checksums.items()):
                if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                    raise BundleValidationError(f"Invalid SHA-256 value for {name}.")
                if _sha256_member(archive, name) != expected:
                    raise BundleValidationError(f"SHA-256 verification failed for {name}.")
            manifest = json.loads(archive.read("job_manifest.json"))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"Cannot read governed bundle: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise BundleValidationError("Unsupported or invalid job manifest.")
    if not isinstance(manifest.get("job_id"), str) or not manifest["job_id"]:
        raise BundleValidationError("Job manifest is missing job_id.")
    if expected_kind is not None and manifest.get("kind") != expected_kind:
        raise BundleValidationError(f"Expected job kind {expected_kind!r}; found {manifest.get('kind')!r}.")
    return VerifiedBundle(resolved, tuple(names), manifest, dict(checksums))
