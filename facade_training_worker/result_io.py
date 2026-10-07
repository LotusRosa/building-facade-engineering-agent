from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def csv_bytes(fieldnames: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_result_zip(output: Path, payloads: dict[str, bytes], files: dict[str, Path] | None = None) -> Path:
    files = files or {}
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    checksums.update({name: sha256_file(path) for name, path in files.items()})
    complete = {**payloads, "checksums.json": json_bytes(checksums)}
    destination = output / "result_bundle.zip"
    temporary = output / ".result_bundle.tmp"
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name in sorted(set(complete) | set(files)):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            if name in complete:
                archive.writestr(info, complete[name])
            else:
                with archive.open(info, "w", force_zip64=True) as target, files[name].open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, destination)
    return destination
