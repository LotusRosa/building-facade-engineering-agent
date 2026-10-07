from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tomllib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CLASS_NAMES = ("crack", "spalling", "hollow")
EXCLUDED_TOP_LEVEL = {
    "build",
    "dist",
    "github_ready",
    ".git",
    ".pytest_cache",
    ".venv",
    "artifact_store",
    "audit",
    "data",
    "exports",
    "inbox",
    "models",
    "projects",
    "trash",
    "venv",
}
EXCLUDED_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
EXCLUDED_DIRECTORY_SUFFIXES = {".egg-info"}
EXCLUDED_SUFFIXES = {
    ".db",
    ".key",
    ".log",
    ".pem",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".zip",
}
EXCLUDED_FILENAMES = {"credentials.json", "secrets.json"}
GITHUB_RELEASE_NAME = "building-facade-engineering-agent"
GITHUB_RELEASE_DIRECTORIES = {"facade_agent", "facade_training_worker", "tests"}
GITHUB_RELEASE_FILES = {
    ".gitignore",
    "ACCUMULATING_GATE_PROTOCOL.md",
    "AGENT_COMPLETE_GPU_HANDOFF.md",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "run_linux.sh",
    "run_windows.ps1",
    "setup_linux_gpu.sh",
    "setup_windows_gpu.ps1",
    "SUPPORTED_ENVIRONMENTS.md",
    "THIRD_PARTY_NOTICES.md",
}
GITHUB_REQUIRED_FILES = {
    ".gitignore",
    "LICENSE",
    "facade_agent/__init__.py",
    "facade_agent/__main__.py",
    "facade_training_worker/__init__.py",
    "pyproject.toml",
    "run_linux.sh",
    "run_windows.ps1",
    "setup_linux_gpu.sh",
    "setup_windows_gpu.ps1",
    "THIRD_PARTY_NOTICES.md",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_signature(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(name, "0") for name in ("crack", "spalling", "hollow", "no_defect"))


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _diverse_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    units: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        unit = row.get("temporal_unit_id") or row.get("capture_group") or row["image_id"]
        units[unit].append(row)
    for unit, values in units.items():
        values.sort(key=lambda row: (_rank(seed, row["image_id"]), row["image_id"]))
    unit_order = sorted(units, key=lambda unit: (_rank(seed, unit), unit))
    ordered: list[dict[str, str]] = []
    while unit_order:
        next_round: list[str] = []
        for unit in unit_order:
            ordered.append(units[unit].pop(0))
            if units[unit]:
                next_round.append(unit)
        unit_order = next_round
    return ordered


def select_role_rows(
    rows: Iterable[dict[str, str]],
    *,
    source_split: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    split = str(source_split).strip()
    candidates = [dict(row) for row in rows if row.get("split") == split]
    if sample_size <= 0 or sample_size > len(candidates):
        raise ValueError(
            f"Sample size must be positive and no larger than source split {split}."
        )
    required = {"image_id", "split", "crack", "spalling", "hollow", "no_defect"}
    for row in candidates:
        if not required.issubset(row) or any(row[name] not in {"0", "1"} for name in CLASS_NAMES + ("no_defect",)):
            raise ValueError("The formal manifest is missing required frozen label fields.")
        selected_count = sum(row[name] == "1" for name in CLASS_NAMES)
        if (row["no_defect"] == "1") == bool(selected_count):
            raise ValueError("The formal manifest contains inconsistent No defect labels.")

    strata: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        strata[_row_signature(row)].append(row)
    raw_quotas = {
        signature: len(values) * sample_size / len(candidates)
        for signature, values in strata.items()
    }
    quotas = {signature: math.floor(value) for signature, value in raw_quotas.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda signature: (-(raw_quotas[signature] - quotas[signature]), signature),
    )
    for signature in order[:remaining]:
        quotas[signature] += 1

    selected: list[dict[str, str]] = []
    for signature in sorted(strata):
        selected.extend(_diverse_order(strata[signature], seed)[: quotas[signature]])
    if len(selected) != sample_size:
        raise RuntimeError("The deterministic sample allocator produced the wrong size.")
    return sorted(selected, key=lambda row: (int(row.get("numeric_image_id") or 0), row["image_id"]))


def select_core_train_rows(
    rows: Iterable[dict[str, str]], *, sample_size: int, seed: int
) -> list[dict[str, str]]:
    """Backward-compatible wrapper for the earlier single-bundle release API."""
    return select_role_rows(
        rows,
        source_split="core_train",
        sample_size=sample_size,
        seed=seed,
    )


def _release_files(agent_root: Path) -> list[Path]:
    output: list[Path] = []
    for path in agent_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"The Agent release cannot contain symbolic links: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(agent_root)
        if (
            relative.parts[0] in EXCLUDED_TOP_LEVEL
            or any(part in EXCLUDED_PARTS for part in relative.parts)
            or any(
                part.casefold().endswith(tuple(EXCLUDED_DIRECTORY_SUFFIXES))
                for part in relative.parts[:-1]
            )
        ):
            continue
        lowered_name = path.name.casefold()
        if (
            path.suffix.casefold() in EXCLUDED_SUFFIXES
            or lowered_name == ".env"
            or lowered_name.startswith(".env.")
            or lowered_name in EXCLUDED_FILENAMES
            or path.name.startswith(".model-storage-probe-")
        ):
            continue
        output.append(path)
    if not output:
        raise ValueError("The Agent source directory contains no releasable files.")
    return sorted(output, key=lambda path: path.relative_to(agent_root).as_posix())


def build_github_release(*, agent_root: Path, output_dir: Path) -> dict[str, Any]:
    """Create a clean source folder and Linux-friendly ZIP for GitHub handoff."""
    agent_root = Path(agent_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not agent_root.is_dir():
        raise ValueError(f"The Agent source directory does not exist: {agent_root}")
    missing = sorted(
        relative
        for relative in GITHUB_REQUIRED_FILES
        if not (agent_root / Path(relative)).is_file()
    )
    if missing:
        raise ValueError(f"The Agent source is missing required release files: {missing}")

    release_dir = output_dir / GITHUB_RELEASE_NAME
    archive_path = output_dir / f"{GITHUB_RELEASE_NAME}.zip"
    checksum_path = archive_path.with_suffix(".zip.sha256")
    occupied = [path for path in (release_dir, archive_path, checksum_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "GitHub release output already exists; choose a new empty output directory: "
            + ", ".join(str(path) for path in occupied)
        )

    files = []
    for source in _release_files(agent_root):
        relative = source.relative_to(agent_root)
        if relative.parts[0] not in GITHUB_RELEASE_DIRECTORIES and relative.as_posix() not in GITHUB_RELEASE_FILES:
            continue
        files.append((source, relative))
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, relative in files:
        destination = release_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with zipfile.ZipFile(
        archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, relative in files:
            archive_name = f"{GITHUB_RELEASE_NAME}/{relative.as_posix()}"
            info = zipfile.ZipInfo.from_file(source, arcname=archive_name)
            info.create_system = 3
            unix_mode = 0o100755 if relative.suffix == ".sh" else 0o100644
            info.external_attr = unix_mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())

    archive_sha256 = _sha256_file(archive_path)
    checksum_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8"
    )
    return {
        "release_dir": str(release_dir),
        "archive": str(archive_path),
        "checksum_file": str(checksum_path),
        "sha256": archive_sha256,
        "file_count": len(files),
    }


def _image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
            continue
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"Multiple source images share an identifier: {path.stem}")
        index[key] = path
    return index


def _write_labeled_bundle(
    *,
    destination: Path,
    selected: list[dict[str, str]],
    source_images: dict[str, Path],
    formal_manifest_sha256: str,
    agent_role: str,
    source_split: str,
    name: str,
    seed: int,
) -> dict[str, Any]:
    resolved_images: list[tuple[dict[str, str], Path, str]] = []
    for row in selected:
        source = source_images.get(row["image_id"].casefold())
        if source is None:
            raise ValueError(f"A selected {source_split} image is missing: {row['image_id']}")
        resolved_images.append((row, source, _sha256_file(source)))

    labels: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    temporal_units: set[str] = set()
    for row, source, digest in resolved_images:
        selected_classes = [class_name for class_name in CLASS_NAMES if row[class_name] == "1"]
        class_counts.update(selected_classes or ["no_defect"])
        temporal_units.add(row.get("temporal_unit_id") or row.get("capture_group") or row["image_id"])
        labels.append(
            {
                "image_id": row["image_id"],
                "filename": source.name,
                "source_split": source_split,
                "temporal_unit_id": row.get("temporal_unit_id", ""),
                "capture_group": row.get("capture_group", ""),
                "labels": selected_classes,
                "no_defect": row["no_defect"] == "1",
                "sha256": digest,
            }
        )
    manifest = {
        "schema_version": 2,
        "name": name,
        "agent_role": agent_role,
        "source_split": source_split,
        "sample_size": len(labels),
        "selection_seed": seed,
        "selection_policy": "proportional exact-label strata with deterministic temporal-unit diversity",
        "classes": list(CLASS_NAMES),
        "class_counts": dict(sorted(class_counts.items())),
        "temporal_unit_count": len(temporal_units),
        "formal_manifest_sha256": formal_manifest_sha256,
    }
    metadata = {
        "classes.json": _json_bytes(list(CLASS_NAMES)),
        "dataset_manifest.json": _json_bytes(manifest),
        "labels.jsonl": (
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels)
        ).encode("utf-8"),
        "README.md": (
            f"# {name}\n\n"
            "This is a labeled example bundle for an Agent trial. Its image count is not "
            "a limit of the Building-Facade Engineering Agent.\n"
        ).encode("utf-8"),
    }
    checksums = {member: _sha256_bytes(content) for member, content in metadata.items()}
    for _, source, digest in resolved_images:
        checksums[f"images/{source.name}"] = digest
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for member, content in sorted(metadata.items()):
            archive.writestr(member, content)
        for _, source, _ in resolved_images:
            archive.write(source, f"images/{source.name}")
        archive.writestr("checksums.json", _json_bytes(checksums))
    return {**manifest, "image_ids": [row["image_id"] for row in labels]}


def _bundle_image_hashes(paths: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for source in paths:
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"An excluded bundle does not exist: {path}")
        with zipfile.ZipFile(path) as archive:
            checksums = json.loads(archive.read("checksums.json"))
        if not isinstance(checksums, dict):
            raise ValueError(f"An excluded bundle has invalid checksums: {path}")
        hashes.update(
            digest
            for member, digest in checksums.items()
            if str(member).startswith("images/") and isinstance(digest, str) and len(digest) == 64
        )
    return hashes


def build_evaluation_snapshot(
    *,
    formal_manifest: Path,
    images_dir: Path,
    destination: Path,
    project_id: str,
    batch_id: str,
    class_ids: dict[str, str],
    current_gate_name: str,
    current_gate_size: int,
    core_safety_size: int,
    excluded_bundles: Iterable[Path] = (),
    seed: int = 20261001,
) -> dict[str, Any]:
    formal_manifest = Path(formal_manifest).resolve()
    images_dir = Path(images_dir).resolve()
    destination = Path(destination).resolve()
    if not formal_manifest.is_file() or not images_dir.is_dir():
        raise ValueError("The frozen manifest and source image directory must exist.")
    if list(class_ids) != list(CLASS_NAMES) or any(not class_ids[name] for name in CLASS_NAMES):
        raise ValueError("Evaluation class IDs must follow the frozen project taxonomy.")

    with formal_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    source_images = _image_index(images_dir)
    forbidden_hashes = _bundle_image_hashes(excluded_bundles)

    def available_rows(split: str, forbidden: set[str]) -> list[dict[str, str]]:
        available: list[dict[str, str]] = []
        for row in rows:
            if row.get("split") != split:
                continue
            source = source_images.get(str(row.get("image_id", "")).casefold())
            if source is None:
                raise ValueError(f"A {split} image is missing: {row.get('image_id')}")
            digest = _sha256_file(source)
            if digest in forbidden:
                continue
            available.append({**row, "_image_path": str(source), "_image_sha256": digest})
        return available

    current_rows = select_role_rows(
        available_rows("adaptation_a_gate", forbidden_hashes),
        source_split="adaptation_a_gate",
        sample_size=current_gate_size,
        seed=seed,
    )
    current_hashes = {row["_image_sha256"] for row in current_rows}
    if len(current_hashes) != len(current_rows):
        raise ValueError("The selected Current Gate contains duplicate image content.")
    core_rows = select_role_rows(
        available_rows("core_safety", forbidden_hashes | current_hashes),
        source_split="core_safety",
        sample_size=core_safety_size,
        seed=seed,
    )
    core_hashes = {row["_image_sha256"] for row in core_rows}
    if len(core_hashes) != len(core_rows) or current_hashes & core_hashes:
        raise ValueError("Current Gate and Core Safety must contain unique image content.")

    split_rows = {"current_gate": current_rows, "core_safety": core_rows}
    labels: list[dict[str, Any]] = []
    image_payloads: dict[str, bytes] = {}
    for split, selected in split_rows.items():
        support = Counter()
        for row in selected:
            selected_classes = [name for name in CLASS_NAMES if row[name] == "1"]
            support.update(selected_classes)
            source = Path(row["_image_path"])
            image_file = f"images/{split}/{source.name}"
            image_payloads[image_file] = source.read_bytes()
            labels.append(
                {
                    "image_id": row["image_id"],
                    "split": split,
                    "image_file": image_file,
                    "image_sha256": row["_image_sha256"],
                    "no_defect": row["no_defect"] == "1",
                    "class_ids": [class_ids[name] for name in selected_classes],
                }
            )
        missing = [name for name in CLASS_NAMES if not support[name]]
        if missing:
            raise ValueError(f"Every Gate needs positive support for every class; {split} lacks {missing}.")

    evaluation_manifest = {
        "schema_version": 2,
        "purpose": "deployment_decision_evidence",
        "project_id": str(project_id),
        "batch_id": str(batch_id),
        "current_gate_name": str(current_gate_name),
        "provided_splits": ["current_gate", "core_safety"],
    }
    classes = [
        {"class_id": class_ids[name], "display_name": name}
        for name in CLASS_NAMES
    ]
    metadata = {
        "evaluation_manifest.json": _json_bytes(evaluation_manifest),
        "classes.json": _json_bytes(classes),
        "labels.jsonl": (
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels)
        ).encode("utf-8"),
    }
    payloads = {**metadata, **image_payloads}
    checksums = {member: _sha256_bytes(content) for member, content in payloads.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for member, content in sorted(payloads.items()):
            archive.writestr(member, content)
        archive.writestr("checksums.json", _json_bytes(checksums))
    bundle_sha256 = _sha256_file(destination)
    sha256_file = destination.with_suffix(destination.suffix + ".sha256")
    sha256_file.write_text(f"{bundle_sha256}  {destination.name}\n", encoding="utf-8")
    return {
        "path": str(destination),
        "sha256": bundle_sha256,
        "sha256_file": str(sha256_file),
        "image_count": len(labels),
        "split_counts": {split: len(selected) for split, selected in split_rows.items()},
    }


def build_linux_trial_release(
    *,
    agent_root: Path,
    formal_manifest: Path,
    images_dir: Path,
    output_dir: Path,
    initial_size: int = 150,
    round_size: int = 50,
    seed: int = 20260930,
) -> dict[str, str]:
    agent_root = Path(agent_root).resolve()
    formal_manifest = Path(formal_manifest).resolve()
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not agent_root.is_dir() or not formal_manifest.is_file() or not images_dir.is_dir():
        raise ValueError("Agent root, frozen manifest, and image directory must exist.")
    with formal_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    initial_rows = select_role_rows(
        rows,
        source_split="core_train",
        sample_size=initial_size,
        seed=seed,
    )
    round_rows = select_role_rows(
        rows,
        source_split="adaptation_a_discovery",
        sample_size=round_size,
        seed=seed,
    )
    initial_ids = {row["image_id"] for row in initial_rows}
    round_ids = {row["image_id"] for row in round_rows}
    if initial_ids & round_ids:
        raise ValueError("Initial and round-one trial samples must be disjoint.")
    source_images = _image_index(images_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    version = "dev"
    pyproject = agent_root / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as stream:
            version = str(tomllib.load(stream).get("project", {}).get("version") or version)
    agent_zip = output_dir / f"building-facade-engineering-agent-linux-v{version}.zip"
    initial_zip = output_dir / f"bfea-trial-initial-{initial_size}.zip"
    round_zip = output_dir / f"bfea-trial-round1-{round_size}.zip"

    with zipfile.ZipFile(agent_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in _release_files(agent_root):
            relative = path.relative_to(agent_root).as_posix()
            archive.write(path, f"building-facade-engineering-agent/{relative}")

    manifest_sha256 = _sha256_file(formal_manifest)
    initial_manifest = _write_labeled_bundle(
        destination=initial_zip,
        selected=initial_rows,
        source_images=source_images,
        formal_manifest_sha256=manifest_sha256,
        agent_role="initial_training",
        source_split="core_train",
        name=f"BFEA trial initial training ({initial_size} images)",
        seed=seed,
    )
    round_manifest = _write_labeled_bundle(
        destination=round_zip,
        selected=round_rows,
        source_images=source_images,
        formal_manifest_sha256=manifest_sha256,
        agent_role="maintenance",
        source_split="adaptation_a_discovery",
        name=f"BFEA trial round one ({round_size} new images)",
        seed=seed,
    )

    agent_sha = _sha256_file(agent_zip)
    initial_sha = _sha256_file(initial_zip)
    round_sha = _sha256_file(round_zip)
    agent_sha_file = agent_zip.with_suffix(agent_zip.suffix + ".sha256")
    initial_sha_file = initial_zip.with_suffix(initial_zip.suffix + ".sha256")
    round_sha_file = round_zip.with_suffix(round_zip.suffix + ".sha256")
    agent_sha_file.write_text(f"{agent_sha}  {agent_zip.name}\n", encoding="utf-8")
    initial_sha_file.write_text(f"{initial_sha}  {initial_zip.name}\n", encoding="utf-8")
    round_sha_file.write_text(f"{round_sha}  {round_zip.name}\n", encoding="utf-8")
    release_manifest = {
        "agent": {"file": agent_zip.name, "sha256": agent_sha},
        "initial_dataset": {
            "file": initial_zip.name,
            "sha256": initial_sha,
            **{key: value for key, value in initial_manifest.items() if key != "image_ids"},
        },
        "round1_dataset": {
            "file": round_zip.name,
            "sha256": round_sha,
            **{key: value for key, value in round_manifest.items() if key != "image_ids"},
        },
        "trial_scope": (
            "one-off 150/50 example trial; image counts are not Agent workflow limits, "
            "and the general Agent remains multi-round"
        ),
    }
    (output_dir / "release_manifest.json").write_bytes(_json_bytes(release_manifest))
    (output_dir / "SHA256SUMS.txt").write_text(
        f"{agent_sha}  {agent_zip.name}\n"
        f"{initial_sha}  {initial_zip.name}\n"
        f"{round_sha}  {round_zip.name}\n",
        encoding="utf-8",
    )
    return {
        "agent_zip": str(agent_zip),
        "initial_dataset_zip": str(initial_zip),
        "round1_dataset_zip": str(round_zip),
        "agent_sha256_file": str(agent_sha_file),
        "initial_dataset_sha256_file": str(initial_sha_file),
        "round1_dataset_sha256_file": str(round_sha_file),
        "release_manifest": str(output_dir / "release_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clean Linux Agent and a one-off initial/round trial data release."
    )
    parser.add_argument("--agent-root", required=True, type=Path)
    parser.add_argument("--formal-manifest", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--initial-size", type=int, default=150)
    parser.add_argument("--round-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    print(
        json.dumps(
            build_linux_trial_release(
                agent_root=args.agent_root,
                formal_manifest=args.formal_manifest,
                images_dir=args.images_dir,
                output_dir=args.output_dir,
                initial_size=args.initial_size,
                round_size=args.round_size,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
