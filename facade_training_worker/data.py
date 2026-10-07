from __future__ import annotations

import hashlib
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    member: str
    sha256: str
    class_ids: tuple[str, ...]
    no_defect: bool


def parse_classes(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("classes.json must be a non-empty array.")
    class_ids: list[str] = []
    for item in value:
        class_id = item.get("class_id") if isinstance(item, dict) else None
        if not isinstance(class_id, str) or not class_id or class_id in class_ids:
            raise ValueError("classes.json contains an invalid or duplicate class_id.")
        class_ids.append(class_id)
    return class_ids


def parse_image_records(rows: Any, class_ids: list[str], bundle_names: Iterable[str]) -> list[ImageRecord]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("labels.jsonl must contain at least one image.")
    names = set(bundle_names)
    allowed = set(class_ids)
    seen: set[str] = set()
    records: list[ImageRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every labels.jsonl row must be an object.")
        image_id = row.get("image_id")
        member = row.get("file")
        digest = row.get("sha256")
        labels = row.get("class_ids")
        no_defect = row.get("no_defect")
        if not isinstance(image_id, str) or not image_id or image_id in seen:
            raise ValueError("labels.jsonl contains an invalid or duplicate image_id.")
        if not isinstance(member, str) or not member.startswith(("images/", "holdout/images/")) or member not in names:
            raise ValueError(f"Image {image_id} refers to an invalid bundle member.")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Image {image_id} has an invalid SHA-256 value.")
        if not isinstance(labels, list) or len(labels) != len(set(labels)) or not set(labels).issubset(allowed):
            raise ValueError(f"Image {image_id} contains invalid class identifiers.")
        if not isinstance(no_defect, bool) or no_defect != (len(labels) == 0):
            raise ValueError(f"Image {image_id} has inconsistent no_defect semantics.")
        seen.add(image_id)
        records.append(ImageRecord(image_id, member, digest, tuple(labels), no_defect))
    return records


def deterministic_multilabel_split(
    records: list[ImageRecord],
    class_ids: list[str],
    *,
    development_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1.")
    if len(records) < 3:
        raise ValueError("At least three images are required for a train/development split.")
    positives = {
        class_id: [record.image_id for record in records if class_id in record.class_ids]
        for class_id in class_ids
    }
    inadequate = [class_id for class_id, values in positives.items() if len(values) < 2]
    if inadequate:
        raise ValueError("Every class needs at least two positive images before training: " + ", ".join(inadequate))

    target_size = min(len(records) - 1, max(1, round(len(records) * development_fraction)))
    rng = random.Random(seed)
    shuffled = [record.image_id for record in records]
    rng.shuffle(shuffled)
    tie_rank = {image_id: index for index, image_id in enumerate(shuffled)}
    by_id = {record.image_id: record for record in records}
    support = {class_id: len(values) for class_id, values in positives.items()}
    desired = {
        class_id: min(count - 1, max(1, round(count * development_fraction)))
        for class_id, count in support.items()
    }
    selected: set[str] = set()
    selected_counts = {class_id: 0 for class_id in class_ids}

    def eligible(record: ImageRecord) -> bool:
        return all(selected_counts[class_id] < support[class_id] - 1 for class_id in record.class_ids)

    while any(selected_counts[class_id] < desired[class_id] for class_id in class_ids):
        candidates = []
        for record in records:
            if record.image_id in selected or not eligible(record):
                continue
            gain = sum(
                1.0 / support[class_id]
                for class_id in record.class_ids
                if selected_counts[class_id] < desired[class_id]
            )
            if gain > 0:
                candidates.append((gain, -tie_rank[record.image_id], record.image_id))
        if not candidates:
            missing = [class_id for class_id in class_ids if selected_counts[class_id] < desired[class_id]]
            raise ValueError("Cannot construct a leakage-free development split for: " + ", ".join(missing))
        image_id = max(candidates)[2]
        selected.add(image_id)
        for class_id in by_id[image_id].class_ids:
            selected_counts[class_id] += 1

    for image_id in shuffled:
        if len(selected) >= target_size:
            break
        record = by_id[image_id]
        if image_id not in selected and eligible(record):
            selected.add(image_id)
            for class_id in record.class_ids:
                selected_counts[class_id] += 1

    train_ids = sorted(set(by_id) - selected)
    development_ids = sorted(selected)
    if not train_ids or not development_ids:
        raise ValueError("Training and development splits must both be non-empty.")
    for class_id in class_ids:
        train_positive = sum(class_id in by_id[image_id].class_ids for image_id in train_ids)
        development_positive = sum(class_id in by_id[image_id].class_ids for image_id in development_ids)
        if train_positive == 0 or development_positive == 0:
            raise ValueError(f"Class {class_id} is not represented in both splits.")
    return train_ids, development_ids


def split_payload(train_ids: list[str], development_ids: list[str], seed: int) -> bytes:
    value = {
        "schema_version": 1,
        "split_strategy": "deterministic_iterative_multilabel_v1",
        "split_seed": seed,
        "train_image_ids": train_ids,
        "development_image_ids": development_ids,
    }
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def extract_images(bundle_path: Path, records: list[ImageRecord], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(bundle_path) as archive:
        for record in records:
            suffix = Path(record.member).suffix.lower()
            target = destination / f"{record.image_id}{suffix}"
            content = archive.read(record.member)
            if hashlib.sha256(content).hexdigest() != record.sha256:
                raise ValueError(f"Image changed after bundle verification: {record.image_id}")
            target.write_bytes(content)
            extracted[record.image_id] = target
    return extracted


class FacadeImageDataset:
    def __init__(
        self,
        records: list[ImageRecord],
        image_paths: dict[str, Path],
        class_ids: list[str],
        transform: Any,
    ) -> None:
        self.records = records
        self.image_paths = image_paths
        self.class_ids = class_ids
        self.class_index = {class_id: index for index, class_id in enumerate(class_ids)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        import torch
        from PIL import Image

        record = self.records[index]
        with Image.open(self.image_paths[record.image_id]) as image:
            tensor = self.transform(image.convert("RGB"))
        target = torch.zeros(len(self.class_ids), dtype=torch.float32)
        for class_id in record.class_ids:
            target[self.class_index[class_id]] = 1.0
        return tensor, target, record.image_id
