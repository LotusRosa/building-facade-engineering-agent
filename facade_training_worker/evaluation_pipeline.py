from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

from . import WORKER_PROTOCOL_VERSION
from .bundle import VerifiedBundle
from .data import extract_images, parse_classes, parse_image_records
from .inference import distributed_inference
from .result_io import json_bytes, jsonl_bytes, write_result_zip


def run_champion_challenger_evaluation(
    *, bundle: VerifiedBundle, output: Path, run_id: str, context: Any, emit: Callable[..., None]
) -> Path | None:
    profile_bytes = bundle.read_bytes("evaluation_profile.json")
    profile = json.loads(profile_bytes)
    models = bundle.read_json("models.json")
    class_ids = parse_classes(bundle.read_json("holdout/classes.json"))
    labels = bundle.read_jsonl("holdout/labels.jsonl")
    adjusted = [
        {**row, "file": f"holdout/{row.get('file') or row['image_file']}"}
        for row in labels
    ]
    records = parse_image_records(adjusted, class_ids, bundle.names)
    label_by_id = {row["image_id"]: row for row in labels}
    split_by_id = {image_id: row["split"] for image_id, row in label_by_id.items()}
    required_splits = set(profile["required_splits"])
    if set(split_by_id.values()) != required_splits:
        raise ValueError("Evaluation labels do not contain exactly the locked holdout splits.")
    workspace = output / ".evaluation_workspace"
    if context.rank == 0:
        workspace.mkdir()
        paths = extract_images(bundle.path, records, workspace / "images")
        with zipfile.ZipFile(bundle.path) as archive:
            (workspace / "champion.pt").write_bytes(archive.read("models/champion.pt"))
            (workspace / "challenger.pt").write_bytes(archive.read("models/challenger.pt"))
        (workspace / "paths.json").write_text(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True), encoding="utf-8")
    context.dist.barrier()
    image_paths = {key: Path(value) for key, value in json.loads((workspace / "paths.json").read_text(encoding="utf-8")).items()}
    predictions: dict[str, Any] = {}
    for role in ("champion", "challenger"):
        inferred = distributed_inference(
            records=records, image_paths=image_paths, class_ids=class_ids,
            checkpoint=workspace / f"{role}.pt", input_size=int(profile["input_size"]),
            context=context, include_features=False,
        )
        if context.rank == 0:
            predictions[role] = inferred
        emit("progress", stage=f"{role}_holdout_inference", percent=50 if role == "champion" else 100)
    result: Path | None = None
    if context.rank == 0:
        rows = []
        for record in sorted(records, key=lambda item: item.image_id):
            for role in ("champion", "challenger"):
                values = predictions[role][record.image_id]["probabilities"]
                rows.append({
                    "image_id": record.image_id,
                    "split": split_by_id[record.image_id],
                    "cohort_id": label_by_id[record.image_id]["cohort_id"],
                    "source_round": label_by_id[record.image_id]["source_round"],
                    "origin_role": label_by_id[record.image_id]["origin_role"],
                    "model_id": models[role]["model_id"],
                    "probabilities": {class_id: float(values[index]) for index, class_id in enumerate(class_ids)},
                })
        prediction_bytes = jsonl_bytes(rows)
        manifest = {
            "schema_version": 1, "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "job_id": bundle.manifest["job_id"], "run_id": run_id,
            "content_fingerprint": bundle.manifest["content_fingerprint"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "champion_model_id": models["champion"]["model_id"],
            "challenger_model_id": models["challenger"]["model_id"],
            "predictions_file": "predictions.jsonl",
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "test_read": False,
        }
        result = write_result_zip(output, {"result_manifest.json": json_bytes(manifest), "predictions.jsonl": prediction_bytes})
    context.dist.barrier()
    if context.rank == 0:
        shutil.rmtree(workspace)
    context.dist.barrier()
    return result
