from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import WORKER_PROTOCOL_VERSION
from .bundle import VerifiedBundle
from .data import deterministic_multilabel_split, extract_images, parse_classes, parse_image_records, split_payload
from .initial_pipeline import _train_seed
from .modeling import load_facade_checkpoint
from .result_io import json_bytes, write_result_zip


def _cycle_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("A frozen Challenger source pool became empty after development reservation.")
    return [rows[index % len(rows)] for index in range(count)]


def run_challenger_update(
    *, bundle: VerifiedBundle, output: Path, run_id: str, context: Any, emit: Callable[..., None]
) -> Path | None:
    profile_bytes = bundle.read_bytes("training_profile.json")
    profile = json.loads(profile_bytes)
    parent = bundle.read_json("parent_champion.json")
    class_ids = parse_classes(bundle.read_json("classes.json"))
    records = parse_image_records(bundle.read_jsonl("labels.jsonl"), class_ids, bundle.names)
    train_ids, development_ids = deterministic_multilabel_split(
        records, class_ids, development_fraction=float(profile["development_fraction"]), seed=int(profile["split_seed"])
    )
    development_set = set(development_ids)
    by_id = {record.image_id: record for record in records}
    development_records = [by_id[image_id] for image_id in development_ids]
    split_bytes = split_payload(train_ids, development_ids, int(profile["split_seed"]))
    workspace = output / ".challenger_workspace"
    weights_dir = output / "weights"
    if context.rank == 0:
        workspace.mkdir()
        paths = extract_images(bundle.path, records, workspace / "images")
        with zipfile.ZipFile(bundle.path) as archive:
            (workspace / "parent.pt").write_bytes(archive.read("parent_champion/checkpoint.pt"))
        weights_dir.mkdir()
        (workspace / "paths.json").write_text(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True), encoding="utf-8")
    context.dist.barrier()
    image_paths = {key: Path(value) for key, value in json.loads((workspace / "paths.json").read_text(encoding="utf-8")).items()}
    effective = dict(profile)
    effective["max_epochs"] = int(profile["epochs"])
    effective["early_stopping_patience"] = int(profile["epochs"]) + 1
    seed_results: list[dict[str, Any]] = []
    thresholds_by_seed: dict[int, dict[str, float]] = {}
    draw_hashes: dict[str, str] = {}
    for seed_index, seed in enumerate(profile["seeds"], 1):
        draw_name = f"training_draws/seed_{seed}.jsonl"
        draw_bytes = bundle.read_bytes(draw_name)
        draw_hashes[str(seed)] = hashlib.sha256(draw_bytes).hexdigest()
        rows = bundle.read_jsonl(draw_name)
        selected: list[Any] = []
        for pool, count in profile["draws_per_epoch"].items():
            if pool == "total":
                continue
            candidates = [row for row in rows if row["source_pool"] == pool and row["image_id"] not in development_set]
            selected.extend(by_id[row["image_id"]] for row in _cycle_rows(candidates, int(count)))
        def parent_factory(_: int) -> Any:
            return load_facade_checkpoint(workspace / "parent.pt", class_ids)[0]
        seed_result, thresholds = _train_seed(
            seed=seed, seed_index=seed_index, profile=effective, class_ids=class_ids,
            train_records=selected, development_records=development_records, image_paths=image_paths,
            weights_dir=weights_dir, context=context, emit=emit, model_factory=parent_factory,
        )
        if context.rank == 0:
            assert seed_result is not None and thresholds is not None
            seed_results.append(seed_result)
            thresholds_by_seed[seed] = thresholds
    result: Path | None = None
    if context.rank == 0:
        selected_seed = sorted(
            ((float(item["metrics"]["development_macro_map"]), int(item["seed"])) for item in seed_results),
            key=lambda value: (-value[0], value[1]),
        )[0][1]
        scores = [float(item["metrics"]["development_macro_map"]) for item in seed_results]
        manifest = {
            "schema_version": 1, "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "job_id": bundle.manifest["job_id"], "run_id": run_id,
            "content_fingerprint": bundle.manifest["content_fingerprint"],
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "parent_champion_model_id": parent["model_id"], "epochs": int(profile["epochs"]),
            "test_read": False, "training_draws_sha256": draw_hashes,
            "seed_results": seed_results, "selected_seed": selected_seed,
            "thresholds": thresholds_by_seed[selected_seed],
            "aggregate_metrics": {
                "development_macro_map_mean": float(np.mean(scores)),
                "development_macro_map_std": float(np.std(scores)),
                "development_macro_map_min": float(np.min(scores)),
                "development_macro_map_max": float(np.max(scores)),
            },
            "development_split_file": "development_split.json",
            "development_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        }
        files = {item["weights_file"]: weights_dir / Path(item["weights_file"]).name for item in seed_results}
        result = write_result_zip(output, {"result_manifest.json": json_bytes(manifest), "development_split.json": split_bytes}, files)
    context.dist.barrier()
    if context.rank == 0:
        shutil.rmtree(workspace)
        shutil.rmtree(weights_dir)
    context.dist.barrier()
    return result
