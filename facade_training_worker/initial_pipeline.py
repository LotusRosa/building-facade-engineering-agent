from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import WORKER_PROTOCOL_VERSION
from .bundle import VerifiedBundle
from .data import (
    FacadeImageDataset,
    deterministic_multilabel_split,
    extract_images,
    parse_classes,
    parse_image_records,
    split_payload,
)
from .metrics import calibrate_thresholds, multilabel_metrics
from .modeling import build_convnext_tiny, build_transforms


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _seed_worker(_: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _amp_dtype(torch: Any) -> Any:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _validate_profile(profile: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "profile_id": "initial_champion_convnext_tiny_768_v2",
        "task": "image_level_multilabel_classification",
        "backbone": "convnext_tiny",
        "input_size": 768,
        "optimizer": "adamw",
        "primary_selection_metric": "development_macro_map",
        "threshold_calibration": "per_class_on_development_only",
        "development_split_strategy": "deterministic_iterative_multilabel_v1",
        "mixed_precision": "auto_bf16_or_fp16",
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"Locked Initial Champion profile field changed: {key}")
    seeds = profile.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3 or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("Initial Champion requires exactly three distinct integer seeds.")
    numeric_positive = (
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "early_stopping_patience",
        "batch_size_per_gpu",
        "gradient_accumulation_steps",
        "max_gradient_norm",
    )
    if any(not isinstance(profile.get(key), (int, float)) or profile[key] <= 0 for key in numeric_positive):
        raise ValueError("Initial Champion profile contains an invalid positive numeric field.")
    if not 0 < float(profile.get("development_fraction", 0)) < 1:
        raise ValueError("Initial Champion development_fraction is invalid.")


def _development_predictions(
    model: Any,
    loader: Any,
    context: Any,
) -> tuple[list[str], np.ndarray | None, np.ndarray | None]:
    import torch

    model.eval()
    local: list[tuple[str, list[int], list[float]]] = []
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            images = images.to(context.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=_amp_dtype(torch)):
                probabilities = torch.sigmoid(model(images)).float().cpu()
            for image_id, target, probability in zip(image_ids, targets, probabilities):
                local.append((str(image_id), target.to(torch.int64).tolist(), probability.tolist()))
    gathered: list[Any] = [None for _ in range(context.world_size)] if context.rank == 0 else []
    context.dist.gather_object(local, gathered if context.rank == 0 else None, dst=0)
    if context.rank != 0:
        return [], None, None
    merged: dict[str, tuple[list[int], list[float]]] = {}
    for rank_rows in gathered:
        for image_id, target, probability in rank_rows:
            existing = merged.get(image_id)
            if existing is not None:
                same_target = existing[0] == target
                same_probability = np.allclose(existing[1], probability, rtol=0.0, atol=1e-6)
                if not same_target or not same_probability:
                    raise RuntimeError(f"Distributed development prediction disagreed for {image_id}.")
            merged[image_id] = (target, probability)
    ordered = sorted(merged)
    targets = np.asarray([merged[image_id][0] for image_id in ordered], dtype=np.int64)
    probabilities = np.asarray([merged[image_id][1] for image_id in ordered], dtype=np.float64)
    return ordered, targets, probabilities


def _save_checkpoint(
    path: Path,
    model: Any,
    *,
    class_ids: list[str],
    profile: dict[str, Any],
    seed: int,
    epoch: int,
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {name: tensor.detach().cpu() for name, tensor in model.module.state_dict().items()}
    payload = {
        "schema_version": 1,
        "checkpoint_format": "facade_convnext_multilabel_v1",
        "backbone": "convnext_tiny",
        "input_size": int(profile["input_size"]),
        "class_ids": class_ids,
        "training_profile_id": profile["profile_id"],
        "seed": seed,
        "best_epoch": epoch,
        "development_metrics": metrics,
        "thresholds": thresholds,
        "model_state_dict": state,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_seed(
    *,
    seed: int,
    seed_index: int,
    profile: dict[str, Any],
    class_ids: list[str],
    train_records: list[Any],
    development_records: list[Any],
    image_paths: dict[str, Path],
    weights_dir: Path,
    context: Any,
    emit: Callable[..., None],
    model_factory: Callable[[int], Any] = build_convnext_tiny,
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    _seed_everything(seed)
    train_transform, development_transform = build_transforms(int(profile["input_size"]))
    train_dataset = FacadeImageDataset(train_records, image_paths, class_ids, train_transform)
    development_dataset = FacadeImageDataset(development_records, image_paths, class_ids, development_transform)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    development_sampler = DistributedSampler(
        development_dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=False,
        drop_last=False,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + context.rank)
    loader_options = {
        "batch_size": int(profile["batch_size_per_gpu"]),
        "num_workers": int(profile["data_loader_workers_per_rank"]),
        "pin_memory": True,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": int(profile["data_loader_workers_per_rank"]) > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_options)
    development_loader = DataLoader(development_dataset, sampler=development_sampler, **loader_options)

    model = model_factory(len(class_ids)).to(context.device)
    model = DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
    )
    positive = np.asarray(
        [sum(class_id in record.class_ids for record in train_records) for class_id in class_ids],
        dtype=np.float32,
    )
    negative = len(train_records) - positive
    pos_weight = np.minimum(negative / positive, float(profile["positive_weight_cap"]))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=context.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    use_scaler = _amp_dtype(torch) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accumulation = int(profile["gradient_accumulation_steps"])
    max_epochs = int(profile["max_epochs"])
    patience = int(profile["early_stopping_patience"])
    checkpoint = weights_dir / f"seed-{seed}.pt"
    best_score = -math.inf
    stale_epochs = 0
    best_metrics: dict[str, float] | None = None
    best_thresholds: dict[str, float] | None = None
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        example_count = 0
        total_steps = len(train_loader)
        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(context.device, non_blocking=True)
            targets = targets.to(context.device, non_blocking=True)
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, total_steps - group_start)
            synchronize = step + 1 == group_start + group_size
            synchronization = nullcontext() if synchronize else model.no_sync()
            with synchronization:
                with torch.autocast(device_type="cuda", dtype=_amp_dtype(torch)):
                    loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss for seed {seed}, epoch {epoch}.")
                scaler.scale(loss / group_size).backward()
            loss_sum += float(loss.detach()) * images.shape[0]
            example_count += images.shape[0]
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(profile["max_gradient_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        totals = torch.tensor([loss_sum, float(example_count)], dtype=torch.float64, device=context.device)
        context.dist.all_reduce(totals)
        training_loss = float(totals[0] / totals[1])
        image_ids, targets_array, probabilities_array = _development_predictions(model, development_loader, context)
        outcome: dict[str, Any] | None = None
        if context.rank == 0:
            if image_ids != sorted(record.image_id for record in development_records):
                raise RuntimeError("Distributed development inference did not cover the frozen split exactly.")
            assert targets_array is not None and probabilities_array is not None
            thresholds = calibrate_thresholds(targets_array, probabilities_array, class_ids)
            metrics = multilabel_metrics(targets_array, probabilities_array, class_ids, thresholds)
            metrics["training_loss"] = training_loss
            score = metrics["development_macro_map"]
            improved = score > best_score
            if improved:
                best_score = score
                stale_epochs = 0
                best_metrics = metrics
                best_thresholds = thresholds
                best_epoch = epoch
                _save_checkpoint(
                    checkpoint,
                    model,
                    class_ids=class_ids,
                    profile=profile,
                    seed=seed,
                    epoch=epoch,
                    metrics=metrics,
                    thresholds=thresholds,
                )
            else:
                stale_epochs += 1
            stop = stale_epochs >= patience
            emit(
                "progress",
                stage="initial_champion_training",
                seed=seed,
                seed_index=seed_index,
                seed_count=len(profile["seeds"]),
                epoch=epoch,
                max_epochs=max_epochs,
                development_macro_map=score,
                best_development_macro_map=best_score,
                training_loss=training_loss,
                early_stop=stop,
                percent=round(100 * ((seed_index - 1) * max_epochs + epoch) / (len(profile["seeds"]) * max_epochs), 2),
            )
            outcome = {"stop": stop, "best_score": best_score}
        message = [outcome]
        context.dist.broadcast_object_list(message, src=0)
        if message[0]["stop"]:
            break

    context.dist.barrier()
    if context.rank != 0:
        return None, None
    if best_metrics is None or best_thresholds is None or not checkpoint.is_file():
        raise RuntimeError(f"Seed {seed} did not produce a valid best checkpoint.")
    result = {
        "seed": seed,
        "weights_file": f"weights/{checkpoint.name}",
        "weights_sha256": _sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "metrics": best_metrics,
    }
    return result, best_thresholds


def build_result_bundle(
    *,
    output: Path,
    run_id: str,
    job_manifest: dict[str, Any],
    profile_bytes: bytes,
    seed_results: list[dict[str, Any]],
    selected_seed: int,
    thresholds: dict[str, float],
    split_bytes: bytes,
    weights_dir: Path,
) -> Path:
    scores = [float(item["metrics"]["development_macro_map"]) for item in seed_results]
    aggregate = {
        "development_macro_map_mean": float(np.mean(scores)),
        "development_macro_map_std": float(np.std(scores)),
        "development_macro_map_min": float(np.min(scores)),
        "development_macro_map_max": float(np.max(scores)),
    }
    manifest = {
        "schema_version": 1,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "job_id": job_manifest["job_id"],
        "run_id": run_id,
        "content_fingerprint": job_manifest["content_fingerprint"],
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "seed_results": seed_results,
        "selected_seed": selected_seed,
        "thresholds": thresholds,
        "aggregate_metrics": aggregate,
        "development_split_file": "development_split.json",
        "development_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "test_read": False,
    }
    byte_payloads = {
        "result_manifest.json": _json_bytes(manifest),
        "development_split.json": split_bytes,
    }
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in byte_payloads.items()}
    weight_sources: dict[str, Path] = {}
    for item in seed_results:
        source = weights_dir / Path(item["weights_file"]).name
        if _sha256_file(source) != item["weights_sha256"]:
            raise RuntimeError(f"Checkpoint changed before result packaging: {source.name}")
        weight_sources[item["weights_file"]] = source
        checksums[item["weights_file"]] = item["weights_sha256"]
    byte_payloads["checksums.json"] = _json_bytes(checksums)
    destination = output / "result_bundle.zip"
    temporary = output / ".result_bundle.tmp"
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name in sorted(set(byte_payloads) | set(weight_sources)):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            if name in byte_payloads:
                archive.writestr(info, byte_payloads[name])
            else:
                with archive.open(info, "w", force_zip64=True) as target, weight_sources[name].open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, destination)
    return destination


def run_initial_champion(
    *,
    bundle: VerifiedBundle,
    output: Path,
    run_id: str,
    context: Any,
    emit: Callable[..., None],
) -> Path | None:
    profile_bytes = bundle.read_bytes("training_profile.json")
    profile = json.loads(profile_bytes)
    _validate_profile(profile)
    job_manifest = bundle.manifest
    classes = bundle.read_json("classes.json")
    class_ids = parse_classes(classes)
    records = parse_image_records(bundle.read_jsonl("labels.jsonl"), class_ids, bundle.names)
    if job_manifest.get("class_count") != len(class_ids) or job_manifest.get("image_count") != len(records):
        raise ValueError("Job manifest counts do not match the immutable dataset.")
    if job_manifest.get("profile_sha256") != hashlib.sha256(profile_bytes).hexdigest():
        raise ValueError("Job manifest profile checksum does not match training_profile.json.")
    train_ids, development_ids = deterministic_multilabel_split(
        records,
        class_ids,
        development_fraction=float(profile["development_fraction"]),
        seed=int(profile["split_seed"]),
    )
    split_bytes = split_payload(train_ids, development_ids, int(profile["split_seed"]))
    workspace = output / ".initial_champion_workspace"
    images_dir = workspace / "images"
    weights_dir = output / "weights"
    if context.rank == 0:
        workspace.mkdir(parents=False, exist_ok=False)
        image_paths = extract_images(bundle.path, records, images_dir)
        weights_dir.mkdir(parents=False, exist_ok=False)
        (workspace / "image_paths.json").write_text(
            json.dumps({key: str(value) for key, value in image_paths.items()}, sort_keys=True),
            encoding="utf-8",
        )
    context.dist.barrier()
    image_paths = {
        key: Path(value)
        for key, value in json.loads((workspace / "image_paths.json").read_text(encoding="utf-8")).items()
    }
    by_id = {record.image_id: record for record in records}
    train_records = [by_id[image_id] for image_id in train_ids]
    development_records = [by_id[image_id] for image_id in development_ids]
    seed_results: list[dict[str, Any]] = []
    thresholds_by_seed: dict[int, dict[str, float]] = {}
    for seed_index, seed in enumerate(profile["seeds"], 1):
        result, thresholds = _train_seed(
            seed=seed,
            seed_index=seed_index,
            profile=profile,
            class_ids=class_ids,
            train_records=train_records,
            development_records=development_records,
            image_paths=image_paths,
            weights_dir=weights_dir,
            context=context,
            emit=emit,
        )
        if context.rank == 0:
            assert result is not None and thresholds is not None
            seed_results.append(result)
            thresholds_by_seed[seed] = thresholds
    result_path: Path | None = None
    if context.rank == 0:
        selected_seed = sorted(
            ((float(item["metrics"][profile["primary_selection_metric"]]), int(item["seed"])) for item in seed_results),
            key=lambda value: (-value[0], value[1]),
        )[0][1]
        result_path = build_result_bundle(
            output=output,
            run_id=run_id,
            job_manifest=job_manifest,
            profile_bytes=profile_bytes,
            seed_results=seed_results,
            selected_seed=selected_seed,
            thresholds=thresholds_by_seed[selected_seed],
            split_bytes=split_bytes,
            weights_dir=weights_dir,
        )
    context.dist.barrier()
    if context.rank == 0:
        shutil.rmtree(workspace)
        shutil.rmtree(weights_dir)
    context.dist.barrier()
    return result_path
