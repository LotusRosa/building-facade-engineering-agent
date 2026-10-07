from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import FacadeImageDataset, ImageRecord
from .modeling import build_transforms, load_facade_checkpoint


def distributed_inference(
    *,
    records: list[ImageRecord],
    image_paths: dict[str, Path],
    class_ids: list[str],
    checkpoint: Path,
    input_size: int,
    context: Any,
    include_features: bool = False,
) -> dict[str, dict[str, Any]] | None:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    _, transform = build_transforms(input_size)
    dataset = FacadeImageDataset(records, image_paths, class_ids, transform)
    sampler = DistributedSampler(dataset, context.world_size, context.rank, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    model, _ = load_facade_checkpoint(checkpoint, class_ids)
    model = model.to(context.device).eval()
    local: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            images = images.to(context.device, non_blocking=True)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                encoded = model.features(images)
                pooled = model.avgpool(encoded)
                normalized = model.classifier[0](pooled)
                features = model.classifier[1](normalized)
                logits = model.classifier[2](features)
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
            feature_values = features.float().cpu().numpy() if include_features else None
            for index, image_id in enumerate(image_ids):
                local.append({
                    "image_id": str(image_id),
                    "target": targets[index].to(torch.int64).tolist(),
                    "probabilities": probabilities[index].tolist(),
                    "features": feature_values[index].tolist() if include_features else None,
                })
    gathered: list[Any] = [None for _ in range(context.world_size)] if context.rank == 0 else []
    context.dist.gather_object(local, gathered if context.rank == 0 else None, dst=0)
    if context.rank != 0:
        return None
    merged: dict[str, dict[str, Any]] = {}
    for part in gathered:
        for row in part:
            previous = merged.get(row["image_id"])
            if previous is not None:
                if previous["target"] != row["target"] or not np.allclose(previous["probabilities"], row["probabilities"], atol=1e-6, rtol=0):
                    raise RuntimeError(f"Distributed inference disagreed for {row['image_id']}.")
            merged[row["image_id"]] = row
    expected = {record.image_id for record in records}
    if set(merged) != expected:
        raise RuntimeError("Distributed inference did not cover the frozen image set exactly.")
    return merged
