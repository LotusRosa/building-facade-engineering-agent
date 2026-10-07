from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from typing import Any


IMAGENET_CHECKPOINT_SHA256_PREFIX = "983f1562"


def cached_imagenet_weights_path() -> Path:
    import torch
    from torchvision.models import ConvNeXt_Tiny_Weights

    filename = Path(urlparse(ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url).path).name
    return Path(torch.hub.get_dir()).resolve() / "checkpoints" / filename


def verify_cached_imagenet_weights() -> dict[str, Any]:
    import hashlib

    checkpoint = cached_imagenet_weights_path()
    if not checkpoint.is_file():
        return {
            "path": str(checkpoint),
            "available": False,
            "sha256": None,
            "sha256_prefix": IMAGENET_CHECKPOINT_SHA256_PREFIX,
            "valid": False,
        }
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return {
        "path": str(checkpoint),
        "available": True,
        "sha256": value,
        "sha256_prefix": IMAGENET_CHECKPOINT_SHA256_PREFIX,
        "valid": value.startswith(IMAGENET_CHECKPOINT_SHA256_PREFIX),
    }


def build_convnext_tiny(class_count: int) -> Any:
    import torch
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    distributed = torch.distributed
    distributed_run = bool(distributed.is_available() and distributed.is_initialized())
    rank = distributed.get_rank() if distributed_run else 0
    model = convnext_tiny(weights=weights) if rank == 0 else None
    if distributed_run:
        distributed.barrier()
        if rank != 0:
            model = convnext_tiny(weights=weights)
    if model is None:
        raise RuntimeError("Torchvision did not construct the configured ConvNeXt-Tiny model.")
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, class_count)
    return model


def load_facade_checkpoint(path: Path, class_ids: list[str]) -> tuple[Any, dict[str, Any]]:
    import torch
    from torchvision.models import convnext_tiny

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported Facade checkpoint schema.")
    if payload.get("checkpoint_format") != "facade_convnext_multilabel_v1":
        raise ValueError("Unsupported Facade checkpoint format.")
    if payload.get("backbone") != "convnext_tiny" or payload.get("class_ids") != class_ids:
        raise ValueError("Checkpoint architecture or frozen class order does not match the job.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Facade checkpoint is missing model_state_dict.")
    model = convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(class_ids))
    model.load_state_dict(state, strict=True)
    return model, payload


def build_transforms(input_size: int) -> tuple[Any, Any]:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train = v2.Compose(
        [
            v2.RandomResizedCrop(input_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            v2.ToImage(),
            v2.ToDtype(dtype=__import__("torch").float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    development = v2.Compose(
        [
            v2.Resize(input_size + 64, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(input_size),
            v2.ToImage(),
            v2.ToDtype(dtype=__import__("torch").float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train, development
