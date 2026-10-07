from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import WORKER_PROTOCOL_VERSION
from .bundle import BundleValidationError, verify_bundle
from .runtime import collect_runtime_report, load_worker_manifest


@dataclass(frozen=True)
class WorkerContract:
    pipeline: str
    expected_kind: str | None
    required_members: tuple[str, ...]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: object
    backend: str
    dist: object


class SingleProcessDist:
    """Small distributed-compatible facade for the supported single-GPU mode."""

    def barrier(self) -> None:
        return None

    def gather_object(self, value: object, output: list[object] | None, dst: int = 0) -> None:
        if output is not None:
            output[:] = [value]

    def all_reduce(self, value: object) -> None:
        return None

    def broadcast_object_list(self, values: list[object], src: int = 0) -> None:
        return None


CONTRACTS = {
    "initial_champion": WorkerContract(
        "initial_champion",
        "initial_champion",
        ("classes.json", "labels.jsonl", "training_profile.json"),
    ),
    "champion_failure_discovery": WorkerContract(
        "champion_failure_discovery",
        "champion_failure_discovery",
        ("classes.json", "labels.jsonl", "champion.json", "champion/checkpoint.pt", "discovery_profile.json"),
    ),
    "challenger_update": WorkerContract(
        "challenger_update",
        "challenger_update",
        ("classes.json", "labels.jsonl", "training_profile.json", "parent_champion.json", "parent_champion/checkpoint.pt", "pool_manifest.json", "failure_review.json"),
    ),
    "champion_challenger_evaluation": WorkerContract(
        "champion_challenger_evaluation",
        None,
        ("evaluation_profile.json", "models.json", "models/champion.pt", "models/challenger.pt", "holdout/classes.json", "holdout/labels.jsonl"),
    ),
}


def _emit(event: str, **payload: object) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _parser(contract: WorkerContract) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Facade governed Worker entry: {contract.pipeline}")
    parser.add_argument("--protocol-version", type=int, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--validate-bundle-only",
        action="store_true",
        help="Verify the immutable bundle without starting CUDA, DDP, training, or inference.",
    )
    return parser


def run(contract: WorkerContract, argv: Sequence[str] | None = None) -> int:
    args = _parser(contract).parse_args(argv)
    if args.protocol_version != WORKER_PROTOCOL_VERSION:
        _emit("error", message=f"Unsupported Worker protocol version: {args.protocol_version}")
        return 2
    try:
        bundle = verify_bundle(
            args.bundle,
            required_members=contract.required_members,
            expected_kind=contract.expected_kind,
        )
    except (BundleValidationError, FileNotFoundError) as error:
        _emit("error", message=str(error))
        return 2
    if args.validate_bundle_only:
        _emit("validation", job_id=bundle.manifest["job_id"], pipeline=contract.pipeline, verified_members=len(bundle.names))
        return 0

    output = args.output.expanduser().resolve()
    if not output.is_dir():
        _emit("error", message="Managed output directory does not exist.")
        return 2
    report = collect_runtime_report(required_gpus=1, pipeline=contract.pipeline)
    if not report["ready"]:
        _emit("error", message="Locked GPU Worker runtime is not ready.", issues=report["issues"])
        return 2
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size < 1 or local_rank < 0 or local_rank >= world_size:
        _emit("error", message="Worker must be launched by the managed launcher with valid rank variables.")
        return 2

    manifest = load_worker_manifest()
    if manifest["pipelines"].get(contract.pipeline) is not True:
        _emit("error", message=f"Production pipeline is not enabled: {contract.pipeline}")
        return 3

    import torch
    import torch.distributed as dist

    if world_size != 1 or local_rank != 0:
        _emit("error", message="This release supports exactly one managed GPU worker process.")
        return 2
    torch.cuda.set_device(0)
    backend = "single-process"
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=world_size,
        device=torch.device("cuda", 0),
        backend=backend,
        dist=SingleProcessDist(),
    )
    try:
        _emit("log", message=f"Initialized single-GPU worker for {contract.pipeline}.", world_size=world_size)
        if contract.pipeline == "initial_champion":
            from .initial_pipeline import run_initial_champion

            result = run_initial_champion(
                bundle=bundle,
                output=output,
                run_id=args.run_id,
                context=context,
                emit=_emit,
            )
        elif contract.pipeline == "champion_failure_discovery":
            from .failure_pipeline import run_failure_discovery

            result = run_failure_discovery(
                bundle=bundle, output=output, run_id=args.run_id, context=context, emit=_emit
            )
        elif contract.pipeline == "challenger_update":
            from .challenger_pipeline import run_challenger_update

            result = run_challenger_update(
                bundle=bundle, output=output, run_id=args.run_id, context=context, emit=_emit
            )
        elif contract.pipeline == "champion_challenger_evaluation":
            from .evaluation_pipeline import run_champion_challenger_evaluation

            result = run_champion_challenger_evaluation(
                bundle=bundle, output=output, run_id=args.run_id, context=context, emit=_emit
            )
        else:
            _emit("error", message=f"Production pipeline implementation is unavailable: {contract.pipeline}")
            return 3
        if context.rank == 0:
            if result is None or not result.is_file():
                raise RuntimeError("Worker pipeline completed without a result bundle.")
            _emit("result", result_bundle_path=str(result.resolve()))
        context.dist.barrier()
        return 0
    except Exception as error:
        _emit("error", message=f"{type(error).__name__}: {error}")
        raise
    finally:
        return None
