from __future__ import annotations

import json
import os


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    try:
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2
        if float(value.item()) != expected:
            raise RuntimeError("Distributed CUDA all-reduce returned an unexpected value.")
        if rank == 0:
            print(json.dumps({"passed": True, "backend": backend, "world_size": world_size, "sum": expected}))
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
