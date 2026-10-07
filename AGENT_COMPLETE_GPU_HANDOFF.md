# Building-Facade Engineering Agent 2.0 — GPU handoff

The conversation-first governance loop and all four real Worker pipelines are implemented. The advanced workbench is available from the lower-right settings drawer. No production metric or checkpoint is embedded in the source tree.

## Supported baseline

- NVIDIA CUDA compute capability 7.0 or newer
- at least 5 GiB reported VRAM; 6 GiB or more recommended
- one visible GPU and one Worker process
- Python 3.12 and the exact locked dependencies

RTX 20-series cards use FP16 with gradient scaling. Newer cards use BF16 when PyTorch reports native support. Input size remains fixed at 768 on every GPU.

## Install and validate

Windows:

```powershell
Set-Location F:\agent
.\setup_windows_gpu.ps1 -Python D:\path\to\python.exe -CudaWheel cu124
```

Linux:

```bash
cd /path/to/agent
PYTHON_BIN=python3 CUDA_WHEEL=cu124 ./setup_linux_gpu.sh
```

The setup installs the locked torchvision environment, downloads the checksum-verified ConvNeXt-Tiny initialization after explicit confirmation, prints Worker diagnostics, and runs a single-GPU smoke check.

## RTX 4090 acceptance

1. Require a passing setup and single-GPU smoke result.
2. Confirm all four pipeline readiness entries are true.
3. Complete a small real Initial Champion run and registration.
   - Before project creation, choose an empty writable models-only folder on the target workstation.
   - After registration, confirm `<model_id>/checkpoint.<ext>` and `<model_id>/manifest.json` are the only files created for that model.
   - Recompute the checkpoint and manifest SHA-256 values and compare them with the database record.
4. Complete one maintenance batch through Failure Discovery and expert review.
5. Train and register one Challenger.
   - Confirm the Challenger is a second model directory under the same locked project root.
   - Confirm its manifest identifies the parent Champion, source batch, taxonomy hash, thresholds, training profile, and checkpoint hash.
6. Evaluate on two disjoint holdouts and obtain verified paired evidence.
7. Exercise Retain and Promote in separate disposable projects.
8. Verify the audit chain and archive logs plus artifact hashes.

The project model folder must not contain datasets, Gate snapshots, training ZIPs, Worker results, logs, API keys, LLM settings, or the ConvNeXt-Tiny pretrained cache. ConvNeXt-Tiny initialization remains part of the environment configuration and may use the standard torchvision cache outside the project model folder.

