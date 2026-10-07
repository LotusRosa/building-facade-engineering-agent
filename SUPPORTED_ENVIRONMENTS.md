# Supported environments

Facade Engineering Agent targets computers with NVIDIA GPUs only.

## Supported operating systems

- Windows 10 or Windows 11, 64-bit
- 64-bit Linux distributions capable of running a supported NVIDIA driver

macOS, WSL-only GPU setups, AMD GPUs, Intel GPUs, and CPU-only training are outside the supported training scope.

## Release behavior

The Agent package stays small and does not bundle CUDA, PyTorch, or model weights. On first use it detects the host runtime and, only after an explicit engineer confirmation, runs the platform setup script and downloads pinned, checksum-verified weights into user-owned storage.

On first launch the Agent performs deterministic checks for:

1. supported operating system and architecture;
2. NVIDIA driver and visible GPU inventory;
3. sufficient GPU memory, RAM, and disk space;
4. the Agent-managed CUDA-enabled PyTorch environment;
5. a short GPU smoke test before training is enabled.

The complete CUDA Toolkit is not an application prerequisite. The managed training environment supplies its version-locked runtime dependencies. A compatible NVIDIA driver remains a host prerequisite and is never installed silently.

If checks fail, the interface remains available for diagnosis, but all training actions are blocked. The Agent must show the failing check and remediation guidance rather than falling back to CPU training.

## Single-GPU policy

The production release uses one visible NVIDIA GPU and one worker process. If several GPUs are present, the runtime selects the first available device unless the engineer explicitly chooses another device.

At least one NVIDIA GPU with compute capability 7.0 and 5 GiB reported VRAM is required. Training, screening, and paired evaluation all use the same single-GPU contract.
