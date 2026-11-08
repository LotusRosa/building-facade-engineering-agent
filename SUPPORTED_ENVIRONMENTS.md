# Windows / Ubuntu

The browser UI uses Python 3.11+ and its standard library on 64-bit Windows and Ubuntu. Python 3.12 is the recorded reference. No Node.js or ML package is required to open the UI. Node is used only for optional frontend tests.

| Component | Scope |
|---|---|
| Browser UI, chat, SQLite, dataset tables | Same implementation on both operating systems |
| Windows launch | start_ui.cmd or run_windows.ps1 |
| Ubuntu launch | bash run_linux.sh or bash start_ui.sh |
| New training environment | Explicit first-run browser wizard; one shared host folder |
| Existing environment | Attach through wizard or select the interpreter in launch scripts |
| Training | NVIDIA CUDA with native BF16; one Worker per GPU |

The create-environment wizard currently selects torch 2.11.0 and torchvision 0.26.0 from the CUDA 12.8 wheel index, plus NumPy and Pillow. It checks CUDA/native BF16 rather than automatically solving every GPU/driver combination. NVIDIA driver installation is outside this setup. An existing compatible environment can be attached without replacing its installed packages. API floors in requirements.worker.txt do not imply that every package/OS combination is validated.

Recorded Ubuntu initial and Standard adaptation runs used Python 3.12.3, torch 2.11.0+cu128, torchvision 0.26.0+cu128 and one RTX 4090 per job on a dual-card host. The current interface/package was checked locally without model execution. See [validation scope](docs/VALIDATION.md); the full Windows GPU and all-method matrix is not established by these checks.

## Launch

```powershell
.\run_windows.ps1 -Python "D:\your_environment\python.exe"
```

```bash
FACADE_PYTHON=/path/to/environment/python bash run_linux.sh
```

The setup_windows_gpu.ps1 and setup_linux_gpu.sh helpers inspect the selected interpreter; the browser wizard performs requested dependency installation and weight preparation. Explicit GPU checks import GPU libraries but do not train a model.

Use `python -m facade_next serve --state runtime --ui-only` for a UI/history session without starting saved workers. Without that option, a ready saved profile can enable its configured workers. The service binds loopback only. Remote users can use their own SSH port forwarding.

## Paths and recovery

Data paths refer to the Agent host. Keep the original runtime, photo and checkpoint paths when updating on that host. Do not open one writable runtime from two services. Frozen absolute paths are not rewritten when moving between Windows and Ubuntu; see [upgrade instructions](docs/UPGRADING.md).

Initial training requires prepared ImageNet weights. Missing weights produce an explicit error; training does not fall back to random initialization. The wizard can download selected weights after the user requests setup. No automatic CPU/precision/image-size fallback, OOM retry or model adoption is performed. `FACADE_DISABLE_MODEL_EXECUTION=1` disables model execution for a host/session.

DeepSeek, OpenAI and compatible LLM connections use the standard library and do not require a GPU. See [LLM setup](docs/LLM_SETUP.md).
