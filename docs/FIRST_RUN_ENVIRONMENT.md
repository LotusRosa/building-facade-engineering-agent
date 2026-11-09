# One training environment for all projects

[中文](FIRST_RUN_ENVIRONMENT_ZH.md)

Open the Agent with the host's Python 3.11+. The UI starts without PyTorch. The first-run wizard, also available under **Settings → Shared training environment**, can create an environment or attach an existing one.

1. Choose **Create an environment in a new folder** and specify a new/empty host folder, or choose **Use an existing environment** and identify the environment directory or its managed parent.
2. Select GPU indices such as `0` or `1`. Use `0,1` only when separate jobs should run on both cards; each GPU has one Worker.
3. Select which ImageNet initialization weights to prepare and start setup/checking. The log reports dependency installation, weight bytes, checksum verification, and failures.
4. When **Shared environment ready** appears, return to chat. Subsequent projects and rounds reuse that profile.

Example folders are `/home/test/FacadeAgentEnvironment` on Ubuntu and `D:\FacadeAgentEnvironment` on Windows. Environment, dataset, and project-state directories are distinct choices.

The create path builds `.venv`, installs the currently configured CUDA 12.8 PyTorch/TorchVision pair and NumPy/Pillow, and checks CUDA/native BF16. It does not install the NVIDIA driver or automatically choose an installer for every hardware/driver combination. See [supported environments](../SUPPORTED_ENVIRONMENTS.md).

If a network download fails, inspect the log, restore connectivity, and retry the same folder. Completed dependencies and verified caches are reused. Training jobs are not automatically rerun. Finish queued/running work before switching the shared environment.

The host profile defaults to `.facade-agent/environment.json` under the user's home. `--environment-config /absolute/path/environment.json` selects an isolated profile. For a history-only or UI session, add `--ui-only` to avoid starting Workers from a saved profile. Setup remains a separately requested operation.

On a fresh Ubuntu host, install `python3-venv` for the selected Python version before using the create-environment option. On Windows, use the 64-bit Python installation with venv/ensurepip support. Python 3.12 is the reference for the pinned CUDA wheels; see [first installation](FIRST_INSTALL.md). Initial training currently uses BF16 in this interactive Worker; the manuscript precision difference is documented in [paper correspondence](MANUSCRIPT_CORRESPONDENCE.md).
