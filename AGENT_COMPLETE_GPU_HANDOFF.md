# GPU handoff

Use README.md for chat-first operation and SUPPORTED_ENVIRONMENTS.md for Windows/Ubuntu launch. Worker entry: `python -m facade_next worker --state runtime --device cuda:0 --execute`; add `--job job_ID` to target one queued task. Queue service: `python -m facade_next serve --state runtime --execute-workers --gpus 0` (or0,1 only when desired).

Native BF16 NVIDIA CUDA required. Existing environment is retained. requirements.worker.txt declares API floors; facade_next/worker_manifest.json records the research reference environment and unverified new-adapter status. No automatic install/download, OOM retry, precision/image reduction, CPU training or adoption.

Initial: Train/Val/S0; maintenance: new Train/Current Gate, cumulative safety and existing Val attached. Use chat to configure only needed tools, confirm the summarized task, wait for actual results, compare and explicitly adopt/Hold with a reason. Checkpoint schema facade_next_v1 binds preprocessing, ordered classes, thresholds and source/role/parent hashes. Explicit legacy conversion is described in docs/SERVER_VALIDATION.md.

Current development performed only static, model-free workflow/browser and package checks on Windows. Ubuntu execution and all real GPU training/inference/weight conversion tests of the new adapter remain pending. No new training or human-time study was started by this handoff. See docs/LOCAL_VALIDATION.json for the actual local record.
