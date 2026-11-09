# Target-host validation

[中文](SERVER_VALIDATION_ZH.md)

Review [supported environments](../SUPPORTED_ENVIRONMENTS.md) first. The UI can configure a shared training environment after opening. To inspect an existing project, follow [upgrade instructions](UPGRADING.md).

```bash
python scripts/check_release.py
python -m unittest discover -s tests
python -m facade_next serve --state runtime_validation --port 8765 --ui-only
```

These commands do not train a model. Once ready on the intended GPU host, explicitly check CUDA/BF16 using that training environment's interpreter:

```bash
/path/to/environment/python -m facade_next gpu-check --execute
```

For a new validation project, supply pre-split Train/Val/S0 and select Core plus one backbone. Confirm the actual data, seed and settings before execution. Initial modeling uses prepared ImageNet weights; missing caches produce a concrete error instead of random initialization.

After inspecting initial evaluation, record and confirm adoption. Then submit new Train/Current Gate for an adaptation task. Choose Standard or another required strategy, review its parent, Val and cumulative acceptance, and explicitly start the work. Each Worker uses one GPU. Do not execute the same writable state through another service.

Verify the actual evidence: roles remain disjoint; the initial checkpoint is selected by Val; adaptation follows its explicit budget and calibration protocol; parent/candidate share acceptance images; the recorded choice matches the active checkpoint and thresholds; a completed gate accumulates only after a human decision. Preserve failure artifacts and do not edit frozen receipts to hide an unsuccessful task.

Existing facade_next results can be read directly. Earlier 768-crop models are not automatically equivalent to the current 1280 full-image model. Supported conversion requires the original training, Val and acceptance provenance and the explicit arguments shown by `python -m facade_next import-model --help`.

A full GPU/backbone/strategy matrix is a separate validation exercise. Model-free UI/package checks do not establish it, and no local model execution is required to update documentation or inspect saved tasks.
