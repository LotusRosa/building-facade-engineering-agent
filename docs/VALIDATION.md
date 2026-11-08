# Validation scope

Local checks on 2026-11-08 use Windows, Python 3.12.7 and a model-import guard. They cover standard-library workflow, HTTP/CSRF, datasets, shared-environment orchestration with simulated subprocesses, frontend behavior, historical snapshot reads and package integrity. No local training, inference, weight loading, GPU query or dependency installation is performed.

After the fixed-role dataset bridge, 165 Python tests ran: 164 passed and one Linux-only test was skipped on Windows. Ten frontend suites passed 58 grouped behavior checks. Package checks verified all required files and manifest hashes. A successful CI run, when available, supplies a separate Windows/Ubuntu result; the presence of a CI file alone is not a passing result.

Nine training-core files were byte-compared against the previous inspected implementation and remained unchanged. Historical user-provided Ubuntu records establish successful initial and Standard Round-A execution on the recorded ConvNeXt-Tiny/RTX4090 case. They do not establish a complete GPU compatibility matrix for all backbones, methods or Windows machines.

The saved-task panel reads existing frozen job configuration, result receipts and recorded decisions. Its frontend/HTTP checks passed; final PNG export acceptance remains a target-server check. No synthetic record or current empty-project view is supplied as a paper experiment screenshot.

For BFD-ML-4K, all 4,000 public image IDs and labels were matched to the frozen allocation, with no missing IDs, label mismatches or cross-role frozen groups. Metadata revision and SHA-256 are pinned in protocols/bfd_ml_4k/protocol.json. Whole image blobs were not downloaded for this metadata audit. The preparation tool verifies the actual selected files when run against a user's downloaded dataset.

The reported timing comparison is one guided single-operator case: initial active time approximately 17 versus 6 minutes and Round-A approximately 11 versus 2 minutes. The raw total reduction is about 71%. Initial time includes environment operations; manual Round-A includes shared-parent preparation, which was not timed separately. Treat this as an observed case rather than a general user-study estimate.

Interface labels map to the paper as follows: Core = initial modeling; Standard · Adaptation = FT; historical distillation = KD; hard-example reweighting = LWS; classifier-head update = Head-only. Core and Standard are stage-specific entries sharing the appropriate standard training implementation, rather than nine independent optimization algorithms.
