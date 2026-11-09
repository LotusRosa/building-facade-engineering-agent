# Correspondence with the earlier public repository

[中文](GITHUB_PARITY_ZH.md)

Baseline: [dcd90cb551ec3e59b61e8188d998a898437a709b](https://github.com/LotusRosa/building-facade-engineering-agent/tree/dcd90cb551ec3e59b61e8188d998a898437a709b). Original file hashes are retained in `GITHUB_BASELINE_FILES.json`. Detailed README subject coverage is in [README_COVERAGE.md](README_COVERAGE.md).

| Earlier file or capability | Current counterpart |
|---|---|
| README / supported environments | Full English README, Chinese entry, runtime scope and shared-environment guide |
| Windows/Ubuntu launch/setup scripts | Same-named launchers; browser-first environment setup; inspect-only helpers documented accurately |
| Independent training Worker | facade_next/worker.py, gpu.py, vendor helpers, requirements.worker.txt and worker_manifest.json |
| Model/data/result protocols | contracts.py, presets.json, frozen roles, task/evaluation/model contracts |
| Photo and annotation import/templates | data.py, intake.py, batch_intake.py and CSV/XLSX templates |
| Published Hugging Face data | Pinned revision, verified fixed-role mapping and prepare_bfd_ml_4k.py |
| Chat, Task Panel and project state | Chat/sidebar interface, shared transactional state, SQLite and queue |
| Comparison, adoption and rollback | Same-round parent/candidate evidence, engineer confirmation/reason and version history |
| Existing project inspection | Same-host facade_next runtime reuse; no automatic earlier facade_agent schema migration |
| Required failure discovery/expert slice review | Intentionally replaced by direct engineer-selected adaptation tasks |
| License, authors, citation and third-party notices | Original LICENSE unchanged; restored full README sections and aligned citation metadata |
| Interface screenshots | Newer original Ubuntu validation images with explicit provenance; old gallery retained in Git history |
| Source and package checks | Required-file/hash checks, model-free Python/frontend suites and Windows/Ubuntu CI |

The earlier implementation remains in Git history. The current default branch is updated through normal appended commits, without rewriting those records. Training-core files retain their verified identities; public documentation and naming describe the current workflow rather than claiming every earlier feature remains implemented.
