# building-facade-engineering-agent

A chat-driven local Agent for initial modeling, adaptation, evaluation and version management of building-facade image classifiers. The same browser interface runs on Windows and Ubuntu.

## Start

Python 3.11+ is required. The UI needs no third-party packages:

```bash
python -m facade_next serve --state runtime --port 8765
```

Open <http://127.0.0.1:8765>. Windows users can run `start_ui.cmd`; Ubuntu users can run `bash run_linux.sh`. The first-run wizard creates or checks one shared training environment on the Agent host for all projects. Use `--ui-only` to view the interface without starting saved workers.

## Workflow

Create a project in chat and supply pre-split data. Initial modeling requires Train, independent Val and initial acceptance S0. Later adaptation rounds require new Train and an independent Current Gate; new Val is optional and existing Val can be reused. The Agent does not split data automatically.

**Core · Initial modeling** starts from ImageNet weights. **Standard · Adaptation** and the optional adaptation strategies continue from the active model. Review the actual data roles and configuration before confirming execution. Training progress and results appear in chat. Adopting a candidate or retaining the parent requires a reason and confirmation.

The project sidebar contains parameters, data, results, saved tasks and version records. Language and model connection controls are in the top-right corner. DeepSeek, OpenAI and compatible APIs are optional; keys remain in service memory.

Companion data: [BFD-ML-4K and fixed-protocol preparation](docs/BFD_ML_4K.md). The published corpus and current experimental allocation are linked by explicit image identities, labels and checksums.

## Existing experiments

Open **Saved tasks** inside the original project to inspect frozen task settings, saved evaluations, recorded costs and actual historical decisions. No retraining or repeat adoption is needed. Export each panel as a native 2600-pixel-wide PNG: `agent_task_configuration.png`, `agent_candidate_evaluation.png`, and `agent_version_record.png`. Missing information stays marked as not recorded.

Preserve the original host's runtime directory, data and checkpoint paths. See [server instructions](docs/SERVER_SCREENSHOTS_CN.md), [data preparation](docs/DATA_PREPARATION.md), [environment setup](docs/FIRST_RUN_ENVIRONMENT.md), [LLM setup](docs/LLM_SETUP.md), and [supported environments](SUPPORTED_ENVIRONMENTS.md).

## Check

```bash
python scripts/check_release.py
```

This checks required files and package hashes without loading a model. See [validation scope](docs/VALIDATION.md). The internal module remains `facade_next` for command and saved-record compatibility. The software package contains no datasets, weights or runtime databases.
