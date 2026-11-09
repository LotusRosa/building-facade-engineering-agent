# Source-package contents

[中文](PACKAGE_CONTENTS_ZH.md)

| Component | Files / directories |
|---|---|
| Product introduction and commands | README.md, README_EN.md, README_ZH.md |
| Windows/Ubuntu launch | start_ui.cmd, run_windows.ps1, run_linux.sh, start_ui.sh |
| Browser service, chat and project state | facade_next/server.py, conversation.py, service.py, store.py, static/ |
| Training queue and GPU Worker | facade_next/worker.py, gpu.py, vendor/, worker_manifest.json |
| Frozen model/task settings | facade_next/contracts.py, presets.json |
| Data import and annotations | facade_next/data.py, intake.py, batch_intake.py, templates/ |
| Shared environment setup | facade_next/environment.py, environment_probe.py, environment_download.py |
| Fixed public-data preparation | protocols/bfd_ml_4k/, scripts/prepare_bfd_ml_4k.py |
| User instructions | [Guide index](USER_GUIDE_INDEX.md) |
| License, authors and citation | LICENSE, CITATION.cff, THIRD_PARTY_NOTICES.md |
| Verification | scripts/check_release.py, tests/, .github/workflows/checks.yml, RELEASE_MANIFEST.json |

The source package excludes dataset photos, pretrained weights, installed environments, runtime databases and private API keys. The release manifest binds packaged source-file hashes. Training-core provenance is recorded in SOURCE_PROVENANCE.json. Validation claims and GPU limits are documented in [VALIDATION.md](VALIDATION.md).
