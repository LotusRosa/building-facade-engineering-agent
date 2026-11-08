"""Read-only package/environment check before Ubuntu or Windows UI testing.

No installation, model import, CUDA query, inference, training or weight loading.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REQUIRED=['README.md','README_EN.md','SUPPORTED_ENVIRONMENTS.md','LICENSE','THIRD_PARTY_NOTICES.md','CITATION.cff',
          'ACCUMULATING_GATE_PROTOCOL.md','AGENT_COMPLETE_GPU_HANDOFF.md','pyproject.toml','requirements.worker.txt',
          'run_linux.sh','start_ui.sh','setup_linux_gpu.sh','run_windows.ps1','start_ui.cmd','setup_windows_gpu.ps1',
          'facade_next/__main__.py','facade_next/cli.py','facade_next/conversation.py','facade_next/intake.py',
          'facade_next/service.py','facade_next/store.py','facade_next/server.py','facade_next/worker.py','facade_next/gpu.py',
          'facade_next/platform_support.py','facade_next/contracts.py','facade_next/presets.json','facade_next/worker_manifest.json',
          'facade_next/static/index.html','facade_next/static/style.css','facade_next/static/app.js','facade_next/static/chat.js',
          'facade_next/vendor/toolkit_methods.py','facade_next/vendor/ewc.py','facade_next/vendor/legacy_engine.py',
          'templates/labels.csv','templates/labels.xlsx','templates/manifest.csv',
          'docs/SERVER_VALIDATION.md','docs/UBUNTU_FIRST_TEST_20261101.md','docs/GITHUB_BASELINE_FILES.json','docs/GITHUB_PARITY.md',
          'scripts/server_acceptance.py','tests/test_conversation.py','tests/test_workflow.py','tests/test_scheduler.py',
          'facade_next/llm.py','facade_next/llm_chat.py','facade_next/static/llm.js','llm.example.env','docs/LLM_SETUP.md',
          'tests/test_llm.py','tests/test_llm_http.py','tests/test_llm_ui.js','docs/DATA_PREPARATION.md',
          'facade_next/batch_intake.py','facade_next/batch_flow.py','tests/test_batch_dialogue.py','tests/test_batch_ui.js','docs/BATCH_DIALOGUE_20261101.md',
          'facade_next/static/workspace.js','facade_next/names.py','tests/test_workspace_ui.js','tests/test_initialization.py','docs/SIDEBAR_CHAT_DESIGN_20261101.md',
          'scripts/check_data_archives.py','docs/UBUNTU_DELIVERY_20261102.md','tests/test_intake_order.py','tests/test_data_archives.py']


REQUIRED += ['facade_next/environment.py','facade_next/environment_probe.py','facade_next/static/environment.js','tests/test_environment.py','tests/test_environment_http.py','tests/test_environment_ui.js','docs/FIRST_RUN_ENVIRONMENT.md']


REQUIRED += ['facade_next/progress.py','tests/test_progress.py','tests/test_progress_ui.js','tests/test_local_results.py','tests/test_result_queries_ui.js','docs/DEV13_FRESH_TEST.md']

REQUIRED += ['facade_next/environment_download.py','tests/test_environment_download.py','tests/test_core_entries.py','tests/test_core_entries_ui.js','docs/DEV14_UPGRADE_CN.md']

REQUIRED += ['facade_next/static/history.js','facade_next/static/history.css','tests/test_history_ui.js','tests/test_history_http.py','tests/test_saved_case_probe.py','scripts/check_saved_case.py','docs/SERVER_SCREENSHOTS_CN.md']


REQUIRED = [name for name in REQUIRED if not name.startswith('docs/')]
REQUIRED += ['.gitattributes','.github/workflows/checks.yml','docs/DATA_PREPARATION.md','docs/FIRST_RUN_ENVIRONMENT.md','docs/LLM_SETUP.md','docs/UPGRADING.md','docs/BFD_ML_4K.md','docs/SERVER_VALIDATION.md','docs/SERVER_SCREENSHOTS_CN.md','docs/GITHUB_PARITY.md','docs/GITHUB_BASELINE_FILES.json','docs/SOURCE_PROVENANCE.json','docs/VALIDATION.md','scripts/prepare_bfd_ml_4k.py','tests/test_dataset_bridge.py','protocols/bfd_ml_4k/roles.csv','protocols/bfd_ml_4k/protocol.json']


def check(root):
    root=Path(root).resolve()
    errors=['Missing '+name for name in REQUIRED if not (root/name).is_file()]
    manifest_path=root/'RELEASE_MANIFEST.json'
    verified=False
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        for name,expected in manifest['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                errors.append('Release file missing/changed: '+name)
        verified=not errors
    for name in ('run_linux.sh','start_ui.sh','setup_linux_gpu.sh'):
        p=root/name
        if p.is_file() and (p.read_bytes().startswith(b'\xef\xbb\xbf') or b'\r\n' in p.read_bytes()):
            errors.append('Bash script must use LF without BOM: '+name)
    packages={}
    for name in ('torch','torchvision','numpy','Pillow'):
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]='MISSING (GPU Worker only; UI still usable)'
    return dict(status='passed' if not errors else 'failed',root=str(root),python=platform.python_version(),
                platform=platform.platform(),required_files=len(REQUIRED),release_manifest_present=manifest_path.is_file(),
                manifest_hashes_verified=verified,packages=packages,errors=errors,
                model_imported=any(n in sys.modules for n in ('torch','torchvision')),automatic_install=False,
                gpu_validation='not performed; run explicit gpu-check/Worker only when ready')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output')
    args=parser.parse_args()
    report=check(args.root)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        Path(args.output).write_text(text,encoding='utf-8')
    print(text)
    if report['status']!='passed' or report['model_imported']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
