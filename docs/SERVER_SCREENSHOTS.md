# Read existing experiments and export panels

[中文](SERVER_SCREENSHOTS_CN.md)

Keep the original host's runtime, data, and checkpoint paths. Verify the new program and inspect existing records:

```bash
python scripts/check_release.py
python scripts/check_saved_case.py --state /absolute/path/to/original/runtime
```

The read-only probe prefers completed ConvNeXt-Tiny, seed `20260921`, Standard adaptation tasks. If several match, inspect their output and optionally pass `--round round_ID`. If the intended task is absent, check the runtime path rather than creating replacement experiment records.

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

Open <http://127.0.0.1:8766> on the host, select the original project, and choose **Saved tasks**. Select the same round/candidate for task configuration, candidate evaluation, and version records.

The panels read frozen jobs, saved result receipts, and actual decisions. They do not retrain or adopt a model. Missing values remain not recorded. The export control produces native 2600-pixel-wide `agent_task_configuration.png`, `agent_candidate_evaluation.png`, and `agent_version_record.png` files.

Use real experiment records for paper panels. The README's user-supplied Ubuntu screenshots illustrate the recorded workflow and retain their original resolution; they are separate from these panel exports. Final panel-export acceptance remains a target-server check.
