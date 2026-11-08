#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${FACADE_PYTHON:-python3}"
# Attach/check an existing environment; never pip install or download assets.
"$PYTHON_BIN" -m facade_next check
if [[ "${1:-}" == "--gpu-check" ]]; then
  "$PYTHON_BIN" -m facade_next gpu-check --execute
fi
printf '%s\n' 'Existing environment retained. Missing dependencies: see SUPPORTED_ENVIRONMENTS.md. No installation performed.'
