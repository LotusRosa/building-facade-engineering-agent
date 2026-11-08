#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${FACADE_PYTHON:-python3}"
"$PYTHON_BIN" -m facade_next check
exec "$PYTHON_BIN" -m facade_next serve --state "${FACADE_STATE:-runtime}" --port "${FACADE_PORT:-8765}" "$@"
