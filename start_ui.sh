#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${FACADE_PYTHON:-python3}" -m facade_next serve --state runtime --port 8765 --open-browser "$@"
