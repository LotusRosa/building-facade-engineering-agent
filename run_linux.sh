#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python -m facade_agent --check
exec python -m facade_agent --host "${FACADE_AGENT_HOST:-127.0.0.1}" --port "${FACADE_AGENT_PORT:-8000}"
