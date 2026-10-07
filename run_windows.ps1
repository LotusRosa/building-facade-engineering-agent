$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m facade_agent --check
python -m facade_agent --host 127.0.0.1 --port 8000
