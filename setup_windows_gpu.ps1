param([string]$Python = $(if ($env:FACADE_PYTHON) {$env:FACADE_PYTHON} else {"python"}), [switch]$GpuCheck)
$ErrorActionPreference="Stop"
Set-Location -LiteralPath $PSScriptRoot
# Environment attachment only. Never installs, downgrades or downloads assets.
& $Python -m facade_next check
if ($LASTEXITCODE -ne 0) {throw "Environment check failed"}
if ($GpuCheck) {& $Python -m facade_next gpu-check --execute; if ($LASTEXITCODE -ne 0) {throw "CUDA/BF16 check failed"}}
Write-Output "Existing environment retained. Missing dependencies: see SUPPORTED_ENVIRONMENTS.md. No installation performed."
