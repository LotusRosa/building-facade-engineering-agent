param([string]$Python = $(if ($env:FACADE_PYTHON) {$env:FACADE_PYTHON} else {"python"}), [int]$Port=8765, [string]$State="runtime", [switch]$EnableGpuWorkers, [string]$Gpus="0")
$ErrorActionPreference="Stop"
Set-Location -LiteralPath $PSScriptRoot
& $Python -m facade_next check
if ($LASTEXITCODE -ne 0) {throw "Environment check failed"}
$runArgs=@("-m","facade_next","serve","--state",$State,"--port",$Port)
if ($EnableGpuWorkers) {$runArgs+=@("--execute-workers","--gpus",$Gpus)}
& $Python @runArgs
exit $LASTEXITCODE
