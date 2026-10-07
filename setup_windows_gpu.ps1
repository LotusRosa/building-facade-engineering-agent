param(
  [string]$Python = "python",
  [ValidateSet("cu118", "cu124")][string]$CudaWheel = "cu124"
)
$ErrorActionPreference = "Stop"
$env:USE_LIBUV = "0"
& $Python -m pip install --upgrade pip
& $Python -m pip install --index-url "https://download.pytorch.org/whl/$CudaWheel" "torch==2.6.0" "torchvision==0.21.0"
& $Python -m pip install "numpy==1.26.4" "scikit-learn==1.5.1" "Pillow==10.4.0"
& $Python -m pip install -e $PSScriptRoot
$report = & $Python -m facade_training_worker.diagnostics --json --required-gpus 1
$report
$gpuCount = (& $Python -c "import torch; print(torch.cuda.device_count())").Trim()
if ([int]$gpuCount -lt 1) { throw "No supported CUDA GPU is visible." }
& $Python -m facade_training_worker.prepare_assets --download
& $Python -m facade_training_worker.launcher --nproc 1 --module facade_training_worker.ddp_smoke
