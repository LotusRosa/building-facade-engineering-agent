#!/usr/bin/env bash
set -euo pipefail
export USE_LIBUV="${USE_LIBUV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL" 'torch==2.6.0' 'torchvision==0.21.0'
"$PYTHON_BIN" -m pip install 'numpy==1.26.4' 'scikit-learn==1.5.1' 'Pillow==10.4.0'
"$PYTHON_BIN" -m pip install -e "$(cd "$(dirname "$0")" && pwd)"
"$PYTHON_BIN" -m facade_training_worker.diagnostics --json --required-gpus 1
GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
test "$GPU_COUNT" -ge 1
"$PYTHON_BIN" -m facade_training_worker.prepare_assets --download
"$PYTHON_BIN" -m facade_training_worker.launcher --nproc 1 --module facade_training_worker.ddp_smoke
