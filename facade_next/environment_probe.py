"""Explicit subprocess probe/download helper; never constructs or loads a model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--weights', nargs='*', choices=['resnet50','convnext_tiny','swin_t'], default=[])
    args = parser.parse_args()
    if not args.execute or os.environ.get('FACADE_DISABLE_MODEL_EXECUTION')=='1':
        raise ValueError('Environment GPU checks require explicit execution and must not be disabled by host policy.')
    if sys.version_info < (3,11) or platform.system() not in ('Windows','Linux') or platform.machine().lower() not in ('amd64','x86_64'):
        raise ValueError('Use Python 3.11+ on 64-bit Windows or Ubuntu.')
    packages = {}
    for name, floor in [('torch',(2,6)),('torchvision',(0,21)),('numpy',(1,26)),('Pillow',(10,4))]:
        version = importlib.metadata.version(name)
        parts = re.match(r'(\d+)\.(\d+)', version)
        if not parts or tuple(map(int,parts.groups())) < floor:
            raise ValueError(name+' is too old: '+version)
        packages[name] = version
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch in this environment.')
    indices = [int(x) for x in args.gpus.split(',')]
    if any(i < 0 or i >= torch.cuda.device_count() for i in indices):
        raise ValueError('A selected GPU index is unavailable.')
    devices = []
    for i in indices:
        with torch.cuda.device(i):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
        if not supported:
            raise ValueError('GPU '+str(i)+' does not support native BF16 required by this training protocol.')
        devices.append(dict(index=i, name=torch.cuda.get_device_name(i), native_bf16=True))
    from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, Swin_T_Weights
    enums = dict(convnext_tiny=ConvNeXt_Tiny_Weights, resnet50=ResNet50_Weights, swin_t=Swin_T_Weights)
    cache = Path(torch.hub.get_dir())/'checkpoints'
    weights = []
    for backbone in args.weights:
        url = enums[backbone].DEFAULT.url
        filename = Path(urlparse(url).path).name
        prefix = filename.rsplit('-',1)[1].split('.')[0]
        path = cache/filename
        if __package__:
            from .environment_download import download_verified
        else:
            from environment_download import download_verified
        def progress(value):
            print('FACADE_WEIGHT_PROGRESS '+json.dumps(dict(backbone=backbone,**value)),file=sys.stderr,flush=True)
        print('Checking/downloading ImageNet weights: '+backbone, file=sys.stderr, flush=True)
        digest=download_verified(url,path,prefix,progress=progress)
        weights.append(dict(backbone=backbone, path=str(path), sha256=digest))
    print(json.dumps(dict(ok=True, python=platform.python_version(), executable=sys.executable,
        packages=packages, cuda=True, devices=devices, weights=weights,
        torch_home=str(Path(torch.hub.get_dir()).absolute().parent))))


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('Environment check failed: '+str(e), file=sys.stderr)
        raise SystemExit(1)
