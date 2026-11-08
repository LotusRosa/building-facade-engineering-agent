# Third-party software and assets

The browser service and CSV/XLSX reader use Python's standard library. The optional GPU Worker uses separately installed [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) (BSD-3-Clause), [TorchVision](https://github.com/pytorch/vision/blob/main/LICENSE) (BSD-3-Clause), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) (BSD-3-Clause) and [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE) (HPND). Their own licenses govern those components. Packages, CUDA dependencies and model weights are not bundled in this source release.

The environment wizard installs dependencies and prepares selected TorchVision ImageNet weights only after the user requests setup. Pretrained assets and source datasets retain their own terms; see [TorchVision model documentation](https://docs.pytorch.org/vision/stable/models.html). This software license grants no additional ImageNet or dataset rights.

Algorithm/model helpers originate from this project's research tools; hashes and provenance are recorded in docs/SOURCE_PROVENANCE.json. The first-party Academic Research License and author attribution are retained in LICENSE. Citation metadata points to the canonical repository.

The companion [BFD-ML-4K dataset](https://huggingface.co/datasets/LotusRosa/BFD-ML-4K) has a separate academic-research license. Its images are not included. The shipped role/index mappings correspond to that dataset and the authors' frozen protocol; see docs/BFD_ML_4K.md for revision and integrity information. Optional Hugging Face CLI downloads use its own terms and dependencies.

No LLM provider SDK or clustering package is required. Optional LLM requests are governed by the chosen service's terms. Node.js is used only for frontend development checks, and GitHub Actions run model-free checks. Any separately redistributed binary environment requires its own dependency and license inventory.
