# Third-Party Notices

This document records direct third-party software, pretrained assets, and
optional external services used by Building-Facade Engineering Agent 2.1.0.
It was reviewed on 2026-10-05.

The standard source release does not bundle the packages, CUDA runtime, or
pretrained weights listed below. The optional environment-preparation scripts
install or download them separately into locations selected or controlled by
the user. Each third-party component remains subject to its own license and
terms. The project's `LICENSE` does not replace or restrict those independent
terms.

## Direct runtime dependencies

The GPU Worker uses the following exact versions from
`facade_training_worker/requirements.lock`:

| Component | Version | Purpose | Upstream license |
| --- | ---: | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch/tree/v2.6.0) | 2.6.0 | Model training and inference | [BSD 3-Clause](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE) |
| [TorchVision](https://github.com/pytorch/vision/tree/v0.21.0) | 0.21.0 | Vision model architecture, transforms, and weight loading | [BSD 3-Clause](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE) |
| [NumPy](https://github.com/numpy/numpy/tree/v1.26.4) | 1.26.4 | Numerical arrays and calculations | [BSD 3-Clause](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt) |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn/tree/1.5.1) | 1.5.1 | Clustering and evaluation utilities | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/COPYING) |
| [Pillow](https://github.com/python-pillow/Pillow/tree/10.4.0) | 10.4.0 | Image decoding and processing | [HPND](https://github.com/python-pillow/Pillow/blob/10.4.0/LICENSE) |

License identifiers in this table are summaries for convenience. The linked
upstream license text controls.

## Development and test dependency

Source-checkout verification uses
[pytest 7.4.4](https://github.com/pytest-dev/pytest/tree/7.4.4), distributed
under the [MIT License](https://github.com/pytest-dev/pytest/blob/7.4.4/LICENSE).
pytest is not required to run the installed Agent interface.

## Pretrained initialization

Initial Champion training can use TorchVision's
`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`. The weights are downloaded separately
to the user's cache only after environment preparation is confirmed; they are
not part of the repository or source release.

TorchVision warns that pretrained weights may carry terms or conditions derived
from the dataset used to train them. Review the
[TorchVision model documentation](https://docs.pytorch.org/vision/main/models.html)
and the applicable ImageNet terms before downloading or using the weights. This
project does not grant rights to those weights or to ImageNet data.

## Optional external model providers

The interface includes configurable endpoint presets for OpenAI, DeepSeek,
OpenRouter, LM Studio, and Ollama. No provider SDK or provider-owned source code
is bundled. If an engineer connects one of these services or tools, requests,
credentials, and submitted text are governed by that provider's current terms,
privacy policy, and deployment configuration. Provider names and trademarks
belong to their respective owners.

## Transitive dependencies and redistributed environments

Package installers may resolve additional transitive dependencies, each under
its own license. Anyone who redistributes a prebuilt virtual environment,
container image, wheel cache, CUDA runtime, pretrained weight file, or other
combined binary distribution must independently inventory those contents and
preserve all required copyright notices, license texts, source offers, and
attributions. Generating an SBOM and a complete license report is recommended
for every such distribution.

This notice is an engineering inventory, not legal advice. Review the actual
upstream terms and obtain legal advice before public or commercial
redistribution.
