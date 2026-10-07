from __future__ import annotations

import argparse
from typing import Sequence

from .modeling import verify_cached_imagenet_weights


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicitly provision locked Worker model assets.")
    parser.add_argument("--download", action="store_true", help="Allow downloading the official torchvision checkpoint.")
    args = parser.parse_args(argv)
    before = verify_cached_imagenet_weights()
    if before["valid"]:
        print(f"READY {before['path']} {before['sha256']}")
        return 0
    if not args.download:
        print(f"MISSING {before['path']} (rerun with --download to explicitly provision it)")
        return 2
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    after = verify_cached_imagenet_weights()
    if not after["valid"]:
        raise RuntimeError("Downloaded ConvNeXt-Tiny initialization did not pass SHA-256 verification.")
    print(f"READY {after['path']} {after['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
