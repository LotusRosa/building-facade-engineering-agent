from __future__ import annotations

import argparse
import json
from pathlib import Path

from .application.sample_dataset import import_sample_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a checksum-verified frozen Core Train sample into a new local project."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--project-name", default="Linux Core Train 200 smoke")
    parser.add_argument("--model-storage-root", required=True, type=Path)
    parser.add_argument(
        "--app-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--confirm-import-frozen-labels", action="store_true")
    args = parser.parse_args()
    result = import_sample_bundle(
        app_root=args.app_root,
        bundle_path=args.bundle,
        project_name=args.project_name,
        model_storage_root=args.model_storage_root,
        confirmed=args.confirm_import_frozen_labels,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
