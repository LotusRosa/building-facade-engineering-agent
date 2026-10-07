from __future__ import annotations

import csv
import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from facade_agent.release import build_evaluation_snapshot


def png_bytes(red: int, green: int, blue: int) -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    raw = bytes((0, red, green, blue))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class LinuxTrialEvaluationReleaseTests(unittest.TestCase):
    def test_builder_emits_disjoint_current_gate_and_core_safety_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            rows: list[dict[str, str]] = []
            signatures = (
                {"crack": "1", "spalling": "0", "hollow": "0", "no_defect": "0"},
                {"crack": "0", "spalling": "1", "hollow": "0", "no_defect": "0"},
                {"crack": "0", "spalling": "0", "hollow": "1", "no_defect": "0"},
                {"crack": "0", "spalling": "0", "hollow": "0", "no_defect": "1"},
                {"crack": "1", "spalling": "1", "hollow": "0", "no_defect": "0"},
            )
            image_number = 1
            for split in ("adaptation_a_gate", "core_safety"):
                for offset, signature in enumerate(signatures):
                    image_id = f"DFWI_{image_number:05d}"
                    payload = png_bytes(image_number, offset, 255 - image_number)
                    (images / f"{image_id}.png").write_bytes(payload)
                    rows.append(
                        {
                            "numeric_image_id": str(image_number),
                            "image_id": image_id,
                            "split": split,
                            "temporal_unit_id": f"T{image_number:04d}",
                            "capture_group": f"T{image_number:04d}",
                            **signature,
                        }
                    )
                    image_number += 1

            manifest = root / "formal.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            excluded_image = images / "DFWI_00005.png"
            excluded = root / "earlier-training.zip"
            with zipfile.ZipFile(excluded, "w") as archive:
                archive.writestr(
                    "checksums.json",
                    json.dumps(
                        {"images/DFWI_00005.png": hashlib.sha256(excluded_image.read_bytes()).hexdigest()}
                    ),
                )

            destination = root / "evaluation.zip"
            result = build_evaluation_snapshot(
                formal_manifest=manifest,
                images_dir=images,
                destination=destination,
                project_id="project_demo",
                batch_id="batch_demo",
                class_ids={
                    "crack": "project_demo_class_001",
                    "spalling": "project_demo_class_002",
                    "hollow": "project_demo_class_003",
                },
                current_gate_name="linux_round_1_gate",
                current_gate_size=4,
                core_safety_size=4,
                excluded_bundles=[excluded],
                seed=20261001,
            )

            self.assertEqual(result["image_count"], 8)
            self.assertEqual(result["split_counts"], {"current_gate": 4, "core_safety": 4})
            self.assertTrue(destination.is_file())
            with zipfile.ZipFile(destination) as archive:
                names = {item.filename for item in archive.infolist() if not item.is_dir()}
                evaluation_manifest = json.loads(archive.read("evaluation_manifest.json"))
                classes = json.loads(archive.read("classes.json"))
                labels = [
                    json.loads(line)
                    for line in archive.read("labels.jsonl").decode("utf-8").splitlines()
                ]
                checksums = json.loads(archive.read("checksums.json"))

                self.assertEqual(
                    evaluation_manifest,
                    {
                        "schema_version": 2,
                        "purpose": "deployment_decision_evidence",
                        "project_id": "project_demo",
                        "batch_id": "batch_demo",
                        "current_gate_name": "linux_round_1_gate",
                        "provided_splits": ["current_gate", "core_safety"],
                    },
                )
                self.assertEqual(
                    classes,
                    [
                        {"class_id": "project_demo_class_001", "display_name": "crack"},
                        {"class_id": "project_demo_class_002", "display_name": "spalling"},
                        {"class_id": "project_demo_class_003", "display_name": "hollow"},
                    ],
                )
                self.assertEqual(set(checksums), names - {"checksums.json"})
                for name, digest in checksums.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

            self.assertNotIn(hashlib.sha256(excluded_image.read_bytes()).hexdigest(), {
                row["image_sha256"] for row in labels
            })
            for split in ("current_gate", "core_safety"):
                split_rows = [row for row in labels if row["split"] == split]
                self.assertEqual(len(split_rows), 4)
                supported = {class_id for row in split_rows for class_id in row["class_ids"]}
                self.assertEqual(
                    supported,
                    {
                        "project_demo_class_001",
                        "project_demo_class_002",
                        "project_demo_class_003",
                    },
                )


if __name__ == "__main__":
    unittest.main()
