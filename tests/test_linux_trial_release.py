from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from facade_agent.release import build_linux_trial_release, select_role_rows


class LinuxTrialReleaseTests(unittest.TestCase):
    def test_agent_source_builds_installable_wheel_with_both_packages(self) -> None:
        agent_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            wheel_dir = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                ],
                cwd=agent_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            wheels = list(wheel_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
            self.assertIn("facade_agent/__init__.py", names)
            self.assertIn("facade_training_worker/__init__.py", names)
            self.assertFalse(any(name.startswith(("data/", "models/")) for name in names))

    def test_linux_launcher_defaults_to_loopback_only(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run_linux.sh").read_text(encoding="utf-8")
        self.assertIn('${FACADE_AGENT_HOST:-127.0.0.1}', script)
        self.assertNotIn('${FACADE_AGENT_HOST:-0.0.0.0}', script)

    def test_selection_is_deterministic_stratified_and_source_specific(self) -> None:
        rows = []
        for index in range(24):
            rows.append(
                {
                    "numeric_image_id": str(index + 1),
                    "image_id": f"DFWI_{index + 1:05d}",
                    "split": "core_train" if index < 12 else (
                        "adaptation_a_discovery" if index < 20 else "final_test"
                    ),
                    "temporal_unit_id": f"T{index // 2:04d}",
                    "capture_group": f"T{index // 2:04d}",
                    "hollow": "1" if index % 4 == 0 else "0",
                    "spalling": "1" if index % 4 == 1 else "0",
                    "crack": "1" if index % 4 == 2 else "0",
                    "no_defect": "1" if index % 4 == 3 else "0",
                }
            )

        first = select_role_rows(
            rows,
            source_split="adaptation_a_discovery",
            sample_size=8,
            seed=20260930,
        )
        second = select_role_rows(
            list(reversed(rows)),
            source_split="adaptation_a_discovery",
            sample_size=8,
            seed=20260930,
        )

        self.assertEqual([row["image_id"] for row in first], [row["image_id"] for row in second])
        self.assertEqual({row["split"] for row in first}, {"adaptation_a_discovery"})
        signatures = {
            (row["hollow"], row["spalling"], row["crack"], row["no_defect"])
            for row in first
        }
        self.assertEqual(len(signatures), 4)

    def test_builder_excludes_runtime_data_and_emits_verified_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = root / "agent"
            (agent / "facade_agent" / "static").mkdir(parents=True)
            (agent / "facade_agent" / "static" / "app.js").write_text("ok", encoding="utf-8")
            (agent / "README.md").write_text("agent", encoding="utf-8")
            (agent / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
            (agent / "data").mkdir()
            (agent / "data" / "private.sqlite3").write_bytes(b"private")
            (agent / ".env").write_text("API_KEY=private", encoding="utf-8")
            (agent / "facade_agent" / "__pycache__").mkdir()
            (agent / "facade_agent" / "__pycache__" / "app.pyc").write_bytes(b"cache")

            images = root / "images"
            images.mkdir()
            rows = []
            for index in range(14):
                image_id = f"DFWI_{index + 1:05d}"
                (images / f"{image_id}.jpg").write_bytes(b"image-" + bytes([index]))
                if index < 6:
                    split = "core_train"
                    split_label = "Core Train"
                elif index < 11:
                    split = "adaptation_a_discovery"
                    split_label = "A Discovery"
                elif index < 13:
                    split = "adaptation_a_gate"
                    split_label = "A Gate"
                else:
                    split = "final_test"
                    split_label = "Final Test"
                rows.append(
                    {
                        "numeric_image_id": str(index + 1),
                        "image_id": image_id,
                        "split": split,
                        "split_label": split_label,
                        "capture_group": f"G{index:03d}",
                        "temporal_unit_id": f"T{index:03d}",
                        "hollow": "1" if index % 4 == 0 else "0",
                        "spalling": "1" if index % 4 == 1 else "0",
                        "crack": "1" if index % 4 == 2 else "0",
                        "no_defect": "1" if index % 4 == 3 else "0",
                    }
                )
            manifest = root / "formal.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            result = build_linux_trial_release(
                agent_root=agent,
                formal_manifest=manifest,
                images_dir=images,
                output_dir=root / "release",
                initial_size=4,
                round_size=3,
                seed=20260930,
            )

            with zipfile.ZipFile(result["agent_zip"]) as archive:
                names = set(archive.namelist())
            self.assertIn("building-facade-engineering-agent/README.md", names)
            self.assertFalse(any("data/" in name or "__pycache__" in name for name in names))
            self.assertNotIn("building-facade-engineering-agent/.env", names)

            with zipfile.ZipFile(result["initial_dataset_zip"]) as archive:
                initial_names = set(archive.namelist())
                initial_manifest = json.loads(archive.read("dataset_manifest.json"))
                initial_labels = [
                    json.loads(line)
                    for line in archive.read("labels.jsonl").decode("utf-8").splitlines()
                ]
            with zipfile.ZipFile(result["round1_dataset_zip"]) as archive:
                round_names = set(archive.namelist())
                round_manifest = json.loads(archive.read("dataset_manifest.json"))
                round_labels = [
                    json.loads(line)
                    for line in archive.read("labels.jsonl").decode("utf-8").splitlines()
                ]
            self.assertEqual(len([name for name in initial_names if name.startswith("images/")]), 4)
            self.assertEqual(len([name for name in round_names if name.startswith("images/")]), 3)
            self.assertEqual(initial_manifest["schema_version"], 2)
            self.assertEqual(initial_manifest["agent_role"], "initial_training")
            self.assertEqual(initial_manifest["source_split"], "core_train")
            self.assertEqual(round_manifest["agent_role"], "maintenance")
            self.assertEqual(round_manifest["source_split"], "adaptation_a_discovery")
            self.assertEqual({row["source_split"] for row in initial_labels}, {"core_train"})
            self.assertEqual(
                {row["source_split"] for row in round_labels},
                {"adaptation_a_discovery"},
            )
            self.assertTrue(
                {row["image_id"] for row in initial_labels}.isdisjoint(
                    {row["image_id"] for row in round_labels}
                )
            )
            self.assertTrue(Path(result["agent_sha256_file"]).is_file())
            self.assertTrue(Path(result["initial_dataset_sha256_file"]).is_file())
            self.assertTrue(Path(result["round1_dataset_sha256_file"]).is_file())
            release_manifest = json.loads(Path(result["release_manifest"]).read_text())
            self.assertEqual(
                set(release_manifest),
                {"agent", "initial_dataset", "round1_dataset", "trial_scope"},
            )
            self.assertIn("one-off", release_manifest["trial_scope"])


if __name__ == "__main__":
    unittest.main()
