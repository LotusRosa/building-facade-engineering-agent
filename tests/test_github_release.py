from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from facade_agent import release


class GitHubReleaseTests(unittest.TestCase):
    def test_builder_creates_clean_runnable_folder_and_linux_zip(self) -> None:
        self.assertTrue(
            hasattr(release, "build_github_release"),
            "The GitHub release builder has not been implemented.",
        )
        builder = getattr(release, "build_github_release")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = root / "source"
            (agent / "facade_agent").mkdir(parents=True)
            (agent / "facade_training_worker").mkdir()
            (agent / "tests").mkdir()
            (agent / "facade_agent" / "__init__.py").write_text("", encoding="utf-8")
            (agent / "facade_agent" / "__main__.py").write_text("", encoding="utf-8")
            (agent / "facade_training_worker" / "__init__.py").write_text("", encoding="utf-8")
            (agent / "tests" / "test_smoke.py").write_text("def test_smoke(): pass\n", encoding="utf-8")
            (agent / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
            (agent / "README.md").write_text("fixture\n", encoding="utf-8")
            (agent / "CITATION.cff").write_text("cff-version: 1.2.0\n", encoding="utf-8")
            (agent / "docs" / "assets").mkdir(parents=True)
            (agent / "docs" / "assets" / "example.png").write_bytes(b"public-figure")
            (agent / "docs" / "private-notes.txt").write_text("not for release", encoding="utf-8")
            (agent / "LICENSE").write_text("research license\n", encoding="utf-8")
            (agent / "THIRD_PARTY_NOTICES.md").write_text(
                "third-party notices\n",
                encoding="utf-8",
            )
            (agent / ".gitignore").write_text("data/\n", encoding="utf-8")
            (agent / "run_linux.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (agent / "setup_linux_gpu.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (agent / "run_windows.ps1").write_text("python -m facade_agent\n", encoding="utf-8")
            (agent / "setup_windows_gpu.ps1").write_text("python -m pip --version\n", encoding="utf-8")

            (agent / "data").mkdir()
            (agent / "data" / "private.sqlite3").write_bytes(b"private")
            (agent / ".venv").mkdir()
            (agent / ".venv" / "secret.bin").write_bytes(b"environment")
            (agent / "facade_agent" / "__pycache__").mkdir()
            (agent / "facade_agent" / "__pycache__" / "app.pyc").write_bytes(b"cache")

            result = builder(agent_root=agent, output_dir=root / "github-ready")

            release_dir = Path(result["release_dir"])
            archive_path = Path(result["archive"])
            checksum_path = Path(result["checksum_file"])
            self.assertTrue((release_dir / "facade_agent" / "__main__.py").is_file())
            self.assertTrue((release_dir / "facade_training_worker" / "__init__.py").is_file())
            self.assertTrue((release_dir / "tests" / "test_smoke.py").is_file())
            self.assertTrue((release_dir / ".gitignore").is_file())
            self.assertTrue((release_dir / "CITATION.cff").is_file())
            self.assertTrue((release_dir / "docs" / "assets" / "example.png").is_file())
            self.assertFalse((release_dir / "docs" / "private-notes.txt").exists())
            self.assertEqual(
                (release_dir / "LICENSE").read_text(encoding="utf-8"),
                "research license\n",
            )
            self.assertEqual(
                (release_dir / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
                "third-party notices\n",
            )
            self.assertFalse((release_dir / "data").exists())
            self.assertFalse((release_dir / ".venv").exists())
            self.assertFalse((release_dir / "facade_agent" / "__pycache__").exists())

            self.assertTrue(archive_path.is_file())
            expected_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            self.assertEqual(result["sha256"], expected_sha256)
            self.assertEqual(
                checksum_path.read_text(encoding="utf-8"),
                f"{expected_sha256}  {archive_path.name}\n",
            )
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                run_linux = archive.getinfo(
                    "building-facade-engineering-agent/run_linux.sh"
                )
                third_party_notices = archive.read(
                    "building-facade-engineering-agent/THIRD_PARTY_NOTICES.md"
                )
            self.assertIn(
                "building-facade-engineering-agent/facade_agent/__main__.py",
                names,
            )
            self.assertIn(
                "building-facade-engineering-agent/LICENSE",
                names,
            )
            self.assertEqual(
                third_party_notices.decode("utf-8").splitlines(),
                ["third-party notices"],
            )
            self.assertFalse(any("/data/" in name or "/.venv/" in name for name in names))
            self.assertTrue((run_linux.external_attr >> 16) & 0o111)


if __name__ == "__main__":
    unittest.main()
