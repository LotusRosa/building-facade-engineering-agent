from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_agent.application.environment import EnvironmentManager
from facade_agent.storage import Store


def inventory(_: Path) -> dict:
    return {
        "platform": {"system": "Test", "release": "1", "machine": "x64"},
        "python": {"version": "3.12", "executable": "python", "supported": True},
        "cpu": {"logical_cores": 16},
        "memory": {"total_bytes": 32 * 1024**3},
        "disk": {"root": "test", "free_bytes": 100, "total_bytes": 200},
        "nvidia": {"available": True, "gpus": [{"index": 0, "name": "GPU", "memory_total_mib": 24564, "driver_version": "1"}], "error": None},
        "torch": {"installed": True, "version": "test", "cuda_available": True, "cuda_version": "test", "device_count": 1, "error": None},
        "capabilities": {"interface_ready": True, "supported_platform": True, "local_gpu_training_ready": True},
        "warnings": [],
    }


class ComputeEnvironmentTests(unittest.TestCase):
    def test_auto_detection_and_persisted_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = Store(root / "agent.sqlite3")
            manager = EnvironmentManager(store, root, inventory)
            self.assertEqual(manager.status()["settings"]["effective_mode"], "local_gpu")
            configured = manager.configure({"mode": "auto", "gpu_devices": [], "cpu_threads": 4})
            self.assertEqual(configured["settings"]["effective_mode"], "local_gpu")
            self.assertEqual(EnvironmentManager(store, root, inventory).status()["settings"]["cpu_threads"], 4)
            self.assertTrue(store.verify_audit_chain()["valid"])

    def test_unavailable_gpu_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = EnvironmentManager(Store(root / "agent.sqlite3"), root, inventory)
            with self.assertRaises(ValueError):
                manager.configure({"mode": "local_gpu", "gpu_devices": [1], "cpu_threads": 0})


if __name__ == "__main__":
    unittest.main()


