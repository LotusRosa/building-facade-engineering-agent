from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_agent import server


class FakeBundleImporter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes, bool]] = []

    def import_initial_bundle(self, *, project_id, bundle_path, actor_id, confirmed):
        self.calls.append(("initial", project_id, Path(bundle_path).read_bytes(), confirmed))
        return {"project_id": project_id, "dataset_id": "dataset-initial", "image_count": 7}

    def import_maintenance_bundle(self, *, batch_id, bundle_path, actor_id, confirmed):
        self.calls.append(("maintenance", batch_id, Path(bundle_path).read_bytes(), confirmed))
        return {"batch_id": batch_id, "dataset_id": "dataset-round", "image_count": 3}


class LabeledBundleHTTPTests(unittest.TestCase):
    def request(self, path: str, body: bytes) -> tuple[int, dict]:
        httpd = server.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=httpd.handle_request, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/zip", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        thread.join(timeout=5)
        httpd.server_close()
        return response.status, payload

    def test_streamed_route_dispatches_initial_and_maintenance_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = FakeBundleImporter()
            inbox = Path(temporary) / "inbox"
            with patch.object(server, "SAMPLE_DATASETS", fake), patch.object(
                server, "BUNDLE_INBOX", inbox
            ):
                status, initial = self.request(
                    "/api/labeled-bundles/import?target=initial&project_id=project-1&confirmed=true",
                    b"initial-zip",
                )
                maintenance_status, maintenance = self.request(
                    "/api/labeled-bundles/import?target=maintenance&batch_id=batch-1&confirmed=true",
                    b"maintenance-zip",
                )

            self.assertEqual(status, 201)
            self.assertEqual(initial["result"]["dataset_id"], "dataset-initial")
            self.assertEqual(maintenance_status, 201)
            self.assertEqual(maintenance["result"]["dataset_id"], "dataset-round")
            self.assertEqual(
                fake.calls,
                [
                    ("initial", "project-1", b"initial-zip", True),
                    ("maintenance", "batch-1", b"maintenance-zip", True),
                ],
            )
            self.assertEqual(list(inbox.glob("*")), [])

    def test_route_requires_explicit_confirmation_before_receiving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = FakeBundleImporter()
            with patch.object(server, "SAMPLE_DATASETS", fake), patch.object(
                server, "BUNDLE_INBOX", Path(temporary) / "inbox"
            ):
                status, payload = self.request(
                    "/api/labeled-bundles/import?target=initial&project_id=project-1",
                    b"not-received",
                )
            self.assertEqual(status, 400)
            self.assertIn("confirmation", payload["error"].lower())
            self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
