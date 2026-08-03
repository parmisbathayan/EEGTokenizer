from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from src.download import DownloadItem, download_one, load_inventory, select_task_files


class FakeResponse(BytesIO):
    def __init__(self, value, status=200):
        super().__init__(value)
        self.status = status


def remote(task, subject, size=10):
    folders = {"NR": "task2 - NR", "TSR": "task3 - TSR"}
    name = f"results{subject}_{task}.mat"
    return {
        "name": name,
        "path": f"/{folders[task]}/Matlab files/{name}",
        "size_bytes": size,
        "download_url": f"https://download.example/{name}",
    }


class DownloadTests(unittest.TestCase):
    def test_inventory_node_and_locked_subjects_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(json.dumps({"node": "wrong", "files": []}))
            with self.assertRaisesRegex(ValueError, "expected ZuCo 1.0"):
                load_inventory(path)

        inventory = {"files": [remote("NR", "ZAB"), remote("NR", "ZDM")]}
        selected = select_task_files(
            inventory, "NR", "/tmp/target", expected_subjects=("ZAB", "ZDM")
        )
        self.assertEqual([item.subject for item in selected], ["ZAB", "ZDM"])
        with self.assertRaisesRegex(ValueError, "locked set"):
            select_task_files(
                inventory, "NR", "/tmp/target", expected_subjects=("ZAB", "ZDM", "ZDN")
            )

    def test_fresh_download_and_existing_file_reuse(self):
        value = b"abcdefghij"
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resultsZAB_NR.mat"
            item = DownloadItem(
                "NR", "ZAB", destination.name, "/remote", len(value),
                "https://download.example/file", str(destination),
            )

            def opener(request, timeout):
                calls.append((request.get_header("Range"), timeout))
                return FakeResponse(value)

            self.assertEqual(download_one(item, opener=opener), "downloaded")
            self.assertEqual(destination.read_bytes(), value)
            self.assertEqual(download_one(item, opener=opener), "existing")
        self.assertEqual(calls, [(None, 120)])

    def test_partial_download_resumes_with_range(self):
        value = b"abcdefghij"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resultsZAB_NR.mat"
            destination.with_suffix(".mat.part").write_bytes(value[:4])
            item = DownloadItem(
                "NR", "ZAB", destination.name, "/remote", len(value),
                "https://download.example/file", str(destination),
            )

            def opener(request, timeout):
                self.assertEqual(request.get_header("Range"), "bytes=4-")
                return FakeResponse(value[4:], status=206)

            self.assertEqual(download_one(item, opener=opener), "downloaded")
            self.assertEqual(destination.read_bytes(), value)

    def test_full_response_restarts_a_partial_file(self):
        value = b"abcdefghij"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resultsZAB_TSR.mat"
            destination.with_suffix(".mat.part").write_bytes(value[:4])
            item = DownloadItem(
                "TSR", "ZAB", destination.name, "/remote", len(value),
                "https://download.example/file", str(destination),
            )

            def opener(request, timeout):
                return FakeResponse(value, status=200)

            self.assertEqual(download_one(item, opener=opener), "downloaded")
            self.assertEqual(destination.read_bytes(), value)

    def test_wrong_existing_size_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resultsZAB_NR.mat"
            destination.write_bytes(b"bad")
            item = DownloadItem(
                "NR", "ZAB", destination.name, "/remote", 10,
                "https://download.example/file", str(destination),
            )
            with self.assertRaisesRegex(ValueError, "wrong size"):
                download_one(item, opener=lambda *_args, **_kwargs: None)
            self.assertEqual(destination.read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
